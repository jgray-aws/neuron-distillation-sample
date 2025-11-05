#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Knowledge Distillation Training Script for Chess Move Evaluation

Trains a student model (Qwen3-0.6B) to classify chess moves using
knowledge distillation from a teacher model (Qwen3-30B-A3B).
"""

from dataclasses import dataclass, field
from typing import Optional
import json

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, HfArgumentParser, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM as TrainingNeuronModelForCausalLM


# =============================================================================
# Distillation Trainer
# =============================================================================
class KnowledgeDistillationTrainer(NeuronTrainer):
    def __init__(self, temperature=4.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute knowledge distillation loss combining:
        - Hard loss: standard cross-entropy with true labels
        - Soft loss: KL divergence between teacher and student logits
        """
        if num_items_in_batch is not None:
            pass
        
        student_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        student_logits = student_outputs.logits
        
        inputs = {k: v.to('xla') if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Hard loss (standard language modeling loss)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            inputs['labels'].view(-1)
        )
        
        # Soft loss (knowledge distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(inputs['teacher_logits'] / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


# =============================================================================
# Chess Dataset
# =============================================================================
class ChessMoveDataset(Dataset):
    """Dataset for chess move evaluation with knowledge distillation."""
    
    def __init__(self, json_file, tokenizer, max_length=2048, model_vocab_size=128256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_vocab_size = model_vocab_size
        
        # Load data from JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter out samples with errors
        self.data = [item for item in self.data if 'error' not in item]
        print(f"Loaded {len(self.data)} valid chess samples")
            
    def create_conversation(self, instruction, input_text):
        """Create conversation format for chess move evaluation."""
        return [
            {
                "role": "system",
                "content": "Classify the better move. Output format: MoveA or MoveB"
            },
            {
                "role": "user",
                "content": input_text
            },
        ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Create conversation format
        conversation = self.create_conversation(item['instruction'], item['input'])
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the prompt with fixed length
        encoded = self.tokenizer.encode_plus(
            formatted_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create fixed-size tensors
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        
        # Ensure input tensors are exactly max_length
        if input_ids.size(0) < self.max_length:
            pad_length = self.max_length - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
        
        # Process labels with fixed length
        label = self.tokenizer.encode(
            item['response']['generated_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        label_tensor = torch.tensor(label)
        
        # Ensure label tensor is exactly max_length
        if label_tensor.size(0) < self.max_length:
            pad_length = self.max_length - label_tensor.size(0)
            label_tensor = torch.cat([label_tensor, torch.full((pad_length,), -100)])
            
        # Process teacher logits with fixed shape
        logits = []
        for token_info in item['response']['token_logits']:
            token_logits = torch.zeros(self.model_vocab_size)
            if isinstance(token_info['indices'], int):
                token_info['indices'] = [token_info['indices']]
                
            for idx, logit in zip(token_info['indices'], token_info['logits']):
                if idx < self.model_vocab_size:
                    token_logits[idx] = logit
            logits.append(token_logits)
        
        # Pad logits to exactly max_length
        while len(logits) < self.max_length:
            logits.append(torch.zeros(self.model_vocab_size))
        logits = logits[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor,
            'teacher_logits': torch.stack(logits)
        }


# =============================================================================
# Training Function
# =============================================================================
def train(script_args, training_args):
    """Main training function."""
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Load chess dataset
    print(f"Loading chess dataset from {script_args.dataset_path}")
    train_dataset = ChessMoveDataset(
        script_args.dataset_path,
        tokenizer,
        max_length=script_args.max_length,
        model_vocab_size=script_args.model_vocab_size
    )
    
    # Load model
    print(f"Loading model: {script_args.model_id}")
    model = TrainingNeuronModelForCausalLM.from_pretrained(
        script_args.model_id,
        trn_config,
        torch_dtype=dtype
    )
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        temperature=script_args.temperature,
        alpha=script_args.alpha,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {script_args.output_model_path}")
    trainer.save_model(script_args.output_model_path)
    tokenizer.save_pretrained(script_args.output_model_path)
    
    print("Training complete!")


# =============================================================================
# Script Arguments
# =============================================================================
@dataclass
class ScriptArguments:
    """Arguments for the training script."""
    
    model_id: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Student model to train"}
    )
    dataset_path: str = field(
        default="data/chess_output.json",
        metadata={"help": "Path to chess dataset JSON file"}
    )
    output_model_path: str = field(
        default="./final_chess_model",
        metadata={"help": "Path to save the final trained model"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    model_vocab_size: int = field(
        default=128256,
        metadata={"help": "Model vocabulary size"}
    )
    temperature: float = field(
        default=4.0,
        metadata={"help": "Temperature for knowledge distillation"}
    )
    alpha: float = field(
        default=0.7,
        metadata={"help": "Weight for soft loss (1-alpha for hard loss)"}
    )


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    train(script_args, training_args)
