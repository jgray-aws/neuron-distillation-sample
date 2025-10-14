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

from dataclasses import dataclass, field
from typing import Optional, Dict, Union
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, DataCollatorForLanguageModeling

from torch.utils.data import Dataset

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM as TrainingNeuronModelForCausalLM
from optimum.neuron import NeuronModelForCausalLM

from util import print_tensor_shapes

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
        # Do something with num_items_in_batch
            pass
        
        print_tensor_shapes(inputs)

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
# Data Loading and Preprocessing Function
# =============================================================================
# NOTE: this section can be adapted to load any dataset you want.
class FixedShapeSentimentDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=2048, model_vocab_size=128256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_vocab_size = model_vocab_size
        
        # Load data from JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the prompt with fixed length
        encoded = self.tokenizer.encode_plus(
            item['prompt'],
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
            label_tensor = torch.cat([label_tensor, torch.full((pad_length,), -100)])  # Use -100 for ignored positions
            
        # Process teacher logits with fixed shape
        logits = []
        for token_info in item['response']['token_logits']:
            token_logits = torch.zeros(self.model_vocab_size)
            if isinstance(token_info['indices'], int):
                token_info['indices'] = [token_info['indices']]
                
            for idx, logit in zip(token_info['indices'], token_info['logits']):
                if idx < self.model_vocab_size:  # Ensure index is within bounds
                    token_logits[idx] = logit
            logits.append(token_logits)
        
        # Pad logits to exactly max_length
        while len(logits) < self.max_length:
            logits.append(torch.zeros(self.model_vocab_size))
        logits = logits[:self.max_length]  # Truncate if necessary
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor,
            'teacher_logits': torch.stack(logits)
        }

# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, training_args):
    # NOTE: Models with a custom modeling implementation (like Llama, Qwen3, Granite, etc) need a `TrainingNeuronConfig`
    # to be passed. This configuration defines modeling customization and distributed training parameters.
    # It is automatically created when using `NeuronTrainingArguments`.
    student_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    student_model = TrainingNeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance and to be able to use larger sequence lengths.
        # attn_implementation="flash_attention_2"
    )
    
    model_vocab_size = student_model.config.vocab_size
    
    # Prepare dataset
    train_dataset = FixedShapeSentimentDataset('output.json', tokenizer, model_vocab_size=model_vocab_size)
    
    # # Training arguments
    # training_args = DistillationTrainingArguments(
    #     output_dir="./distillation_output",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=1,  # Process one sample at a time
    #     gradient_accumulation_steps=4,
    #     warmup_steps=500,
    #     learning_rate=5e-5,
    #     logging_dir="./logs",
    #     logging_steps=100,
    #     save_strategy="epoch",
    #     temperature=4.0,
    #     alpha=0.7,
    #     # Add memory optimization arguments
    #     gradient_checkpointing=True,
    #     optim="adamw_torch",
    #     dataloader_num_workers=0,
    #     dataloader_pin_memory=True,
    #     remove_unused_columns=False
    # )

    # The NeuronSFTTrainer will use `format_dolly` to format the dataset and `lora_config` to apply LoRA on the
    # model.
    distillation_trainer = KnowledgeDistillationTrainer(
        args=training_args,
        model=student_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    distillation_trainer.train()
    trainer.save_model("./final_distilled_model")

# =============================================================================
# Defining the script-specific arguments
# =============================================================================
@dataclass
class ScriptArguments:
    temperature: Optional[float] = field(default=4.0),
    alpha: Optional[float] = field(default=0.7)
    model_id: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    
    )


# =============================================================================
# Main Function
# =============================================================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Set chat template for Llama 3.1 format
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'user' %}"
        "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'assistant' %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    )

    print(training_args.trn_config)

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        training_args=training_args,
    )