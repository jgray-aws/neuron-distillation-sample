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
        Only computed on non-masked tokens (where labels != -100)
        """
        if num_items_in_batch is not None:
            pass
        
        student_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        student_logits = student_outputs.logits
        
        # Move tensors to XLA device
        device = student_logits.device
        labels = inputs['labels'].to(device)
        
        # Shift logits and labels for next-token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Create mask for valid positions (where labels != -100)
        mask = (shift_labels != -100).float()
        
        # Hard loss (standard cross-entropy with true labels)
        hard_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        hard_loss = (hard_loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
        
        # Soft loss (knowledge distillation) - reconstruct sparse teacher logits on-the-fly
        batch_size, seq_len, vocab_size = shift_logits.shape
        
        # Only create teacher logits tensor for positions with labels (much more memory efficient)
        # We'll compute KL loss only on non-masked positions
        teacher_logits_sparse_batch = inputs['teacher_logits_sparse']
        num_prompt_tokens_batch = inputs['num_prompt_tokens']
        num_full_tokens_batch = inputs['num_full_tokens']
        
        # Initialize teacher logits with very small values (log-space)
        shift_teacher_logits = torch.full_like(shift_logits, -1e10)
        
        # Fill in sparse teacher logits for each sample in batch
        for batch_idx in range(batch_size):
            num_prompt = num_prompt_tokens_batch[batch_idx]
            num_full = num_full_tokens_batch[batch_idx]
            num_answer = num_full - num_prompt
            sparse_logits = teacher_logits_sparse_batch[batch_idx]
            
            for token_idx, sparse_logit in enumerate(sparse_logits):
                if token_idx < num_answer:
                    # Position in the shifted sequence (accounting for the shift)
                    position = num_prompt + token_idx
                    if position < seq_len:
                        for vocab_idx, logit_val in zip(sparse_logit['indices'], sparse_logit['logits']):
                            if vocab_idx < vocab_size:
                                shift_teacher_logits[batch_idx, position, vocab_idx] = logit_val
        
        # Compute soft loss with temperature scaling
        student_soft = F.log_softmax(shift_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
        
        # KL divergence per position
        kl_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='none'
        ).sum(dim=-1)
        
        # Apply mask and average
        soft_loss = (kl_loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
        soft_loss = soft_loss * (self.temperature ** 2)
        
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
        
        # Pre-process all data to avoid CPU bottleneck during training
        print("Pre-processing dataset (this may take a minute)...")
        self._preprocess_data()
            
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

    def _preprocess_data(self):
        """Pre-process all samples to avoid CPU bottleneck during training."""
        self.processed_samples = []
        
        for idx, item in enumerate(self.data):
            # Create conversation format
            conversation = self.create_conversation(item['instruction'], item['input'])
            
            # Apply chat template to get the prompt
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Extract answer from teacher's response
            teacher_response = item['response']['generated_text']
            if 'assistant' in teacher_response:
                answer_part = teacher_response.split('assistant')[-1].strip()
                if '<think>' in answer_part:
                    answer_part = answer_part.split('</think>')[-1].strip()
                answer = answer_part
            else:
                answer = item['expected_output']
            
            # Build full sequence
            full_text = formatted_prompt + answer
            
            # Tokenize
            prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # Truncate if needed
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[:self.max_length]
            
            # Create tensors
            input_ids = torch.tensor(full_tokens, dtype=torch.long)
            labels = torch.full((len(full_tokens),), -100, dtype=torch.long)
            if len(prompt_tokens) < len(full_tokens):
                labels[len(prompt_tokens):] = input_ids[len(prompt_tokens):]
            attention_mask = torch.ones(len(full_tokens), dtype=torch.long)
            
            # Pad to max_length
            pad_length = self.max_length - len(full_tokens)
            if pad_length > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])
            
            # Store sparse teacher logits (only non-zero values) to save memory
            teacher_logits_sparse = []
            for token_info in item['response']['token_logits']:
                indices = token_info['indices'] if isinstance(token_info['indices'], list) else [token_info['indices']]
                logits = token_info['logits'] if isinstance(token_info['logits'], list) else [token_info['logits']]
                teacher_logits_sparse.append({
                    'indices': indices,
                    'logits': logits
                })
            
            self.processed_samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'teacher_logits_sparse': teacher_logits_sparse,
                'num_prompt_tokens': len(prompt_tokens),
                'num_full_tokens': len(full_tokens)
            })
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(self.data)} samples...")
        
        print(f"Pre-processing complete! {len(self.processed_samples)} samples ready.")
    
    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, idx):
        """Get pre-processed sample - keep teacher logits sparse to save memory."""
        sample = self.processed_samples[idx]
        
        return {
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'labels': sample['labels'],
            'teacher_logits_sparse': sample['teacher_logits_sparse'],
            'num_prompt_tokens': sample['num_prompt_tokens'],
            'num_full_tokens': sample['num_full_tokens']
        }


# =============================================================================
# Training Function
# =============================================================================
def train(script_args, training_args):
    """Main training function."""
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    
    # Custom data collator that handles teacher logits
    class ChessDataCollator:
        """Custom collator that keeps teacher logits sparse to save memory."""
        
        def __init__(self, max_length, vocab_size):
            self.max_length = max_length
            self.vocab_size = vocab_size
        
        def __call__(self, features):
            # Handle list of lists (from NeuronTrainer batching)
            if features and isinstance(features[0], list):
                features = [item for sublist in features for item in sublist]
            
            # Stack tensors into batch
            batch = {
                'input_ids': torch.stack([f['input_ids'] for f in features]),
                'attention_mask': torch.stack([f['attention_mask'] for f in features]),
                'labels': torch.stack([f['labels'] for f in features]),
                'teacher_logits_sparse': [f['teacher_logits_sparse'] for f in features],
                'num_prompt_tokens': [f['num_prompt_tokens'] for f in features],
                'num_full_tokens': [f['num_full_tokens'] for f in features]
            }
            return batch
    
    data_collator = ChessDataCollator(
        max_length=script_args.max_length,
        vocab_size=script_args.model_vocab_size
    )
    
    # Load chess dataset
    print(f"Loading chess dataset from {script_args.dataset_path}")
    train_dataset = ChessMoveDataset(
        script_args.dataset_path,
        tokenizer,
        max_length=script_args.max_length,
        model_vocab_size=script_args.model_vocab_size
    )
    
    # Debug: Print a sample to verify data processing
    print("\n" + "="*80)
    print("DEBUG: Sample data item")
    print("="*80)
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Teacher logits (sparse): {len(sample['teacher_logits_sparse'])} tokens")
    print(f"Number of non-masked label positions: {(sample['labels'] != -100).sum().item()}")
    print(f"Sample input (decoded): {tokenizer.decode(sample['input_ids'][:100])}")
    label_tokens = sample['labels'][sample['labels'] != -100]
    if len(label_tokens) > 0:
        print(f"Sample labels (decoded): {tokenizer.decode(label_tokens)}")
    
    # Test the collator
    print("\nTesting data collator with 2 samples...")
    test_batch = [train_dataset[0], train_dataset[1]]
    try:
        collated = data_collator(test_batch)
        print(f"Collated batch keys: {collated.keys()}")
        print(f"Collated input_ids shape: {collated['input_ids'].shape}")
        print(f"Collated teacher_logits_sparse: {len(collated['teacher_logits_sparse'])} samples")
        print("✓ Data collator test passed!")
    except Exception as e:
        print(f"✗ Data collator test failed: {e}")
        raise
    print("="*80 + "\n")
    
    # Load model
    print(f"Loading model: {script_args.model_id}")
    model = TrainingNeuronModelForCausalLM.from_pretrained(
        script_args.model_id,
        trn_config,
        torch_dtype=dtype
    )
    
    # Initialize trainer
    # Override the method that removes unused columns to keep teacher_logits
    class KnowledgeDistillationTrainerWithLogits(KnowledgeDistillationTrainer):
        def _remove_unused_columns(self, dataset, description):
            """Override to keep teacher_logits column."""
            # Don't remove any columns - we need teacher_logits
            return dataset
    
    # Disable automatic column removal by setting remove_unused_columns=False
    training_args.remove_unused_columns = False
    
    trainer = KnowledgeDistillationTrainerWithLogits(
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
    print(f"Temperature: {script_args.temperature}, Alpha: {script_args.alpha}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {script_args.output_model_path}")
    trainer.save_model(script_args.output_model_path)
    tokenizer.save_pretrained(script_args.output_model_path)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Model saved to: {script_args.output_model_path}")
    print("="*80)


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
