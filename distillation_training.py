import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Union

class SentimentDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=256, model_vocab_size=128256):
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
        
        # Tokenize the prompt
        encoded = self.tokenizer.encode_plus(
            item['prompt'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Use the generated text as labels
        label = self.tokenizer.encode(
            item['response']['generated_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Process teacher logits
        logits = []
        for token_info in item['response']['token_logits']:
            token_logits = torch.zeros(self.model_vocab_size)
            if isinstance(token_info['indices'], int):
                token_info['indices'] = [token_info['indices']]
                
            for idx, logit in zip(token_info['indices'], token_info['logits']):
                token_logits[idx] = logit
            logits.append(token_logits)
        
        # Pad logits to max_length
        while len(logits) < self.max_length:
            logits.append(torch.zeros(self.model_vocab_size))
        
        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'labels': torch.tensor(label),
            'teacher_logits': torch.stack(logits[:self.max_length])
        }

class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, temperature=4.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student outputs
        student_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        student_logits = student_outputs.logits
        
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

@dataclass
class DistillationTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=4.0)
    alpha: Optional[float] = field(default=0.7)

def main():
    # Initialize tokenizer and models
    student_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16
    )
    model_vocab_size = student_model.config.vocab_size
    
    # Prepare dataset
    train_dataset = SentimentDataset('output_student.json', tokenizer, model_vocab_size=model_vocab_size)
    
    # Training arguments
    training_args = DistillationTrainingArguments(
        output_dir="./distillation_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Process one sample at a time
        gradient_accumulation_steps=4,
        warmup_steps=500,
        learning_rate=5e-5,
        fp16=True,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        temperature=4.0,
        alpha=0.7,
        # Add memory optimization arguments
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=1,
        dataloader_pin_memory=True
    )
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        temperature=training_args.temperature,
        alpha=training_args.alpha
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model("./final_distilled_model")

if __name__ == "__main__":
    main()
