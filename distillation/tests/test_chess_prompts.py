#!/usr/bin/env python3
"""
Test script to find the optimal prompt for chess move evaluation.
Goal: Get the model to output ONLY the move string (e.g., "MoveA" or "MoveB")
to minimize logits for student model training.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
    python test_chess_prompts.py
"""

import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

torch.manual_seed(0)

# Configuration
model_path = "Qwen/Qwen3-30B-A3B"
traced_model_path = "/home/ubuntu/traced_model/Qwen3-30B-A3B/"
num_test_samples = 3  # Test with just a few samples

# Load a few samples from the dataset
print("Loading test samples from MATE_DATASET...")
dataset_stream = load_dataset("OutFlankShu/MATE_DATASET", split="train", streaming=True)
test_samples = []
for i, sample in enumerate(dataset_stream):
    if 'instruction' in sample and 'input' in sample and 'output' in sample:
        test_samples.append(sample)
    if len(test_samples) >= num_test_samples:
        break

print(f"Loaded {len(test_samples)} test samples\n")

# Load model and tokenizer
print("Loading compiled model...")
model = NeuronQwen3MoeForCausalLM(traced_model_path)
model.load(traced_model_path)
tokenizer = AutoTokenizer.from_pretrained(traced_model_path)
# Load generation config from original model path (not traced path)
generation_config = GenerationConfig.from_pretrained(model_path)
print("Model loaded!\n")

# ============================================================================
# PROMPT TEMPLATES TO TEST
# ============================================================================
# Edit these to test different approaches. The goal is to get output like:
# "MoveA" or "MoveB" with nothing else.
# ============================================================================

PROMPT_TEMPLATES = [
    {
        "name": "Original (verbose)",
        "system": lambda instruction: instruction,
        "user": lambda input_text: input_text,
    },
    {
        "name": "Strict single-word instruction",
        "system": lambda instruction: "You are a chess move evaluator. Respond with ONLY 'MoveA' or 'MoveB'. No explanation, no punctuation, no additional text.",
        "user": lambda input_text: input_text,
    },
    {
        "name": "Format-constrained",
        "system": lambda instruction: "You are a chess evaluator. Your response must be exactly one of these two strings: MoveA or MoveB",
        "user": lambda input_text: f"{input_text}\n\nAnswer (MoveA or MoveB):",
    },
    {
        "name": "JSON format",
        "system": lambda instruction: "You are a chess evaluator. Respond with valid JSON containing only a 'move' field with value 'MoveA' or 'MoveB'.",
        "user": lambda input_text: f"{input_text}\n\nJSON:",
    },
    {
        "name": "Direct question",
        "system": lambda instruction: "Answer with only 'MoveA' or 'MoveB'.",
        "user": lambda input_text: f"{input_text}\n\nWhich move is better?",
    },
    {
        "name": "Classification task",
        "system": lambda instruction: "Classify the better move. Output format: MoveA or MoveB",
        "user": lambda input_text: input_text,
    },
    {
        "name": "Minimal system prompt",
        "system": lambda instruction: "Output: MoveA or MoveB only.",
        "user": lambda input_text: input_text,
    },
    {
        "name": "Example-based (few-shot style)",
        "system": lambda instruction: "You evaluate chess moves. Examples:\nInput: [position] MoveA vs MoveB\nOutput: MoveA\n\nNow evaluate:",
        "user": lambda input_text: input_text,
    },
]

# ============================================================================
# TEST FUNCTION
# ============================================================================

def create_conversation(system_msg, user_msg):
    """Create conversation format for the model."""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def test_prompt_template(template, samples):
    """Test a prompt template with the given samples."""
    print(f"\n{'='*80}")
    print(f"Testing: {template['name']}")
    print(f"{'='*80}")
    
    results = []
    
    for idx, sample in enumerate(samples):
        try:
            # Create conversation with this template
            system_msg = template['system'](sample['instruction'])
            user_msg = template['user'](sample['input'])
            conversation = create_conversation(system_msg, user_msg)
            
            # Format and tokenize
            formatted_chat = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(formatted_chat, padding=True, return_tensors="pt")
            
            # Generate
            generation_model = HuggingFaceGenerationAdapter(model)
            outputs = generation_model.generate(
                inputs.input_ids,
                generation_config=generation_config,
                attention_mask=inputs.attention_mask,
                max_length=model.config.neuron_config.max_length,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True
            )
            
            # Extract ONLY the newly generated tokens (not the input)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[0]
            new_tokens = generated_tokens[input_length:]  # Only the new tokens
            
            generated_text = tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Count tokens generated (excluding input)
            output_length = len(new_tokens)
            
            # Count unique logits (to see how constrained the output is)
            num_logit_positions = len(outputs.scores) if outputs.scores else 0
            
            results.append({
                'sample_idx': idx,
                'expected': sample['output'],
                'generated': generated_text,
                'output_tokens': output_length,
                'logit_positions': num_logit_positions,
            })
            
            print(f"\nSample {idx + 1}:")
            print(f"  Expected:  {sample['output']}")
            print(f"  Generated: {generated_text}")
            print(f"  Output tokens: {output_length}")
            print(f"  Logit positions: {num_logit_positions}")
            
        except Exception as e:
            print(f"\nSample {idx + 1}: ERROR - {str(e)}")
            results.append({
                'sample_idx': idx,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'-'*80}")
    print(f"Summary for '{template['name']}':")
    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_tokens = sum(r['output_tokens'] for r in successful) / len(successful)
        avg_logits = sum(r['logit_positions'] for r in successful) / len(successful)
        print(f"  Average output tokens: {avg_tokens:.1f}")
        print(f"  Average logit positions: {avg_logits:.1f}")
        print(f"  Success rate: {len(successful)}/{len(results)}")
        
        # Check if outputs match expected format
        correct_format = sum(
            1 for r in successful 
            if r['generated'].strip() in ['MoveA', 'MoveB'] or 
               r['expected'] in r['generated']
        )
        print(f"  Correct format: {correct_format}/{len(successful)}")
    else:
        print(f"  All samples failed!")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHESS PROMPT OPTIMIZATION TEST")
    print("="*80)
    print(f"\nTesting {len(PROMPT_TEMPLATES)} prompt templates with {len(test_samples)} samples each")
    print("\nGoal: Minimize output tokens and logit positions while maintaining accuracy")
    
    all_results = {}
    
    for template in PROMPT_TEMPLATES:
        results = test_prompt_template(template, test_samples)
        all_results[template['name']] = results
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    for name, results in all_results.items():
        successful = [r for r in results if 'error' not in r]
        if successful:
            avg_tokens = sum(r['output_tokens'] for r in successful) / len(successful)
            avg_logits = sum(r['logit_positions'] for r in successful) / len(successful)
            print(f"\n{name}:")
            print(f"  Avg tokens: {avg_tokens:.1f}, Avg logits: {avg_logits:.1f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nBased on the results above, choose the template with:")
    print("1. Lowest average output tokens")
    print("2. Lowest average logit positions")
    print("3. Highest correct format rate")
    print("\nUpdate the notebook's create_conversation() function with the winning template.")
    print("\nIf none of these work well, try these additional approaches:")
    print("  - Add explicit examples in the system prompt")
    print("  - Use temperature=0 for more deterministic output")
    print("  - Add stop tokens to halt generation early")
    print("  - Use a different chat template format")
    print("  - Post-process to extract just the move from longer output")
