import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<|finetune_right_pad_id|>"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def create_conversation(sample):
    system_message = (
        "You are a sentiment classifier. You take input strings and return the sentiment of POSITIVE, NEGATIVE, or NEUTRAL. Only return the sentiment."
    )
    return [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user", 
                "content": sample
            },
        ]


def process_text(input_text):
    input_text = create_conversation(input_text)

    formatted_chat = tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)
    # Tokenize input
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)
    
    # print(inputs)
    # Generate with output logits
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    # Get generated tokens and scores
    generated_tokens = outputs.sequences[0]
    token_logits = outputs.scores
    
    # Convert tokens to text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Process logits for each token
    token_logits_list = []
    for logits in token_logits:        
        # Get non-zero probabilities and their indices
        # Filter out -inf values from logits
        finite_mask = torch.isfinite(logits[0])
        finite_indices = torch.nonzero(finite_mask).squeeze().tolist()
        finite_logits = logits[0][finite_mask]
        
        
        # Create dictionary with non-zero probabilities and their indices
        token_info = {
            'indices': finite_indices,
            'logits': finite_logits.tolist()
        }
        token_logits_list.append(token_info)

    print(generated_text)
    
    return {
        'generated_text': generated_text,
        'token_logits': token_logits_list
    }

# Read input file and process each line
results = []
with open('../data/dataset.txt', 'r') as f:
    for line in f:
        # Skip empty lines
        if line.strip():
            try:
                result = process_text(line.strip())
                results.append({
                    'prompt': line.strip(),
                    'response': result
                })
                # print(f"Processed prompt: {line[:50]}...")  # Progress indicator
            except Exception as e:
                print(f"Error processing prompt: {line[:50]}...")
                print(f"Error message: {str(e)}")
                results.append({
                    'prompt': line.strip(),
                    'error': str(e)
                })

# Write results to output file
with open('../data/output.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processing complete! Processed {len(results)} prompts. Results written to output.json")
