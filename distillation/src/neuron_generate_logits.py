import torch
import json
import argparse

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

model_path = "/home/ubuntu/models--Qwen--Qwen3-30B-A3B"
traced_model_path = "/home/ubuntu/traced_model/Qwen3-30B-A3B/"

torch.manual_seed(0)


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

def parse_args():
    parser = argparse.ArgumentParser(description='Process text dataset through Qwen3 MoE model')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset.txt',
        help='Path to input dataset text file (default: dataset.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Path to output JSON file (default: output.json)'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    neuron_config = MoENeuronConfig(
        tp_degree=8,
        batch_size=1,
        max_context_length=128,
        seq_len=1024,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        output_scores=True,
        output_logits=True
    )
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronQwen3MoeForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    model = NeuronQwen3MoeForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)
    
    results = []
    with open(args.dataset, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                try:
                    input_text = create_conversation(line.strip())
                    formatted_chat = tokenizer.apply_chat_template(
                        input_text,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                    )
                    inputs = tokenizer(formatted_chat, padding=True, return_tensors="pt")
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
                    print(outputs)
                    generated_tokens = outputs.sequences[0]
                    token_logits = outputs.scores
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    print(generated_text)
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
                    print(token_logits_list)
                    results.append({
                        'prompt': line.strip(),
                        'response': token_logits_list
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
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processing complete! Processed {len(results)} prompts. Results written to output.json")
