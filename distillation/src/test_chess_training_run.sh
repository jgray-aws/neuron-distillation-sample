#!/bin/bash
# Quick test of chess distillation training
# Runs just 5 steps to verify everything works

set -e

echo "================================"
echo "Chess Training Test"
echo "================================"

# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate

# Set environment variables
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX="1"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS="3"
export MALLOC_ARENA_MAX="64"
export WORLD_SIZE="8"
export WANDB_DISABLED="true"  # Disable wandb logging

# Training parameters
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="test_chess_model"
DATASET_PATH="data/chess_output.json"
MAX_STEPS=5  # Just 5 steps for testing

echo ""
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS (test mode)"
echo ""

# Run training
torchrun \
    --nproc_per_node 2 \
    src/distill_chess_neuron_torchrun.py \
    --model_id "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_model_path "./final_chess_model" \
    --num_train_epochs 1 \
    --do_train \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --bf16 \
    --zero_1 False \
    --tensor_parallel_size 2 \
    --warmup_steps 2 \
    --pipeline_parallel_size 1 \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir

echo ""
echo "================================"
echo "Training test complete!"
echo "================================"
