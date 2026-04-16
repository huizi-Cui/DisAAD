#!/bin/bash

# Configuration
BASE_DIR="/path/to/workspace"
DATA_GEN_PATH="$BASE_DIR/datasets/70b/final_sft_tqa"

# Training configuration
TRAIN_DATA_JSON="${DATA_GEN_PATH}.raw_data.json"
MODEL_TO_TRAIN="llama3.2-3b"
OUTPUT_DIR="$BASE_DIR/exp_results/llama70b_proxy_3b_model"
GPU_IDS="1,2"

# Training hyperparameters
EPOCHS=1
BATCH_SIZE=2
NUM_SAMPLES=100

echo "Starting training process..."
echo "Input dataset: $TRAIN_DATA_JSON"

# Run training script
python /path/to/acl_scripts/train_newdald_new.py \
    --train_dataset_path "$TRAIN_DATA_JSON" \
    --scoring_model_name "$MODEL_TO_TRAIN" \
    --output_path "$OUTPUT_DIR" \
    --gpu_ids "$GPU_IDS" \
    --train_num_samples $NUM_SAMPLES \
    --epochs $EPOCHS \
    --seed 42 \
    --cache_dir "cache"

echo "Training completed. Model saved at: $OUTPUT_DIR"
