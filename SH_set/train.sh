#!/bin/bash

# Configuration
# Base directories
BASE_DIR="/path/to/workspace"
CACHE_DIR="$BASE_DIR/sft_data"

# Step 1: Mixed dataset configuration
OTHER_DATASET="tqa"
WILD_SAMPLES= 70
OTHER_SAMPLES=70
ORI_DATA_PATH="$BASE_DIR/datasets/mixed_questions/wild_tqa"

# Step 2: Model generation configuration
BASE_MODEL="llama2-7b"
OUTPUT_SFT_PATH="$BASE_DIR/datasets/7b/final_sft_tqa"
GEN_NUM_SAMPLES=140
GPU_IDS="1,2,3,4"

# OpenAI API configuration (for args.what_to_do="api")
API_MODEL="gpt-4-0613"
API_KEY="YOUR_API_KEY"

# Step 2: Generate multiple responses using local model
echo "Starting Stage 2: Generate multiple responses (local)..."
python /path/to/acl_scripts/data_builder.py \
    --what_to_do local \
    --base_model_name $BASE_MODEL \
    --ori_dataset_path $ORI_DATA_PATH \
    --sft_dataset_path $OUTPUT_SFT_PATH \
    --num_samples $GEN_NUM_SAMPLES \
    --gpuids "$GPU_IDS" \
    --cache_dir $CACHE_DIR \
    --do_temperature \
    --temperature 0.7 \
    --top_p 0.9 \
    --seed 42

echo "Process completed. Final SFT data at: $OUTPUT_SFT_PATH"
