#!/bin/bash

# Configuration
# GPU IDs to use
GPU_ID="1,2"
export CUDA_VISIBLE_DEVICES=$GPU_ID

NUM_SAMPLES=800
DATASET="tqa"
MODE="one_pass"

# Models and their types (format: "model_name:model_type")
MODELS=(
    "llama2_70b:black"
    "llama3_3b:base"
    "sft_llama3_3b_llama2_70b:sft"
)

# Loop through each model
for ITEM in "${MODELS[@]}"
do
    # Parse model name and type
    MODEL=$(echo $ITEM | cut -d':' -f1)
    TYPE=$(echo $ITEM | cut -d':' -f2)

    echo "================================================"
    echo "Processing model: $MODEL (type: $TYPE)"
    echo "================================================"

    # Stage 1: Generate responses and compute uncertainty with proxy model
    echo ">>> Stage 1: Generate responses and compute uncertainty..."
    python /path/to/acl_scripts/generate_black.py \
        --model_name "$MODEL" \
        --model_type "$TYPE" \
        --dataset_name "$DATASET" \
        --mode "$MODE" \
        --gene 1 \
        --num_samples $NUM_SAMPLES \
        --gpuid "$GPU_ID"

    # Stage 2: Correctness annotation using BLEURT
    echo ">>> Stage 2: BLEURT correctness annotation..."
    python /path/to/acl_scripts/generate_black.py \
        --model_name "$MODEL" \
        --model_type "$TYPE" \
        --dataset_name "$DATASET" \
        --mode "$MODE" \
        --generate_gt 1 \
        --num_samples $NUM_SAMPLES \
        --gpuid "$GPU_ID"

    # Stage 3: Correctness annotation using LLM
    echo ">>> Stage 3: LLM correctness annotation..."
    python /path/to/acl_scripts/generate_black.py \
        --model_name "$MODEL" \
        --model_type "$TYPE" \
        --dataset_name "$DATASET" \
        --mode "$MODE" \
        --generate_gt_llm 1 \
        --num_samples $NUM_SAMPLES \
        --gpuid "$GPU_ID"

    echo "Model $MODEL processing complete."
done

# Stage 4: Generate evaluation comparison table
echo ">>> Generating comparison table..."
python /path/to/acl_scripts/one_eval.py \
    --config_name "llama3_3b_llama2_70b-tqa" \
    --num_samples $NUM_SAMPLES

echo "All processes completed."
