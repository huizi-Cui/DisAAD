# DisAAD: Improving Logits-based Detector without Logits from Black-box LLMs

This repository contains the official code for the paper **Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation**, accepted at **ACL 2026**.

## Method Overview

DisAAD is a two-stage method for black-box uncertainty quantification, where the first stage is distribution-aligned adversarial distillation for obtaining the proxy model, and the second stage is proxy-guided LLM uncertainty quantification based on evidential deep learning.

- **Stage 1 (Data Collection & Adversarial Distillation)**: Systematically collect outputs from the target black-box LLM across diverse prompts to create a comprehensive distillation dataset. Train a proxy model with a generation-discrimination architecture to approximate the high-probability regions of the target output distribution. 
- **Stage 2 (Uncertainty Estimation)**: Utilize the distilled proxy model to reproduce the response of target LLM and estimate real-time uncertainty via evidential deep learning.

## Project Structure

```
./
├── scripts/                 # Core code
│   ├── generate_black.py       # Black-box LLM response generation & uncertainty estimation
│   ├── one_eval.py             # One-pass evaluation with BLEURT
│   ├── one_eval_llm.py         # One-pass evaluation with LLM
│   ├── train_disaad.py         # Adversarial distillation training
│   ├── data_builder.py         # SFT dataset construction
│   ├── metrics.py              # Uncertainty metrics (AU, EU, entropy)
│   ├── model.py                # Model loading utilities
│   └── utils.py                # Helper functions
├── SH_set/                     # Workflow scripts
│   ├── data_build.sh           # Stage 1-1: data collection
│   ├── train.sh                # Stage 1-2: proxy model training
│   └── one_eval.sh             # Stage 2: uncertainty estimation
├── datasets/                   # Dataset storage
│   ├── 70b/                     # Distillation data of 70B model
│   ├── GPT4/                    # Distillation data of GPT4
│   └── mixed_questions/        # Mixed prompts for generation
├── blackbox_outputs/           # Black-box LLM outputs (one-pass & multi-pass)
├── cache/                      # HuggingFace model cache
└── results*/                   # Training logs & fine-tuned models
    └── {model_name}_proxy_{size}_model/  # Trained proxy models
```

## Model Preparation

Download required models:

### Base models
```bash
- meta-llama/Llama-2-{7b,13b,70b}-chat-hf
- meta-llama/Llama-3-{1B,3B,8B,70B}-Instruct
- Qwen/Qwen3-32B-Instruct
- gpt-4-0613
- openai-community/gpt2
```
### Evaluation models
```bash
- lucadiliello/BLEURT-20
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2
```

## Usage

### Stage 1: Data Collection & Adversarial Distillation

Collect outputs from black-box LLM and train the proxy model:

```bash
sh SH_set/data_build.sh  # Generate responses and create distillation dataset
sh SH_set/train.sh      # Train proxy model via adversarial distillation
```

### Stage 2: Uncertainty Estimation

Use the distilled proxy model to estimate response uncertainty:

```bash
sh SH_set/one_eval.sh   # Evaluate uncertainty estimation performance
```

## Citation

```bibtex
@inproceedings{cui2026estimating,
  title={Estimating the Black-box {LLM} Uncertainty with Distribution-Aligned Adversarial Distillation},
  author={Cui, Huizi and Ma, Huan and Wang, Qilin and Gao, Yuhang and Zhang, Changqing},
  booktitle={The 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```
