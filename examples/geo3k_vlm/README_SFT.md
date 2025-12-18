# Qwen3-VL SFT Training on Geo3K Dataset

This directory contains scripts for fine-tuning Vision-Language Models (VLMs) using Supervised Fine-Tuning (SFT) on the Geo3K geometry dataset.

## Overview

The script supports:
- **Models**: Qwen2.5-VL-3B-Instruct, Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct
- **Backends**: Megatron (default) and FSDP
- **Training Mode**: Supervised Fine-Tuning (SFT) only
- **Dataset**: Geo3K with image URLs

## Key Features

### SFT-Specific Configuration
- Uses `slime.rollout.sft_rollout.generate_rollout` for data processing
- Supports multimodal data (images + text) through `prepare_model_inputs()`
- Proper handling of image tokens in loss computation
- Chat template application for conversational data

### Modified Components

#### 1. `run_geo3k_vlm.sh`
Main training script that:
- Downloads model and dataset automatically
- Configures SFT training parameters
- Supports both Megatron and FSDP backends
- Handles multimodal data with `--multimodal-keys '{"image": "images"}'`

#### 2. `slime/rollout/sft_rollout.py`
Enhanced rollout function that:
- Uses `prepare_model_inputs()` for proper VLM tokenization
- Handles image tokens correctly
- Generates loss masks for SFT training
- Stores multimodal inputs in sample objects

## Usage

### Basic Usage (Megatron Backend, 8 GPUs)

```bash
cd examples/geo3k_vlm
./run_geo3k_vlm.sh
```

### Using FSDP Backend

```bash
SLIME_SCRIPT_TRAIN_BACKEND=fsdp ./run_geo3k_vlm.sh
```

### Specifying a Different Model

```bash
# Use Qwen3-VL-4B-Instruct
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct ./run_geo3k_vlm.sh

# Use Qwen3-VL-8B-Instruct
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./run_geo3k_vlm.sh

# Use Qwen3-VL-30B-A3B-Instruct (MoE model)
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-30B-A3B-Instruct ./run_geo3k_vlm.sh
```

### Custom Number of GPUs

```bash
SLIME_SCRIPT_NUM_GPUS=4 ./run_geo3k_vlm.sh
```

### Using External Ray Cluster

```bash
SLIME_SCRIPT_EXTERNAL_RAY=1 ./run_geo3k_vlm.sh
```

### With Wandb Logging

```bash
WANDB_API_KEY=your_api_key_here ./run_geo3k_vlm.sh
```

### Combining Multiple Options

```bash
SLIME_SCRIPT_TRAIN_BACKEND=fsdp \
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_NUM_GPUS=8 \
WANDB_API_KEY=your_api_key \
./run_geo3k_vlm.sh
```

## Configuration Details

### Training Parameters

```bash
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/datasets/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 32
   --global-batch-size 128
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)
```

### Optimizer Settings

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5                      # Learning rate
   --lr-decay-style cosine        # Cosine learning rate schedule
   --min-lr 1e-6                  # Minimum learning rate
   --lr-warmup-fraction 0.1       # 10% warmup
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)
```

### Backend-Specific Settings

#### Megatron Backend (Default)
- Tensor parallelism: 1
- Pipeline parallelism: 1
- Sequence parallelism: enabled
- Dynamic batch size with max 4096 tokens per GPU
- Flash attention backend
- Gradient checkpointing with recompute-granularity=full

#### FSDP Backend
- Gradient checkpointing enabled
- Flash Attention 3
- Custom weight update buffer size

## Dataset Format

The Geo3K dataset should be in Parquet format with the following structure:

```python
{
    "problem": "...",           # Text description of the geometry problem
    "answer": "...",            # Expected answer
    "images": [<image_url>]     # List of image URLs
}
```

The script automatically:
1. Downloads the dataset from HuggingFace: `chenhegu/geo3k_imgurl`
2. Applies chat template to format conversations
3. Processes images through the VLM processor
4. Generates proper loss masks for training

## Checkpointing

- **Save directory**: `/root/models/${MODEL_NAME}_slime/`
- **Save interval**: Every 100 iterations
- **Load checkpoint**: `/root/models/${MODEL_NAME}` (for Megatron backend)

## Evaluation

- **Eval interval**: Every 100 training steps
- **Test set**: `/root/datasets/${DATASET_LOCAL_NAME}/test.parquet`
- **Samples per eval**: 1
- **Max response length**: 4096 tokens

## Multimodal Data Handling

The script uses Slime's `prepare_model_inputs()` function which:
1. Processes vision information from the dataset
2. Applies chat template to messages
3. Tokenizes text and inserts image tokens
4. Returns both input_ids and multimodal_inputs

Key feature: **Image tokens are masked out in loss computation** (loss_mask=0), ensuring the model only learns from text responses.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--rollout-batch-size` or `--global-batch-size`
- Reduce `--max-tokens-per-gpu`
- Enable gradient checkpointing (already enabled)

### NVLink Detection
The script automatically detects NVLink and sets `NCCL_NVLS_ENABLE` accordingly.

### Image Loading Issues
Ensure images in the dataset are accessible. The script expects image URLs that can be downloaded.

### Ray Cluster Issues
If using external Ray, make sure:
```bash
export MASTER_ADDR=<your_ray_head_address>
SLIME_SCRIPT_EXTERNAL_RAY=1 ./run_geo3k_vlm.sh
```

## Differences from RL Training

This SFT version differs from the original RL script in:

1. **Training objective**: SFT loss instead of GRPO/PPO
2. **No sampling**: Single ground-truth response per prompt
3. **No reward model**: Uses supervised labels only
4. **No inference engine**: No SGLang server needed
5. **Simplified evaluation**: Direct model inference
6. **Different script**: Uses `train_async.py` instead of `train.py`

## References

- Original RL script: `run_geo3k_vlm.sh` (RL training version)
- Text-only SFT example: `scripts/run-qwen3-4B-base-sft.sh`
- Megatron-Bridge VLM finetuning: `finetune_qwen_vl.py`

## Citation

If you use this script, please cite:
```bibtex
@article{slime2024,
  title={SLIME: Scalable Large-scale Inference and Model Evolution},
  author={Your Team},
  year={2024}
}
```

