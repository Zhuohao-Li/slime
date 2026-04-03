# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slime is an LLM post-training framework for RL scaling, designed to connect Megatron-LM (training) with SGLang (inference/rollout). It supports 30+ model architectures including Qwen3, DeepSeek V3, GLM4, Llama 3, Kimi-K2, and others. The framework uses Ray for distributed orchestration across training and rollout workers.

**Key capabilities:**
- High-performance RL training (PPO, GRPO, GSPO, Reinforce++)
- Flexible data generation with custom rollout and reward functions
- Support for both synchronous and asynchronous training modes
- Multi-turn agent training with tool calling support
- Colocation mode for memory-efficient single-GPU training

## Common Commands

### Linting and Formatting

```bash
# Run all pre-commit checks (ruff, autoflake, isort, black)
pre-commit run --all-files --show-diff-on-failure --color=always

# Install pre-commit hooks
pre-commit install
```

Style rules: line length 119 (black/isort), isort uses black profile. Ruff selects E, F, B, UP rules (ignores E402, E501).

### Testing

```bash
# Run full test suite
pytest

# Run a single test
pytest tests/test_qwen3_4B_ppo.py

# Run by marker
pytest -m "unit"
pytest -m "not skipduringci"
```

Test markers: `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`.

Note: Most tests require GPU hardware and launch full training runs. Tests in `tests/` are typically standalone scripts that configure and run training via subprocess.

### Multi-Node Setup (Ray Cluster)

```bash
# On head node (node 0)
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# On worker nodes
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8

# Submit job from head node
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
  -- python3 train.py [args...]

# Stop Ray cluster
ray stop --force
```

Environment variables for multi-node (especially in Docker/SLURM):
```bash
export SLIME_HOST_IP=$(hostname -I | awk '{print $1}')
export GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2}')
export NCCL_SOCKET_IFNAME=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2}')
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2}')
```

### Installation

```bash
pip install -e .          # Basic install
pip install -e ".[fsdp]"  # With FSDP extras (torch>=2.0)
```

### Training

```bash
# Synchronous training (standard)
python train.py <args>

# Async training (overlaps rollout with training)
python train_async.py <args>

# Example scripts in scripts/
bash scripts/run-qwen3-4B.sh
```

### Model Conversion

```bash
# HuggingFace <-> Megatron distributed format
python tools/convert_hf_to_torch_dist.py
python tools/convert_torch_dist_to_hf.py
```

## Architecture

### Three-Layer Design

```
Training Layer     →  Megatron-LM or HF+FSDP backends (reads data, trains, syncs params)
Data Buffer        →  Bridge for prompt/rollout/reward management
Rollout Layer      →  SGLang engines + Router (generates responses + computes rewards)
```

All layers are coordinated via **Ray** actors and placement groups.

### Core Package (`slime/`)

- **`backends/`** — Training backend implementations
  - `megatron_utils/` — Megatron-LM backend: actor training, loss computation (PPO/GRPO), checkpointing, weight sync, model-specific HF converters
  - `fsdp_utils/` — HuggingFace + FSDP backend: simpler alternative to Megatron, includes data packing and fused MoE kernels
  - `sglang_utils/` — SGLang inference engine wrapper
- **`rollout/`** — Data generation and reward computation
  - `sglang_rollout.py` — Main rollout generation using SGLang
  - `rm_hub/` — Reward model implementations (math, code, GPQA, etc.)
  - `filter_hub/` — Dynamic sampling filters
  - `generate_hub/` — Generation strategies
  - `data_source.py` — Data loading interface
- **`ray/`** — Distributed orchestration
  - `placement_group.py` — GPU allocation and placement groups
  - `rollout.py` — `RolloutManager` (Ray remote actor coordinating rollout)
  - `train_actor.py` — Base `TrainRayActor`
  - `actor_group.py` — Actor group coordination
- **`router/`** — Request routing middleware
- **`utils/`** — Shared utilities
  - `arguments.py` — Central CLI argument parsing (Megatron args, SGLang args prefixed `--sglang-`, slime-specific args)
  - `ppo_utils.py` — PPO/GRPO advantage estimation, KL divergence, policy loss
  - `types.py` — Core types (`Sample`, `RolloutBatch`, etc.)
  - `data.py` — Dataset loading (JSONL, Parquet)
  - `logging_utils.py` — WandB and TensorBoard integration

### Plugins (`slime_plugins/`)

- **`rollout_buffer/`** — FastAPI server for async trajectory generation with auto-discovered task-specific generators
- **`models/`** — Model-specific implementations (GLM4, Qwen3-Next)
- **`megatron_bridge/`**, **`mbridge/`** — Megatron bridge integration

### Entry Points

- `train.py` — Synchronous training loop: create placement groups → init rollout manager with SGLang → init actor/critic → train loop (rollout → weight offload → train → eval)
- `train_async.py` — Async training: overlaps next rollout generation with current training step

### Key Argument Categories

Arguments are parsed in `slime/utils/arguments.py`:
- **Cluster**: `--actor-num-nodes`, `--actor-num-gpus-per-node`, `--rollout-num-gpus`, `--rollout-num-gpus-per-engine`, `--colocate`, `--offload-train`, `--offload-rollout`
- **Training**: `--train-backend` (megatron/fsdp), `--advantage-estimator` (grpo/gspo/reinforce++/ppo), `--kl-loss-coef`, `--use-dynamic-batch-size`, `--max-tokens-per-gpu`
- **Rollout**: `--prompt-data`, `--rm-type`, `--rollout-batch-size`, `--n-samples-per-prompt`, `--rollout-temperature`, `--rollout-max-response-len`
- **SGLang**: Prefixed with `--sglang-*` (e.g., `--sglang-mem-fraction-static`, `--sglang-context-length`)
- **Router**: Prefixed with `--router-*` (e.g., `--router-balance-abs-threshold`)
- **Evaluation**: `--eval-interval`, `--eval-prompt-data`, `--n-samples-per-eval-prompt`
- **Checkpointing**: `--hf-checkpoint`, `--ref-load`, `--load`, `--save`, `--save-interval`, `--ckpt-format` (torch/torch_dist)
- **Debugging**: `--debug-rollout-only`, `--debug-train-only`, `--save-debug-rollout-data`, `--load-debug-rollout-data`

### Key Technical Concepts

- **Colocation mode** (`--colocate`): Training and inference share the same GPUs with memory offloading between phases. Requires `--sglang-mem-fraction-static` (typically 0.8) to prevent OOM.
- **True on-policy mode**: Ensures identical log probs between SGLang rollout and Megatron training engines
- **Off-policy distillation**: Teacher-student learning within on-policy training loop
- **Weight sync**: After each training step, updated weights are pushed from Megatron/FSDP to SGLang engines
- **Dynamic batching** (`--use-dynamic-batch-size`): Intelligently packs samples to maximize GPU utilization with `--max-tokens-per-gpu`
- **Data packing**: All training uses variable-length packed sequences (varlen/thd), so `--seq-length` doesn't limit model context
- **Rollout-Train relationship**: `(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`
- **Custom functions**: Support `--custom-generate-function-path` and `--custom-rm-path` for multi-turn agents and tool calling

## Debugging

### Separate Component Debugging

```bash
# Debug inference only (no Megatron)
python train.py --debug-rollout-only [args...]

# Debug training only (no SGLang)
python train.py --debug-train-only [args...]

# Save rollout data for reproducible training debugging
python train.py --save-debug-rollout-data /path/data_{rollout_id}.pt [args...]

# Load saved rollout data (automatically sets debug-train-only)
python train.py --load-debug-rollout-data /path/data_{rollout_id}.pt [args...]
```

### Precision Alignment Checklist

**First training step:**
1. Check if generated rollout text is coherent (not garbled)
   - If garbled: verify checkpoint loading, weight sync, parameter names match parallelism strategy
   - For pretrained models: try instruct version to rule out model issues
2. Verify `log_probs == ref_log_probs` (KL divergence = 0) and values are small
   - If not equal: may need `--attention-backend flash` for Transformer Engine stability under CP
   - If values large (>1): check training config or data chat template alignment
3. When `--num-steps-per-rollout == 1`: check KL divergence = 0 and `grad_norm` is small
   - May need `--moe-permute-fusion` for MoE models

**Second training step:**
- For colocation mode: verify second step loads correctly without OOM

### SGLang IMA (Illegal Memory Access) Debugging

```bash
# Enable blocking mode to pinpoint error
CUDA_LAUNCH_BLOCKING=1 python train.py [args...]

# Toggle speculative decoding and CUDA graph
# IMA often appears in padding/cuda graph replay or draft model differences

# Disable deepep if used
# Can cause IMA issues

# Use CUDA Core Dump (see vLLM blog: "CUDA Core Dump: An Effective Tool...")
```

## Code Style

- Python >=3.10, line length 119
- Formatting: black + isort (black profile)
- Linting: ruff (E, F, B, UP rules) + autoflake (removes unused imports)
- Known first-party packages: `slime`, `slime_plugins`
- Known third-party: `megatron`, `wandb`, `ray`, `transformers`
