# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slime is an LLM post-training framework for RL scaling, designed to connect Megatron-LM (training) with SGLang (inference/rollout). It supports 30+ model architectures including Qwen3, DeepSeek V3, GLM4, Llama 3, Kimi-K2, and others. The framework uses Ray for distributed orchestration across training and rollout workers.

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
- **Cluster**: `--actor-num-nodes`, `--rollout-num-gpus`, `--colocate`, `--offload-train`, `--offload-rollout`
- **Training**: `--train-backend` (megatron/fsdp), `--advantage-estimator`, `--kl-loss-coef`
- **Rollout**: `--prompt-data`, `--rm-type`, `--rollout-batch-size`
- **SGLang**: Prefixed with `--sglang-*`
- **Evaluation**: `--eval-interval`, `--eval-prompt-data`

### Key Technical Concepts

- **Colocation mode** (`--colocate`): Training and inference share the same GPUs with memory offloading between phases
- **True on-policy mode**: Ensures identical log probs between SGLang rollout and Megatron training engines
- **Off-policy distillation**: Teacher-student learning within on-policy training loop
- **Weight sync**: After each training step, updated weights are pushed from Megatron/FSDP to SGLang engines

## Code Style

- Python >=3.10, line length 119
- Formatting: black + isort (black profile)
- Linting: ruff (E, F, B, UP rules) + autoflake (removes unused imports)
- Known first-party packages: `slime`, `slime_plugins`
- Known third-party: `megatron`, `wandb`, `ray`, `transformers`
