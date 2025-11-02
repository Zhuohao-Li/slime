#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Script configuration (equivalent to Python ScriptArgs)
MODE=${SLIME_SCRIPT_MODE:-"normal"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-4B"}
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-1}
NUM_GPUS_PER_NODE=${SLIME_SCRIPT_NUM_GPUS_PER_NODE:-8}
HARDWARE=${SLIME_SCRIPT_HARDWARE:-"H100"}
EXTRA_ARGS=${SLIME_SCRIPT_EXTRA_ARGS:-""}
MULTI_EVAL=${SLIME_SCRIPT_MULTI_EVAL:-0}
TRUE_ON_POLICY=${SLIME_SCRIPT_ENABLE_TRUE_ON_POLICY:-0}
DYNAMIC_SAMPLING=${SLIME_SCRIPT_DYNAMIC_SAMPLING:-0}
ENABLE_EVAL=${SLIME_SCRIPT_ENABLE_EVAL:-1}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUN_ID=$(date +%Y%m%d_%H%M%S)_$(openssl rand -hex 3)

DATA_DIR=/home/data/workgroup/zhuohao/data
WANDB_KEY=dfbfb48c275f2d5182d9d3fb6ce84c71d752c39c

CKPT_ARGS=(
   --hf-checkpoint /root/${MODEL_NAME}
   --ref-load /root/${MODEL_NAME}
)

# Set response length based on mode
if [ "$MODE" = "debug_minimal" ]; then
    ROLLOUT_MAX_RESPONSE_LEN=100
else
    ROLLOUT_MAX_RESPONSE_LEN=32768
fi

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 16
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature 0.8
   --global-batch-size 1024
   --balance-data
)

# Dynamic sampling (if enabled)
if [ "$DYNAMIC_SAMPLING" = "1" ] && [ "$MODE" != "debug_minimal" ]; then
    ROLLOUT_ARGS+=(
        --over-sampling-batch-size 64
        --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    )
fi

# Evaluation configuration
EVAL_ARGS=()
if [ "$MODE" != "debug_minimal" ] && [ "$ENABLE_EVAL" = "1" ]; then
    EVAL_MAX_RESPONSE_LEN=32768
    EVAL_ARGS+=(--eval-interval 20)
    
    if [ "$MULTI_EVAL" = "1" ]; then
        # Create temporary eval config file
        EVAL_CONFIG_FILE="/tmp/eval_config_${RUN_ID}.yaml"
        cat > "$EVAL_CONFIG_FILE" << EOF
eval:
  defaults:
    max_response_len: ${EVAL_MAX_RESPONSE_LEN}
    top_p: 0.7
  datasets:
    - name: aime
      path: ${DATA_DIR}/aime-2024/aime-2024.jsonl
      rm_type: deepscaler
      n_samples_per_eval_prompt: 16
    - name: gpqa
      path: /root/datasets/gpqa_diamond/gpqa_eval.jsonl
      rm_type: gpqa
      n_samples_per_eval_prompt: 2
    - name: ifbench
      path: /root/datasets/IFBench/IFBench_eval.jsonl
      rm_type: ifbench
      n_samples_per_eval_prompt: 1
EOF
        EVAL_ARGS+=(--eval-config "$EVAL_CONFIG_FILE")
    else
        EVAL_ARGS+=(
            --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
            --n-samples-per-eval-prompt 16
            --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}
            --eval-top-p 0.7
        )
    fi
fi

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # --optimizer deepspeed_cpu_adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Wandb configuration (equivalent to U.get_default_wandb_args)
WANDB_ARGS=()
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS+=(
        --use-wandb
        --wandb-project slime-dev-mcore-fsdp
        --wandb-group qwen3-4B-fsdp-revise
        --wandb-key ${WANDB_API_KEY}
        --disable-wandb-random-suffix
    )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-chunked-prefill-size 4096
)

FSDP_ARGS=(
   --train-backend fsdp
   --attn-implementation flash_attention_2
   --gradient-checkpointing
   --update-weights-bucket-size $((512 * 1024 * 1024))
)

MISC_ARGS=(
   --actor-num-nodes ${NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --colocate
   --offload-train-mode move
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --use-fault-tolerance
   --save-debug-rollout-data /root/shared_data/${RUN_ID}/{rollout_id}.pt
)

# True on-policy configuration
TRUE_ON_POLICY_ARGS=()
TRUE_ON_POLICY_ENVS=""
if [ "$TRUE_ON_POLICY" = "1" ]; then
    TRUE_ON_POLICY_ARGS+=(
        --sglang-enable-deterministic-inference
        --sglang-rl-on-policy-target fsdp
        --sglang-attention-backend fa3
        --attn-implementation flash_attention_3
        --deterministic-mode
        --true-on-policy-mode
    )
    TRUE_ON_POLICY_ENVS=',"NCCL_ALGO":"allreduce:tree","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"'
fi

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats

# Build the runtime environment JSON with proper variable substitution
# Note: Removed PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True because it's incompatible with torch_memory_saver used by SGLang
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}\"${TRUE_ON_POLICY_ENVS}
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${FSDP_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${TRUE_ON_POLICY_ARGS[@]} \
   ${EXTRA_ARGS}

# Cleanup temporary files
if [ -n "$EVAL_CONFIG_FILE" ] && [ -f "$EVAL_CONFIG_FILE" ]; then
    rm -f "$EVAL_CONFIG_FILE"
fi
