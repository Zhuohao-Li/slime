#!/bin/bash

# Qwen3-VL-8B RL training on geo3k dataset using megatron backend
# Note: This requires megatron-bridge VLM support

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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model and dataset if not present
mkdir -p /root/models /root/datasets
if [ ! -d "/root/models/Qwen3-VL-8B-Instruct" ]; then
    hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /root/models/Qwen3-VL-8B-Instruct
fi
if [ ! -d "/root/datasets/geo3k_imgurl" ]; then
    hf download --repo-type dataset chenhegu/geo3k_imgurl --local-dir /root/datasets/geo3k_imgurl
fi


MODEL_ARGS=(
   --model-name qwen3-vl
)

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-VL-8B-Instruct
   --load /root/Qwen3-VL-8B-Instruct
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/geo3k_imgurl/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
#    --eval-interval 50
#    --eval-prompt-data geo3k-test /root/datasets/geo3k_imgurl/test.parquet
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 4096
#    --eval-top-k 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project geo3k-vlm-megatron
   --wandb-group qwen3-vl-8b-geo3k
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   
   # colocate training and rollout on same GPUs
   --colocate
)

NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/tmp/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --train-backend megatron \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}

