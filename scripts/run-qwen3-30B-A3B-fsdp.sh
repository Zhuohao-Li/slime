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
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B
   # --ref-load /root/Qwen3-30B-A3B
   --load /root/Qwen3-30B-A3B
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
)

# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
#    --n-samples-per-eval-prompt 16
#    --eval-max-response-len 16384
#    --eval-top-p 0.7
# )

PERF_ARGS=(
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-moe
   --wandb-group qwen3-30B-A3B-fsdp
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-cuda-graph-max-bs 512
)

FSDP_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 512000000
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --sglang-attention-backend fa3
   --fsdp-cpu-offload
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --use-fault-tolerance
   --dump-details /root/shared_data/qwen3-30B-A3B-fsdp/dump_details
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
     }
   }' \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${FSDP_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]}
