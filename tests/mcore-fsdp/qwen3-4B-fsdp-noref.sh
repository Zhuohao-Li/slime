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

CKPT_ARGS=(
#    --hf-checkpoint /root/models/Qwen3-4B-Instruct-2507
#    --ref-load /root/models/Qwen3-4B-Instruct-2507
    --hf-checkpoint /root/Qwen3-4B
    # --ref-load /root/Qwen3-4B
    --load /root/Qwen3-4B
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 64
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # Uncomment to use DeepSpeed CPU Adam
   # --optimizer deepspeed_cpu_adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   # Set equal to the number of GPUs per node for colocated mode
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-mcore-fsdp
   --wandb-group qwen3-4B-fsdp-1116-noref
   --wandb-key ${WANDB_KEY}
)

FSDP_ARGS=(
   # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
   # --fsdp-full-params  # Uncomment this line to enable full params mode
   --train-backend fsdp
   # Set the bucket size for weight update
   --update-weight-buffer-size $((512 * 1024 * 1024)) # 512MB
   # --attn-implementation flash_attention_2
   --gradient-checkpointing
   --sglang-attention-backend fa3
   --attn-implementation flash_attention_3
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
   --offload-train-mode move \
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --use-fault-tolerance
   --dump-details /root/shared_data/qwen3-4B-fsdp-1116-noref/dump_details
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
     }
   }' \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${FSDP_ARGS[@]} \
   ${MISC_ARGS[@]}
