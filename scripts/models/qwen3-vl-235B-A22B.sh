MODEL_ARGS=(
   --model-name Qwen3-VL-235B-A22B-Instruct
   --swiglu
   --num-layers 94
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 64
   --num-query-groups 4
   --init-method-std 0.02
   --norm-epsilon 1e-06
   --rotary-base 5000000
   --vocab-size 151936
   --seq-length 262144
   --use-rotary-position-embeddings
   --normalization "RMSNorm"
   --qk-layernorm
   --group-query-attention
   --disable-bias-linear
   --kv-channels 128

   # moe
   --moe-ffn-hidden-size 1536
   --moe-router-topk 8
   --num-experts 128
)
