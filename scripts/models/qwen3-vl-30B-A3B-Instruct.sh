MODEL_ARGS=(
   --model-name Qwen3-VL-30B-A3B-Instruct
   --swiglu
   --num-layers 48
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 32
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
   --untie-embeddings-and-output-weights

   # moe
   --moe-ffn-hidden-size 768
   --moe-router-topk 8
   --num-experts 128
)
