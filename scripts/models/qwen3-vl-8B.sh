MODEL_ARGS=(
   # Language model parameters (from text_config)
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 5000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
   
   # Qwen3-VL specific parameters
   # Position embedding type for multimodal RoPE
   --position-embedding-type mrope
   --attention-dropout 0.0
   --attention-softmax-in-fp32
   
   # Vision model parameters (from vision_config)
   # Note: These parameters may be automatically loaded from HuggingFace config
   # by megatron-bridge provider, but are included here for explicit configuration
   # --patch-size 16
   # --temporal-patch-size 2
   # --spatial-merge-size 2
   # --num-position-embeddings 2304
   # --out-hidden-size 4096
   
   # Vision token IDs (from config.json root level)
   # --image-token-id 151655
   # --video-token-id 151656
   # --vision-start-token-id 151652
   # --vision-end-token-id 151653
   # --bos-token-id 151643
   # --eos-token-id 151645
   
   # Multimodal RoPE section [temporal, height, width]
   --mrope-section 24 20 20
   
   # Language max sequence length
   # --language-max-sequence-length 2048
)

