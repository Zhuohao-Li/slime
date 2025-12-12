# Using Megatron-Bridge to train Qwen3-VL

In latest slime image, update mcore to dev branch:

```
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout dev && \
    pip install -e .
```
convert hf checkpoints to megatron:
```
python -m torch.distributed.run --nproc_per_node=1 convert_checkpoints.py   import \
    --hf-model /root/Qwen3-VL-8B-Instruct \
    --megatron-path /root/checkpoints/qwen3vl8b
```
train:
```
python -m torch.distributed.run --nproc_per_node=8 \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_8b_finetune_config \
    --pretrained-checkpoint /root/checkpoints/qwen3vl8b \
    model.tensor_model_parallel_size=1 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    train.train_iters=2
```

train on geo3k:
```
torchrun --nproc-per-node=8 examples/recipes/qwen_vl/finetune_qwen_vl.py \
    --pretrained-checkpoint /root/checkpoints/qwen3vl8b \
    --recipe qwen3_vl_8b_finetune_config \
    +dataset.maker_kwargs.path_or_dataset=/root/datasets/geo3k_imgurl \
    +dataset.maker_kwargs.data_files.train=train.parquet \
    train.global_batch_size=16 \
    train.train_iters=8 \
    logger.wandb_project=mbridge-qwen3-vl \
    logger.wandb_save_dir=/root/checkpoints/wandb \
    checkpoint.save=/root/checkpoints/wandb/mbridge-qwen3-vl \
    checkpoint.fully_parallel_save=True \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1
```

## FAQ

if encounter cudnn issue, refer to this issue reported from mcore: [#issue](https://github.com/nvidia/megatron-lm/issues/1882)