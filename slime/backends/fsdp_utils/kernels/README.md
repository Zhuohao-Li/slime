# 测试结果

- HuggingFace原生实现

```
Running forward pass...
--------------------
Next Token Prediction:
Token ID: 151667
Token: '<think>'
Logprob: -1.6689286894688848e-06
Forward Time: 4.513868 seconds (4513.87 ms)
--------------------

Testing backward pass...
--------------------
Backward Pass Debug Info:
Loss value: 4014080.0
Loss dtype: torch.bfloat16
Loss shape: torch.Size([])
--------------------
Running backward...
--------------------
✓ Backward pass completed - custom backward functions were called
Gradient Statistics:
  Total gradient norm: 3602429321.977151
  Params with gradients: 12876
  Max gradient value: 390070272.000000
  Min gradient value: -285212672.000000
Backward Time: 4.975255 seconds (4975.26 ms)
```

- triton实现

```python
Running forward pass...
--------------------
Next Token Prediction:
Token ID: 151667
Token: '<think>'
Logprob: -1.7881377516459906e-06
Forward Time: 2.859184 seconds (2859.18 ms)
--------------------

Testing backward pass...
--------------------
Backward Pass Debug Info:
Loss value: 5865472.0
Loss dtype: torch.bfloat16
Loss shape: torch.Size([])
--------------------
Running backward...
--------------------
✓ Backward pass completed - custom backward functions were called
Gradient Statistics:
  Total gradient norm: 3507420610.003704
  Params with gradients: 18867
  Max gradient value: 402653184.000000
  Min gradient value: -281018368.000000
Backward Time: 2.917407 seconds (2917.41 ms)
```


##  测试代码

```python
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 关键：导入并应用patch，使模型使用自定义的MoE实现
from slime.backends.fsdp_utils.models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe


def apply_fsdp2(model, mesh=None, cpu_offload=False):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
           or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
        "reshard_after_forward": True,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def main():
    # 获取分布式信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}/{world_size}] Starting on GPU {local_rank}")

    model_path = "/sgl-workspace/Qwen3-30B-A3B"
    # model_path = "/mnt/o1_alicloud/personal/zzl/hf_checkpoint/Qwen3-4B"

    if rank == 0:
        print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "<|im_start|>system\nYou are a helpful assistant. Please put the answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nStefan goes to a restaurant to eat dinner with his family. They order an appetizer that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for the waiter, what is the total amount of money that they spend at the restaurant?<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(tokens, dtype=torch.long).cuda()

    # 关键步骤：在加载模型前应用patch，替换transformers的MoE实现为自定义实现
    enable_self_moe = os.environ.get("ENABLE_SELF_MOE", "false").lower() == "true"
    if rank == 0:
        if enable_self_moe:
            print("Applying custom MoE patch...")
        else:
            print("Using default transformers MoE implementation (ENABLE_SELF_MOE=false)")

    if enable_self_moe:
        apply_true_on_policy_patch_for_qwen3_moe()
        if rank == 0:
            print("✓ Custom MoE patch applied - will use fused_experts_impl with custom forward/backward")
    else:
        if rank == 0:
            print("✓ Using standard MoE implementation")

    # Use bfloat16 as it is common for newer models
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,  # 添加这行，避免在 CPU 上加载 float32
    )

    # 先不移动到 CUDA，让 FSDP 处理
    model.train()

    if rank == 0:
        print("Applying FSDP...")
    model = apply_fsdp2(model, cpu_offload=False)

    if rank == 0:
        print("Running forward pass...")

    # Measure forward pass time
    torch.cuda.synchronize()
    forward_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids=tokens.unsqueeze(0))
        logits = outputs.logits.squeeze(0)
        logits = logits.to(torch.float32)

        # Calculate log probabilities
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        # Logits shape: [batch_size, seq_len, vocab_size]
        # We want the logits for the last token in the sequence to predict the next token
        next_token_logits = logits[-1, :]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        # Get top 1
        top1_logprob, top1_token_id = torch.max(log_probs, dim=-1)
        top1_token = tokenizer.decode(top1_token_id)

    torch.cuda.synchronize()
    forward_end = time.perf_counter()
    forward_time = forward_end - forward_start

    if rank == 0:
        print("-" * 20)
        print(f"Next Token Prediction:")
        print(f"Token ID: {top1_token_id.item()}")
        print(f"Token: '{top1_token}'")
        print(f"Logprob: {top1_logprob.item()}")
        print(f"Forward Time: {forward_time:.6f} seconds ({forward_time*1000:.2f} ms)")
        print("-" * 20)

    # Test backward pass to verify gradients computation
    if rank == 0:
        print("\nTesting backward pass...")
    model.zero_grad()

    # Forward with gradients enabled
    outputs = model(input_ids=tokens.unsqueeze(0))
    loss = outputs.logits.sum()

    if rank == 0:
        print("-" * 20)
        print(f"Backward Pass Debug Info:")
        print(f"Loss value: {loss.item()}")
        print(f"Loss dtype: {loss.dtype}")
        print(f"Loss shape: {loss.shape}")
        print("-" * 20)
        print("Running backward...")

    # Measure backward pass time
    torch.cuda.synchronize()
    backward_start = time.perf_counter()

    loss.backward()

    torch.cuda.synchronize()
    backward_end = time.perf_counter()
    backward_time = backward_end - backward_start

    if rank == 0:
        print("-" * 20)
        print("✓ Backward pass completed - custom backward functions were called")

        # Print gradient statistics
        total_grad_norm = 0.0
        num_params_with_grad = 0
        max_grad = float('-inf')
        min_grad = float('inf')

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                num_params_with_grad += 1
                max_grad = max(max_grad, param.grad.max().item())
                min_grad = min(min_grad, param.grad.min().item())

        total_grad_norm = total_grad_norm ** 0.5

        print(f"Gradient Statistics:")
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        print(f"  Params with gradients: {num_params_with_grad}")
        print(f"  Max gradient value: {max_grad:.6f}")
        print(f"  Min gradient value: {min_grad:.6f}")
        print(f"Backward Time: {backward_time:.6f} seconds ({backward_time*1000:.2f} ms)")
        print("-" * 20)

    # 清理
    torch.distributed.barrier()
    if rank == 0:
        print("All ranks completed successfully!")


if __name__ == "__main__":
    # 设置环境变量（这些会被 torchrun 自动设置，但保留作为默认值）
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 不要手动设置这些，让 torchrun 自动设置
    # os.environ["WORLD_SIZE"] = "2"
    # os.environ["RANK"] = "0"
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "29500"

    # 获取 LOCAL_RANK（由 torchrun 自动设置）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 初始化进程组
    torch.distributed.init_process_group(backend="nccl")

    # deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode(
        # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
        # and disabling it will make it aligned
        enable_bmm=False,
    )

    try:
        main()
    finally:
        # 清理进程组
        torch.distributed.destroy_process_group()

```

# GateUpProj

## forward

| 变量名 | 维度 | 含义                                                                          |
|--------|------|-----------------------------------------------------------------------------|
| `hidden_states` | `(num_tokens, D_in)` | 输入的隐藏状态，D_in 为输入维度                                                          |
| `w1` | `(E, N, D_in)` | Gate-Up 投影权重矩阵，E 为专家数量，N 为中间层维度. 一般来说N = intermedia_size * 2，因为融合了gate和up操作 |
| `topk_ids` | `(num_tokens, topk)` | Top-K 路由索引，每个 token 对应 topk 个专家的 ID                                         |

$$\text{intermediate\_cache1}[t \times \text{topk} + k] = \text{hidden\_states}[t] \times W_1^{(\text{expert\_id})}$$

即`(D_in,) @ (N, D_in).T = (N,)`

($W_1^{(\text{expert\_id})}$ 的形状为 `(N, D_in)`)

输出结果`intermediate_cache1` 的形状为 `(num_tokens * topk, N)`


## backward

输入grad_output的维度为 $(num_token \times topk, N)$ 
 
- grad_hidden_states (∂L/∂x)

$$\frac{\partial L}{\partial x[t]} = \sum_{k=0}^{\text{topk}-1} \text{grad}_y[t \cdot \text{topk} + k] \cdot W_1^{(\text{expert\_id})}$$

**维度：** `(N,) @ (N, D_in) = (D_in,)`

代码实现为

```python
for t in range(curr_num_tokens):
    for k in range(topk):
        expert_id = curr_topk_ids[t, k].item()
        grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
        W1_e = w1[expert_id]  # shape: (N, D_in)
        # grad_x: (N,) @ (N, D_in) -> (D_in,)
        grad_hidden_states[begin_chunk_idx + t] += grad_y_tk @ W1_e
```

---

- grad_w1 (∂L/∂W1)

$$\frac{\partial L}{\partial W_1^{(e)}} = \sum_{(t,k): \text{expert\_id}[t,k]=e} \text{grad}_y[t \cdot \text{topk} + k] \otimes x[t]$$

**维度：** `outer((N,), (D_in,)) = (N, D_in)`

代码实现为

```python
for t in range(curr_num_tokens):
    for k in range(topk):
        expert_id = curr_topk_ids[t, k].item()
        x_t = curr_hidden_states[t]  # shape: (D_in,)
        grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
        # grad_W1: outer(grad_y_tk, x_t) -> (N, D_in)
        # Convert to float32 for accumulation to avoid bfloat16 precision loss
        grad_w1[expert_id] = torch.outer(grad_y_tk, x_t).to(torch.float32)
```
---


# DownProjFunction

## forward

| 变量名 | 维度 | 含义                                               |
|--------|------|--------------------------------------------------|
| `intermediate_cache2` | `(num_tokens * topk, N//2)` | 经过 SiLU 激活和乘法后的中间结果，这里的N/2实际上就是immediate_size的大小 |
| `w2` | `(E, hidden_size, N//2)` | Down 投影权重矩阵，E 为专家数量                              |
| `topk_weights` | `(num_tokens, topk)` | Top-K 路由权重                                       |
| `topk_ids` | `(num_tokens, topk)` | Top-K 路由索引                                       |

- forward计算公式

$$\text{intermediate\_cache3}[t, k] = \text{topk\_weights}[t, k] \times (\text{intermediate\_cache2}[t \times \text{topk} + k] \times W_2^{(\text{expert\_id})})$$

`intermediate_cache3` 的形状为 `(num_tokens, topk, hidden_size)`

其中：
- $W_2^{(\text{expert\_id})}$ 的形状为 `(hidden_size, N//2)`
- 矩阵乘法：`(N//2,) @ (hidden_size, N//2).T = (hidden_size,)`

**关键特性：**
- 该阶段 **会乘以** `topk_weights`
- 输出维度从 `(num_tokens * topk, N//2)` 重组为 `(num_tokens, topk, hidden_size)`，便于后续的聚合操作



## backward

- input: grad_output
- **维度**: `(num_tokens, topk, hidden_size)`
- **含义**: 损失函数对输出的梯度

#### 1 grad_intermediate_cache2 (∂L/∂x)
- **维度**: `(num_tokens * topk, intermediate_size)`
- **计算公式**:
$$\frac{\partial L}{\partial x[t \cdot \text{topk} + k]} = w[t,k] \cdot \left(\frac{\partial L}{\partial y[t,k]} @ W_2^{(\text{expert\_id})}\right)$$

**维度分析**: 
- `grad_y[t,k]`: `(hidden_size,)`
- `W2[expert_id]`: `(hidden_size, intermediate_size)`
- `grad_y @ W2`: `(intermediate_size,)`
- 乘以标量 `topk_weight`：`(intermediate_size,)` 

- python实现

```python
for t in range(curr_num_tokens):
    for k in range(topk):
        expert_id = curr_topk_ids[t, k].item()
        grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
        W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)
        weight_tk = curr_topk_weights[t, k]  # scalar

        grad_intermediate_cache2[(begin_chunk_idx + t) * topk + k] = weight_tk * (grad_y_tk @ W2_e)

```

#### 2 grad_w2 (∂L/∂W2)
- **计算公式**:
$$\frac{\partial L}{\partial W_2^{(e)}} = \sum_{t,k: \text{expert\_id}[t,k]=e} w[t,k] \cdot \text{outer}\left(\frac{\partial L}{\partial y[t,k]}, x[t \cdot \text{topk} + k]\right)$$

**维度分析**: 
- `outer(grad_y, x)`: `outer((hidden_size,), (intermediate_size,)) = (hidden_size, intermediate_size)`
- 乘以标量 `topk_weight`: `(hidden_size, intermediate_size)` 
- 输出维度: `(E, hidden_size, intermediate_size)`


- python实现

```python
for t in range(curr_num_tokens):
    for k in range(topk):
        expert_id = curr_topk_ids[t, k].item()
        grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
        x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
        weight_tk = curr_topk_weights[t, k]  # scalar

        grad_w2[expert_id] += (weight_tk * torch.outer(grad_y_tk, x_tk)).to(torch.float32)
```

#### 3 grad_topk_weights (∂L/∂w)

- **计算公式**:
$$\frac{\partial L}{\partial w[t,k]} = \frac{\partial L}{\partial y[t,k]} \cdot (x[t \cdot \text{topk} + k] @ W_2^{(\text{expert\_id})}{}^T)$$

**解释**: 前向公式为 $y = w \cdot (x @ W^T)$，对 $w$ 求导得到 $\frac{\partial y}{\partial w} = x @ W^T$

**维度分析**:
- `x @ W2.T`: `(intermediate_size,) @ (intermediate_size, hidden_size) = (hidden_size,)`
- `dot(grad_y, forward_unweighted)`: `sum((hidden_size,) * (hidden_size,)) = scalar` 
- 输出维度：`(num_tokens, topk)`

- python实现

```python
for t in range(curr_num_tokens):
    for k in range(topk):
        expert_id = curr_topk_ids[t, k].item()
        grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
        x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
        W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)

        # Compute forward output before weighting: x @ w2.T
        forward_output_unweighted = x_tk @ W2_e.T  # shape: (hidden_size,)

        # grad_topk_weights: dot product
        grad_topk_weights[begin_chunk_idx + t, k] += grad_y_tk @ forward_output_unweighted
```

# 并行策略

## 一层并行

首先，不难看到所有的计算都是沿着 num_token 方向进行的，因此可以按照 num_token维度做一次分块

```bash
CHUNK_SIZE = 64 * 1024

# Initialize gradient tensors
grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
grad_w2 = torch.zeros_like(w2)
grad_topk_weights = torch.zeros_like(topk_weights)

# Process in chunks to match forward pass
for chunk in range((num_tokens // CHUNK_SIZE) + 1):
    begin_chunk_idx, end_chunk_idx = (
        chunk * CHUNK_SIZE,
        min((chunk + 1) * CHUNK_SIZE, num_tokens),
    )

    curr_num_tokens = end_chunk_idx - begin_chunk_idx
    if curr_num_tokens == 0:
        continue

    curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
    curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
    curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
    curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]
```

## 二层并行

### hidden_state

以GateProjFunction的Backward阶段计算$grad_hidden_size$为例子

本质上是要计算 $gard\_output \times w1e$

$gard\_output$ 的维度是 `(num_tokens * topk, N)`
$w1e$ 的维度是 `(E, N, hidden_size )`

网格的配置为

```python
# sorted_token_ids.shape[0] 是该chunk内token的数量
# N 是 w1 矩阵中第1维的维度(和上面的表示不同)

E, N, K = weight.shape
num_tokens = topk_ids.shape[0]

# 1. 网格配置 - 二维网格处理 (M, N) 维度
grid_input = lambda META: (
    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])  # M 方向的块数，沿着num_tokens方向计算
    * triton.cdiv(N, META["BLOCK_SIZE_N"]),  # N 方向的块数
)
```

> 这里有个问题是：GateUpProj时w1的维度为(E, N, hidden_size)，但是DownProj的时候w2的维度是 (E, hidden_size, N//2)，在GateUpProj阶段是沿着第一维去做计算，但是DownProj阶段是否需要沿着第二维去做计算呢？
> 可以但是由于N和hidden_state基本在一个数量级，所以没必要，参考forward阶段，两者都是按照第一维去做计算





枚举$K$维度去进行计算

```python

# Iterate over K dimension
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Current K offsets
    curr_offs_k = k * BLOCK_SIZE_K + offs_k

    # Load weight block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
    # weight: shape (E, N, K)
    weight_ptrs = (
        weight_ptr
        + off_experts * stride_we
        + offs_n[:, None] * stride_wn
        + curr_offs_k[None, :] * stride_wk
    )
    w = tl.load(
        weight_ptrs,
        mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
        other=0.0,
    )

    # Compute contribution: grad_out @ weight
    # grad_out: (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # w: (BLOCK_SIZE_N, BLOCK_SIZE_K)
    # result: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    contribution = tl.dot(grad_out, w)

    # Atomic add to grad_input because different N blocks contribute to same K
    grad_input_ptrs = grad_input_ptr + (
        (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
    )
    grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
    tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)

```


### w1/w2矩阵

### topk_weight矩阵