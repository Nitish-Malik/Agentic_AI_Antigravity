# Performance & Optimization: AMP, Profiling, and Distributed Training

This cheat sheet covers the advanced efficiency and scaling techniques tested in the **PyTorch Certified Associate (PTCA)** exam.

---

## 1. Automatic Mixed Precision (AMP)

AMP runs parts of your training loop in lower precision (like `float16` or `bfloat16`) to speed up execution and reduce GPU memory consumption, while keeping critical parameters (like weight updates and gradients) in `float32` to preserve accuracy.

### Code Implementation
To prevent gradient underflow (where small gradients become zero in `float16`), you must use a **`GradScaler`** alongside the **`autocast`** context manager.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 1. Initialize the Gradient Scaler (only needed for float16)
scaler = torch.cuda.amp.GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    
    # 2. Forward pass under autocast context manager
    with torch.amp.autocast(device_type=device, dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    # 3. Backward pass: Scale the loss to prevent gradient underflow
    scaler.scale(loss).backward()
    
    # 4. scaler.step() replaces optimizer.step() - unscales gradients internally
    scaler.step(optimizer)
    
    # 5. Update the scale factor for the next iteration
    scaler.update()
```

> [!IMPORTANT]
> Do NOT use `autocast` during model evaluation (`model.eval()`) unless you explicitly want to test mixed precision speed. During inference, you can simply use `with torch.inference_mode():` and optionally cast the inputs/model to `torch.float16` manually if speed is key.

---

## 2. Execution Profiling (`torch.profiler`)

The PyTorch Profiler allows you to analyze CPU and GPU execution times, identify bottlenecks in your code, and track memory allocation.

```python
import torch
import torch.profiler

model = MyModel().cuda()
inputs = torch.randn(64, 3, 224, 224).cuda()

# 1. Context manager for profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,      # Tracks input tensor shapes across operators
    profile_memory=True,     # Tracks VRAM/RAM allocations
    with_stack=True          # Tracks python call stacks
) as prof:
    # Use record_function to mark specific blocks in the trace
    with torch.profiler.record_function("model_inference"):
        outputs = model(inputs)

# 2. Print or inspect results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 3. Export tracing file for Chrome DevTools visualization (chrome://tracing)
prof.export_chrome_trace("trace.json")
```

---

## 3. Distributed Training Concepts

When scaling PyTorch models to multiple GPUs, you have two primary options: `DataParallel` (DP) and `DistributedDataParallel` (DDP).

### DP vs. DDP Comparison

| Feature | `DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|---|---|---|
| **Architecture** | Single-process, multi-threaded. | Multi-process (one process per GPU). |
| **GIL Bottleneck** | Yes (limited by Python GIL). | No (separate processes bypass GIL). |
| **Scaling** | Single-node only. | Multi-node and single-node scaling. |
| **Communication** | Replicates model on each step; high overhead. | Syncs gradients via Ring-AllReduce at backward pass. |
| **Recommendation** | Legacy; easy to implement but slow. | **Recommended standard** for all scaling. |

### DDP Core Implementation Steps

1.  **Initialize Process Group**: Setup communication backend (`nccl` for GPUs, `gloo` for CPUs).
    ```python
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method="env://")
    ```
2.  **Use `DistributedSampler`**: Ensures each GPU process gets a unique subset of data.
    ```python
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    ```
3.  **Wrap model in DDP**:
    ```python
    local_rank = int(os.environ["LOCAL_RANK"])
    model = MyModel().to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    ```
4.  **Launch via CLI**: Run training script using the PyTorch torchrun tool:
    ```bash
    torchrun --nproc_per_node=4 train_ddp.py
    ```
