# PyTorch Certified Associate (PTCA) Preparation Roadmap

Welcome to your study workspace for the **PyTorch Certified Associate (PTCA)** exam. This roadmap is structured around the four core domains of the official Linux Foundation exam blueprint.

---

## 📋 Exam Blueprint & Checklist

### 1. PyTorch Fundamentals (38%)
- [ ] **Tensors**: Creation (`torch.tensor`, `zeros`, `ones`, `randn`, `arange`), Attributes (`shape`, `dtype`, `device`), Operations (arithmetic, matrix multiplication `@`, indexing, slicing).
- [ ] **Reshaping & Manipulation**: `view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, concatenation/stacking (`cat`, `stack`).
- [ ] **Autograd & Computation Graphs**: `requires_grad=True`, `backward()`, gradient accumulation, disabling gradient calculations (`torch.no_grad()`, `torch.inference_mode()`).
- [ ] **Device Basics**: `torch.device`, moving tensors/models to CPU/CUDA/MPS (`.to(device)`), checking availability (`torch.cuda.is_available()`).

### 2. Performance & Optimization (26%)
- [ ] **Automatic Mixed Precision (AMP)**: Scaling floating-point representation (`torch.amp.autocast`, `torch.cuda.amp.GradScaler`) to optimize memory and speed.
- [ ] **Execution Profiling**: Tracking execution time and memory usage (`torch.profiler.profile`, `record_function`).
- [ ] **Distributed Training Concepts**: Understanding scalability via `DistributedDataParallel` (DDP) versus standard `DataParallel`.

### 3. Model Development (20%)
- [ ] **Neural Network Modules**: Subclassing `nn.Module`, defining structures in `__init__`, implementing forward passes in `forward()`.
- [ ] **Common Layers & Activations**: `nn.Linear`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.Dropout`, activation functions (`nn.ReLU`, `nn.Sigmoid`, `nn.Softmax`).
- [ ] **Loss Functions**: `nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.BCELoss`, `nn.BCEWithLogitsLoss`.
- [ ] **Optimizers**: `torch.optim.SGD`, `torch.optim.Adam`, updating weights via `optimizer.step()`, resetting gradients via `optimizer.zero_grad()`.

### 4. Data Handling (16%)
- [ ] **Custom Datasets**: Subclassing `torch.utils.data.Dataset`, implementing `__len__()` and `__getitem__()`.
- [ ] **DataLoaders**: Batching, shuffling, using `num_workers` for parallel preprocessing, and `pin_memory=True` for faster GPU transfers.
- [ ] **Transforms**: Applying preprocessing and data augmentation transforms.

---

## 📂 Study Files in this Workspace
To help you master these topics, the following files have been prepared in your workspace:

| File | Type | Description |
|---|---|---|
| [fundamentals.md](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/cheat_sheets/fundamentals.md) | Cheat Sheet | Core tensor operations, indexing, slicing, autograd, and devices. |
| [model_development.md](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/cheat_sheets/model_development.md) | Cheat Sheet | `nn.Module` lifecycle, layers, losses, and optimization workflows. |
| [data_handling.md](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/cheat_sheets/data_handling.md) | Cheat Sheet | Subclassing Datasets, configuring DataLoaders, and Transforms. |
| [performance_optimization.md](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/cheat_sheets/performance_optimization.md) | Cheat Sheet | Mixed precision (AMP), Profiler, and Distributed Training. |
| [demo_training_pipeline.py](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/demo_training_pipeline.py) | Code Lab | Practical implementation of model building, data pipeline, and training loop. |
| [demo_optimization.py](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/demo_optimization.py) | Code Lab | Practical implementation of `torch.amp` and `torch.profiler`. |
| [practice_quiz.py](file:///home/malik/.gemini/antigravity/scratch/pytorch_certification_prep/practice_quiz.py) | App | Interactive multiple-choice quiz covering realistic exam questions. |
