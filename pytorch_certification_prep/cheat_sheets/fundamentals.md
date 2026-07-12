# PyTorch Fundamentals: Tensors, Autograd, & Devices

This cheat sheet covers the essential PyTorch fundamentals tested in the **PyTorch Certified Associate (PTCA)** exam.

---

## 1. Tensor Creation and Attributes

Tensors are the multi-dimensional arrays that form the core data structure in PyTorch.

```python
import torch
import numpy as np

# 1. Direct creation
t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # Infers dtype (float32)

# 2. Factory functions
t_zeros = torch.zeros(3, 3)                # 3x3 of 0.0
t_ones = torch.ones(2, 3, dtype=torch.int32) # 2x3 of 1 (int32)
t_rand = torch.randn(4, 4)                 # Standard normal distribution (mean=0, std=1)
t_arange = torch.arange(start=0, end=10, step=2) # [0, 2, 4, 6, 8]
t_linspace = torch.linspace(0, 10, steps=5) # [0., 2.5, 5., 7.5, 10.]

# 3. From NumPy arrays (shares memory!)
np_arr = np.array([1, 2, 3])
t_from_np = torch.from_numpy(np_arr)       # Modifying np_arr modifies t_from_np
t_as_tensor = torch.as_tensor(np_arr)      # Similar to from_numpy (prefers memory sharing)

# To break memory sharing, use clone() or copy:
t_copy = torch.tensor(np_arr)              # Copies data

# 4. Cloning existing tensors
t_like = torch.ones_like(t1)               # Copies shape and dtype of t1, fills with 1s
```

### Core Attributes
*   **`.shape`** (or `.size()`): Returns a `torch.Size` tuple representing dimensions.
*   **`.dtype`**: Data type of elements (e.g., `torch.float32`, `torch.int64`, `torch.float16`).
*   **`.device`**: Where the tensor is stored (`cpu`, `cuda:0`, `mps`).
*   **`.layout`**: Memory layout (default is `torch.strided`).

---

## 2. Tensor Operations

### Arithmetic & Reductions
```python
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])

# Element-wise operations
add_res = x + y         # torch.add(x, y)
sub_res = x - y         # torch.sub(x, y)
mul_res = x * y         # torch.mul(x, y) - Element-wise multiplication
div_res = x / y         # torch.div(x, y)

# In-place operations (denoted by trailing underscore '_')
x.add_(y)               # Modifies x directly

# Reductions
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
t.sum()                 # Scalar sum: 10.0
t.mean(dim=0)           # Column means: [2.0, 3.0]
t.std(dim=1)            # Row standard deviations
```

### Matrix Multiplication
*   **1D Vectors (Dot Product)**: `torch.dot(vec1, vec2)`
*   **2D Matrices**: `torch.matmul(A, B)` or `A @ B` or `torch.mm(A, B)`.
*   **Batched Matrix Multiplication**: `torch.bmm(A_batched, B_batched)` (requires 3D inputs: `[batch, n, m]`).

---

## 3. Reshaping, Slicing, and Manipulating

```python
x = torch.randn(2, 3, 4)

# 1. view vs. reshape
# view() requires contiguous memory; reshape() works on non-contiguous by copying if needed.
x_view = x.view(6, 4)
x_reshape = x.reshape(-1, 2) # -1 infers dimension size automatically

# 2. Transpose & Permute
# transpose(dim0, dim1) swaps two dimensions
x_transposed = x.transpose(0, 1) # Shape: (3, 2, 4)
# permute(*dims) reorders all dimensions
x_permuted = x.permute(2, 0, 1)  # Shape: (4, 2, 3)

# Note: transpose and permute return non-contiguous views. Use .contiguous() if view() is needed next.

# 3. Squeeze & Unsqueeze
y = torch.randn(1, 3, 1, 4)
y_squeezed = y.squeeze()       # Removes all dimensions of size 1 -> Shape: (3, 4)
y_sq_dim2 = y.squeeze(dim=2)   # Removes specific dimension if size is 1 -> Shape: (1, 3, 4)
y_unsqueezed = y_squeezed.unsqueeze(dim=0) # Inserts dimension of size 1 at dim 0 -> Shape: (1, 3, 4)

# 4. Cat & Stack
a = torch.randn(2, 3)
b = torch.randn(2, 3)
cat_dim0 = torch.cat([a, b], dim=0)   # Concatenates along existing dim -> Shape: (4, 3)
stack_dim0 = torch.stack([a, b], dim=0) # Stacks along new dimension -> Shape: (2, 2, 3)
```

---

## 4. Autograd & Computation Graph

PyTorch uses a dynamic Directed Acyclic Graph (DAG) to track computations and calculate gradients automatically.

```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 4 * x

# Compute gradients via backpropagation
y.backward()

# Access gradient d(y)/d(x) = 3*x^2 + 4
print(x.grad) # 3*(2.0)^2 + 4 = 16.0
```

### Key Autograd Rules
1.  **Leaf Tensors**: Tensors created directly (not as outputs of operations). Only leaf tensors with `requires_grad=True` accumulate gradients in their `.grad` field by default.
2.  **`backward()`**: Must be called on a scalar tensor (e.g., loss). If called on a vector/matrix, you must pass a matching gradient tensor: `y.backward(torch.ones_like(y))`.
3.  **Gradient Accumulation**: PyTorch accumulates gradients in `.grad` rather than overwriting them. You must call `optimizer.zero_grad()` or reset `.grad = None` before each training backward pass.

### Context Managers for Inference
To disable gradient calculations and save memory/speed up execution:

```python
# 1. torch.inference_mode() - RECOMMENDED (Newer, faster, safer)
with torch.inference_mode():
    y = x * 2 # No tracking graph created, operates directly

# 2. torch.no_grad() - Legacy, but still widely used
with torch.no_grad():
    y = x * 2
```

---

## 5. Hardware Device Management

PyTorch models and tensors must reside on the same device to perform computations.

```python
# Detect available accelerator
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon GPU
else:
    device = torch.device("cpu")

# Move tensors to device
x = torch.randn(2, 3).to(device)
# Alternative syntax:
x = torch.randn(2, 3, device=device)

# Move models to device (in-place operation for models)
model = MyNeuralNetwork().to(device)
```

> [!WARNING]
> While moving a model `model.to(device)` is performed in-place (modifies the existing object), moving a tensor `x.to(device)` returns a **new tensor** on that device. You must assign it: `x = x.to(device)`.
