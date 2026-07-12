# Model Development: nn.Module, Layers, Losses, & Optimizers

This cheat sheet covers PyTorch neural network construction and training loop structures tested in the **PyTorch Certified Associate (PTCA)** exam.

---

## 1. Subclassing `nn.Module`

In PyTorch, all neural network models subclass `torch.nn.Module`.

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        # Always call the super constructor
        super().__init__()
        
        # Define layers as instance variables
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Define the forward computation flow
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

> [!NOTE]
> Never call `model.forward(x)` directly. Always call the model as a callable: `model(x)`. This ensures that all hooks (like forward and backward pre/post hooks) registered on the module are executed correctly.

---

## 2. Common Neural Network Layers

PyTorch provides pre-built layers in the `torch.nn` module.

### Linear/Dense Layers
*   **`nn.Linear(in_features, out_features, bias=True)`**: Applies a linear transformation: $y = xA^T + b$.

### Convolutional & Pooling Layers
*   **`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`**: 2D convolution for image inputs of shape `[batch, channels, height, width]`.
*   **`nn.MaxPool2d(kernel_size, stride=None, padding=0)`**: Downsamples inputs by taking the maximum value.
*   **`nn.Flatten(start_dim=1, end_dim=-1)`**: Flattens dimensions (usually from Conv2D output shape `[batch, C, H, W]` to linear input shape `[batch, C*H*W]`).

### Regularization Layers
*   **`nn.Dropout(p=0.5)`**: Randomly zeroes elements during training to prevent overfitting.
*   **`nn.BatchNorm2d(num_features)`**: Normalizes activations across the batch/spatial dimensions.

### Activation Functions
*   **`nn.ReLU()`**: Rectified Linear Unit ($f(x) = \max(0, x)$).
*   **`nn.Sigmoid()`**: Sigmoid function, outputs values in $(0, 1)$.
*   **`nn.Softmax(dim)`**: Softmax normalization, turns logits into probability distributions.

---

## 3. Loss Functions (`nn.modules.loss`)

Choosing the correct loss function depends on your model task:

| Loss Function | Use Case | Target Type | Notes |
|---|---|---|---|
| **`nn.MSELoss()`** | Regression | Real numbers | Mean Squared Error. |
| **`nn.L1Loss()`** | Regression | Real numbers | Mean Absolute Error. |
| **`nn.BCELoss()`** | Binary Classification | Float probability in $[0,1]$ | Requires output of `nn.Sigmoid()`. |
| **`nn.BCEWithLogitsLoss()`** | Binary Classification | Float logits (raw scores) | Integrates sigmoid internally for numerical stability. **Preferred over BCELoss**. |
| **`nn.CrossEntropyLoss()`** | Multi-class Classification | Integer class indices ($0$ to $C-1$) | Integrates `nn.LogSoftmax()` and `nn.NLLLoss()` internally. **Expects raw logits** (do not apply Softmax at the end of the model). |

---

## 4. Optimizers (`torch.optim`)

Optimizers update the model parameters (weights and biases) based on computed gradients.

```python
import torch.optim as optim

model = SimpleClassifier(10, 20, 2)

# 1. Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 2. Adam (Adaptive Moment Estimation)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

---

## 5. The Canonical Training and Evaluation Loops

You must manage model state flags (`train()` vs. `eval()`) during the cycle.

```python
# --- TRAINING LOOP ---
model.train() # Enable training mode (enables Dropout, BatchNorm updates)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 1. Zero out any accumulated gradients
        optimizer.zero_grad() # Alternatively: for param in model.parameters(): param.grad = None
        
        # 2. Forward pass
        outputs = model(inputs)
        
        # 3. Compute loss
        loss = criterion(outputs, targets)
        
        # 4. Backward pass (computes gradients)
        loss.backward()
        
        # 5. Update weights
        optimizer.step()

# --- EVALUATION LOOP ---
model.eval() # Disable training mode (turns off Dropout, freezes BatchNorm stats)

total_loss = 0.0
with torch.inference_mode(): # Disable gradient tracking
    for inputs, targets in val_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
```

> [!IMPORTANT]
> Always call `model.train()` before starting training, and `model.eval()` before performing validation or inference. Forgetting this can lead to unexpected model behaviors due to layers like `nn.Dropout` or `nn.BatchNorm2d` retaining their training behaviors.
