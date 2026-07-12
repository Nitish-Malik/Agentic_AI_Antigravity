# Data Handling: Datasets, DataLoaders, and Transforms

This cheat sheet covers PyTorch's data ingestion pipelines tested in the **PyTorch Certified Associate (PTCA)** exam.

---

## 1. Custom Datasets

To create a custom dataset, subclass `torch.utils.data.Dataset` and override three key methods:

```python
import torch
from torch.utils.data import Dataset

class CustomNumericDataset(Dataset):
    def __init__(self, data_features, data_labels, transform=None):
        """
        Initialize data source (paths, arrays, databases).
        """
        self.features = torch.tensor(data_features, dtype=torch.float32)
        self.labels = torch.tensor(data_labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Fetch a single sample (features, label) at index `idx`.
        """
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
```

---

## 2. The DataLoader

The `torch.utils.data.DataLoader` wraps a `Dataset` and provides an iterable over the dataset, adding features like batching, shuffling, and multi-threaded loading.

```python
from torch.utils.data import DataLoader

dataset = CustomNumericDataset(features_np, labels_np)

dataloader = DataLoader(
    dataset,
    batch_size=32,       # Number of samples per batch
    shuffle=True,        # Set to True for training, False for validation/test
    num_workers=4,       # Number of subprocesses to use for data loading
    pin_memory=True,     # Allocates samples in page-locked memory for faster GPU copies
    drop_last=False,     # If True, drops the last incomplete batch if dataset size isn't divisible by batch_size
    timeout=0            # If positive, the timeout value for collecting a batch from workers
)
```

### Performance Best Practices
*   **`num_workers`**: Setting `num_workers > 0` enables asynchronous data loading using separate CPU threads. A standard guideline is to set it to `4 * num_gpus` or `os.cpu_count()`.
*   **`pin_memory=True`**: When training on a GPU (`cuda`), setting `pin_memory=True` speeds up data transfers from host (CPU RAM) to device (GPU VRAM) because PyTorch can bypass standard page checks.
*   **`shuffle=True`**: Essential during training to prevent the model from learning dependencies based on the ordering of samples. Set `shuffle=False` for validation and testing to ensure consistent, deterministic evaluation.

---

## 3. Data Transformations (`torchvision.transforms`)

Transforms are used to perform data preprocessing and data augmentation.

```python
import torchvision.transforms as transforms

# Define a pipeline of transforms using Compose
train_transforms = transforms.Compose([
    transforms.ToPILImage(),             # Convert tensor or ndarray to PIL Image
    transforms.Resize((224, 224)),       # Resize image to 224x224
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation: random horizontal flip
    transforms.ToTensor(),               # Convert PIL Image or numpy array to PyTorch Tensor (scales to [0.0, 1.0])
    transforms.Normalize(                # Normalize with mean and standard deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Key Transform Behaviors
1.  **`ToTensor()`**:
    *   Converts a PIL Image or numpy array (shape: `H x W x C` in range `[0, 255]`) to a float tensor of shape `C x H x W`.
    *   **Scales** the values from the range `[0, 255]` to the range `[0.0, 1.0]`.
2.  **`Normalize(mean, std)`**:
    *   Normalizes an input image tensor: `output = (input - mean) / std`.
    *   Operates **in-place** on the input tensor. Must be applied *after* `ToTensor()`.

---

## 4. Tensor Shape Conventions

| Area | Format | Dimensions | Example Shape |
|---|---|---|---|
| **Single Image** | `[Channels, Height, Width]` | 3D | `[3, 224, 224]` (RGB Image) |
| **Batch of Images** | `[Batch, Channels, Height, Width]` | 4D | `[64, 3, 224, 224]` |
| **Linear Batch** | `[Batch, Features]` | 2D | `[64, 768]` (Fully connected input) |
| **Sequence Batch** | `[Batch, SeqLen, Features]` | 3D | `[32, 100, 512]` (NLP input) |
