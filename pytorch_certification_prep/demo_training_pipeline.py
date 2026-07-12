#!/usr/bin/env python3
"""
PTCA Practice Lab: End-to-End Training and Evaluation Pipeline.
This script demonstrates:
1. Subclassing torch.utils.data.Dataset
2. Defining a model with torch.nn.Module
3. Running a standard training loop (model.train(), zero_grad(), backward(), step())
4. Running a standard validation loop (model.eval(), torch.inference_mode())
5. Saving and loading model parameters (state_dict)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Custom Dataset Definition
# ==========================================
class MockClassificationDataset(Dataset):
    def __init__(self, num_samples=500, input_dim=20, num_classes=3):
        # Generate random mock features and labels
        self.features = torch.randn(num_samples, input_dim)
        # Random class targets (integers from 0 to num_classes-1)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==========================================
# 2. Model Architecture
# ==========================================
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # Layers definition
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Forward pass flow
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print("--- Starting PyTorch Training Pipeline Lab ---")

    # Hyperparameters
    input_dim = 20
    hidden_dim = 64
    num_classes = 3
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.01

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using hardware accelerator device: {device}")

    # Data Ingestion
    train_dataset = MockClassificationDataset(num_samples=800, input_dim=input_dim, num_classes=num_classes)
    val_dataset = MockClassificationDataset(num_samples=200, input_dim=input_dim, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss (criterion), and optimizer
    model = SimpleMLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    # CrossEntropyLoss expects raw logits and integer target indices
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for epoch in range(num_epochs):
        model.train() # Set flag: Enable Dropout/BatchNorm updates
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            # Move data to target device
            inputs, targets = inputs.to(device), targets.to(device)

            # A. Reset gradients
            optimizer.zero_grad()

            # B. Forward pass
            outputs = model(inputs)

            # C. Compute loss
            loss = criterion(outputs, targets)

            # D. Backward pass (computes gradients)
            loss.backward()

            # E. Step optimizer (updates parameters)
            optimizer.step()

            # Track training metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # ==========================================
    # 4. Evaluation Loop
    # ==========================================
    model.eval() # Set flag: Disable Dropout, freeze BatchNorm stats
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.inference_mode(): # Disable gradient tracking (saves memory and speeds up runtime)
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    avg_val_loss = val_loss / total_val
    avg_val_acc = (correct_val / total_val) * 100
    print(f"Validation Complete | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")

    # ==========================================
    # 5. Model Serialization (Saving & Loading)
    # ==========================================
    model_path = "simple_mlp.pth"
    print(f"Saving model state dictionary to {model_path}...")
    torch.save(model.state_dict(), model_path)

    # Re-loading model to verify parameters
    new_model = SimpleMLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    print("Loading weights into a fresh model instance...")
    new_model.load_state_dict(torch.load(model_path, weights_only=True))
    new_model.eval()
    print("Model successfully saved, loaded, and verified!")

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == "__main__":
    main()
