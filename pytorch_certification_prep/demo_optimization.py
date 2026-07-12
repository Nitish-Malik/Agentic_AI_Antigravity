#!/usr/bin/env python3
"""
PTCA Practice Lab: Model Performance & Optimization.
This script demonstrates:
1. Using Automatic Mixed Precision (AMP) with torch.amp.autocast and GradScaler
2. Profiling model execution time and memory via torch.profiler.profile
3. Tagging specific blocks using torch.profiler.record_function
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler

class LargeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a heavy convolutional layer sequence
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 56 * 56, 10)
        )

    def forward(self, x):
        return self.features(x)

def run_amp_training_step(model, optimizer, criterion, inputs, targets, device, scaler):
    """
    Executes a single training iteration utilizing Automatic Mixed Precision (AMP).
    """
    optimizer.zero_grad()
    
    # autocast type selection: float16 for CUDA, bfloat16 for CPU (if supported/available)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    # 1. Forward pass runs in lower precision under autocast context manager
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    # 2. Backward pass with scaled loss (only scaled on CUDA with GradScaler)
    if device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
        
    return loss.item()

def main():
    print("--- Starting PyTorch Performance & Optimization Lab ---")
    
    # Hardware acceleration detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # Initialize components
    model = LargeConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate GradScaler (only relevant/functional on CUDA/GPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Mock Batch
    batch_size = 16
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # Warmup step to initialize lazy parameters/drivers
    print("Running warmup pass...")
    run_amp_training_step(model, optimizer, criterion, inputs, targets, device, scaler)
    
    # ==========================================
    # Profiler Configuration
    # ==========================================
    print("\nStarting execution profiling...")
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Use record_function to label code blocks in the trace profile
        with torch.profiler.record_function("amp_training_iteration"):
            loss_val = run_amp_training_step(model, optimizer, criterion, inputs, targets, device, scaler)
            
    print(f"Training step complete. Loss: {loss_val:.4f}")
    
    # Print the profiler summary
    print("\n--- Profiler Execution Summary (Sorted by Total Time) ---")
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    
    # Optional: Save Chrome trace to view on chrome://tracing
    trace_path = "profiler_trace.json"
    print(f"\nExporting detailed chrome trace to {trace_path}...")
    prof.export_chrome_trace(trace_path)
    import os
    if os.path.exists(trace_path):
        os.remove(trace_path) # Clean up generated trace to keep workspace neat
        print("Trace exported and verified (cleaned up).")

if __name__ == "__main__":
    main()
