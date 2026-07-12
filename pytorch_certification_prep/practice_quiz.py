#!/usr/bin/env python3
"""
PTCA Practice Quiz: PyTorch Certified Associate Mock Exam.
An interactive CLI tool to test your PyTorch knowledge across the four domains:
1. PyTorch Fundamentals (38%)
2. Performance & Optimization (26%)
3. Model Development (20%)
4. Data Handling (16%)
"""

import sys

# Questions Database
QUESTIONS = [
    {
        "domain": "PyTorch Fundamentals",
        "question": "What is the primary difference between torch.view() and torch.reshape()?",
        "options": [
            "view() copies the underlying tensor data, whereas reshape() always returns a memory-sharing view.",
            "view() is restricted to contiguous tensors and always returns a memory-sharing view. reshape() can operate on non-contiguous tensors, automatically copying data to a new memory block if necessary.",
            "view() is exclusively used for 2D matrices, whereas reshape() supports arbitrary dimensions.",
            "There is no operational difference; they are exact aliases in PyTorch."
        ],
        "answer": "B",
        "explanation": "view() returns a new tensor with the same data but different shape, and requires the underlying tensor to be contiguous in memory. reshape() will return a view if possible, but if the tensor is non-contiguous, it copies the data to make it contiguous, making it safer but slightly more memory-intensive in some cases."
    },
    {
        "domain": "PyTorch Fundamentals",
        "question": "If you have a tensor x on CPU and a model instance on CPU, which of the following is correct regarding device transfers?",
        "options": [
            "Calling x.to('cuda') moves the tensor in-place; no assignment is needed.",
            "Calling model.to('cuda') returns a new model instance; you must assign it: model = model.to('cuda').",
            "Calling x.to('cuda') returns a new tensor; you must assign it: x = x.to('cuda'). Calling model.to('cuda') is an in-place operation.",
            "Both tensor and model device transfers are always in-place."
        ],
        "answer": "C",
        "explanation": "In PyTorch, model.to(device) is an in-place operation that moves all model parameters and buffers. However, tensor.to(device) is NOT in-place; it returns a new tensor copy on the target device, so you must assign it: x = x.to(device)."
    },
    {
        "domain": "PyTorch Fundamentals",
        "question": "Which context manager should be preferred for disabling gradient calculations during model evaluation (inference) in PyTorch (v1.9+)?",
        "options": [
            "torch.no_grad()",
            "torch.inference_mode()",
            "torch.autograd.set_detect_anomaly(True)",
            "torch.enable_grad(False)"
        ],
        "answer": "B",
        "explanation": "While torch.no_grad() is widely used, torch.inference_mode() is a newer, faster, and safer context manager introduced in PyTorch 1.9. It offers extreme performance improvements by disabling tracking entirely, bypassing view tracking checks, and preventing tensor version counter increments."
    },
    {
        "domain": "PyTorch Fundamentals",
        "question": "If you have three tensors of shape [2, 3] and stack them using torch.stack(..., dim=0), what is the shape of the resulting tensor?",
        "options": [
            "[6, 3]",
            "[2, 9]",
            "[3, 2, 3]",
            "[2, 3, 3]"
        ],
        "answer": "C",
        "explanation": "torch.stack concatenates a sequence of tensors along a NEW dimension. Stacking three tensors of shape [2, 3] along dim=0 creates a new dimension of size 3 at index 0, resulting in [3, 2, 3]. (Conversely, torch.cat concatenates along an EXISTING dimension, which would yield [6, 3] if dim=0)."
    },
    {
        "domain": "Model Development",
        "question": "Why is it best practice to execute a forward pass by calling model(x) instead of model.forward(x)?",
        "options": [
            "model(x) compiles the PyTorch forward pass to native C++ on the fly.",
            "Calling forward() directly bypasses hooks (pre/post-forward and backward hooks) registered on the module.",
            "model(x) automatically moves the input tensor x to the correct device.",
            "forward() is a private method and cannot be accessed outside the class definition."
        ],
        "answer": "B",
        "explanation": "The __call__ method of nn.Module handles the execution of registered forward and backward hooks, and then calls the forward() method. Calling model.forward(x) directly bypasses all registered hooks, which can break tools like profilers, weight trackers, or activation savers."
    },
    {
        "domain": "Model Development",
        "question": "When using nn.CrossEntropyLoss, what activation function should you apply to the final output layer of your neural network?",
        "options": [
            "nn.Softmax(dim=-1)",
            "nn.LogSoftmax(dim=-1)",
            "No activation function (return raw logits)",
            "nn.Sigmoid()"
        ],
        "answer": "C",
        "explanation": "nn.CrossEntropyLoss combines LogSoftmax and NLLLoss (Negative Log Likelihood Loss) in a single class. Therefore, the model should return raw unnormalized scores (logits) directly. Applying Softmax at the end of the model when using CrossEntropyLoss is redundant and degrades gradient calculation stability."
    },
    {
        "domain": "Model Development",
        "question": "What is the function of optimizer.zero_grad() (or parameter.grad = None) in a PyTorch training loop?",
        "options": [
            "It clears model weights, resetting the neural network to randomly initialized states.",
            "It resets gradient buffers to zero, preventing gradients from accumulating across separate mini-batches.",
            "It turns off gradient calculation during the validation phase.",
            "It resets the learning rate scheduler to its initial learning rate."
        ],
        "answer": "B",
        "explanation": "By default, PyTorch accumulates (adds) gradients in param.grad during each .backward() call. To prevent gradients from mixing across different batches, you must reset them using optimizer.zero_grad() at the beginning of each iteration."
    },
    {
        "domain": "Performance & Optimization",
        "question": "In Automatic Mixed Precision (AMP) training, what does a GradScaler do?",
        "options": [
            "It scales neural network layers dynamically to prevent activation explosion.",
            "It multiplies loss values by a scale factor to prevent gradient underflow in float16 representation, then unscales them before the optimizer step.",
            "It reduces the batch size when GPU memory is full.",
            "It scales learning rates depending on gradient magnitude."
        ],
        "answer": "B",
        "explanation": "In float16 (half-precision), small gradient values can underflow to zero. A GradScaler multiplies the loss by a scale factor before backward() to inflate the gradient values, prevents them from underflowing, and then scales them back down (unscaling) before weight updates are applied by the optimizer."
    },
    {
        "domain": "Performance & Optimization",
        "question": "Which of the following is NOT a reason to prefer DistributedDataParallel (DDP) over DataParallel (DP) for multi-GPU training?",
        "options": [
            "DP is single-process and multi-threaded, meaning it suffers from Python's Global Interpreter Lock (GIL) bottleneck.",
            "DP replicates the model on each device on every step, leading to substantial overhead.",
            "DDP cannot scale to multiple machines (nodes), whereas DP can.",
            "DDP uses efficient Ring-AllReduce collective communications to synchronize gradients."
        ],
        "answer": "C",
        "explanation": "This statement is false, making it the correct answer. DDP supports scaling to multiple machines (nodes) and multiple processes. DP is limited to a single machine only. DDP is widely preferred over DP for all multi-GPU workflows."
    },
    {
        "domain": "Performance & Optimization",
        "question": "Which PyTorch tool is designed to identify execution bottlenecks, operator timings, and device memory allocations?",
        "options": [
            "torch.utils.tensorboard",
            "torch.profiler.profile",
            "torch.autograd.detect_anomaly",
            "torch.cuda.memory_summary"
        ],
        "answer": "B",
        "explanation": "torch.profiler.profile is PyTorch's built-in profiler that monitors execution time and memory use across CPU and GPU operators, allowing developers to identify bottleneck hotspots."
    },
    {
        "domain": "Data Handling",
        "question": "When subclassing torch.utils.data.Dataset, which set of methods MUST be implemented?",
        "options": [
            "__init__, forward, backward",
            "__init__, __len__, __getitem__",
            "__init__, __iter__, __next__",
            "__init__, setup, teardown"
        ],
        "answer": "B",
        "explanation": "A map-style PyTorch dataset must implement __init__ (initialization), __len__ (total sample count), and __getitem__ (retrieval of a sample at a given index)."
    },
    {
        "domain": "Data Handling",
        "question": "What is the primary benefit of setting pin_memory=True in a PyTorch DataLoader?",
        "options": [
            "It shuffles dataset indices in place to speed up randomized batching.",
            "It copies data samples directly to GPU VRAM.",
            "It allocates CPU tensors in page-locked (pinned) memory, allowing for much faster, asynchronous copying of tensors to GPU memory.",
            "It locks the model weights in CPU cache."
        ],
        "answer": "C",
        "explanation": "pin_memory=True tells the DataLoader to allocate host (CPU) memory in page-locked (pinned) memory. This allows host-to-device transfers to bypass CPU copies and use faster Direct Memory Access (DMA) transfers, improving GPU training pipeline speed."
    }
]

def run_quiz():
    print("=" * 60)
    print("      PyTorch Certified Associate (PTCA) Practice Quiz")
    print("=" * 60)
    print("Test your knowledge. Enter A, B, C, or D for each question.\n")
    
    score = 0
    total = len(QUESTIONS)
    
    for idx, item in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {idx}/{total} [{item['domain']}]")
        print("-" * 50)
        print(item["question"])
        print()
        
        choices = ["A", "B", "C", "D"]
        for letter, option in zip(choices, item["options"]):
            print(f"  {letter}) {option}")
            
        print()
        
        while True:
            user_ans = input("Your Answer (A/B/C/D) or 'q' to quit: ").strip().upper()
            if user_ans == 'Q':
                print("\nQuiz terminated. Good luck with your studies!")
                sys.exit(0)
            if user_ans in choices:
                break
            print("Invalid input. Please enter A, B, C, or D.")
            
        if user_ans == item["answer"]:
            print("\n✅ CORRECT!")
            score += 1
        else:
            print(f"\n❌ INCORRECT. The correct answer was {item['answer']}.")
            
        print(f"\nExplanation:\n{item['explanation']}")
        print("=" * 60)
        input("Press Enter for the next question...")
        
    print("\n" + "=" * 60)
    print(f"Quiz Complete! Your Score: {score}/{total} ({score/total * 100:.1f}%)")
    print("=" * 60)
    if score == total:
        print("Outstanding! You got a perfect score. You are well prepared!")
    elif score >= int(total * 0.8):
        print("Great job! You have a solid understanding of PyTorch basics.")
    else:
        print("Review the cheat sheets and practice code labs, then try again!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_quiz()
