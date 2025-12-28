"""GPU Accelerators Example.

Demonstrates how to:
- Request specific GPU types (A100, H100, etc.)
- Detect available hardware on the remote instance
- Run GPU-accelerated computations
"""

from skyward import AWS, NVIDIA, ComputePool, compute, instance_info, InstanceInfo


@compute
def instance_information() -> InstanceInfo:
    """Get information about available GPUs."""
    return instance_info()


@compute
def matrix_multiply(size: int) -> dict:
    """Benchmark matrix multiplication on GPU vs CPU."""
    import time

    import torch

    results = {}

    # CPU benchmark
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    start = time.time()
    _ = torch.matmul(a_cpu, b_cpu)
    results["cpu_time"] = time.time() - start

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        a_gpu = torch.randn(size, size, device="cuda")
        b_gpu = torch.randn(size, size, device="cuda")

        # Warmup
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        results["gpu_time"] = time.time() - start
        results["speedup"] = results["cpu_time"] / results["gpu_time"]

    return results


@compute
def train_simple_model(epochs: int) -> dict:
    """Train a simple neural network on GPU."""
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # Synthetic data (like MNIST)
    x = torch.randn(1000, 784, device=device)
    y = torch.randint(0, 10, (1000,), device=device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {
        "device": device,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "epochs": epochs,
    }


if __name__ == "__main__":
    # =================================================================
    # Pool with NVIDIA T4 GPU
    # =================================================================
    with ComputePool(
        provider=AWS(),
        accelerator='T4',
        pip=["torch"],
        spot="always",
    ) as pool:
        # Get GPU information
        info = instance_information() >> pool
        print(f"GPU Info: {info}")

        # Benchmark matrix multiplication
        benchmark = matrix_multiply(4096) >> pool
        print(f"Matmul 4096x4096:")
        print(f"  CPU: {benchmark['cpu_time']:.3f}s")
        if "gpu_time" in benchmark:
            print(f"  GPU: {benchmark['gpu_time']:.3f}s")
            print(f"  Speedup: {benchmark['speedup']:.1f}x")

        # Train a simple model
        result = train_simple_model(epochs=100) >> pool
        print(f"Training on {result['device']}:")
        print(f"  Initial loss: {result['initial_loss']:.4f}")
        print(f"  Final loss: {result['final_loss']:.4f}")

    # =================================================================
    # Other GPU options (commented for reference)
    # =================================================================
    # NVIDIA.H100      # Latest and fastest
    # NVIDIA.A100      # Great for training
    # NVIDIA.L4        # Cost-effective inference
    # NVIDIA.T4        # Budget option
    # NVIDIA.V100      # Older but capable
