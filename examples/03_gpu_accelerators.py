"""GPU Accelerators Example.

Demonstrates how to:
- Request specific GPU types (A100, H100, etc.)
- Detect available hardware on the remote instance
- Run GPU-accelerated computations
"""

import skyward as sky


@sky.compute
def instance_information() -> sky.InstanceInfo:
    """Get information about available GPUs."""
    return sky.instance_info()


@sky.compute
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


@sky.compute
def heavy_matrix_ops(size: int, iterations: int) -> dict:
    """Heavy matrix operations that take ~5s on CPU.

    Performs repeated matrix multiplications and element-wise operations
    to create a workload that clearly shows GPU acceleration benefits.
    """
    import time

    import torch

    results = {}

    # CPU benchmark - repeated matmuls and ops
    print(f"Starting CPU benchmark ({size}x{size}, {iterations} iterations)...")
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    start = time.time()
    result = a_cpu
    for i in range(iterations):
        result = torch.matmul(result, b_cpu)
        result = torch.relu(result)
        result = result / result.norm()  # Normalize to prevent overflow
    cpu_time = time.time() - start
    results["cpu_time"] = cpu_time
    print(f"CPU: {cpu_time:.2f}s")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        print(f"Starting GPU benchmark ({size}x{size}, {iterations} iterations)...")
        a_gpu = torch.randn(size, size, device="cuda")
        b_gpu = torch.randn(size, size, device="cuda")

        # Warmup
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        start = time.time()
        result = a_gpu
        for i in range(iterations):
            result = torch.matmul(result, b_gpu)
            result = torch.relu(result)
            result = result / result.norm()
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        results["gpu_time"] = gpu_time
        results["speedup"] = cpu_time / gpu_time
        print(f"GPU: {gpu_time:.2f}s (speedup: {results['speedup']:.1f}x)")

    return results


@sky.compute
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
    with sky.ComputePool(
        provider=sky.VastAI(),
        image=sky.Image(pip=["torch", "numpy"]),
        accelerator=sky.accelerators.L40S(),
        allocation="spot-if-available",
    ) as pool:
        # Get GPU information
        info = instance_information() @ pool
        print(f"GPU Info: {info}")

        # Heavy benchmark - ~5s on CPU vs ~0.1s on GPU
        print("\n" + "=" * 60)
        print("Heavy Matrix Operations Benchmark")
        print("=" * 60)
        heavy = heavy_matrix_ops(size=4096, iterations=50) >> pool
        print(f"\nResults:")
        print(f"  CPU time:  {heavy['cpu_time']:.2f}s")
        if "gpu_time" in heavy:
            print(f"  GPU time:  {heavy['gpu_time']:.2f}s")
            print(f"  Speedup:   {heavy['speedup']:.1f}x")

        # Quick matmul benchmark
        print("\n" + "=" * 60)
        print("Single Matmul 4096x4096")
        print("=" * 60)
        benchmark = matrix_multiply(4096) >> pool
        print(f"  CPU: {benchmark['cpu_time']:.3f}s")
        if "gpu_time" in benchmark:
            print(f"  GPU: {benchmark['gpu_time']:.3f}s")
            print(f"  Speedup: {benchmark['speedup']:.1f}x")

        # Train a simple model
        print("\n" + "=" * 60)
        print("Neural Network Training (100 epochs)")
        print("=" * 60)
        result = train_simple_model(epochs=100) >> pool
        print(f"  Device: {result['device']}")
        print(f"  Initial loss: {result['initial_loss']:.4f}")
        print(f"  Final loss: {result['final_loss']:.4f}")
