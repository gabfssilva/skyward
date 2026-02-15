"""GPU Accelerators — choose and use specific GPUs."""

import skyward as sky


@sky.compute
def gpu_info() -> sky.InstanceInfo:
    """Return information about this instance."""
    return sky.instance_info()


@sky.compute
def matrix_benchmark(size: int) -> dict:
    """Benchmark matrix multiplication on GPU vs CPU."""
    import time

    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.perf_counter()
    torch.matmul(a, b)
    cpu_time = time.perf_counter() - start

    if torch.cuda.is_available():
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        start = time.perf_counter()
        torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        return {"cpu": cpu_time, "gpu": gpu_time, "speedup": cpu_time / gpu_time}

    return {"cpu": cpu_time}


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.VastAI(),
        accelerator=sky.accelerators.L40S(),
        image=sky.Image(pip=["torch"]),
    ) as pool:
        info = gpu_info() >> pool
        print(f"Instance: {info}")

        result = matrix_benchmark(4096) >> pool
        cpu_time, gpu_time, speedup = result['cpu'], result['gpu'], result['speedup']
        print(f"CPU: {cpu_time:.3f}s | GPU: {gpu_time:.3f}s | Speedup: {speedup:.0f}x")
