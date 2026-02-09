"""Hello Skyward - GPU vs CPU Benchmark.

A simple introduction to Skyward that demonstrates:
- Running compute on the cloud
- Comparing GPU vs CPU performance
- The dramatic speedup GPUs provide for parallel workloads

    ┌──────────────────────────────────────┐
    │  CPU: ████████████████████ 2.5s      │
    │  GPU: █ 0.05s                        │
    │                                      │
    │  Speedup: ~50x                       │
    └──────────────────────────────────────┘
"""
from time import sleep

import skyward as sky


@sky.compute
def benchmark(matrix_size: int, iterations: int) -> dict:
    """Benchmark matrix operations on GPU vs CPU.

    Matrix multiplication is the canonical GPU benchmark because:
    - O(n³) operations vs O(n²) memory = compute-bound
    - GPUs are literally designed for this (tensor cores)
    - Shows real-world speedups for ML/scientific computing
    """
    import time
    from functools import partial

    import jax
    from jax import lax, random

    @partial(jax.jit, static_argnames=["n"])
    def matmul_chain(a, b, n):
        def body(_, c):
            return jax.nn.relu(c @ b)
        return lax.fori_loop(0, n, body, a)

    def bench_on(device, a, b, n):
        a, b = jax.device_put((a, b), device)
        matmul_chain(a, b, 1).block_until_ready()  # warmup + compile

        start = time.perf_counter()
        matmul_chain(a, b, n).block_until_ready()
        return time.perf_counter() - start

    # Generate matrices
    k1, k2 = random.split(random.key(0))
    a = random.normal(k1, (matrix_size, matrix_size))
    b = random.normal(k2, (matrix_size, matrix_size))

    results = {"matrix_size": matrix_size, "iterations": iterations}

    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]

    results["cpu_time"] = bench_on(cpu, a, b, iterations)
    results["gpu_time"] = bench_on(gpu, a, b, iterations)
    results["speedup"] = results["cpu_time"] / results["gpu_time"]
    results["gpu_name"] = gpu.device_kind

    return results

def format_results(r: dict) -> None:
    """Pretty-print benchmark results."""
    cpu_time, gpu_time = r["cpu_time"], r["gpu_time"]
    bar_width = 40
    gpu_bar = max(1, int(bar_width * gpu_time / cpu_time))

    print(f"\n{'═' * 50}")
    print(f"  Matrix: {r['matrix_size']}×{r['matrix_size']} | Iterations: {r['iterations']}")
    print(f"{'═' * 50}")
    print(f"  CPU: {cpu_time:.3f}s")
    print(f"  GPU: {gpu_time:.3f}s ({r['gpu_name']})")
    print()
    print(f"  CPU │{'█' * bar_width}│ {cpu_time:.2f}s")
    print(f"  GPU │{'█' * gpu_bar}│ {gpu_time:.2f}s")
    print()
    print(f"  ⚡ Speedup: {r['speedup']:.1f}x")
    print(f"{'═' * 50}\n")


@sky.pool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4G(),
    image=sky.Image(
        pip=["jax[cuda12]"],
        skyward_source="local",
        metrics=sky.metrics.Default()
    ),
    max_hourly_cost=0.5,
    ttl=240
)
def main():
    return benchmark(matrix_size=4096, iterations=50) >> sky

if __name__ == "__main__":
    format_results(main())
