"""MultiPool: provision multiple pools in parallel, compare T4 vs L4."""

from skyward import AWS, ComputePool, Image, MultiPool, compute

@compute
def matmul_bench(size: int) -> float:
    import time

    import jax.numpy as jnp

    x = jnp.ones((size, size))

    for _ in range(10):
        (x @ x).block_until_ready()  # warmup

    t0 = time.perf_counter()
    for _ in range(100):
        (x @ x).block_until_ready()

    return time.perf_counter() - t0

if __name__ == "__main__":
    image = Image(pip=["jax[cuda12]"])

    with MultiPool(
        ComputePool(provider=AWS(), image=image, accelerator="T4"),
        ComputePool(provider=AWS(), image=image, accelerator="L4"),
    ) as (T4, L4):
        time_t4, time_l4 = matmul_bench(4096) >> T4, matmul_bench(4096) >> L4
        print(f"T4: {time_t4:.2f}s | L4: {time_l4:.2f}s | Speedup: {time_t4 / time_l4:.1f}x")
