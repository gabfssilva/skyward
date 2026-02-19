"""GCP TPU — JAX matrix multiply on Cloud TPU.

Demonstrates using Google Cloud TPUs with Skyward:
- Provisioning TPU VMs via the Cloud TPU API
- Running JAX code on TPU hardware
- Comparing TPU vs CPU performance

Requires:
    - GCP project with TPU API enabled
    - TPU quota (check with: gcloud compute regions describe <region>)
    - google-cloud-tpu installed: uv add "skyward[gcp]"

    ┌──────────────────────────────────────┐
    │  CPU: ████████████████████ 1.2s      │
    │  TPU: ██ 0.05s                       │
    │                                      │
    │  Speedup: ~24x                       │
    └──────────────────────────────────────┘
"""

import skyward as sky


@sky.compute
def tpu_benchmark(matrix_size: int, iterations: int) -> dict:
    """Benchmark matrix operations on TPU vs CPU."""
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
        matmul_chain(a, b, 1).block_until_ready()

        start = time.perf_counter()
        matmul_chain(a, b, n).block_until_ready()
        return time.perf_counter() - start

    k1, k2 = random.split(random.key(0))
    a = random.normal(k1, (matrix_size, matrix_size))
    b = random.normal(k2, (matrix_size, matrix_size))

    results = {"matrix_size": matrix_size, "iterations": iterations}

    cpu = jax.devices("cpu")[0]
    results["cpu_time"] = bench_on(cpu, a, b, iterations)

    tpu_devices = jax.devices("tpu")
    if tpu_devices:
        tpu = tpu_devices[0]
        results["tpu_time"] = bench_on(tpu, a, b, iterations)
        results["speedup"] = results["cpu_time"] / results["tpu_time"]
        results["tpu_name"] = tpu.device_kind
    else:
        results["tpu_time"] = 0.0
        results["speedup"] = 0.0
        results["tpu_name"] = "no TPU found"

    return results


def format_results(r: dict) -> None:
    """Pretty-print benchmark results."""
    cpu_time, tpu_time = r["cpu_time"], r["tpu_time"]
    bar_width = 40
    tpu_bar = max(1, int(bar_width * tpu_time / cpu_time)) if cpu_time > 0 else 1

    print(f"\n{'═' * 50}")
    print(f"  Matrix: {r['matrix_size']}×{r['matrix_size']} | Iterations: {r['iterations']}")
    print(f"{'═' * 50}")
    print(f"  CPU: {cpu_time:.3f}s")
    print(f"  TPU: {tpu_time:.3f}s ({r['tpu_name']})")
    print()
    print(f"  CPU │{'█' * bar_width}│ {cpu_time:.2f}s")
    print(f"  TPU │{'█' * tpu_bar}│ {tpu_time:.2f}s")
    print()
    print(f"  Speedup: {r['speedup']:.1f}x")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.GCP(zone="us-central1-a"),
        accelerator="v5litepod-1",
        image=sky.Image(
            pip=["jax[tpu]"],
        ),
        provision_timeout=600,
    ) as pool:
        format_results(tpu_benchmark(matrix_size=4096, iterations=50) >> pool)
