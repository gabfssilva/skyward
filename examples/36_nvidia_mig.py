"""NVIDIA MIG (Multi-Instance GPU) Example.

Partitions a single A100 80GB into two 3g.40gb MIG instances and runs
independent matrix multiplication benchmarks on each partition. Each
loky subprocess sees only its assigned MIG device via CUDA_VISIBLE_DEVICES,
ensuring full isolation — separate compute units, memory, and L2 cache.

Usage:
    python examples/36_nvidia_mig.py
"""

from time import perf_counter

import skyward as sky

PARTITIONS = 2
PROFILE = "3g.40gb"


@sky.compute
def matmul_bench(iterations: int, size: int) -> dict:
    """Run matrix multiplications on the assigned MIG partition."""
    import os
    import subprocess
    from time import perf_counter as _pc

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = sky.instance_info()

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # warm-up
    for _ in range(5):
        torch.mm(a, b)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = _pc()
    for _ in range(iterations):
        torch.mm(a, b)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = _pc() - t0

    tflops = (2 * size**3 * iterations) / elapsed / 1e12

    mig_instances = ""
    try:
        mig_instances = subprocess.check_output(
            ["nvidia-smi", "mig", "-lgi"], text=True,
        ).strip()
    except Exception:
        pass

    return {
        "worker": info.worker,
        "node": info.node,
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "mig_instances": mig_instances,
        "iterations": iterations,
        "matrix_size": size,
        "elapsed": round(elapsed, 3),
        "tflops": round(tflops, 2),
    }


if __name__ == "__main__":
    ITERATIONS = 500
    MATRIX_SIZE = 4096

    with sky.ComputePool(
        provider=sky.Verda(),
        nodes=1,
        accelerator=sky.accelerators.H100(),
        worker=sky.Worker(concurrency=PARTITIONS, executor="process"),
        plugins=[
            sky.plugins.torch(),
            sky.plugins.mig(profile=PROFILE),
        ],
        allocation='spot',
    ) as pool:
        print(
            f"Running matmul benchmark on {PARTITIONS} MIG partitions "
            f"({PROFILE}), {ITERATIONS}x {MATRIX_SIZE}x{MATRIX_SIZE}"
        )

        tasks = [matmul_bench(ITERATIONS, MATRIX_SIZE) for _ in range(PARTITIONS)]
        start = perf_counter()
        results = list(sky.gather(*tasks, stream=True) >> pool)
        wall = perf_counter() - start

        print(f"\n{'worker':>6}  {'device':>40}  {'time':>6}  {'TFLOPS':>7}")
        print("-" * 68)
        for r in sorted(results, key=lambda x: x["worker"]):
            print(
                f"  {r['worker']:>4}  {r['cuda_device']:>40}  "
                f"{r['elapsed']:>5.2f}s  {r['tflops']:>6.2f}"
            )

        print(f"\nWall time: {wall:.1f}s (both partitions ran in parallel)")

        if results[0].get("mig_instances"):
            print(f"\n{results[0]['mig_instances']}")
