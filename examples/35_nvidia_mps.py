"""NVIDIA MPS (Multi-Process Service) Example.

Runs ResNet-50 inference concurrently on a single T4 GPU to demonstrate
MPS throughput gains.  Each task loads the model, runs N forward passes on
random ImageNet-shaped tensors, and reports images/sec plus peak GPU
utilisation.  With MPS the tasks share one GPU context and the SM scheduler
can overlap their kernels — without MPS the CUDA runtime time-slices them.

Usage:
    python examples/35_nvidia_mps.py
"""

from time import perf_counter

import skyward as sky


@sky.compute
def resnet_inference(task_id: int, batches: int, batch_size: int) -> dict:
    """Run ResNet-50 inference and report throughput."""
    import subprocess
    from time import perf_counter as _pc

    import torch
    from torchvision.models import resnet50  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnet50(weights=None).to(device).eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    # warm-up
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = _pc()
    with torch.no_grad():
        for _ in range(batches):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = _pc() - t0

    images = batches * batch_size
    throughput = images / elapsed

    # grab GPU utilisation from nvidia-smi
    gpu_util = "n/a"
    mem_used = "n/a"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = out.split(", ")
        gpu_util = f"{parts[0]}%"
        mem_used = f"{parts[1]} MiB"
    except Exception:
        pass

    return {
        "task_id": task_id,
        "device": device,
        "images": images,
        "elapsed": round(elapsed, 2),
        "throughput": round(throughput, 1),
        "gpu_util": gpu_util,
        "mem_used": mem_used,
    }


if __name__ == "__main__":
    CONCURRENCY = 8
    BATCHES = 200
    BATCH_SIZE = 1

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=1,
        accelerator="T4",
        worker=sky.Worker(concurrency=CONCURRENCY, executor='process'),
        plugins=[
            sky.plugins.torch(),
            sky.plugins.mps(active_thread_percentage=100 // CONCURRENCY),
        ],
        image=sky.Image(pip=["torchvision"]),
    ) as pool:
        total_tasks = CONCURRENCY * 2

        print(
            f"Running {total_tasks} concurrent ResNet-50 inference tasks "
            f"({BATCHES}×{BATCH_SIZE} images each, concurrency={CONCURRENCY})"
        )
        start = perf_counter()

        tasks = [
            resnet_inference(i, BATCHES, BATCH_SIZE)
            for i in range(total_tasks)
        ]
        results = list(sky.gather(*tasks, stream=True) >> pool)

        wall = perf_counter() - start

        print(f"\n{'task':>4}  {'images':>6}  {'time':>6}  {'img/s':>7}  {'gpu':>5}  {'vram'}")
        print("-" * 52)
        for r in sorted(results, key=lambda x: x["task_id"]):
            print(
                f"  {r['task_id']:>2}  {r['images']:>6}  "
                f"{r['elapsed']:>5.1f}s  {r['throughput']:>6.1f}  "
                f"{r['gpu_util']:>5}  {r['mem_used']}"
            )

        total_images = sum(r["images"] for r in results)
        print(f"\nTotal: {total_images} images in {wall:.1f}s wall "
              f"({total_images / wall:.0f} img/s aggregate)")
