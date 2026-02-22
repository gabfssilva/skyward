"""Worker Executors — thread vs process execution backends."""

from time import monotonic

import skyward as sky


# --8<-- [start:cpu_burn]
@sky.compute
def cpu_burn(task_id: int) -> dict:
    """CPU-intensive task: tight numerical loop for ~10 seconds."""
    start = monotonic()
    total = 0.0
    i = 0
    while monotonic() - start < 10:
        total += (i ** 0.5) * ((i + 1) ** 0.5)
        i += 1
    elapsed = monotonic() - start
    return {"task_id": task_id, "iterations": i, "elapsed": round(elapsed, 1)}
# --8<-- [end:cpu_burn]


if __name__ == "__main__":
    total = 20

    # --8<-- [start:thread_executor]
    # Thread executor (default) — tasks run as threads in the worker process.
    # Supports streaming, low overhead, ideal for I/O-bound and GIL-releasing workloads.
    with sky.ComputePool(
        provider=sky.AWS(),
        worker=sky.Worker(concurrency=2),  # executor="thread" is the default
        nodes=3,
    ) as pool:
        results = sky.gather(*(cpu_burn(i) for i in range(total)), stream=True)
        for r in (results >> pool):
            print(f"[thread] Task {r['task_id']}: {r['iterations']:,} iters in {r['elapsed']}s")
    # --8<-- [end:thread_executor]

    # --8<-- [start:process_executor]
    # Process executor — each task runs in a separate OS process.
    # Bypasses the GIL, so pure-Python CPU-bound work uses all available cores.
    with sky.ComputePool(
        provider=sky.AWS(),
        worker=sky.Worker(concurrency=2, executor="process"),
        nodes=3,
    ) as pool:
        results = sky.gather(*(cpu_burn(i) for i in range(total)), stream=True)
        for r in (results >> pool):
            print(f"[process] Task {r['task_id']}: {r['iterations']:,} iters in {r['elapsed']}s")
    # --8<-- [end:process_executor]
