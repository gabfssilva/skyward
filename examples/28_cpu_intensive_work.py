"""CPU-Intensive Work with Concurrency.

Distributes CPU-heavy tasks across 5 nodes using concurrent execution.
Each task runs a tight numerical loop for ~30 seconds.
"""

from time import monotonic

import skyward as sky


@sky.compute
def cpu_burn(task_id: int) -> dict:
    start = monotonic()
    total = 0.0
    i = 0
    while monotonic() - start < 30:
        total += (i ** 0.5) * ((i + 1) ** 0.5)
        i += 1
    elapsed = monotonic() - start
    return {"task_id": task_id, "iterations": i, "elapsed": round(elapsed, 1)}


if __name__ == "__main__":
    total = 50

    with sky.ComputePool(
        provider=sky.AWS(),
        concurrency=2,
        nodes=5,
        max_inflight=total,
    ) as pool:
        tasks = sky.gather(*(cpu_burn(i) for i in range(total)), stream=True)
        results = tasks.with_timeout(60 * 10) >> pool

        for r in results:
            print(f"Task {r['task_id']}: {r['iterations']:,} iterations in {r['elapsed']}s")
