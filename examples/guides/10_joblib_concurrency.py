"""Joblib Concurrency — distribute joblib tasks across cloud nodes."""

from time import perf_counter, sleep

from joblib import Parallel, delayed

import skyward as sky


def slow_task(x):
    """A slow task that takes 5 seconds."""
    sleep(5)
    return x * 2


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=10,
        worker=sky.Worker(concurrency=10),
        plugins=[sky.plugins.joblib()],
    ) as pool:
        t0 = perf_counter()

        results = Parallel(n_jobs=-1)(
            delayed(slow_task)(i) for i in range(2000)
        )

        elapsed = perf_counter() - t0

        print("Tasks: 2000 | Nodes: 10 | Concurrency: 10")
        print("Effective workers: 100")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {2000 / elapsed:.2f} tasks/s")
        print(f"Ideal time: {2000 / 100 * 5:.0f}s")
        print(f"Efficiency: {(2000 / 100 * 5) / elapsed * 100:.1f}%")
