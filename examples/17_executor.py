"""SkywardExecutor - ThreadPoolExecutor drop-in for cloud compute.

This example shows how to use SkywardExecutor as a drop-in replacement
for concurrent.futures.ThreadPoolExecutor, but with automatic cloud
instance provisioning.
"""

from concurrent.futures import as_completed
from time import sleep

import skyward as sky


def slow_task(x: int) -> int:
    """A CPU-bound task that takes time to complete."""
    print(f"Task {x} starting")
    sleep(5)
    print(f"Task {x} done")
    return x * 2


if __name__ == "__main__":
    # Executor works exactly like ThreadPoolExecutor
    with sky.Executor(
        provider=sky.AWS(),
        nodes=5,
        concurrency=5,
    ) as executor:
        # Option 1: submit individual tasks
        futures = [executor.submit(slow_task, i) for i in range(25)]

        # Option 2: use as_completed for results as they finish
        for future in as_completed(futures):
            print(f"Result: {future.result()}")

        # Option 3: use map for ordered results
        results = list(executor.map(slow_task, range(25, 50)))
        print(f"Mapped results: {results}")
