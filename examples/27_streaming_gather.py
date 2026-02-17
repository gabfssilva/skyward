"""Streaming Gather Example.

Demonstrates gather(stream=True) which yields results as they complete,
instead of waiting for all tasks to finish before returning.

This is useful when:
- Tasks have varying execution times and you want to process early results
- You want to display progress as results arrive
- You're feeding results into a pipeline that can start with partial data
"""

import time
from random import randint

import skyward as sky


@sky.compute
def simulate_work(task_id: int, duration: float) -> dict:
    """Simulate a task that takes variable time."""
    time.sleep(duration)
    return {"task_id": task_id, "duration": duration}


def rand_delay() -> float:
    return randint(1, 10) / 10

if __name__ == "__main__":
    with sky.ComputePool(provider=sky.AWS()) as pool:
        tasks = [
            simulate_work(i, rand_delay()) for i in range(50)
        ]

        # =================================================================
        # Standard gather: waits for ALL tasks before returning
        # =================================================================
        print("--- gather (default) ---")
        start = time.monotonic()
        results = sky.gather(*tasks) >> pool
        elapsed = time.monotonic() - start
        print(f"All results at once after {elapsed:.1f}s: {results}")

        # =================================================================
        # Streaming gather: yields each result as it completes
        # =================================================================
        print("\n--- gather(stream=True) ---")
        start = time.monotonic()
        for result in sky.gather(*tasks, stream=True) >> pool:
            elapsed = time.monotonic() - start
            print(f"  [{elapsed:.1f}s] Got: {result}")
