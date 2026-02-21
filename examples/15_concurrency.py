"""Concurrency Example.

Demonstrates how to use high concurrency with map_async
to process many tasks across multiple nodes.
"""

from time import sleep

import skyward as sky


@sky.compute
def heavy_stuff(x: int, y: int) -> int:
    print("That's one expensive sum.")
    sleep(10)
    return x + y


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(), cpu=4, worker=sky.Worker(concurrency=10), nodes=5,
    ) as pool:
        # Process 100 tasks across 5 nodes with 10 concurrent slots each (50 total)
        results = sky.conc.map_async(lambda x: heavy_stuff(x, x) >> pool, list(range(100)))
        print(list(results))
