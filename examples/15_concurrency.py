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
        provider=sky.AWS(),
        vcpus=4,
        worker=sky.Worker(concurrency=250),
        nodes=5
    ) as pool:
        results = sky.gather(*[heavy_stuff(i, i * 2) for i in range(5000) ], stream=True) >> pool

        for result in results:
            print(result)
