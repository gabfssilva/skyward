import sys
from time import sleep

from loguru import logger

import skyward
import skyward.conc

# Enable debug logs
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss.SSS} | {message}")


@skyward.compute
def heavy_stuff(x: int, y: int) -> int:
    print("That's one expensive sum.")
    sleep(10)
    return x + y


if __name__ == "__main__":
    with skyward.ComputePool(provider=skyward.AWS(), cpu=4, concurrency=10, nodes=5) as pool:
        results = skyward.conc.map_async(lambda x: heavy_stuff(x, x) >> pool, list(range(100)))
        print(list(results))
