"""Auto-Scaling — start small, grow under load, shrink when idle."""

from time import sleep

import skyward as sky
from skyward.api import Worker


@sky.compute
def heavy_work(seconds: int) -> str:
    sleep(seconds)
    info = sky.instance_info()
    return f"node-{info.node}"


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=(1, 4),
        worker=Worker(concurrency=2),
        vcpus=1,
        memory_gb=1,
        autoscale_idle_timeout=10,
        autoscale_cooldown=20
    ) as pool:
        print(f"initial: {pool.current_nodes()} node(s)")

        futures = [heavy_work(30) > pool for _ in range(50)]

        sleep(120)
        print(f"under load: {pool.current_nodes()} node(s)")

        nodes_used = {f.result() for f in futures}
        print(f"work ran on: {nodes_used}")

        sleep(60)
        print(f"after idle: {pool.current_nodes()} node(s)")
