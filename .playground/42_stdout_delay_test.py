"""Stdout delay investigation.

Prints a message every second from each container node to observe
how quickly the output reaches the local console.

Usage:
    uv run python examples/42_stdout_delay_test.py
"""

from time import sleep

import skyward as sky


@sky.function
def tick(node_label: str, count: int = 10) -> str:
    info = sky.instance_info()
    for i in range(1, count + 1):
        print(f"[{node_label}] tick {i}/{count} (node {info.node})")
        sleep(1)
    return f"{node_label} done"


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.Container(),
        nodes=2,
        memory_gb=1,
        vcpus=1,
    ) as pool:
        results = tick("worker") @ pool

        for r in results:
            print(r)
