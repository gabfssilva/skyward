"""Local Development with Containers.

Runs compute tasks on local containers instead of cloud instances.
No cloud credentials or GPU required — perfect for development and CI.

Supports Docker, podman, nerdctl, and Apple's container CLI
via the ``binary`` field (defaults to "docker").

Requirements:
    - A container runtime running locally (Docker, podman, etc.)
    - SSH key at ~/.ssh/id_ed25519 (or id_rsa)
"""
from time import sleep

import skyward as sky


@sky.compute
def hello(n: int) -> str:
    sleep(0.5)

    match n % 3:
        case 0:
            return "hello, 🌎!"
        case 1:
            return "hello, 🌍!"
        case _:
            return "hello, 🌏!"

if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.Container(),
        nodes=3,
        memory_gb=0.2,
        vcpus=0.2
    ) as pool:
        results = sky.gather(*(hello(i) for i in range(30))) >> pool

        for r in results:
            print(r)
