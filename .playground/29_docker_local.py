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


@sky.function
def hello(n: int) -> str:
    sleep(5)

    match n % 3:
        case 0:
            return "hello, 🌎!"
        case 1:
            return "hello, 🌍!"
        case _:
            return "hello, 🌏!"

if __name__ == "__main__":
    with sky.Compute(
        provider=sky.Container(),
        nodes=sky.Nodes(min=2, desired=2, max=3),
        memory_gb=1,
        vcpus=1,
        image=sky.Image(pip=['torch', 'scipy', 'marimo'])
    ) as compute:
        all_results = hello(10) @ compute

        results = sky.gather(*(hello(i) for i in range(30))) >> compute

        for r in all_results:
            print(r)
