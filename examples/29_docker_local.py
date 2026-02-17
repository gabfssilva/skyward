"""Local Development with Docker.

Runs compute tasks on local Docker containers instead of cloud instances.
No cloud credentials or GPU required — perfect for development and CI.

Requirements:
    - Docker running locally
    - SSH key at ~/.ssh/id_ed25519 (or id_rsa)
"""

import skyward as sky


@sky.compute
def hello(node_id: int) -> dict:
    import os
    import socket

    return {
        "node": node_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.Docker(image="ubuntu:24.04"),
        nodes=3,
        logging=sky.LogConfig(level="DEBUG"),
    ) as pool:
        results = sky.gather(*(hello(i) for i in range(3))) >> pool

        for r in results:
            print(f"Node {r['node']}: hostname={r['hostname']} pid={r['pid']}")
