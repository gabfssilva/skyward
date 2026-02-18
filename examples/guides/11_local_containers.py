"""Local Containers — test your compute functions without cloud costs."""

import skyward as sky


@sky.compute
def hello() -> str:
    """A simple function that reports where it's running."""
    import socket

    return f"Hello from {socket.gethostname()}"


@sky.compute
def check_env() -> str:
    """Read an environment variable set via Image."""
    import os

    return os.environ.get("MY_VAR", "not set")


@sky.compute
def node_info() -> dict:
    """Report this node's position in the cluster."""
    info = sky.instance_info()
    assert info is not None
    return {
        "node": info.node,
        "total_nodes": info.total_nodes,
        "is_head": info.is_head,
    }


@sky.compute
def shard_sum(data: list[int]) -> int:
    """Sum only this node's shard of the data."""
    local = sky.shard(data)
    return sum(local)


if __name__ == "__main__":
    # 1. Basic: single container, no dependencies
    with sky.ComputePool(provider=sky.Container(), nodes=1) as pool:
        print(hello() >> pool)

    # 2. Image with env vars
    with sky.ComputePool(
        provider=sky.Container(),
        nodes=1,
        image=sky.Image(env={"MY_VAR": "it works"}),
    ) as pool:
        print(check_env() >> pool)

    # 3. Multi-node: broadcast and shard
    with sky.ComputePool(provider=sky.Container(), nodes=3) as pool:
        results = node_info() @ pool
        for r in results:
            print(f"  Node {r['node']}/{r['total_nodes']} (head={r['is_head']})")

        partial_sums = shard_sum(list(range(100))) @ pool
        print(f"  Total: {sum(partial_sums)}")
