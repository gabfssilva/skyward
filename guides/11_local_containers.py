"""Local Containers — test your compute functions without cloud costs."""

import skyward as sky


@sky.function
def hello() -> str:
    """A simple function that reports where it's running."""
    import socket

    return f"Hello from {socket.gethostname()}"


@sky.function
def check_env() -> str:
    """Read an environment variable set via Image."""
    import os

    return os.environ.get("MY_VAR", "not set")


@sky.function
def node_info() -> dict:
    """Report this node's position in the cluster."""
    info = sky.instance_info()
    assert info is not None
    return {
        "node": info.node,
        "total_nodes": info.total_nodes,
        "is_head": info.is_head,
    }


@sky.function
def shard_sum(data: list[int]) -> int:
    """Sum only this node's shard of the data."""
    local = sky.shard(data)
    return sum(local)


if __name__ == "__main__":
    # 1. Basic: single container, no dependencies
    with sky.Compute(provider=sky.Container(), nodes=1) as compute:
        print(hello() >> compute)

    # 2. Image with env vars
    with sky.Compute(
        provider=sky.Container(),
        nodes=1,
        image=sky.Image(env={"MY_VAR": "it works"}),
    ) as compute:
        print(check_env() >> compute)

    # 3. Multi-node: broadcast and shard
    with sky.Compute(provider=sky.Container(), nodes=3) as compute:
        results = node_info() @ compute
        for r in results:
            print(f"  Node {r['node']}/{r['total_nodes']} (head={r['is_head']})")

        partial_sums = shard_sum(list(range(100))) @ compute
        print(f"  Total: {sum(partial_sums)}")
