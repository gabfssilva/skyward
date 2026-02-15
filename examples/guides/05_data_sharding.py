"""Data Sharding — distribute data across nodes automatically."""

import random

import skyward as sky


@sky.compute
def train_on_shard(full_x: list, full_y: list) -> dict:
    """Train on this node's shard of the data."""
    import numpy as np

    x, y = sky.shard(full_x, full_y, shuffle=True, seed=42)
    info = sky.instance_info()
    assert info is not None

    x_arr = np.array(x)
    return {
        "node": info.node,
        "shard_size": len(x),
        "mean": float(x_arr.mean()),
    }


@sky.compute
def show_shard_types() -> dict:
    """Demonstrate that shard() preserves types."""
    import numpy as np

    info = sky.instance_info()
    assert info is not None

    sharded_list = sky.shard(list(range(100)))
    sharded_tuple = sky.shard(tuple(range(100)))
    sharded_array = sky.shard(np.arange(100))

    return {
        "node": info.node,
        "list": type(sharded_list).__name__,
        "tuple": type(sharded_tuple).__name__,
        "array": type(sharded_array).__name__,
    }


if __name__ == "__main__":
    random.seed(42)
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(1000)]
    Y = [random.randint(0, 9) for _ in range(1000)]

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=4,
        image=sky.Image(pip=["numpy"]),
    ) as pool:
        results = train_on_shard(X, Y) @ pool
        for r in results:
            print(f"  Node {r['node']}: {r['shard_size']} samples, mean={r['mean']:.3f}")

        types = show_shard_types() @ pool
        for t in types:
            print(f"  Node {t['node']}: list={t['list']}, tuple={t['tuple']}, array={t['array']}")
