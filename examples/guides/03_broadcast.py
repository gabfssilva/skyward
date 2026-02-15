"""Broadcast — execute on every node in the pool."""

import skyward as sky


@sky.compute
def process_partition(data: list[int]) -> dict:
    """Process this node's shard of the data."""
    info = sky.instance_info()
    assert info is not None
    local_data = sky.shard(data)

    return {
        "node": info.node,
        "size": len(local_data),
        "sum": sum(local_data),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=4,
    ) as pool:
        data = list(range(1000))
        results = process_partition(data) @ pool

        total = 0
        for r in results:
            print(f"  Node {r['node']}: {r['size']} items, sum={r['sum']}")
            total += r["sum"]

        print(f"\nTotal: {total} (expected: {sum(data)})")
