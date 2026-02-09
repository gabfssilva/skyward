"""Broadcasting Example.

Demonstrates the @ operator for executing a function on ALL nodes in the pool.
"""

import skyward as sky


@sky.compute
def process_partition(data: list[int]) -> dict:
    """Process a partition of data on this node."""
    pool = sky.instance_info()

    # shard() automatically gives this node its portion
    local_data = sky.shard(data)

    print(f"Processing {len(local_data)} items...")

    return {
        "node": pool.node,
        "partition_size": len(local_data),
        "partition_sum": sum(local_data),
        "partition_range": (min(local_data), max(local_data)) if local_data else None,
        "info": pool
    }


@sky.pool(
    provider=sky.AWS(),
    nodes=4,
    accelerator=sky.accelerators.T4G(),
    image=sky.Image(skyward_source='local'),
)
def main():
    print("\nProcessing data across all nodes...")
    data = list(range(1000))  # Full dataset
    partition_results = process_partition(data) @ sky

    total_sum = 0
    for result in partition_results:
        print(
            f"  Node {result['node']}: "
            f"{result['partition_size']} items, "
            f"sum={result['partition_sum']}"
        )
        total_sum += result["partition_sum"]

        print(f"  info: {result['info']}")

    print(f"\nTotal sum across all nodes: {total_sum}")
    print(f"Expected sum: {sum(data)}")


if __name__ == "__main__":
    main()
