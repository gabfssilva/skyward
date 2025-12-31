"""Broadcasting Example.

Demonstrates the @ operator for executing a function on ALL nodes in the pool.
"""

from skyward import ComputePool, compute, instance_info, Verda, AWS


@compute
def process_partition(data: list[int]) -> dict:
    """Process a partition of data on this node."""
    from skyward import shard

    pool = instance_info()

    # shard() automatically gives this node its portion
    local_data = shard(data)

    print(f"Processing {len(local_data)} items...")

    return {
        "node": pool.node,
        "partition_size": len(local_data),
        "partition_sum": sum(local_data),
        "partition_range": (min(local_data), max(local_data)) if local_data else None,
    }


if __name__ == "__main__":
    with ComputePool(
        provider=AWS(),
        nodes=4,
        accelerator='T4',
        spot="always",
    ) as pool:
        print("\nProcessing data across all nodes...")
        data = list(range(1000))  # Full dataset
        partition_results = process_partition(data) @ pool

        total_sum = 0
        for result in partition_results:
            print(
                f"  Node {result['node']}: "
                f"{result['partition_size']} items, "
                f"sum={result['partition_sum']}"
            )
            total_sum += result["partition_sum"]

        print(f"\nTotal sum across all nodes: {total_sum}")
        print(f"Expected sum: {sum(data)}")
