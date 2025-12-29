"""Broadcasting Example.

Demonstrates the @ operator for executing a function on ALL nodes in the pool.
This is useful for:
- Pre-loading models/data on all nodes
- Running initialization code
- Collecting information from all nodes
- Distributed map operations
"""
from time import sleep

from skyward import AWS, NVIDIA, ComputePool, compute, instance_info


@compute
def node_info() -> dict:
    """Get information about this node."""
    pool = instance_info()
    return {
        "node": pool.node,
        "total_nodes": pool.total_nodes,
        "is_head": pool.is_head,
        "head_addr": pool.head_addr,
    }


@compute
def warm_cache(model_path: str) -> str:
    """Simulate loading a model into cache on each node."""
    import time

    pool = instance_info()

    # Simulate model loading time
    time.sleep(0.5)

    return f"Node {pool.node}: cached {model_path}"


@compute
def initialize_node() -> dict:
    """Initialize node-specific resources."""
    import os

    pool = instance_info()

    # Each node can set up its own resources
    work_dir = f"/tmp/node_{pool.node}"
    os.makedirs(work_dir, exist_ok=True)

    return {
        "node": pool.node,
        "work_dir": work_dir,
        "initialized": True,
    }


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
    # =================================================================
    # Pool with 4 nodes, each with an A100 GPU
    # =================================================================
    with ComputePool(
        provider=AWS(),
        nodes=4,
        accelerator='T4',
        spot="always",
    ) as pool:
        # =================================================================
        # Get info from ALL nodes using @ operator
        # =================================================================
        print("Getting info from all nodes...")
        infos = node_info() @ pool  # Returns tuple of results, one per node
        for info in infos:
            head_marker = " (HEAD)" if info["is_head"] else ""
            print(f"  Node {info['node']}/{info['total_nodes']}{head_marker}")

        # =================================================================
        # Pre-load model cache on ALL nodes
        # =================================================================
        print("\nWarming cache on all nodes...")
        cache_results = warm_cache("/models/llama-7b") @ pool
        for result in cache_results:
            print(f"  {result}")

        # =================================================================
        # Initialize resources on ALL nodes
        # =================================================================
        print("\nInitializing all nodes...")
        init_results = initialize_node() @ pool
        for result in init_results:
            print(f"  Node {result['node']}: {result['work_dir']}")

        # =================================================================
        # Distributed data processing
        # =================================================================
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
