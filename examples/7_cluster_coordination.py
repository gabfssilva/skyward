"""Cluster Coordination Example.

Demonstrates how to use compute_pool() to:
- Access cluster topology information
- Coordinate between head and worker nodes
- Implement distributed patterns

The head node (node 0) typically aggregates results or coordinates work.
"""

from skyward import AWS, ComputePool, compute, instance_info, shard


@compute
def get_cluster_info() -> dict:
    """Get detailed cluster information."""
    pool = instance_info()

    return {
        "node": pool.node,
        "total_nodes": pool.total_nodes,
        "is_head": pool.is_head,
        "accelerators": pool.accelerators,
        "total_accelerators": pool.total_accelerators,
        "head_addr": pool.head_addr,
        "head_port": pool.head_port,
        "job_id": pool.job_id,
    }


@compute
def distributed_sum(values: list[int]) -> dict:
    """Demonstrate distributed reduce pattern."""
    pool = instance_info()

    # Each node processes its shard
    local_values = shard(values)
    local_sum = sum(local_values)

    result = {
        "node": pool.node,
        "local_count": len(local_values),
        "local_sum": local_sum,
    }

    # In a real scenario, you'd use collective operations
    # to aggregate across nodes. Here we just return local results.
    if pool.is_head:
        result["role"] = "head"
    else:
        result["role"] = "worker"

    return result


@compute
def worker_with_role(task_id: int) -> dict:
    """Different behavior based on node role."""
    import time

    pool = instance_info()

    if pool.is_head:
        # Head node might do coordination
        time.sleep(0.1)  # Simulate coordination overhead
        return {
            "node": pool.node,
            "role": "coordinator",
            "task_id": task_id,
            "action": "coordinating other nodes",
        }
    else:
        # Worker nodes do computation
        result = sum(i**2 for i in range(task_id * 1000))
        return {
            "node": pool.node,
            "role": "worker",
            "task_id": task_id,
            "result": result,
        }


@compute
def map_reduce_example(data: list[int], operation: str) -> dict:
    """Classic map-reduce pattern."""
    pool = instance_info()

    # Map phase: each node processes its shard
    local_data = shard(data)

    if operation == "sum":
        local_result = sum(local_data)
    elif operation == "product":
        result = 1
        for x in local_data:
            result *= x
        local_result = result
    elif operation == "count":
        local_result = len(local_data)
    else:
        local_result = 0

    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "operation": operation,
        "input_size": len(local_data),
        "local_result": local_result,
    }


if __name__ == "__main__":
    with ComputePool(
        provider=AWS(),
        nodes=4,
        spot="always",
    ) as pool:
        # =================================================================
        # Get cluster information from all nodes
        # =================================================================
        print("Cluster Information:")
        cluster_info = get_cluster_info() @ pool

        for info in cluster_info:
            role = "HEAD" if info["is_head"] else "WORKER"
            print(f"  Node {info['node']} ({role}): {info['accelerators']} accelerators")

        head_info = cluster_info[0]
        print(f"\nCluster: {head_info['total_nodes']} nodes, job_id={head_info['job_id'][:8]}...")

        # =================================================================
        # Distributed sum with role awareness
        # =================================================================
        print("\nDistributed Sum:")
        data = list(range(1000))
        sum_results = distributed_sum(data) @ pool

        total = 0
        for r in sum_results:
            print(f"  {r['role'].capitalize()} {r['node']}: {r['local_count']} items, sum={r['local_sum']}")
            total += r["local_sum"]

        print(f"  Total: {total} (expected: {sum(data)})")

        # =================================================================
        # Different behavior per role
        # =================================================================
        print("\nRole-based behavior:")
        role_results = worker_with_role(task_id=42) @ pool

        for r in role_results:
            if r["role"] == "coordinator":
                print(f"  Node {r['node']} (coordinator): {r['action']}")
            else:
                print(f"  Node {r['node']} (worker): computed result={r['result']}")

        # =================================================================
        # Map-Reduce Pattern
        # =================================================================
        print("\nMap-Reduce Pattern:")
        for op in ["sum", "count"]:
            mr_results = map_reduce_example(list(range(100)), op) @ pool
            aggregated = sum(r["local_result"] for r in mr_results)
            print(f"  {op}: {aggregated}")
