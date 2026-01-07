"""
VastAI Multi-Node with Overlay Networks.

VastAI overlay networks enable NCCL-based distributed training
on marketplace GPUs by creating virtual LANs for direct
instance-to-instance communication.
"""

import skyward as sky


@sky.compute
@sky.integrations.torch()
def distributed_train(batch_data: list) -> dict:
    """Distributed training with NCCL via overlay network."""
    import torch
    import torch.distributed as dist

    # Initialize distributed (env vars set by torch integration)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get cluster info
    info = sky.instance_info()

    # Simple all-reduce example
    tensor = torch.tensor([rank + 1.0]).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected: 1 + 2 + 3 + 4 = 10 for 4 nodes
    expected_sum = sum(range(1, world_size + 1))

    dist.destroy_process_group()

    return {
        "rank": rank,
        "world_size": world_size,
        "node": info.node,
        "is_head": info.is_head,
        "overlay_ip": info.peers[info.node].ip if info.peers else "N/A",
        "all_reduce_result": tensor.item(),
        "expected": expected_sum,
        "nccl_working": tensor.item() == expected_sum,
    }


@sky.compute
def get_network_info() -> dict:
    """Get network interface information."""
    import subprocess

    # Get eth0 (overlay) IP
    result = subprocess.run(
        ["ip", "-4", "addr", "show", "eth0"],
        capture_output=True,
        text=True,
    )

    return {
        "eth0_info": result.stdout.strip(),
        "instance_info": sky.instance_info().__dict__ if sky.instance_info() else None,
    }


def main():
    # VastAI with overlay networking for multi-node NCCL
    provider = sky.VastAI(
        geolocation="US",  # Filter to US for lower latency
        min_reliability=0.95,  # High reliability for distributed training
        bid_multiplier=1.3,  # Bid 30% above minimum
        use_overlay=True,  # Enable overlay networking (default)
    )

    with sky.ComputePool(
        provider=provider,
        accelerator="RTX 4090",
        nodes=4,  # Multi-node automatically creates overlay
        image=sky.Image(
            pip=["torch"],
            env={"NCCL_DEBUG": "INFO"},  # Debug NCCL communication
        ),
        allocation="spot-if-available",
    ) as pool:
        # First, verify overlay network is working
        print("Checking overlay network connectivity...")
        network_info = get_network_info() @ pool  # Broadcast to all nodes

        for i, info in enumerate(network_info):
            print(f"\nNode {i}:")
            print(f"  eth0: {info['eth0_info'][:100]}...")

        # Run distributed training
        print("\n" + "=" * 50)
        print("Running distributed training with NCCL...")
        print("=" * 50 + "\n")

        results = distributed_train([]) @ pool  # Broadcast to all nodes

        # Check results
        all_working = all(r["nccl_working"] for r in results)

        for r in results:
            status = "OK" if r["nccl_working"] else "FAILED"
            print(
                f"Node {r['node']}: rank={r['rank']}, "
                f"all_reduce={r['all_reduce_result']}, "
                f"expected={r['expected']}, "
                f"status={status}"
            )

        print(f"\nNCCL Communication: {'SUCCESS' if all_working else 'FAILED'}")


def main_simple():
    """Simpler example without distributed training."""

    @sky.compute
    def node_info() -> dict:
        import socket

        info = sky.instance_info()
        return {
            "hostname": socket.gethostname(),
            "node": info.node if info else -1,
            "total_nodes": info.total_nodes if info else 0,
            "is_head": info.is_head if info else False,
        }

    # VastAI automatically sets up overlay for nodes > 1
    with sky.ComputePool(
        provider=sky.VastAI(geolocation="US"),
        accelerator="RTX 3090",
        nodes=2,
        allocation="spot-if-available",
    ) as pool:
        # Broadcast to all nodes
        results = node_info() @ pool

        print("Cluster nodes:")
        for r in results:
            head = " (HEAD)" if r["is_head"] else ""
            print(f"  {r['hostname']}: node {r['node']}/{r['total_nodes']}{head}")


if __name__ == "__main__":
    print("=== VastAI Multi-Node Overlay Example ===\n")
    main()
