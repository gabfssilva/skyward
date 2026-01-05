"""Data Sharding Example.

Demonstrates how to distribute data across nodes using:
- shard(): Automatically partition arrays/lists for each node
- DistributedSampler: PyTorch-compatible sampler for DataLoader

Sharding ensures each node processes only its portion of the data,
enabling efficient parallel processing.
"""

import skyward as sky


@sky.compute
def train_on_shard(full_x: list, full_y: list) -> dict:
    """Train on this node's shard of the data."""
    import numpy as np

    # shard() automatically divides data for this node
    # All nodes use the same seed, so sharding is deterministic
    x, y = sky.shard(full_x, full_y, shuffle=True, seed=42)

    pool = sky.instance_info()

    # Convert to numpy for computation
    x_arr = np.array(x)
    y_arr = np.array(y)

    # Simple "training": compute statistics
    return {
        "node": pool.node,
        "shard_size": len(x),
        "x_mean": float(x_arr.mean()),
        "x_std": float(x_arr.std()),
        "y_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_arr, return_counts=True))},
    }


@sky.compute
def demonstrate_shard_types() -> dict:
    """Show that shard() preserves types."""
    import numpy as np

    pool = sky.instance_info()

    # List sharding
    my_list = list(range(100))
    sharded_list = sky.shard(my_list)

    # Tuple sharding
    my_tuple = tuple(range(100))
    sharded_tuple = sky.shard(my_tuple)

    # NumPy array sharding
    my_array = np.arange(100)
    sharded_array = sky.shard(my_array)

    return {
        "node": pool.node,
        "list_type": type(sharded_list).__name__,
        "list_len": len(sharded_list),
        "tuple_type": type(sharded_tuple).__name__,
        "tuple_len": len(sharded_tuple),
        "array_type": type(sharded_array).__name__,
        "array_len": len(sharded_array),
    }


@sky.compute
def train_with_dataloader(epochs: int, batch_size: int) -> dict:
    """Use DistributedSampler with PyTorch DataLoader."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pool = sky.instance_info()

    # Create synthetic dataset
    x = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(x, y)

    # DistributedSampler automatically shards for this node
    sampler = sky.DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Simple model
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    total_samples = 0
    final_loss = 0.0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_samples += len(batch_x)

        final_loss = epoch_loss / len(loader)

    return {
        "node": pool.node,
        "samples_per_node": len(sampler),
        "total_samples_processed": total_samples,
        "final_loss": final_loss,
    }


@sky.pool(
    provider=sky.AWS(),
    nodes=4,
    image=sky.Image(pip=["numpy", "torch"]),
    allocation="spot-if-available",
)
def main():
    # Generate full dataset
    import random

    random.seed(42)
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(10000)]
    Y = [random.randint(0, 9) for _ in range(10000)]

    # =================================================================
    # Basic sharding with shard()
    # =================================================================
    print("Training on sharded data...")
    results = train_on_shard(X, Y) @ sky

    for r in results:
        print(f"  Node {r['node']}: {r['shard_size']} samples, mean={r['x_mean']:.3f}")

    # =================================================================
    # Type preservation demonstration
    # =================================================================
    print("\nType preservation:")
    type_results = demonstrate_shard_types() @ sky

    for r in type_results:
        print(
            f"  Node {r['node']}: "
            f"list={r['list_type']}({r['list_len']}), "
            f"array={r['array_type']}({r['array_len']})"
        )

    # =================================================================
    # DistributedSampler with DataLoader
    # =================================================================
    print("\nTraining with DistributedSampler:")
    dl_results = train_with_dataloader(epochs=5, batch_size=32) @ sky

    for r in dl_results:
        print(
            f"  Node {r['node']}: "
            f"{r['samples_per_node']} samples, "
            f"loss={r['final_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
