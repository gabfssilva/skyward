"""PyTorch Distributed Training Example.

Demonstrates a complete distributed training setup with PyTorch:
- Multi-node, multi-GPU training
- DistributedDataParallel (DDP)
- Skyward's DistributedSampler
- Automatic distributed environment setup

Skyward automatically configures MASTER_ADDR, MASTER_PORT, WORLD_SIZE,
RANK, and initializes the process group.
"""

from skyward import (
    AWS,
    NVIDIA,
    ComputePool,
    DistributedSampler,
    compute,
    instance_info,
    shard,
)


@compute
def train_model(epochs: int, batch_size: int, learning_rate: float) -> dict:
    """Train a neural network with distributed data parallelism."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset

    pool = instance_info()

    # =================================================================
    # Model Definition
    # =================================================================
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.layers(x)

    # =================================================================
    # Setup Device and Model
    # =================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleNet().to(device)

    # Wrap with DDP for distributed training
    if dist.is_initialized():
        model = DDP(model)

    # =================================================================
    # Create Dataset and DataLoader
    # =================================================================
    # Synthetic dataset (in production, load real data)
    n_samples = 10000
    x = torch.randn(n_samples, 100)
    y = torch.randint(0, 10, (n_samples,))

    dataset = TensorDataset(x, y)

    # DistributedSampler shards data across nodes
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # =================================================================
    # Training Setup
    # =================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # =================================================================
    # Training Loop
    # =================================================================
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        accuracy = 100.0 * correct / total

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        # Only head node prints progress
        if pool.is_head:
            print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

    # =================================================================
    # Return Results
    # =================================================================
    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "device": device,
        "samples_per_node": len(sampler),
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1],
        "epochs_trained": epochs,
    }


@compute
def evaluate_model(test_data: list, test_labels: list) -> dict:
    """Evaluate model on test data (run on head node)."""
    import torch
    import torch.nn as nn

    pool = instance_info()

    # Only head node evaluates
    if not pool.is_head:
        return {"node": pool.node, "role": "worker", "skipped": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstruct model (in production, load from checkpoint)
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # Convert test data
    x = torch.tensor(test_data, dtype=torch.float32).to(device)
    y = torch.tensor(test_labels, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        _, predicted = output.max(1)
        accuracy = 100.0 * predicted.eq(y).sum().item() / len(y)

    return {
        "node": pool.node,
        "role": "evaluator",
        "test_samples": len(test_labels),
        "test_accuracy": accuracy,
    }


if __name__ == "__main__":
    # =================================================================
    # Multi-Node Training
    # =================================================================
    with ComputePool(
        provider=AWS(),
        nodes=2,
        accelerator=NVIDIA.A100,
        pip=["torch"],
        spot="always",
    ) as pool:
        print("=" * 60)
        print("Starting Distributed Training")
        print("=" * 60)

        # Train on all nodes (broadcast)
        results = train_model(
            epochs=10,
            batch_size=64,
            learning_rate=0.001,
        ) @ pool

        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)

        for r in results:
            role = "HEAD" if r["is_head"] else "WORKER"
            print(
                f"Node {r['node']} ({role}): "
                f"{r['samples_per_node']} samples, "
                f"loss={r['final_loss']:.4f}, "
                f"acc={r['final_accuracy']:.2f}%"
            )

        # Evaluation (synthetic test data)
        import random

        test_x = [[random.gauss(0, 1) for _ in range(100)] for _ in range(1000)]
        test_y = [random.randint(0, 9) for _ in range(1000)]

        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)

        eval_results = evaluate_model(test_x, test_y) @ pool

        for r in eval_results:
            if not r.get("skipped"):
                print(f"Test accuracy: {r['test_accuracy']:.2f}%")
