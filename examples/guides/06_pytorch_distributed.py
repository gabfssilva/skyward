"""PyTorch Distributed — multi-node training with DDP."""

import skyward as sky


@sky.compute
@sky.integrations.torch
def train(epochs: int, batch_size: int, lr: float) -> dict:
    """Train a neural network with DistributedDataParallel."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    info = sky.instance_info()
    assert info is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    if dist.is_initialized():
        model = DDP(model)

    x = torch.randn(10000, 100)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    avg_loss = 0.0
    accuracy = 0.0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        stats = torch.tensor([epoch_loss, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = stats[0].item() / (len(loader) * info.total_nodes)
        accuracy = 100.0 * stats[1].item() / stats[2].item()

        if info.is_head:
            print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.1f}%")

    return {
        "node": info.node,
        "is_head": info.is_head,
        "final_loss": avg_loss,
        "final_accuracy": accuracy,
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=2,
        accelerator="T4",
        image=sky.Image(pip=["torch"]),
    ) as pool:
        results = train(epochs=5, batch_size=64, lr=0.001) @ pool

        for r in results:
            role = "HEAD" if r["is_head"] else "WORKER"
            loss, acc = r['final_loss'], r['final_accuracy']
            print(f"Node {r['node']} ({role}): loss={loss:.4f}, acc={acc:.1f}%")
