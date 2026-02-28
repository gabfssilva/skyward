"""NVIDIA MIG — train independent models on isolated GPU partitions."""

import skyward as sky

PARTITIONS = 2
PROFILE = "3g.40gb"


@sky.compute
def train_on_partition(epochs: int, lr: float) -> dict:
    """Train a small CNN on a MIG partition."""
    import os

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    info = sky.instance_info()
    device = torch.device("cuda")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    x = torch.randn(5000, 784, device=device)
    y = torch.randint(0, 10, (5000,), device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100.0 * correct / total
    return {
        "worker": info.worker,
        "partition": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "final_loss": round(epoch_loss / len(loader), 4),
        "accuracy": round(accuracy, 1),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.Verda(),
        nodes=1,
        accelerator=sky.accelerators.A100(),
        worker=sky.Worker(concurrency=PARTITIONS, executor="process"),
        image=sky.Image(pip=["torch"]),
        plugins=[sky.plugins.mig(profile=PROFILE)],
    ) as pool:
        tasks = [train_on_partition(epochs=10, lr=1e-3) for _ in range(PARTITIONS)]
        results = list(sky.gather(*tasks, stream=True) >> pool)

        for r in sorted(results, key=lambda x: x["worker"]):
            print(
                f"Worker {r['worker']}: partition {r['partition'][:20]}... "
                f"loss={r['final_loss']}  acc={r['accuracy']}%"
            )
