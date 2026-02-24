"""PyTorch Model Roundtrip.

Demonstrates that PyTorch nn.Module instances serialize seamlessly
through cloudpickle — both as arguments and return values.

    ┌────────┐  cloudpickle   ┌────────┐  pickle         ┌────────┐
    │ Local  │ ── model() ──▶ │ Worker │ ── trained ──▶  │ Local  │
    │  build model            │  train on cloud           │  evaluate
    └────────┘                └────────┘                  └────────┘

The entire nn.Module (architecture + weights) travels through the wire.
No manual state_dict save/load, no checkpoint files — just pickle.

IMPORTANT: Pin the remote torch version to match the local one. Torch
tensors pickled with version X may not deserialize correctly on version Y.
"""

import torch
import torch.nn as nn

import skyward as sky

TORCH_VERSION = torch.__version__.split("+")[0]


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_synthetic_mnist(n_train: int = 5000, n_test: int = 1000) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Generate synthetic MNIST-like data for demonstration."""
    gen = torch.Generator().manual_seed(42)
    x_train = torch.randn(n_train, 784, generator=gen)
    y_train = torch.randint(0, 10, (n_train,), generator=gen)
    x_test = torch.randn(n_test, 784, generator=gen)
    y_test = torch.randint(0, 10, (n_test,), generator=gen)
    return x_train, y_train, x_test, y_test


@sky.compute
def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    """Train the model on the remote worker and return it with learned weights."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loader = DataLoader(
        TensorDataset(x.to(device), y.to(device)),
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

        acc = 100.0 * correct / total
        print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / len(loader):.4f}, acc={acc:.1f}%")

    # Diagnostic: fingerprint weights before returning
    with torch.no_grad():
        fingerprint = sum(p.sum().item() for p in model.parameters())
    print(f"  [remote] weight fingerprint: {fingerprint:.6f}")

    return model.cpu()


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    """Evaluate the trained model locally."""
    model.eval()
    output = model(x)
    predictions = output.argmax(1)
    accuracy = 100.0 * (predictions == y).float().mean().item()

    print(f"  test samples : {len(y)}")
    print(f"  test accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = make_synthetic_mnist()

    # Build an untrained model locally
    model = MNISTClassifier()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Before training:")
    evaluate(model, x_test, y_test)

    # Ship the model to the cloud, train, get it back
    with sky.ComputePool(
        provider=sky.Container(),
        vcpus=4,
        memory_gb=2,
        image=sky.Image(
            pip=[f"torch=={TORCH_VERSION}"],
            pip_indexes=[
                sky.PipIndex(
                    url="https://download.pytorch.org/whl/cpu",
                    packages=["torch"],
                ),
            ],
        ),
    ) as pool:
        print("\nTraining remotely...")
        trained_model: nn.Module = train(model, x_train, y_train, epochs=10, lr=1e-3) >> pool

    # Diagnostic: fingerprint weights after receiving
    with torch.no_grad():
        fingerprint = sum(p.sum().item() for p in trained_model.parameters())
    print(f"\n[local] weight fingerprint: {fingerprint:.6f}")
    print(f"[local] type: {type(trained_model)}")
    print(f"[local] state_dict keys: {list(trained_model.state_dict().keys())[:3]}...")

    print("\nAfter training:")
    evaluate(trained_model, x_test, y_test)
