"""PyTorch Model Roundtrip.

Demonstrates that PyTorch nn.Module instances serialize seamlessly
through cloudpickle — both as arguments and return values.

The entire nn.Module (architecture + weights) travels through the wire.
No manual state_dict save/load, no checkpoint files — just pickle.
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


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST and flatten images to 784-d vectors."""
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    y_train = train_ds.targets
    x_test = test_ds.data.float().view(-1, 784) / 255.0
    y_test = test_ds.targets
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
    x_train, y_train, x_test, y_test = load_mnist()

    model = MNISTClassifier()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Before training:")
    evaluate(model, x_test, y_test)

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

    print("\nAfter training:")
    evaluate(trained_model, x_test, y_test)
