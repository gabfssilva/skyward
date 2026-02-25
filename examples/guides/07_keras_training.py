"""Keras Training — train a model on cloud GPUs with Keras 3."""

from __future__ import annotations

import keras
from keras import layers

import skyward as sky


@sky.compute
def train_mnist(epochs: int = 5, batch_size: int = 128) -> dict:
    """Train an MLP on this node's shard of MNIST."""
    info = sky.instance_info()
    assert info is not None

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    x_train, y_train = sky.shard(x_train, y_train, shuffle=True, seed=42)

    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return {
        "node": info.node,
        "samples": len(x_train),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "test_accuracy": float(test_acc),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator="T4",
        nodes=2,
        plugins=[sky.plugins.jax(), sky.plugins.keras(backend="jax")],
    ) as pool:
        results = train_mnist() @ pool

        for r in results:
            node, samples = r['node'], r['samples']
            train_acc, test_acc = r['final_accuracy'], r['test_accuracy']
            print(f"  Node {node}: {samples} samples, acc={train_acc:.2%}, test={test_acc:.2%}")
