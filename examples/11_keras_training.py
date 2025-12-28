"""Keras 3 Training Example.

Demonstrates training with Keras 3, which is backend-agnostic:
- Works with JAX, PyTorch, or TensorFlow backends
- Uses Skyward's shard() for data distribution
- Runs on GPU with automatic backend configuration

This example uses JAX backend for best performance.
"""

from skyward import AWS, NVIDIA, ComputePool, compute, instance_info, shard


@compute
def get_keras_info() -> dict:
    """Get Keras configuration information."""
    import keras

    return {
        "keras_version": keras.__version__,
        "backend": keras.backend.backend(),
        "devices": [str(d) for d in keras.distribution.list_devices()],
    }


@compute
def train_classifier(epochs: int, batch_size: int) -> dict:
    """Train a simple classifier with Keras."""
    import keras
    from keras import layers

    pool = instance_info()

    # =================================================================
    # Model Definition
    # =================================================================
    model = keras.Sequential(
        [
            layers.Input(shape=(100,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # =================================================================
    # Generate and Shard Data
    # =================================================================
    import numpy as np

    # Full dataset (would be loaded from files in production)
    n_samples = 10000
    x_full = np.random.randn(n_samples, 100).astype(np.float32)
    y_full = np.random.randint(0, 10, n_samples)

    # Shard data for this node
    x_train, y_train = shard(x_full, y_full, shuffle=True, seed=42)

    # =================================================================
    # Training
    # =================================================================
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1 if pool.is_head else 0,
    )

    # =================================================================
    # Results
    # =================================================================
    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "samples_trained": len(x_train),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }


@compute
def train_autoencoder(epochs: int, latent_dim: int) -> dict:
    """Train an autoencoder with Keras."""
    import keras
    from keras import layers

    import numpy as np

    pool = instance_info()

    # =================================================================
    # Autoencoder Model
    # =================================================================
    # Encoder
    encoder_input = layers.Input(shape=(784,))
    x = layers.Dense(256, activation="relu")(encoder_input)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu")(x)
    encoder = keras.Model(encoder_input, latent, name="encoder")

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(decoder_input)
    x = layers.Dense(256, activation="relu")(x)
    decoder_output = layers.Dense(784, activation="sigmoid")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder
    autoencoder_input = layers.Input(shape=(784,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )

    # =================================================================
    # Synthetic MNIST-like data
    # =================================================================
    n_samples = 5000
    x_full = np.random.rand(n_samples, 784).astype(np.float32)

    # Shard for this node
    x_train = shard(x_full, shuffle=True, seed=42)

    # =================================================================
    # Training
    # =================================================================
    history = autoencoder.fit(
        x_train,
        x_train,  # Autoencoder reconstructs input
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=1 if pool.is_head else 0,
    )

    return {
        "node": pool.node,
        "latent_dim": latent_dim,
        "samples_trained": len(x_train),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }


if __name__ == "__main__":
    # =================================================================
    # Keras with JAX Backend
    # =================================================================
    with ComputePool(
        provider=AWS(),
        nodes=2,
        accelerator=NVIDIA.A100,
        pip=["keras>=3.0", "jax[cuda12]"],
        env={"KERAS_BACKEND": "jax"},
        spot="always",
    ) as pool:
        print("=" * 60)
        print("Keras Configuration")
        print("=" * 60)

        # Get Keras info from one node
        info = get_keras_info() >> pool
        print(f"Keras {info['keras_version']} with {info['backend']} backend")
        print(f"Devices: {info['devices']}")

        # =================================================================
        # Train Classifier
        # =================================================================
        print("\n" + "=" * 60)
        print("Training Classifier")
        print("=" * 60)

        classifier_results = train_classifier(epochs=10, batch_size=64) @ pool

        print("\nResults:")
        for r in classifier_results:
            role = "HEAD" if r["is_head"] else "WORKER"
            print(
                f"  Node {r['node']} ({role}): "
                f"{r['samples_trained']} samples, "
                f"acc={r['final_accuracy']:.2%}, "
                f"val_acc={r['final_val_accuracy']:.2%}"
            )

        # =================================================================
        # Train Autoencoder
        # =================================================================
        print("\n" + "=" * 60)
        print("Training Autoencoder")
        print("=" * 60)

        ae_results = train_autoencoder(epochs=20, latent_dim=32) @ pool

        print("\nResults:")
        for r in ae_results:
            print(
                f"  Node {r['node']}: "
                f"latent_dim={r['latent_dim']}, "
                f"loss={r['final_loss']:.4f}, "
                f"val_loss={r['final_val_loss']:.4f}"
            )
