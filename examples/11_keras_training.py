"""Vision Transformer (ViT) for MNIST Classification.

Demonstrates training a ViT with Keras 3 on Skyward:
- Pure Keras 3 implementation (backend-agnostic)
- Uses Skyward's shard() for data distribution
- Runs on GPU with JAX backend

Architecture:
    MNIST 28x28 → Patches 4x4 (49 patches) → Embedding + Pos → Transformer × N → [CLS] → 10 classes
"""
from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers

import skyward as sky

@sky.compute
@sky.integrations.keras(backend="jax")
def train_vit(
    vit_config: ViTConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict:
    """Train Vision Transformer on MNIST shard."""
    keras.config.disable_interactive_logging()

    vit_config = vit_config or ViTConfig()
    training_config = training_config or TrainingConfig()
    pool = sky.instance_info()

    # Load and shard data
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist()
    x_train, y_train = sky.shard(x_train_full, y_train_full, shuffle=True, seed=42)

    # Build model
    model = ViT(vit_config)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if pool.is_head:
        model.summary()

    # Train
    history = model.fit(
        x_train,
        y_train,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        verbose=0,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return {
        "node": pool.node,
        "is_head": pool.is_head,
        "samples_trained": len(x_train),
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

@sky.pool(
    provider=sky.AWS(),
    nodes=3,
    accelerator=sky.Accelerator.NVIDIA.T4(lambda _: 2 >= _ >= 1),
    image=sky.Image(
        pip=["keras>=3.2", "jax[cuda12]"],
        env={"KERAS_BACKEND": "jax"},
    ),
)
def clustered_train() -> tuple[dict, ...]:
    return train_vit() @ sky

@dataclass(frozen=True, slots=True)
class ViTConfig:
    image_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 128
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.1


class PatchEmbedding(layers.Layer):
    """Extract patches and project to embedding space with [CLS] token."""

    def __init__(self, embed_dim: int, patch_size: int, num_patches: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="position_embedding",
        )
        super().build(input_shape)

    def call(self, images):
        batch_size = keras.ops.shape(images)[0]

        # (B, H, W, C) -> (B, num_patches, embed_dim)
        patches = self.projection(images)
        patches = keras.ops.reshape(patches, (batch_size, -1, self.embed_dim))

        # Prepend [CLS] token
        cls_tokens = keras.ops.broadcast_to(
            self.cls_token, (batch_size, 1, self.embed_dim)
        )
        embeddings = keras.ops.concatenate([cls_tokens, patches], axis=1)

        return embeddings + self.position_embedding


class TransformerBlock(layers.Layer):
    """Single transformer encoder block with pre-norm architecture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Attention
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)

        # FFN
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(mlp_dim, activation="gelu")
        self.dropout2 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(embed_dim)
        self.dropout3 = layers.Dropout(dropout)

    def call(self, x, training=None):
        # Self-attention with residual
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, norm_x, training=training)
        x = x + self.dropout1(attn_out, training=training)

        # FFN with residual
        norm_x = self.norm2(x)
        ffn_out = self.dense1(norm_x)
        ffn_out = self.dropout2(ffn_out, training=training)
        ffn_out = self.dense2(ffn_out)
        return x + self.dropout3(ffn_out, training=training)


class ViT(keras.Model):
    """Vision Transformer for image classification.

    Args:
        config: Model architecture configuration.

    Example:
        >>> config = ViTConfig(embed_dim=128, num_layers=6)
        >>> model = ViT(config)
        >>> model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        >>> model.fit(x_train, y_train)
    """

    def __init__(self, config: ViTConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or ViTConfig()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            embed_dim=self.config.embed_dim,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )
        self.embedding_dropout = layers.Dropout(self.config.dropout)

        # Transformer encoder
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_dim=self.config.mlp_dim,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ]

        # Classification head
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.head_dense = layers.Dense(self.config.mlp_dim, activation="gelu")
        self.head_dropout = layers.Dropout(self.config.dropout)
        self.classifier = layers.Dense(self.config.num_classes, activation="softmax")

    def call(self, inputs, training=None):
        # Patch embedding
        x = self.patch_embedding(inputs)
        x = self.embedding_dropout(x, training=training)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Extract [CLS] token and classify
        x = self.final_norm(x)
        cls_output = x[:, 0]

        x = self.head_dense(cls_output)
        x = self.head_dropout(x, training=training)
        return self.classifier(x)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "config": {
                "image_size": self.config.image_size,
                "patch_size": self.config.patch_size,
                "num_classes": self.config.num_classes,
                "embed_dim": self.config.embed_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "mlp_dim": self.config.mlp_dim,
                "dropout": self.config.dropout,
            },
        }

def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dimension
    x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
    x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, axis=-1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    results = clustered_train()

    print("\nResults:")
    for r in results:
        role = "HEAD" if r["is_head"] else "WORKER"
        print(
            f"  Node {r['node']} ({role}): "
            f"{r['samples_trained']} samples, "
            f"acc={r['final_accuracy']:.2%}, "
            f"val_acc={r['final_val_accuracy']:.2%}, "
            f"test_acc={r['test_accuracy']:.2%}"
        )
