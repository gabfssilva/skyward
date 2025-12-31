"""Vision Transformer (ViT) for MNIST Classification.

Demonstrates training a ViT with Keras 3 on Skyward:
- Pure Keras 3 implementation (backend-agnostic)
- Uses Skyward's shard() for data distribution
- Runs on GPU with JAX backend

Architecture:
    MNIST 28x28 → Patches 4x4 (49 patches) → Embedding + Pos → Transformer × N → [CLS] → 10 classes
"""
from typing import Literal, LiteralString

from skyward import AWS, ComputePool, compute, distributed, instance_info, shard, Verda, Accelerator, Image


# =============================================================================
# ViT Components
# =============================================================================


def create_patch_embedding(embed_dim: int, patch_size: int, num_patches: int):
    """Create patch embedding layer."""
    import keras
    from keras import layers

    class PatchEmbedding(layers.Layer):
        def __init__(self, embed_dim: int, patch_size: int, num_patches: int, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            self.num_patches = num_patches

            # Patch extraction via Conv2D
            self.projection = layers.Conv2D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid",
            )

            # Learnable [CLS] token
            self.cls_token = self.add_weight(
                shape=(1, 1, embed_dim),
                initializer="random_normal",
                trainable=True,
                name="cls_token",
            )

            # Positional embedding (num_patches + 1 for CLS)
            self.position_embedding = self.add_weight(
                shape=(1, num_patches + 1, embed_dim),
                initializer="random_normal",
                trainable=True,
                name="position_embedding",
            )

        def call(self, images):
            import keras

            batch_size = keras.ops.shape(images)[0]

            # (B, H, W, C) -> (B, num_patches, embed_dim)
            patches = self.projection(images)
            patches = keras.ops.reshape(patches, (batch_size, -1, self.embed_dim))

            # Prepend [CLS] token
            cls_tokens = keras.ops.broadcast_to(
                self.cls_token, (batch_size, 1, self.embed_dim)
            )
            embeddings = keras.ops.concatenate([cls_tokens, patches], axis=1)

            # Add positional embedding
            return embeddings + self.position_embedding

    return PatchEmbedding(embed_dim, patch_size, num_patches)


def create_transformer_block(embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
    """Create a single transformer encoder block."""
    from keras import layers

    class TransformerBlock(layers.Layer):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_dim: int,
            dropout: float,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_dim = mlp_dim

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
            self.ffn = [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]

        def call(self, x, training=None):
            # Self-attention with residual
            norm_x = self.norm1(x)
            attn_out = self.attention(norm_x, norm_x, training=training)
            x = x + self.dropout1(attn_out, training=training)

            # FFN with residual
            norm_x = self.norm2(x)
            ffn_out = norm_x
            for layer in self.ffn:
                ffn_out = layer(ffn_out, training=training) if hasattr(layer, "training") else layer(ffn_out)
            return x + ffn_out

    return TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)


def build_vit(
    image_size: int = 28,
    patch_size: int = 4,
    num_classes: int = 10,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    mlp_dim: int = 128,
    dropout: float = 0.1,
):
    """Build Vision Transformer model."""
    import keras
    from keras import layers

    num_patches = (image_size // patch_size) ** 2  # 49 for 28x28 with 4x4 patches

    # Input
    inputs = layers.Input(shape=(image_size, image_size, 1))

    # Patch embedding
    x = create_patch_embedding(embed_dim, patch_size, num_patches)(inputs)
    x = layers.Dropout(dropout)(x)

    # Transformer encoder blocks
    for _ in range(num_layers):
        x = create_transformer_block(embed_dim, num_heads, mlp_dim, dropout)(x)

    # Extract [CLS] token and classify
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_output = x[:, 0]  # [CLS] token

    # Classification head
    x = layers.Dense(mlp_dim, activation="gelu")(cls_output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="vit_mnist")


# =============================================================================
# Training Function
# =============================================================================


@compute
# @distributed.keras(backend="jax")
def train_vit(
    epochs: int = 10,
    batch_size: int = 128,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    mlp_dim: int = 128,
    dropout: float = 0.1,
) -> dict:
    """Train Vision Transformer on MNIST."""
    import keras
    import numpy as np

    keras.config.disable_interactive_logging()

    pool = instance_info()

    # =========================================================================
    # Load MNIST
    # =========================================================================
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dimension
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_train_full = np.expand_dims(x_train_full, axis=-1)  # (60000, 28, 28, 1)

    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    # =========================================================================
    # Shard Data
    # =========================================================================
    x_train, y_train = shard(x_train_full, y_train_full, shuffle=True, seed=42)

    # =========================================================================
    # Build Model
    # =========================================================================
    model = build_vit(
        image_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout=dropout,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if pool.is_head:
        model.summary()

    # =========================================================================
    # Train
    # =========================================================================
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    )

    # =========================================================================
    # Evaluate on Test Set
    # =========================================================================
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    with ComputePool(
        provider=AWS(),
        # nodes=2,
        accelerator=Accelerator.NVIDIA.T4(count=0.2),
        image=Image(
            pip=["keras>=3.2", "jax[cuda12]"],
            env={"KERAS_BACKEND": "jax"},
        ),
        spot="always",
        timeout=1200,
    ) as pool:
        print("=" * 60)
        print("Vision Transformer (ViT) - MNIST Classification")
        print("=" * 60)

        results = train_vit(
            epochs=10,
            batch_size=128,
            embed_dim=64,
            num_heads=4,
            num_layers=4,
            mlp_dim=128,
            dropout=0.1,
        ) @ pool

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
