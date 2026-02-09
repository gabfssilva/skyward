"""JAX Distributed Training Test.

Trains a simple MLP on MNIST using pure JAX data parallelism across nodes.
No Keras — isolates JAX multi-host distributed training.

With jit + sharded data + replicated params, JAX automatically inserts
all-reduce for gradients (no explicit pmean needed).
"""

import skyward as sky


@sky.compute
@sky.integrations.jax()
def train_distributed() -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import jit, random, value_and_grad

    pool = sky.instance_info()

    print(f"node={pool.node} process={jax.process_index()}/{jax.process_count()}")
    print(f"node={pool.node} devices={jax.devices()} local={jax.local_devices()}")

    # --- Mesh for data parallelism ---
    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(f"node={pool.node} mesh={mesh}")

    # --- Data: MNIST (flatten 28x28 → 784) ---
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x_all = mnist.data.astype(np.float32) / 255.0
    y_all = mnist.target.astype(np.int32)

    x_local, y_local = sky.shard(x_all[:60000], y_all[:60000], shuffle=True, seed=42)
    x_test, y_test = x_all[60000:], y_all[60000:]

    print(f"node={pool.node} local_train={len(x_local)} test={len(x_test)}")

    # --- Model: 784 → 128 → 10 (replicated params) ---
    key = random.PRNGKey(42)
    k1, k2 = random.split(key)

    params = jax.device_put({
        "w1": random.normal(k1, (784, 128)) * 0.01,
        "b1": jnp.zeros(128),
        "w2": random.normal(k2, (128, 10)) * 0.01,
        "b2": jnp.zeros(10),
    }, replicated)

    def predict(params, x):
        h = jnp.maximum(0, x @ params["w1"] + params["b1"])
        logits = h @ params["w2"] + params["b2"]
        return logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)

    def loss_fn(params, x, y):
        log_probs = predict(params, x)
        return -jnp.mean(log_probs[jnp.arange(len(y)), y])

    # jit auto-inserts all-reduce: sharded data → replicated grads → replicated params
    @jit
    def train_step(params, x, y):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
        return params, loss

    # --- Train ---
    batch_size = 256
    n_batches = len(x_local) // batch_size
    epochs = 5
    avg_loss = 0.0

    for epoch in range(epochs):
        perm = np.random.RandomState(epoch).permutation(len(x_local))
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            local_x = jnp.array(x_local[idx])
            local_y = jnp.array(y_local[idx])

            global_x = jax.make_array_from_process_local_data(data_sharding, local_x)
            global_y = jax.make_array_from_process_local_data(data_sharding, local_y)

            params, batch_loss = train_step(params, global_x, global_y)
            epoch_loss += float(batch_loss)

        avg_loss = epoch_loss / n_batches
        if pool.node == 0:
            print(f"epoch={epoch} loss={avg_loss:.4f}")

    # --- Eval ---
    test_logits = predict(params, jnp.array(x_test))
    preds = jnp.argmax(test_logits, axis=-1)
    accuracy = float(jnp.mean(preds == jnp.array(y_test)))
    print(f"node={pool.node} test_accuracy={accuracy:.4f}")

    return {
        "node": pool.node,
        "train_samples": len(x_local),
        "test_accuracy": accuracy,
        "final_loss": avg_loss,
    }


@sky.pool(
    provider=sky.AWS(),
    nodes=2,
    accelerator=sky.accelerators.T4G(),
    image=sky.Image(
        pip=["jax[cuda12]==0.8.2", "scikit-learn"],
        skyward_source="local",
    ),
)
def main():
    results = train_distributed() @ sky

    print("\nResults:")
    for r in results:
        print(
            f"  Node {r['node']}: "
            f"{r['train_samples']} samples, "
            f"loss={r['final_loss']:.4f}, "
            f"test_acc={r['test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
