"""Distributed MicroGPT in JAX.

Karpathy's MicroGPT (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
reimplemented in idiomatic JAX with distributed data parallelism via Skyward.

The original gist trains a character-level GPT in pure Python (~200 lines, zero deps).
This version replaces the custom autograd with JAX's jit + value_and_grad, processes
full sequences with causal masks instead of one token at a time, and distributes
training across multiple GPUs with automatic gradient all-reduce.
"""

import skyward as sky


@sky.compute
@sky.integrations.jax()
@sky.stdout(only="head")
def train_microgpt(
    n_layer: int = 4,
    n_embd: int = 64,
    n_head: int = 4,
    block_size: int = 32,
    batch_size: int = 64,
    num_steps: int = 2000,
    lr: float = 3e-4,
    temperature: float = 0.5,
    num_samples: int = 20,
    data_url: str = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import jit, random, value_and_grad

    info = sky.instance_info()
    assert info is not None
    print(f"node={info.node} devices={jax.devices()} local={jax.local_devices()}")

    # --- Tokenizer ---
    import os
    import tempfile
    import urllib.request

    data_path = os.path.join(tempfile.gettempdir(), "microgpt_data.txt")
    if not os.path.exists(data_path):
        urllib.request.urlretrieve(data_url, data_path)

    with open(data_path) as f:
        docs = [line.strip() for line in f if line.strip()]
    np.random.RandomState(42).shuffle(docs)

    uchars = sorted(set("".join(docs)))
    bos = len(uchars)
    pad = len(uchars) + 1
    vocab_size = len(uchars) + 2  # chars + bos + pad
    char_to_id = {ch: i for i, ch in enumerate(uchars)}

    def encode(doc: str) -> list[int]:
        return [bos] + [char_to_id[ch] for ch in doc] + [bos]

    def decode(token_ids: list[int]) -> str:
        return "".join(uchars[t] for t in token_ids if t != bos)

    # Shard documents across nodes
    local_docs = sky.shard(docs)
    print(
        f"node={info.node} vocab={vocab_size}"
        f" total_docs={len(docs)} local_docs={len(local_docs)}"
    )

    # Pre-tokenize and pad/truncate to (num_docs, block_size+1) for input/target pairs
    def tokenize_all(documents: list[str]) -> np.ndarray:
        seq_len = block_size + 1  # +1 for target shift
        sequences = []
        for doc in documents:
            toks = encode(doc)
            toks = toks[:seq_len] if len(toks) > seq_len else toks + [pad] * (seq_len - len(toks))
            sequences.append(toks)
        return np.array(sequences, dtype=np.int32)

    all_tokens = tokenize_all(local_docs)
    print(f"node={info.node} token_matrix={all_tokens.shape}")

    # --- Model ---
    head_dim = n_embd // n_head

    def init_params(key):
        def normal(k, shape):
            return random.normal(k, shape) * 0.08

        keys = random.split(key, 3 + n_layer * 6)
        ki = iter(keys)
        params = {
            "wte": normal(next(ki), (vocab_size, n_embd)),
            "wpe": normal(next(ki), (block_size, n_embd)),
            "layers": [
                {
                    "attn_wq": normal(next(ki), (n_embd, n_embd)),
                    "attn_wk": normal(next(ki), (n_embd, n_embd)),
                    "attn_wv": normal(next(ki), (n_embd, n_embd)),
                    "attn_wo": normal(next(ki), (n_embd, n_embd)),
                    "mlp_fc1": normal(next(ki), (n_embd, 4 * n_embd)),
                    "mlp_fc2": normal(next(ki), (4 * n_embd, n_embd)),
                }
                for _ in range(n_layer)
            ],
            "lm_head": normal(next(ki), (n_embd, vocab_size)),
        }
        return params

    def rmsnorm(x):
        ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
        return x * jnp.rsqrt(ms + 1e-5)

    def attention(x, layer_params, mask):
        b, t, c = x.shape
        q = x @ layer_params["attn_wq"]
        k = x @ layer_params["attn_wk"]
        v = x @ layer_params["attn_wv"]

        q = q.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        attn = jnp.where(mask[:t, :t], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, c)
        return out @ layer_params["attn_wo"]

    def mlp(x, layer_params):
        h = jax.nn.relu(x @ layer_params["mlp_fc1"])
        return h @ layer_params["mlp_fc2"]

    def transformer_block(x, layer_params, mask):
        x = x + attention(rmsnorm(x), layer_params, mask)
        x = x + mlp(rmsnorm(x), layer_params)
        return x

    def gpt(params, tokens):
        _, t = tokens.shape
        tok_emb = params["wte"][tokens]
        pos_emb = params["wpe"][jnp.arange(t)]
        x = rmsnorm(tok_emb + pos_emb)

        mask = jnp.tril(jnp.ones((block_size, block_size), dtype=bool))
        for layer_params in params["layers"]:
            x = transformer_block(x, layer_params, mask)

        logits = x @ params["lm_head"]
        return logits

    # --- Training ---
    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    params = jax.device_put(init_params(random.PRNGKey(42)), replicated)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"node={info.node} params={n_params:,}")

    def loss_fn(params, inputs, targets):
        logits = gpt(params, inputs)                    # (B, T, vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)                                    # (B, T)
        mask = targets != pad
        return -jnp.where(mask, target_log_probs, 0.0).sum() / jnp.maximum(mask.sum(), 1)

    @jit
    def train_step(params, m_state, v_state, inputs, targets, step):
        loss, grads = value_and_grad(loss_fn)(params, inputs, targets)

        # Adam with linear LR decay
        lr_t = lr * (1.0 - step / num_steps)
        beta1, beta2, eps = 0.85, 0.99, 1e-8

        def adam_update(p, g, m, v):
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m_new / (1 - beta1 ** (step + 1))
            v_hat = v_new / (1 - beta2 ** (step + 1))
            p_new = p - lr_t * m_hat / (jnp.sqrt(v_hat) + eps)
            return p_new, m_new, v_new

        updates = jax.tree.map(adam_update, params, grads, m_state, v_state)
        params = jax.tree.map(lambda u: u[0], updates)
        m_state = jax.tree.map(lambda u: u[1], updates)
        v_state = jax.tree.map(lambda u: u[2], updates)

        return params, m_state, v_state, loss

    # Init Adam state
    m_state = jax.tree.map(jnp.zeros_like, params)
    v_state = jax.tree.map(jnp.zeros_like, params)

    # Training loop
    rng = np.random.RandomState(42 + info.node)
    log_interval = max(1, num_steps // 20)
    final_loss = 0.0

    for step in range(num_steps):
        # Sample batch
        idx = rng.randint(0, len(all_tokens), size=batch_size)
        batch = all_tokens[idx]
        inputs_np = batch[:, :-1]   # (B, block_size)
        targets_np = batch[:, 1:]   # (B, block_size)

        local_inputs = jnp.array(inputs_np)
        local_targets = jnp.array(targets_np)
        inputs = jax.make_array_from_process_local_data(data_sharding, local_inputs)
        targets = jax.make_array_from_process_local_data(data_sharding, local_targets)

        params, m_state, v_state, loss = train_step(
            params, m_state, v_state, inputs, targets, jnp.float32(step)
        )

        final_loss = float(loss)
        if step % log_interval == 0 or step == num_steps - 1:
            print(f"step {step+1:5d}/{num_steps} | loss {final_loss:.4f}")

    # --- Inference (head node only) ---
    samples = []
    if info.node == 0:
        print(f"\n--- generating {num_samples} samples (temperature={temperature}) ---")

        @jit
        def get_logits(params, token_buf, length):
            logits = gpt(params, token_buf[None, :])  # (1, block_size, vocab)
            return logits[0, length - 1]

        sample_key = random.PRNGKey(1337)
        for si in range(num_samples):
            token_buf = jnp.full((block_size,), pad, dtype=jnp.int32)
            token_buf = token_buf.at[0].set(bos)
            length = 1

            sample_key, subkey = random.split(sample_key)

            for _ in range(block_size - 1):
                logits = get_logits(params, token_buf, length)
                scaled = logits / temperature
                subkey, gen_key = random.split(subkey)
                token = random.categorical(gen_key, scaled)
                if int(token) == bos:
                    break
                token_buf = token_buf.at[length].set(token)
                length += 1

            name = decode(token_buf[1:length].tolist())
            samples.append(name)
            print(f"  {si+1:2d}. {name}")

    return {
        "node": info.node,
        "train_docs": len(local_docs),
        "final_loss": final_loss,
        "samples": samples,
    }


def format_results(results: list[dict]) -> None:
    for r in results:
        node = r["node"]
        loss = r["final_loss"]
        docs = r["train_docs"]
        print(f"  Node {node}: {docs} docs, final_loss={loss:.4f}")

    head = next(r for r in results if r["node"] == 0)
    if head.get("samples"):
        print("\nGenerated samples:")
        for i, s in enumerate(head["samples"], 1):
            print(f"  {i:2d}. {s}")


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.VastAI(),
        nodes=2,
        accelerator=sky.accelerators.RTX_4090(),
        image=sky.Image(pip=["jax[cuda13]"]),
    ) as pool:
        results = train_microgpt() @ pool

        print("\nResults:")
        format_results(list(results))
