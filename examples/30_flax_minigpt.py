"""Distributed MicroGPT in JAX — Shower Thoughts edition.

Karpathy's MicroGPT reimplemented with Flax NNX and modern transformer
components: learnable LayerNorm, GeLU activation, weight tying, dropout,
bfloat16 mixed precision, and Optax optimizer with warmup + cosine decay +
gradient clipping.

Architecture based on the official JAX Stack miniGPT tutorial
(https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html),
adapted for character-level training on r/Showerthoughts with distributed
data parallelism via Skyward.

The model learns punctuation, capitalization, and sentence structure from
scratch — all at the character level.
"""

import skyward as sky


@sky.compute
@sky.integrations.jax()
@sky.stdout(only="head")
def train_microgpt(
    n_layer: int = 6,
    n_embd: int = 256,
    n_head: int = 8,
    block_size: int = 256,
    batch_size: int = 128,
    num_steps: int = 5000,
    lr: float = 3e-4,
    dropout_rate: float = 0.1,
    temperature: float = 0.8,
    num_samples: int = 20,
    data_url: str = "https://skeeto.s3.amazonaws.com/share/showerthoughts",
) -> dict:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax import random

    compute_dtype = jnp.bfloat16
    param_dtype = jnp.float32

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
        raw = f.read()

    entries = raw.split("\n%\n")
    docs = []
    for entry in entries:
        lines = entry.strip().split("\n")
        thought_lines = [
            line for line in lines
            if not line.startswith("—") and not line.startswith("\u2014")
        ]
        thought = " ".join(thought_lines).strip()
        if thought:
            docs.append(thought)
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

    def tokenize_all(documents: list[str]) -> np.ndarray:
        seq_len = block_size + 1
        sequences = []
        for doc in documents:
            toks = encode(doc)
            toks = toks[:seq_len] if len(toks) > seq_len else toks + [pad] * (seq_len - len(toks))
            sequences.append(toks)
        return np.array(sequences, dtype=np.int32)

    all_tokens = tokenize_all(local_docs)
    print(f"node={info.node} token_matrix={all_tokens.shape}")

    # --- Model (Flax NNX, bfloat16 mixed precision) ---

    class TransformerBlock(nnx.Module):
        def __init__(
            self, n_embd: int, n_head: int, *,
            rngs: nnx.Rngs, dropout_rate: float = 0.1,
        ):
            self.attn = nnx.MultiHeadAttention(
                num_heads=n_head,
                in_features=n_embd,
                dropout_rate=dropout_rate,
                decode=False,
                dtype=compute_dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.norm1 = nnx.LayerNorm(
                num_features=n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.norm2 = nnx.LayerNorm(
                num_features=n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.fc1 = nnx.Linear(
                n_embd, 4 * n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.fc2 = nnx.Linear(
                4 * n_embd, n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
            self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        def __call__(self, x, mask, *, deterministic: bool = True):
            # Pre-norm attention with residual
            h = self.norm1(x)
            h = self.attn(h, mask=mask, deterministic=deterministic)
            h = self.dropout1(h, deterministic=deterministic)
            x = x + h

            # Pre-norm FFN with GeLU and residual
            h = self.norm2(x)
            h = nnx.gelu(self.fc1(h))
            h = self.fc2(h)
            h = self.dropout2(h, deterministic=deterministic)
            x = x + h
            return x

    class MicroGPT(nnx.Module):
        def __init__(
            self, vocab_size: int, block_size: int, n_embd: int,
            n_head: int, n_layer: int, *,
            rngs: nnx.Rngs, dropout_rate: float = 0.1,
        ):
            self.token_emb = nnx.Embed(
                vocab_size, n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.pos_emb = nnx.Embed(
                block_size, n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )
            self.blocks = nnx.List([
                TransformerBlock(
                    n_embd, n_head,
                    rngs=rngs, dropout_rate=dropout_rate,
                )
                for _ in range(n_layer)
            ])
            self.final_norm = nnx.LayerNorm(
                num_features=n_embd,
                dtype=compute_dtype, param_dtype=param_dtype,
                rngs=rngs,
            )

        def __call__(self, tokens, *, deterministic: bool = True):
            _, t = tokens.shape
            # one_hot @ embedding — sharding-safe, computed in bfloat16
            emb_w = self.token_emb.embedding[...].astype(compute_dtype)
            tok_emb = jax.nn.one_hot(tokens, vocab_size, dtype=compute_dtype) @ emb_w
            pos_emb = self.pos_emb.embedding[...].astype(compute_dtype)[jnp.arange(t)]
            x = tok_emb + pos_emb

            mask = jnp.tril(jnp.ones((t, t), dtype=bool))
            for block in self.blocks:
                x = block(x, mask, deterministic=deterministic)

            x = self.final_norm(x)
            # weight tying — logits in float32 for numerical stability
            return (x @ emb_w.T).astype(jnp.float32)

    # --- Setup ---
    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch"),
    )
    replicated = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(),
    )

    model = MicroGPT(
        vocab_size, block_size, n_embd, n_head, n_layer,
        rngs=nnx.Rngs(42), dropout_rate=dropout_rate,
    )

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"node={info.node} params={n_params:,} dtype={compute_dtype}")

    # Replicate model state across devices
    graphdef, state = nnx.split(model)
    state = jax.device_put(state, replicated)
    model = nnx.merge(graphdef, state)

    # Pre-load token matrix on device (avoids per-step CPU→GPU transfer)
    all_tokens_jnp = jax.device_put(jnp.array(all_tokens), replicated)

    # Optax: warmup + cosine decay + gradient clipping + AdamW
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=num_steps // 10,
        decay_steps=num_steps,
        end_value=lr / 10,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Replicate optimizer state
    graphdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated)
    optimizer = nnx.merge(graphdef_opt, opt_state)

    # --- Training ---

    @nnx.jit
    def train_step(model, optimizer, inputs, targets):
        def loss_fn(model):
            logits = model(inputs, deterministic=False)  # float32 logits
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets,
            )
            mask = targets != pad
            return jnp.where(mask, loss, 0.0).sum() / jnp.maximum(mask.sum(), 1)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Sample batches on-device via JAX RNG (avoids numpy → jax per step)
    sample_key = random.PRNGKey(42 + info.node)
    n_docs = all_tokens_jnp.shape[0]
    log_interval = max(1, num_steps // 20)
    final_loss = 0.0

    for step in range(num_steps):
        sample_key, subkey = random.split(sample_key)
        idx = random.randint(subkey, (batch_size,), 0, n_docs)
        batch = all_tokens_jnp[idx]
        inputs_local = batch[:, :-1]
        targets_local = batch[:, 1:]

        inputs = jax.make_array_from_process_local_data(data_sharding, inputs_local)
        targets = jax.make_array_from_process_local_data(data_sharding, targets_local)

        loss = train_step(model, optimizer, inputs, targets)

        final_loss = float(loss)
        if step % log_interval == 0 or step == num_steps - 1:
            print(f"step {step+1:5d}/{num_steps} | loss {final_loss:.4f}")

    # --- Inference (head node only) ---
    samples = []
    if info.node == 0:
        print(
            f"\n--- generating {num_samples} shower thoughts"
            f" (temperature={temperature}) ---"
        )

        @nnx.jit
        def forward(model, tokens):
            return model(tokens, deterministic=True)

        gen_key = random.PRNGKey(1337)
        for si in range(num_samples):
            token_buf = jnp.full((block_size,), pad, dtype=jnp.int32)
            token_buf = token_buf.at[0].set(bos)
            length = 1

            gen_key, subkey = random.split(gen_key)

            for _ in range(block_size - 1):
                logits = forward(model, token_buf[None, :])
                next_logits = logits[0, length - 1] / temperature
                subkey, step_key = random.split(subkey)
                token = random.categorical(step_key, next_logits)
                if int(token) == bos:
                    break
                token_buf = token_buf.at[length].set(token)
                length += 1

            thought = decode(token_buf[1:length].tolist())
            samples.append(thought)
            print(f"  {si+1:2d}. {thought}")

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
        print("\nGenerated shower thoughts:")
        for i, s in enumerate(head["samples"], 1):
            print(f"  {i:2d}. {s}")


if __name__ == "__main__":
    with  sky.ComputePool(
        # sky.Spec(
        #     provider=sky.AWS(),
        #     accelerator=sky.accelerators.T4G(),
        #     ttl=1200
        # ),
        sky.Spec(
            provider=sky.AWS(),
            accelerator=sky.accelerators.L4(),
            ttl=1200
        ),
        sky.Spec(
            provider=sky.AWS(),
            accelerator=sky.accelerators.L40S(),
            ttl=1200
        ),
        image=sky.Image(pip=["jax[cuda12]", "flax"]),
    ) as pool:
        results = train_microgpt().with_timeout(1130) @ pool

        print("\nResults:")
        format_results(list(results))
