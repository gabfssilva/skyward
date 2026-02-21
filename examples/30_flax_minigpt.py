"""Distributed MicroGPT in JAX — Recipe edition.

Small GPT trained from scratch with Flax NNX: pre-norm transformer blocks,
GeLU, weight tying, dropout, bfloat16 mixed precision, and Optax optimizer
with warmup + cosine decay + gradient clipping.

Trains on the RecipeNLG dataset (~248K recipes) with GPT-2's BPE tokenizer
(tiktoken) and distributed data parallelism via Skyward. Tokens are
concatenated into a single flat array on GPU, with random window sampling
during training — zero padding waste.

The model learns recipe structure — titles, ingredient measurements, and
cooking instructions — all from scratch.
"""

from __future__ import annotations

import ast
import os
import tempfile
import urllib.request

import pandas as pd

import skyward as sky

DATA_URL = (
    "https://huggingface.co/datasets/Default-Box/recipe_nlg-trim"
    "/resolve/main/train.csv"
)


@sky.compute
@sky.integrations.jax()
@sky.stdout(only="head")
def train_microgpt(
    n_layer: int = 6,
    n_embd: int = 256,
    n_head: int = 8,
    block_size: int = 512,
    batch_size: int = 64,
    num_epochs: int = 2,
    lr: float = 3e-4,
    dropout_rate: float = 0.1,
    temperature: float = 0.8,
    num_samples: int = 10,
) -> dict:
    from functools import reduce

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import tiktoken
    from jax import random

    recipes = load_recipes(DATA_URL)

    compute_dtype = jnp.bfloat16
    param_dtype = jnp.float32

    info = sky.instance_info()
    assert info is not None
    print(f"node={info.node} devices={jax.devices()} local={jax.local_devices()}")

    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch"),
    )
    replicated = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(),
    )

    docs: list[str] = (
        recipes["title"] + "\n" + recipes["ingredients"] + "\n" + recipes["directions"]
    ).tolist()
    local_docs = sky.shard(docs)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    vocab_size = enc.n_vocab

    encoded = enc.encode_batch(local_docs)
    flat_tokens = [t for doc_toks in encoded for t in (eot, *doc_toks)]
    flat_tokens.append(eot)

    tokens_gpu = jax.device_put(
        jnp.array(flat_tokens, dtype=jnp.int32), replicated,
    )
    n_tokens = tokens_gpu.shape[0]
    print(
        f"node={info.node} vocab={vocab_size}"
        f" total_docs={len(docs)} local_docs={len(local_docs)}"
        f" tokens={n_tokens:,}"
    )

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, n_tokens // tokens_per_step)
    num_steps = num_epochs * steps_per_epoch
    print(
        f"node={info.node} epochs={num_epochs}"
        f" steps_per_epoch={steps_per_epoch} total_steps={num_steps}"
    )

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
            h = self.norm1(x)
            h = self.attn(h, mask=mask, deterministic=deterministic)
            h = self.dropout1(h, deterministic=deterministic)
            x = x + h

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
            emb_w = self.token_emb.embedding[...]
            batch_spec = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch", None, None),
            )
            tok_emb = emb_w.astype(compute_dtype).at[tokens].get(
                out_sharding=batch_spec,
            )
            pos_emb = self.pos_emb(jnp.arange(t))

            mask = jnp.tril(jnp.ones((t, t), dtype=bool))
            x = reduce(
                lambda h, block: block(h, mask, deterministic=deterministic),
                self.blocks,
                tok_emb + pos_emb,
            )
            x = self.final_norm(x)
            return x.astype(jnp.float32) @ emb_w.astype(jnp.float32).T

    model = MicroGPT(
        vocab_size, block_size, n_embd, n_head, n_layer,
        rngs=nnx.Rngs(42), dropout_rate=dropout_rate,
    )

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"node={info.node} params={n_params:,} dtype={compute_dtype}")

    graphdef, state = nnx.split(model)
    state = jax.device_put(state, replicated)
    model = nnx.merge(graphdef, state)

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

    graphdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated)
    optimizer = nnx.merge(graphdef_opt, opt_state)

    def get_window(start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(tokens_gpu, (start,), (block_size + 1,))

    @nnx.jit
    def train_step(model, optimizer, inputs, targets):
        def loss_fn(model):
            logits = model(inputs, deterministic=False)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets,
            ).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    sample_key = random.PRNGKey(42 + info.node)
    max_start = n_tokens - block_size - 1
    log_interval = max(1, num_steps // 20)
    final_loss = 0.0

    for step in range(num_steps):
        sample_key, subkey = random.split(sample_key)
        starts = random.randint(subkey, (batch_size,), 0, max_start)
        batch = jax.vmap(get_window)(starts)
        inputs_local = batch[:, :-1]
        targets_local = batch[:, 1:]

        inputs = jax.make_array_from_process_local_data(data_sharding, inputs_local)
        targets = jax.make_array_from_process_local_data(data_sharding, targets_local)

        loss = train_step(model, optimizer, inputs, targets)

        final_loss = float(loss)
        if step % log_interval == 0 or step == num_steps - 1:
            print(f"step {step+1:5d}/{num_steps} | loss {final_loss:.4f}")

    samples: list[str] = []
    if info.node == 0:
        print(
            f"\n--- generating {num_samples} recipes"
            f" (temperature={temperature}) ---"
        )

        @nnx.jit
        def forward(model, tokens):
            return model(tokens, deterministic=True)

        gen_key = random.PRNGKey(1337)
        for si in range(num_samples):
            token_buf = jnp.full((block_size,), eot, dtype=jnp.int32)
            length = 1

            gen_key, subkey = random.split(gen_key)

            for _ in range(block_size - 1):
                logits = forward(model, token_buf[None, :])
                next_logits = logits[0, length - 1] / temperature
                subkey, step_key = random.split(subkey)
                token = random.categorical(step_key, next_logits)
                if int(token) == eot:
                    break
                token_buf = token_buf.at[length].set(token)
                length += 1

            recipe = enc.decode(token_buf[1:length].tolist())
            samples.append(recipe)
            lines = recipe.split("\n")
            body = "\n".join(f"      {line}" for line in lines[1:])
            print(f"\n  {si+1:2d}. {lines[0]}\n{body}")

    return {
        "node": info.node,
        "train_docs": len(local_docs),
        "final_loss": final_loss,
        "samples": samples,
    }


def _parse_list_col(s: object) -> str:
    try:
        match ast.literal_eval(str(s)):
            case list(items):
                return "\n".join(items)
            case other:
                return str(other)
    except (ValueError, SyntaxError):
        return ""


def load_recipes(url: str) -> pd.DataFrame:
    path = os.path.join(tempfile.gettempdir(), "recipe_nlg.csv")
    if not os.path.exists(path):
        print(f"Downloading recipes from {url}...")
        urllib.request.urlretrieve(url, path)

    print("Parsing recipes...")
    df: pd.DataFrame = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["ingredients"] = df["ingredients"].apply(_parse_list_col)
    df["directions"] = df["directions"].apply(_parse_list_col)
    df = pd.DataFrame(df[["title", "ingredients", "directions"]].dropna())
    df = pd.DataFrame(df[(df["ingredients"] != "") & (df["directions"] != "")])

    print(f"Loaded {len(df):,} recipes")
    return df.reset_index(drop=True)


def _format_recipe(index: int, recipe: str) -> str:
    lines = recipe.split("\n")
    body = "\n".join(f"      {line}" for line in lines[1:])
    return f"\n  {index:2d}. {lines[0]}\n{body}"


def format_results(results: list[dict]) -> None:
    summary = "\n".join(
        f"  Node {r['node']}: {r['train_docs']:,} docs, final_loss={r['final_loss']:.4f}"
        for r in results
    )
    print(summary)

    head = next(r for r in results if r["node"] == 0)
    if samples := head.get("samples"):
        print("\nGenerated recipes:")
        print("\n".join(_format_recipe(i, r) for i, r in enumerate(samples, 1)))


if __name__ == "__main__":
    with sky.ComputePool(
        sky.Spec(
            provider=sky.RunPod(),
            accelerator=sky.accelerators.RTX_4090(),
            ttl=2400
        ),
        image=sky.Image(pip=["jax[cuda12]", "flax", "pandas==2.3.3", "tiktoken"]),
    ) as pool:
        results = train_microgpt().with_timeout(2400) @ pool

        print("\nResults:")
        format_results(list(results))
