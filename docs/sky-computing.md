# Sky Computing

> "We don't ask that readers accept Sky Computing as inevitable, merely as not impossible."
>
> — *The Sky Above The Clouds*, Berkeley 2022

## The Evolution Pattern

Technology ecosystems follow a predictable pattern. Telephony, the Internet, and personal computers all started with single providers but evolved into competitive markets with **universal compatibility standards**.

| Era | Started As | Evolved To |
|-----|------------|------------|
| Telephony | AT&T monopoly | Carrier interoperability |
| Internet | Proprietary networks (CompuServe, AOL) | Universal TCP/IP |
| PCs | IBM dominance | Open x86 ecosystem |
| Cloud | AWS, GCP, Azure silos | **?** |

The paper [*The Sky Above The Clouds*][paper] (Berkeley, 2022) argues that cloud computing, barely 15 years old, is still in its pre-standards phase. Each provider is a silo with proprietary APIs.

The proposed answer: **Sky Computing** — a compatibility layer above the clouds.

## The Vision

Sky Computing proposes:

1. **Intercloud Broker**: Routes requests to the best provider based on cost, availability, or performance
2. **Compatibility APIs**: Single interface that abstracts provider differences
3. **Economic Arbitrage**: Automatically chooses the cheapest option
4. **Multi-cloud Resilience**: Transparent failover between providers

## How Skyward Uses These Ideas

Skyward focuses on a specific problem: **running Python functions on remote GPUs without worrying about where**.

### Multi-Provider Selection

Specify multiple providers — Skyward picks the best one:

```python
@sky.pool(
    provider=['aws', 'verda'],
    selection='cheapest',
    accelerator='T4',
)
def main():
    result = train(data) >> sky
```

Selection strategies:

| Strategy | Behavior |
|----------|----------|
| `"first"` | Use first provider in list (default) |
| `"cheapest"` | Compare prices, pick lowest |
| `"available"` | First provider with matching instances |
| `callable` | Your custom logic |

### Automatic Fallback

If a provider fails to provision, Skyward tries the next one:

```python
pool = ComputePool(
    provider=[AWS(), DigitalOcean(), Verda()],
    accelerator="H100",
)
# If AWS fails → tries DigitalOcean → tries Verda
```

No retry logic needed — it's handled for you.

### Unified Resource Specification

`accelerator="H100"` works across all providers:

```python
# AWS
ComputePool(provider=AWS(), accelerator="H100")

# Verda
ComputePool(provider=Verda(), accelerator="H100")

# Both work — Skyward finds the right instance type
```

## What Skyward Adds: Ephemeral Compute

While the Berkeley paper focuses on interoperability, Skyward adds a specific philosophy: **ephemeral compute**.

GPU infrastructure should exist only during your job. Not before, not after.

```python
with ComputePool(provider=AWS(), accelerator="H100", nodes=4) as pool:
    metrics = train(dataset) >> pool
# Instances terminated automatically
```

The context manager guarantees cleanup even if your code fails.

## Simple Execution API

Skyward uses Python operators for remote execution:

```python
result = train(data) >> pool       # Execute on one worker
results = init() @ pool            # Broadcast to all workers
a, b = (f1() & f2()) >> pool       # Parallel execution
```

## Distributed Training

Built-in support for multi-node training:

```python
@sky.integrations.torch(backend="nccl")
@sky.compute
def train():
    # MASTER_ADDR, RANK, WORLD_SIZE already configured
    model = DDP(MyModel().cuda())
    ...

@sky.pool(provider=AWS(), nodes=4, accelerator="A100")
def main():
    results = train() @ sky  # Runs on all 4 nodes
```

Also available: `@sky.integrations.keras()`, `@sky.integrations.jax()`, `@sky.integrations.transformers()`.

## Joblib Integration

Distribute scikit-learn workloads:

```python
with sky.integrations.JoblibPool(provider=AWS(), nodes=4, concurrency=4):
    grid = GridSearchCV(estimator, param_grid, n_jobs=-1)
    grid.fit(X, y)  # Distributed across 16 workers
```

## Skyward vs SkyPilot

Both draw from the Sky Computing vision. The difference is in programming model:

**SkyPilot**: You launch *jobs*. Define a task in YAML, run `sky launch`, and manage clusters. Your script runs on the cluster. Good for batch pipelines where jobs are self-contained units.

**Skyward**: You call *functions*. Your local Python orchestrates everything — functions are transparently executed remotely and return results to your code. Good for interactive development where you iterate quickly.

| | Skyward | SkyPilot |
|-|---------|----------|
| Unit of work | Python function | Shell command/script |
| Results | Returned to your code | Written to storage |
| Cluster lifecycle | Ephemeral (per `with` block) | Managed (persist or auto-down) |
| Iteration | Change code, rerun | Change YAML, relaunch |

Choose SkyPilot when your workload is "run this script end-to-end." Choose Skyward when you want remote compute to feel like a local function call.

## Further Reading

- [The Sky Above The Clouds][paper] — The original Berkeley paper (2022)
- [SkyPilot][skypilot] — Berkeley's reference implementation
- [Sky Computing Lab][lab] — Research lab at UC Berkeley

Skyward documentation:

- [Core Concepts](concepts.md) — Programming model and ephemeral compute
- [Getting Started](getting-started.md) — Installation and first steps
- [Providers](providers.md) — AWS, DigitalOcean, Verda configuration

[paper]: https://arxiv.org/abs/2205.07147
[skypilot]: https://github.com/skypilot-org/skypilot
[lab]: https://sky.cs.berkeley.edu/
