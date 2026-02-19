# Sky Computing

> "We don't ask that readers accept Sky Computing as inevitable, merely as not impossible."
>
> — *The Sky Above The Clouds*, Berkeley 2022

Technology ecosystems follow a predictable pattern. Telephony started with AT&T's monopoly and evolved into carrier interoperability. The Internet began with proprietary networks — CompuServe, AOL, Prodigy — and converged on universal TCP/IP. Personal computers started with IBM dominance and opened into the x86 ecosystem. In each case, single providers or closed systems gave way to competitive markets with compatibility standards.

Cloud computing, barely 15 years old, is still in its pre-standards phase. Each provider is a silo with proprietary APIs: AWS, GCP, and Azure each have their own compute, storage, and networking interfaces. The paper [*The Sky Above The Clouds*][paper] (Berkeley, 2022) argues that this is the natural starting point of a technology ecosystem, not its final state. The proposed answer is **Sky Computing** — a compatibility layer above the clouds that routes workloads to the best provider based on cost, availability, or performance, without locking users to a single vendor.

## The Vision

Sky Computing proposes an intercloud broker that abstracts provider differences behind a unified interface. You describe what you need — compute, storage, accelerators — and the broker finds the best option across all available clouds. If one provider is cheaper, the workload goes there. If another has better availability for the GPU you need, it goes there instead. If a provider has an outage, the broker fails over transparently.

This is the same idea behind carrier portability in telephony (keep your phone number when you switch carriers) or TCP/IP in networking (your application doesn't know whether the packet travels over fiber, copper, or wireless). The abstraction boundary moves up: applications talk to the compatibility layer, the compatibility layer talks to providers.

## How Skyward Uses These Ideas

Skyward focuses on a specific problem within this vision: **running Python functions on remote accelerators without worrying about where**. The full Sky Computing vision includes storage interoperability, network optimization, and global job scheduling across clouds. Skyward narrows the scope to compute orchestration for ML workloads — but within that scope, it implements the key ideas from the paper.

### Provider Portability

All providers implement the same `CloudProvider` protocol — five methods (`prepare`, `provision`, `get_instance`, `terminate`, `teardown`) that map to the pool lifecycle. Switching between providers is a one-line change:

```python
# Development: local containers, zero cost
with sky.ComputePool(provider=sky.Container(), nodes=2) as pool:
    result = train(data) >> pool

# Production: real GPUs on AWS
with sky.ComputePool(provider=sky.AWS(), accelerator="H100", nodes=4) as pool:
    result = train(data) >> pool

# Same code, different cloud
with sky.ComputePool(provider=sky.RunPod(), accelerator="H100", nodes=4) as pool:
    result = train(data) >> pool
```

The `@sky.compute` functions, the operators, the `Image` specification — everything stays identical across providers. Your code doesn't know which cloud it's running on, and it doesn't need to. This is the practical realization of the Sky Computing idea: you're not locked to a single cloud, and moving between them doesn't require rewriting anything.

### Unified Resource Specification

You describe hardware needs in logical terms — `"A100"`, `sky.accelerators.H100(count=4)` — and the provider translates that into whatever its API requires. An "A100" on AWS is a `p4d.24xlarge` instance. On RunPod, it's a GPU pod with a specific `gpuTypeId`. On VastAI, it's a marketplace offer filtered by GPU model. The translation from a logical accelerator name to a provider-specific resource involves resolving instance types, memory variants, multi-GPU configurations, and availability constraints. The accelerator catalog centralizes this complexity so that `sky.accelerators.A100(count=4)` resolves correctly on any provider that supports it.

### Economic Arbitrage

The `allocation` parameter is a simple form of economic optimization. `"spot-if-available"` (the default) requests discounted spot capacity first and falls back to on-demand if none is available. `"spot"` always uses spot instances for maximum savings. `"cheapest"` compares all options and picks the lowest-cost one. These aren't as sophisticated as a full intercloud broker optimizing across providers simultaneously, but they capture the core idea: let the system find the best price for the hardware you need, rather than manually comparing instance types and pricing pages.

## What Skyward Adds: Ephemeral Compute

While the Berkeley paper focuses on interoperability, Skyward adds a specific philosophy: **ephemeral compute**. Accelerator infrastructure should exist only during your job — not before, not after. The `ComputePool` context manager guarantees this: provision on enter, destroy on exit, cleanup guaranteed even if your code throws an exception.

```python
with sky.ComputePool(provider=sky.AWS(), accelerator="H100", nodes=4) as pool:
    metrics = train(dataset) @ pool
# all instances terminated — no idle costs, no forgotten machines
```

This model fits ML workloads naturally. Training runs, fine-tuning jobs, hyperparameter sweeps, batch inference — these are all tasks with a beginning and an end. There are no machines to forget about, no environments that drift over time, no idle costs accumulating overnight. The pool's lifetime is the job's lifetime.

## Skyward vs SkyPilot

Both projects draw from the Sky Computing vision. The difference is in programming model.

**SkyPilot** is job-oriented. You define a task in YAML — a script, resource requirements, and storage mounts — and run `sky launch` to submit it. SkyPilot provisions a cluster, runs your script, and optionally tears it down. The cluster can persist between jobs (for iterative development) or auto-shutdown after idle time. Results are written to storage (S3, GCS, etc.) rather than returned to your code. SkyPilot excels at batch pipelines where jobs are self-contained units — you submit a script, it runs on the best available cloud, and the output lands in a storage bucket.

**Skyward** is function-oriented. You decorate a Python function with `@sky.compute` and dispatch it with `>>` or `@`. Your local Python process orchestrates everything — functions are transparently executed remotely and return results directly to your code. There's no YAML, no job submission, no separate storage layer for results. The programming model is a function call that happens to execute on a remote GPU. Skyward excels at interactive development where you iterate quickly and want remote compute to feel like a local operation.

The choice depends on your workflow. If your workload is "run this script end-to-end and store the results," SkyPilot is the right tool. If you want to call a function on a remote GPU and get the result back in a variable — composing remote computation with local logic, running experiments interactively, iterating on training code — Skyward is the right tool.

## Further Reading

- [The Sky Above The Clouds][paper] — The original Berkeley paper (2022)
- [SkyPilot][skypilot] — Berkeley's reference implementation
- [Sky Computing Lab][lab] — Research lab at UC Berkeley
- [Core Concepts](concepts.md) — Skyward's programming model and ephemeral compute
- [Getting Started](getting-started.md) — Installation and first steps
- [Providers](providers.md) — AWS, RunPod, VastAI, Verda, and Container configuration

[paper]: https://arxiv.org/abs/2205.07147
[skypilot]: https://github.com/skypilot-org/skypilot
[lab]: https://sky.cs.berkeley.edu/
