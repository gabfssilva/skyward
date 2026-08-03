<p align="center">
  <img src="logo_sky.png" alt="Skyward" width="400">
</p>

<p align="center">
  <strong>Cloud accelerators with a single API</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/v/skyward.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/skyward/"><img src="https://img.shields.io/pypi/pyversions/skyward.svg" alt="Python"></a>
  <a href="https://github.com/gabfssilva/skyward/actions"><img src="https://img.shields.io/github/actions/workflow/status/gabfssilva/skyward/tests.yml" alt="Tests"></a>
  <a href="https://github.com/gabfssilva/skyward/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabfssilva/skyward.svg" alt="License"></a>
</p>

<p align="center">
  <img src="demo.gif" alt="Skyward Demo" width="800">
</p>

Skyward is a Python control plane for accelerator compute. Describe the machines, image, providers, and runtime in Python; submit ordinary functions as tasks; and let the daemon reconcile the requested state with the machines that exist. The same SDK can use an embedded daemon or a remote one.

---

```python
import skyward as sky


@sky.function
def train(batch: list[float]) -> float:
    return sum(batch) / len(batch)


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(),
    nodes=4,
    plugins=[sky.plugins.Torch()],
) as compute:
    result = train([1.0, 2.0, 3.0]) >> compute
```

`@sky.function` creates a lazy `Pending` value. The `>>` operator submits it to one node and waits for its result. The context manager creates a Compute resource, waits for it to become ready, and releases it on exit by default.

---

## A single API. Any cloud.

Provider accounts describe a login and its non-secret configuration. Changing the provider does not change the function being submitted.

=== "AWS"
    ```python
    with sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.H100()) as compute:
        result = train(data) >> compute
    ```
=== "VastAI"
    ```python
    with sky.Compute(provider=sky.VastAI(), accelerator=sky.accelerators.H100()) as compute:
        result = train(data) >> compute
    ```
=== "RunPod"
    ```python
    with sky.Compute(provider=sky.RunPod(), accelerator=sky.accelerators.H100()) as compute:
        result = train(data) >> compute
    ```
=== "GCP"
    ```python
    with sky.Compute(provider=sky.GCP(), accelerator=sky.accelerators.H100()) as compute:
        result = train(data) >> compute
    ```

The daemon keeps provider accounts and never includes their credentials in a provider response. Offers are fetched through the provider adapter and cached according to that provider's freshness interval. A Compute may contain several `sky.Spec` values; the control plane selects a matching offer according to `selection`.

---

## Fully customizable.

Define the remote environment declaratively. Python version, packages, system dependencies, environment variables, user-code files, volumes, and ports are part of the Compute definition.

=== "Image"
    ```python
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        image=sky.Image(
            python="3.12",
            pip=["torch", "transformers", "my-internal-lib"],
            apt=["ffmpeg", "libsndfile1"],
            pip_indexes=[
                sky.PipIndex(
                    url="https://pypi.internal.co/simple",
                    packages=("my-internal-lib",),
                ),
            ],
        ),
    ) as compute:
        result = train(data) >> compute
    ```
=== "Plugins"
    ```python
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        plugins=[
            sky.plugins.Torch(),
            sky.plugins.Accelerate(config={"mixed_precision": "bf16"}),
        ],
    ) as compute:
        result = train(data) >> compute
    ```
=== "Volumes"
    ```python
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        volumes=[
            sky.Volume(bucket="my-dataset", mount="/data"),
            sky.Volume(bucket="my-checkpoints", mount="/checkpoints", read_only=False),
        ],
    ) as compute:
        result = train(data) >> compute
    ```
=== "Ports"
    ```python
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        ports=[sky.Port(remote=8080, local=8080)],
    ) as compute:
        serve() >> compute
    ```

Plugins are immutable values serialized into the Compute definition. Built-in plugins configure the image, bootstrap, worker lifetime, task execution, or client-side integration. See [Plugins](plugins/index.md).

---

## Operators for real workloads.

The operators choose how a lazy call becomes a task.

| Operator | Result |
|----------|--------|
| `task() >> compute` | Run one task on one node and wait for its result |
| `task() @ compute` | Create one execution per ready node admitted for the broadcast |
| `task() > compute` | Start asynchronously and return a `Future` |
| `(task_a() & task_b()) >> compute` | Submit independent tasks and collect their results |
| `sky.gather(task_a(), task_b(), stream=True) >> compute` | Yield grouped results as executions finish |
| `@sky.stream` plus `>>` | Consume a generator result item by item |

```python
@sky.function
def evaluate(weights: bytes) -> float:
    return score(weights)


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(),
    nodes=4,
    plugins=[sky.plugins.Torch()],
) as compute:
    weights = train(data) >> compute
    scores = evaluate(weights) @ compute
    pending_score = evaluate(weights) > compute
    score = pending_score.result()
```

All calls are persisted as tasks by the daemon. A task keeps its identity across physical execution retries; the SDK's `Future` points to that stable task.

---

## Declarative Compute.

`Compute` combines one or more placement `Spec` values with the desired node bounds, image, plugins, executor, options, ports, volumes, and lifecycle policy.

```python
with sky.Compute(
    sky.Spec(provider=sky.VastAI(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.AWS(), accelerator=sky.accelerators.H100()),
    selection="cheapest",
    nodes=sky.Nodes(desired=2, max=8),
    delete_on_exit=False,
    name="training",
) as compute:
    train(data) >> compute
```

The definition is the desired state. The daemon stores it, gives it a generation, and reports observed state separately. A process can leave a Compute running and another process can attach to it:

```python
with sky.Compute.attached("training") as compute:
    evaluate(weights) >> compute
```

With no `url`, the SDK runs the control plane in the current process against the default SQLite database. With `url` or `SKYWARD_URL`, it uses the same HTTP API against a remote daemon.

---

## Runtime information and distributed state.

Code running on a node can ask for its topology without contacting the client:

```python
@sky.function
def rank_info() -> dict[str, object]:
    info = sky.instance_info()
    return {
        "rank": info.rank,
        "nodes": info.nodes,
        "worker": info.worker,
        "is_head": info.is_head,
    }
```

`sky.shard()` makes a deterministic contiguous slice for the current rank. `sky.dict`, `sky.set`, `sky.counter`, `sky.queue`, `sky.registry`, `sky.barrier`, and `sky.lock` provide named state shared by the nodes of one Compute. Values are serialized; collection names identify the shared state.

---

## Local development.

The `Container` provider exercises the same SSH bootstrap and control-plane path without cloud credentials:

```python
with sky.Compute(provider=sky.Container()) as compute:
    result = train([1.0, 2.0, 3.0]) >> compute
```

The provider requires Docker by default. For function-only tests, call the original implementation through `.local`:

```python
result = train.local([1.0, 2.0, 3.0])
```

---

## Get started.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Install and run](getting-started.md)** — Install the SDK and run the first task
- :material-lightbulb: **[Core concepts](concepts.md)** — Lazy calls, Compute resources, tasks, and leases
- :material-server-network: **[Architecture](architecture.md)** — The daemon, control plane, and node runtime
- :material-cloud: **[Providers](providers.md)** — Provider accounts and accelerator offers
- :material-puzzle: **[Plugins](plugins/index.md)** — PyTorch, JAX, Keras, HuggingFace, and more
- :material-api: **[API reference](reference/pool.md)** — Public Python types and functions

</div>
