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

Skyward is a Python library for ephemeral accelerator compute. Spin up cloud accelerators (GPUs, TPUs, Trainium, and more), run your ML training code, and tear them down automatically. No infrastructure to manage.

---

```python
import skyward as sky

@sky.function
def train(data):
    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(784, 128), 
        nn.ReLU(), 
        nn.Linear(128, 10)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters())

    for batch in data:
        loss = nn.functional.cross_entropy(model(batch.cuda()), targets.cuda())
        loss.backward()
        optimizer.step()
    
    return model.state_dict()

with sky.Compute(
    provider=sky.AWS(), 
    accelerator=sky.accelerators.H100(), 
    nodes=4,
    plugins=[sky.plugins.torch()]
) as compute:
    result = train(my_data) @ compute  # broadcast to all 4 nodes
```

---

## A single API. Any cloud.

Write your function once. Run it on any provider by changing a single argument.

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

---

## Fully customizable.

Define your remote environment declaratively. Python version, packages, system deps, env vars, file syncing — all in one place.

=== "Packages"
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
                    name="private",
                    url="https://pypi.internal.co/simple",
                    packages=["my-internal-lib"],
                ),
            ],
        ),
    ) as compute:
        result = train(data) >> compute
    ```
=== "Plugins"
    ```python
    from contextlib import contextmanager
    from skyward.api.plugin import Plugin

    @contextmanager
    def wandb_tracking(info):
        import wandb
        wandb.init(project="my-project", group=f"node-{info.node_index}")
        yield
        wandb.finish()

    wandb_plugin = (
        Plugin.create("wandb")
        .with_image_transform(lambda img, _: replace(img, pip=(*img.pip, "wandb")))
        .with_around_app(wandb_tracking)
    )

    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        plugins=[sky.plugins.torch(), wandb_plugin],
    ) as compute:
        result = train(data) >> compute
    ```
=== "Metrics"
    ```python
    with sky.Compute(
        provider=sky.AWS(),
        accelerator=sky.accelerators.H100(),
        image=sky.Image(
            metrics=[
                sky.metrics.CPU(interval=1),
                sky.metrics.GPU(interval=2),
                sky.metrics.GPUMemory(interval=2),
                sky.metrics.GPUTemp(interval=5),
                sky.metrics.Disk("/data", interval=10),
            ],
        ),
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

---

## Simple operators. Real workloads.

No job configs. No submission scripts. Python operators dispatch work.

| Operator | What it does |
|----------|-------------|
| `train() >> compute` | Run on a single node |
| `train() @ compute` | Broadcast to **all** nodes |
| `task_a() & task_b() >> compute` | Run in parallel, collect results |
| `train() > compute` | Fire and forget — returns a `Future[T]` |

```python
with sky.Compute(
    provider=sky.AWS(), 
    accelerator=sky.accelerators.H100(), 
    nodes=4,
    plugins=[sky.plugins.torch()]
) as compute:
    # preprocess on one node, train on all, evaluate async
    data = preprocess(raw) >> compute
    weights = train(data) @ compute
    future = evaluate(weights) > compute

    # parallelize independent work
    metrics, report = (compute_metrics() & generate_report()) >> compute
```

---

## Spot instances without the headache.

Save 60–90% on compute. Skyward handles spot allocation, preemption detection, and automatic node replacement. You just pick a strategy.

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(),
    nodes=4,
    allocation="spot",  # or "on-demand", "spot-if-available"
) as compute:
    result = train(data) @ compute
    # node preempted? already replaced. your code doesn't change.
```

---

## The cheapest GPU across clouds.

Define multiple specs across providers. Skyward picks the cheapest available option.

```python
with sky.Compute(
    sky.Spec(provider=sky.VastAI(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.AWS(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.RunPod(), accelerator=sky.accelerators.H100()),
    selection="cheapest",
) as compute:
    result = train(data) @ compute
```

---

## Batteries included.

Plugins configure distributed runtimes, install dependencies, and handle framework-specific setup. You just pass them in.

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(),
    nodes=4,
    plugins=[
        sky.plugins.torch(), 
        sky.plugins.accelerate({"fsdp": {"sharding_strategy": "FULL_SHARD"}})
    ],
) as compute:
    result = finetune(model, dataset) @ compute
```

<div class="grid cards" markdown>

- **PyTorch** — DDP, FSDP, NCCL backend
- **Accelerate** — HuggingFace Trainer, DeepSpeed, FSDP
- **JAX** — Multi-host distributed initialization
- **Keras 3** — Backend-agnostic data parallelism
- **Joblib** — Drop-in parallel backend for scikit-learn
- **cuML** — GPU-accelerated scikit-learn estimators

</div>

---

## Get started.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Install & run](getting-started.md)** — Up and running in 5 minutes
- :material-lightbulb: **[Core concepts](concepts.md)** — Functions, operators, and pools
- :material-cloud: **[Providers](providers.md)** — AWS, GCP, RunPod, VastAI, and more
- :material-puzzle: **[Plugins](plugins/index.md)** — PyTorch, JAX, Keras, HuggingFace
- :material-server-network: **[Distributed training](distributed-training.md)** — Scale to many nodes
- :material-api: **[API reference](reference/pool.md)** — Full autodoc of all public types

</div>
