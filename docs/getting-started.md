# Getting started

This page covers installation, a local first run, provider credentials, and a first cloud task. Skyward's Python API is synchronous; the control plane underneath persists Computes, tasks, nodes, and events.

Skyward requires Python 3.12 or higher. A client that creates an embedded Compute needs the `client` and `server` extras. The `Container` provider also needs Docker or Podman-compatible container tooling.

## Installation

Install the SDK and the embedded control plane with `uv`:

```bash
uv add "skyward[client,server]"
```

Or with pip:

```bash
pip install "skyward[client,server]"
```

The optional extras are split by role:

```bash
uv add "skyward[client]"    # SDK for a remote daemon
uv add "skyward[server]"    # embedded or standalone daemon
uv add "skyward[cli]"       # sky command
uv add "skyward[notebook]"  # Jupyter kernel provisioning
uv add "skyward[tui]"       # terminal rendering
uv add "skyward[storage]"   # user-facing object storage helpers
uv add "skyward[all]"       # every extra below plus every provider
```

Three providers need an SDK the daemon does not install by itself, and each has an extra of its own. The other twelve speak plain HTTP and are always available:

```bash
uv add "skyward[aws]"        # aioboto3
uv add "skyward[gcp]"        # cryptography, for the service-account signature
uv add "skyward[salad]"      # salad-cloud-sdk and websockets
uv add "skyward[providers]"  # all three at once
```

A provider whose SDK is missing is not registered: `sky providers list` omits it, and asking for it names the extra that brings it back.

Frameworks belong in the remote image or in a plugin. They do not need to be installed locally just to define a function.

## Your first local Compute

The `Container` provider uses local containers as machines. It has no credentials and exercises the same SSH, bootstrap, worker, and task paths as a cloud provider.

Create `hello.py`:

```python
import socket

import skyward as sky


@sky.function
def hello() -> str:
    """This function runs on a node, not in the calling process."""
    return f"Hello from {socket.gethostname()}!"


with sky.Compute(provider=sky.Container()) as compute:
    print(hello() >> compute)
```

Run it:

```bash
uv run python hello.py
```

The `with` block creates a Compute definition, waits until its node is ready, submits one task, and marks the Compute for deletion when the block exits. The task result is returned to the calling process.

## Provider credentials

Provider factories resolve credentials in the calling process. Explicit arguments take precedence over environment variables. The selected control plane stores the provider account so that the daemon can fetch offers and provision machines; provider read responses do not include credentials.

Configure only the providers you use. The complete factory list is documented in [Providers](providers.md).

### AWS

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

The factory also reads the standard AWS shared credentials file and instance credentials. The account needs permissions to manage the instances and their networking. SSM permissions are optional when direct SSH is available.

### GCP

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project
```

The JSON file contents are passed to the provider account. The path itself is not stored as a credential.

### RunPod

```bash
export RUNPOD_API_KEY=your_api_key
```

### VastAI

```bash
export VAST_API_KEY=your_api_key
```

### Verda

```bash
export VERDA_CLIENT_ID=your_client_id
export VERDA_CLIENT_SECRET=your_client_secret
```

## Your first cloud task

Replace `sky.Container()` with a cloud provider and put heavy imports inside the decorated function:

```python
import skyward as sky


@sky.function
def gpu_info() -> dict[str, object]:
    import torch

    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4(),
    image=sky.Image(pip=["torch"]),
    allocation="spot_if_available",
) as compute:
    print(gpu_info() >> compute)
```

`Image(pip=["torch"])` installs PyTorch on the node during bootstrap. The local process only needs the Skyward client. `allocation="spot_if_available"` tries the preferred lower-cost market and can use on-demand capacity when the preferred market is unavailable. The other values are `"spot"`, `"on_demand"`, and `"cheapest"`.

## Parallel execution

`>>` submits one task. Use `&` or `sky.gather()` for independent calls:

```python
import skyward as sky


@sky.function
def square(value: int) -> int:
    return value * value


with sky.Compute(provider=sky.Container()) as compute:
    results = sky.gather(square(1), square(2), square(3)) >> compute
    print(results)  # [1, 4, 9]

    a, b, c = (square(4) & square(5) & square(6)) >> compute
    print(a, b, c)  # 16 25 36
```

The calls are submitted together. Results from the ordinary group are returned in submission order. `sky.gather(..., stream=True, ordered=False)` returns an iterator in completion order.

For a collection of inputs, `Compute.map()` submits one pending call per item and returns results in input order:

```python
with sky.Compute(provider=sky.Container()) as compute:
    results = compute.map(lambda value: square(value), range(10))
```

## Streaming results

Generator functions use `@sky.stream`, which produces a `Streaming` value. The result is an iterator and is pulled from the node as the caller consumes it.

```python
from collections.abc import Iterator

import skyward as sky


@sky.stream
def tokens(prompt: str) -> Iterator[str]:
    for token in prompt.split():
        yield token


with sky.Compute(provider=sky.Container()) as compute:
    for token in tokens("streamed output") >> compute:
        print(token)
```

A streaming task is not resumable. Closing the iterator closes the remote generator.

## Multi-node Compute

Set `nodes` and broadcast a call with `@`. The broadcast freezes the ready node set when the task is admitted and returns results in rank order.

```python
import skyward as sky


@sky.function
def rank_info() -> dict[str, object]:
    info = sky.instance_info()
    return {
        "node": info.node,
        "rank": info.rank,
        "nodes": info.nodes,
        "is_head": info.is_head,
    }


with sky.Compute(provider=sky.AWS(), nodes=4) as compute:
    for result in rank_info() @ compute:
        print(result)
```

`Info` also exposes `peers`, `worker`, `workers_per_node`, `total_workers`, `global_worker_index`, `host`, `head_addr`, `head_port`, and `job_id`. See [Distributed training](distributed-training.md).

For a fixed Compute that can start with partial readiness, use `sky.Nodes`:

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=sky.Nodes(desired=8, min=4),
) as compute:
    train() @ compute
```

The lower bound controls readiness. The reconciler can add capacity up to the upper bound as queued work requires it. A tuple is shorthand for an elastic range: `nodes=(2, 16)` means a lower bound of 2 and an upper bound of 16. See [Reconciliation and provisioning](provision-controllers.md).

## Persistent Computes and remote daemons

By default, the SDK runs an embedded daemon against `~/.skyward/skyward.sqlite`. Set `url` or `SKYWARD_URL` to use a daemon running elsewhere:

```python
with sky.Compute(
    provider=sky.AWS(),
    nodes=4,
    url="http://127.0.0.1:7590",
    name="training",
    delete_on_exit=False,
) as compute:
    train() >> compute
```

The daemon keeps the Compute after the process exits. A later process can attach by name or id without restating its definition:

```python
with sky.Compute.attached("training", url="http://127.0.0.1:7590") as compute:
    evaluate() >> compute
```

The SDK claims and renews a lease while it owns the Compute. `Compute.attached()` does not delete on exit by default.

## Local testing

Every decorated function keeps its original implementation on `.local`:

```python
value = gpu_info.local()
```

This calls the function immediately and bypasses the daemon, serialization, and remote image. Use the `Container` provider when the bootstrap and task path also need coverage.

## Troubleshooting

If the SDK reports that the client or daemon dependencies are missing, install the role-specific extras:

```bash
uv add "skyward[client,server]"
```

If a Compute does not become ready, inspect the provider credentials, offer availability, SSH reachability, and the image bootstrap dependencies. Increase the client-side wait with `sky.Options(ready_timeout=...)`:

```python
with sky.Compute(
    provider=sky.AWS(),
    options=sky.Options(ready_timeout=1800),
) as compute:
    train() >> compute
```

## Next steps

- [Core concepts](concepts.md) — Lazy calls, Compute definitions, tasks, executions, and leases
- [Architecture](architecture.md) — Embedded and remote control planes
- [Providers](providers.md) — Accounts, offers, and provider capabilities
- [CLI](cli.md) — Manage the daemon and Computes from the terminal
- [Distributed collections](distributed-collections.md) — Shared state across nodes
