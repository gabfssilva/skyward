# Getting started

This page covers installation, credential setup, and your first remote computation. By the end, you'll have run a Python function on a cloud instance and seen the full lifecycle — provision, execute, return, tear down — in action.

Skyward requires Python 3.12 or higher and credentials for at least one cloud provider (AWS, RunPod, VastAI, or Verda). For local development and testing without cloud credentials, the `Container` provider works with Docker or Podman.

## Installation

The recommended way to install Skyward is with `uv`:

```bash
uv add skyward
```

Or with pip:

```bash
pip install skyward
```

Skyward also provides optional extras for framework-specific type hints and dependencies. These are not required — the frameworks themselves only need to be installed on the remote workers via the `Image` — but the extras add local type stubs and tooling support:

```bash
uv add skyward[pytorch]      # PyTorch type hints
uv add skyward[huggingface]  # HuggingFace type hints
uv add skyward[aws]          # AWS (boto3) type hints
uv add skyward[all]          # All extras
```

## Provider credentials

Each provider needs credentials before Skyward can provision instances. You only need to configure the provider you intend to use.

### AWS

Skyward uses standard AWS credential resolution — the same chain that `boto3` and the AWS CLI use. The simplest approach is environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

You can also use `aws configure` to write credentials to `~/.aws/credentials`, or rely on IAM roles if running from an EC2 instance. Any method that `boto3` recognizes will work.

The AWS provider needs `ec2:*` permissions for instance management and `iam:PassRole` for instance profiles. SSM permissions (`ssm:*`) are optional but recommended — they enable Session Manager connectivity as a fallback when direct SSH isn't available.

### RunPod

```bash
export RUNPOD_API_KEY=your_api_key
```

Get your API key from **Settings > API Keys** at [runpod.io](https://www.runpod.io/).

### VastAI

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

Get your API key at [cloud.vast.ai/account](https://cloud.vast.ai/account/).

### Verda

```bash
export VERDA_CLIENT_ID=your_client_id
export VERDA_CLIENT_SECRET=your_client_secret
```

## Your first remote function

Create a file called `hello.py`:

```python
import skyward as sky

@sky.compute
def hello() -> str:
    """This function runs on a remote instance."""
    import socket
    return f"Hello from {socket.gethostname()}!"

with sky.ComputePool(provider=sky.AWS()) as pool:
    result = hello() >> pool
    print(result)
```

Run it:

```bash
uv run python hello.py
```

When you execute this, Skyward provisions an EC2 instance, opens an SSH tunnel to it, installs Python and Skyward on the remote machine via an idempotent bootstrap script, serializes your `hello` function with cloudpickle, sends it over the tunnel, executes it on the remote instance, serializes the result back, and returns it to your local process. When the `with` block exits, the instance is terminated. The entire lifecycle — from bare metal to running Python to cleanup — happens automatically.

The output will look something like this:

```
[ClusterProvisioned] Cluster ready in us-east-1
[InstanceLaunched] Launching instance i-0abc123...
[InstanceRunning] Instance running (52.1.2.3)
[InstanceProvisioned] Instance provisioned, starting bootstrap
[BootstrapPhase] Phase 'apt' started
[BootstrapPhase] Phase 'apt' completed (12s)
[BootstrapPhase] Phase 'pip' started
[BootstrapPhase] Phase 'pip' completed (45s)
[InstanceBootstrapped] Bootstrap complete
[NodeReady] Node 0 ready
[ClusterReady] Cluster ready with 1 node(s)
Hello from ip-172-31-0-1!
[TaskCompleted] Task completed in 2.3s
[ClusterDestroyed] Cluster terminated
```

Each line is an event from the pool's lifecycle. The sequence reflects the stages described in [Core Concepts](concepts.md) — the pool actor asks the provider to launch an instance, the instance actor polls until it's running, opens the SSH tunnel, runs the bootstrap script phase by phase, starts the worker, and reports ready. After the task completes, everything is torn down.

## Your first accelerator job

To run on a GPU, add the `accelerator` and `image` parameters to the pool. The `accelerator` specifies what hardware you need, and the `image` describes what software should be installed on the remote worker:

```python
import skyward as sky

@sky.compute
def gpu_info() -> dict:
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

with sky.ComputePool(
    provider=sky.AWS(),
    accelerator=sky.accelerators.T4(),
    image=sky.Image(pip=["torch"]),
    allocation="spot",
) as pool:
    info = gpu_info() >> pool
    print(f"GPU: {info['device_name']}")
    print(f"CUDA devices: {info['device_count']}")
```

Notice that `torch` is imported *inside* the function, not at the top of the file. This is intentional: `torch` doesn't need to be installed on your local machine — it only needs to exist on the remote worker, where the function actually runs. The `Image(pip=["torch"])` tells Skyward to install it there during bootstrap. This pattern — importing heavy dependencies inside `@sky.compute` functions — keeps your local environment lightweight.

The `allocation="spot"` parameter requests spot instances, which are typically 60-90% cheaper than on-demand. If spot capacity isn't available, the pool will fail rather than fall back. Use `allocation="spot-if-available"` (the default) to automatically fall back to on-demand pricing.

## Parallel execution

A single `>>` sends one computation to one node. When you have multiple independent tasks, `gather()` dispatches them all concurrently:

```python
import skyward as sky

@sky.compute
def square(x: int) -> int:
    return x * x

with sky.ComputePool(provider=sky.AWS()) as pool:
    results = sky.gather(square(1), square(2), square(3)) >> pool
    print(results)  # (1, 4, 9)
```

The `&` operator does the same thing with a fixed set of computations and full type inference:

```python
    a, b, c = (square(4) & square(5) & square(6)) >> pool
    print(a, b, c)  # 16 25 36
```

Both approaches dispatch tasks to the pool's nodes via round-robin scheduling. For a deeper look at parallel execution patterns, see the [Parallel Execution guide](guides/parallel-execution.md).

## Multi-node clusters

To scale beyond a single instance, set `nodes` on the pool. The `@` operator broadcasts a function to every node:

```python
import skyward as sky

@sky.compute
def worker_info() -> dict:
    info = sky.instance_info()
    return {
        "node": info.node,
        "total": info.total_nodes,
        "is_head": info.is_head,
    }

with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    results = worker_info() @ pool
    for r in results:
        print(f"Node {r['node']}/{r['total']} (head={r['is_head']})")
```

Where `>>` sends work to one node, `@` sends it to all of them. Each node runs the same function independently, but `sky.instance_info()` returns different values on each one — node index, total count, head status — so the function can adapt its behavior based on where it's running. This is the foundation for distributed training, data-parallel processing, and any workload that benefits from multiple machines. See [Broadcast](guides/broadcast.md) for more.

## Local testing

During development, you'll want to test your functions without provisioning any cloud infrastructure. Every `@sky.compute` function exposes the original, unwrapped version via `.local`:

```python
result = my_function.local(test_data)  # executes immediately, no cloud
```

This bypasses the lazy computation entirely — no `PendingCompute`, no serialization, no pool required. It's the fastest way to iterate on function logic before sending it to the cloud.

For integration testing with the full Skyward lifecycle (serialization, bootstrap, worker execution) but without cloud costs, use the `Container` provider:

```python
import skyward as sky

with sky.ComputePool(provider=sky.Container(), nodes=2) as pool:
    result = hello() >> pool  # runs in local Docker containers
```

## Verbose logging

Skyward logs lifecycle events through Python's standard `logging` module under the `"skyward"` logger. To see detailed debug output:

```python
import logging

logging.getLogger("skyward").setLevel(logging.DEBUG)
```

This will show SSH connection details, bootstrap script output, serialization sizes, and actor message traces — useful for diagnosing connectivity or performance issues.

## Troubleshooting

If the pool fails with **"No instances available"**, the provider couldn't find capacity for the requested hardware. Try a different region, a different accelerator, or use `allocation="spot-if-available"` to fall back to on-demand pricing.

**"Permission denied"** errors typically mean your IAM or API credentials don't have the required permissions. For AWS, verify that your role or user has `ec2:*` and `iam:PassRole`.

If bootstrap takes too long or **times out**, the most common cause is a large `pip` list — installing PyTorch from scratch on a fresh instance takes time. You can increase the timeout with `provision_timeout=7200` on the pool, or reduce the number of pip dependencies.

**Connection issues on AWS** usually mean the instance doesn't have outbound internet access (needed for package installation) or the security group doesn't allow the SSH connection. Skyward creates a temporary security group during `prepare()`, but VPC configurations can override this. Enabling SSM access provides a fallback connectivity path that doesn't require open inbound ports.

## Next steps

- **[Core Concepts](concepts.md)** — The programming model: lazy computation, operators, ephemeral pools
- **[Providers](providers.md)** — Detailed configuration for AWS, RunPod, VastAI, Verda, and Container
- **[Distributed Training](distributed-training.md)** — Multi-node training with PyTorch, Keras, JAX
- **[Plugins](plugins/index.md)** — Framework plugins for PyTorch, JAX, Keras, joblib, and scikit-learn
