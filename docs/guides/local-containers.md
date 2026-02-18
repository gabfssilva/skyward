# Local Containers

When developing with Skyward, the normal feedback loop involves provisioning cloud instances, waiting for bootstrap, running your function, and tearing everything down. This works, but it's slow and costs money — not ideal when you're iterating on a function's logic, debugging serialization issues, or validating that your `Image` installs the right dependencies.

The `Container` provider solves this by running the exact same orchestration locally. Instead of launching EC2 instances or GPU pods, it starts Docker containers on your machine. The pool lifecycle is identical: SSH tunnels, bootstrap scripts, worker processes, cluster formation. The only difference is that "the cloud" is your laptop.

This means you can develop and test your `@sky.compute` functions locally, then switch to `sky.AWS()` or `sky.RunPod()` when you're ready for real hardware — with confidence that the behavior will be the same.

## Prerequisites

You need a container runtime installed and running:

- **Docker** (default): `docker` CLI available in PATH
- **Podman**: pass `binary="podman"` to `sky.Container()`
- **nerdctl**: pass `binary="nerdctl"`

## Basic Usage

The `Container` provider is a drop-in replacement for any cloud provider. The only change is the `provider` parameter:

```python
--8<-- "examples/guides/11_local_containers.py:1:13"
```

```python
--8<-- "examples/guides/11_local_containers.py:42:45"
```

Behind the scenes, Skyward builds a lightweight Docker image with SSH access, starts a container, opens an SSH tunnel, bootstraps the environment, and runs the worker — the same pipeline that runs on a real cloud instance. The function is serialized with cloudpickle, sent over the tunnel, executed inside the container, and the result comes back.

## Testing Your Image

One of the most common sources of failures in cloud runs is a misconfigured `Image`: a missing pip package, a wrong environment variable, a Python version mismatch. The Container provider lets you validate all of this locally before spending time and money on cloud provisioning.

```python
--8<-- "examples/guides/11_local_containers.py:15:21"
```

```python
--8<-- "examples/guides/11_local_containers.py:47:53"
```

If the function returns `"it works"`, you know the `env` field in your Image is being injected correctly. The same applies to `pip` (install packages and import them inside the function), `apt` (install system tools and shell out to them), and `includes` (sync local modules and import them).

## Multi-Node Locally

The Container provider supports `nodes > 1`. Each node becomes a separate container, and they form a real Casty cluster — with a head node, peer discovery, and inter-node networking via a Docker bridge network. This lets you test broadcast, data sharding, and distributed collections without any cloud infrastructure.

```python
--8<-- "examples/guides/11_local_containers.py:23:40"
```

```python
--8<-- "examples/guides/11_local_containers.py:55:63"
```

Each container gets its own `instance_info()` with the correct `node`, `total_nodes`, and `is_head` values. `sky.shard()` works as expected — each node processes its portion of the data. This is the same behavior you'd see on a 3-node AWS cluster, just running on `localhost`.

## Configuration

The `Container` dataclass accepts a few parameters beyond the default:

```python
sky.Container(
    image="ubuntu:24.04",   # base Docker image (default)
    binary="docker",        # container runtime CLI
    network="my-network",   # shared Docker network name (optional)
)
```

- `image` — The base Docker image. Skyward builds an SSH-enabled layer on top of it. Use a different base if your functions need system libraries that take long to install via `apt`.
- `binary` — The container runtime. Useful for environments where Docker isn't available (Podman in rootless mode, nerdctl with containerd).
- `network` — By default, each pool creates its own isolated Docker network and tears it down on exit. If you set a shared network name, multiple concurrent pools reuse the same network — useful when running parallel test suites.

Resource limits (`vcpus` and `memory_gb`) are set on the pool, not the provider:

```python
sky.ComputePool(
    provider=sky.Container(),
    nodes=2,
    vcpus=1,
    memory_gb=1,
)
```

These map directly to Docker's `--cpus` and `--memory` flags.

## CI/CD

The Container provider works anywhere Docker runs — including CI environments. Since GitHub Actions runners have Docker pre-installed, you can run integration tests against real Skyward pools without cloud credentials.

A typical pytest setup uses a session-scoped fixture so that the pool is provisioned once and reused across tests:

```python
import pytest
import skyward as sky

@pytest.fixture(scope="session")
def pool():
    with sky.ComputePool(
        provider=sky.Container(network="skyward-ci"),
        nodes=2,
        vcpus=1,
        memory_gb=1,
    ) as p:
        yield p
```

Then test your compute functions against the fixture:

```python
@sky.compute
def double(x: int) -> int:
    return x * 2

def test_single_execution(pool):
    assert double(5) >> pool == 10

def test_broadcast(pool):
    results = double(5) @ pool
    assert all(r == 10 for r in results)
```

This runs real containers, real SSH tunnels, real serialization — the full Skyward stack — in your CI pipeline.

## What You Can and Can't Test Locally

The Container provider replicates the full orchestration pipeline: provisioning, SSH, bootstrap, worker startup, cluster formation, task dispatch, and teardown. Most of what matters for correctness — serialization, Image configuration, data sharding, distributed collections, broadcast — works identically.

What it **doesn't** replicate:

- **GPU execution** — containers don't have accelerators (unless your machine has GPUs and you use `--gpus` via a custom setup). Code that calls `torch.cuda` will see no devices.
- **Spot preemption** — there's no concept of spot interruption in local containers. Preemption handling can't be tested this way.
- **Cloud-specific networking** — VPCs, security groups, and cross-AZ latency don't exist locally. Multi-node communication is over a Docker bridge, which is effectively zero-latency.
- **Real provisioning time** — containers start in seconds. Cloud instances take minutes. Timeout tuning needs real cloud testing.

The general pattern: test logic and integration locally with `Container`, test performance and infrastructure with a real cloud provider.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/11_local_containers.py
```

---

**What you learned:**

- **`sky.Container()`** runs the full Skyward stack locally using Docker containers.
- **Same orchestration** — SSH, bootstrap, workers, cluster formation — just on localhost.
- **Image validation** — test pip, apt, env, and includes before deploying to cloud.
- **Multi-node locally** — broadcast, shard, and distributed collections work with `nodes > 1`.
- **CI-friendly** — session-scoped pytest fixtures give you real integration tests without cloud costs.
