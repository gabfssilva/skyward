# Core concepts

Skyward's programming model is built on a simple idea: **computation and location should be separate concerns**. You write ordinary Python functions. You decide where they run — on which cloud, on how many machines, on which accelerators — at the call site, not inside the function itself. This separation is what makes the same code work on a laptop, a single GPU, or a cluster of eight H100s.

This page walks through the concepts that make this possible. For how the daemon behind it actually works, see [Architecture](architecture.md).

## Lazy computation

The central abstraction is `@sky.function`. It transforms a regular function into a **lazy computation** — calling the function no longer executes it. Instead, it produces a `Pending` object: a frozen, serializable description of *what* to compute, without committing to *where* or *when*.

```python
import skyward as sky


@sky.function
def train(epochs: int) -> float:
    model = build_model()
    return model.fit(epochs=epochs)


pending = train(10)  # nothing runs — returns Pending[float]
```

In functional programming, this pattern is known as reifying an **effect**. Calling `train(10)` doesn't produce a result — it produces a *description* of a computation that, when interpreted, will produce a result. The side effects (provisioning a GPU, transferring data, executing remotely) are not performed at the call site. They are deferred to an interpreter — here, the compute and its operators. This is the same idea behind `IO` in Haskell, `Effect` in ZIO, or `suspend` in Kotlin: separating the description of a program from its execution so the runtime can decide how, where, and when to run it.

Because `Pending` is a value and not a running process, it can be serialized, sent over the network, and executed on a remote machine. It can also be composed: combined with other computations via `&`, collected into a `gather()`, or stored and dispatched later. None of this triggers execution. The program you're building is a data structure that only materializes when you commit to a target with `>>` or `@`.

The generic type is preserved throughout — `train(10)` produces `Pending[float]`, and dispatching it returns `float`. Your type checker sees the same types whether the function runs locally or on a remote GPU.

Nothing is pickled at call time either. Serialization happens at dispatch, so building a thousand `Pending` values costs a thousand small dataclasses, not a thousand cloudpickle payloads.

To bypass all of this and run the original function directly — for testing, debugging, or local profiling — every decorated function exposes the unwrapped version via `.local`:

```python
result = train.local(10)  # executes immediately, returns float
```

`@sky.function(timeout=600)` sets a deadline for the call, and `.with_timeout(600)` overrides it per dispatch.

## Compute

A `Compute` is a context manager representing a set of cloud machines with a defined lifecycle. When you enter the block, Skyward provisions the machines, installs dependencies, and establishes connectivity. When you exit — normally or through an exception — everything is torn down.

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=4,
    image=sky.Image(pip=["torch", "transformers"]),
) as compute:
    result = train(10) >> compute
# all machines terminated
```

This is what it means for compute to be **ephemeral**: the infrastructure exists only for the duration of the work. There are no machines to forget about, no environments that drift over time, no idle costs accumulating overnight. The compute's lifetime is the job's lifetime.

This model fits ML workloads naturally. Training runs, fine-tuning jobs, hyperparameter sweeps, batch inference — these are all tasks with a beginning and an end. `Compute` captures that shape: provision what you need, do the work, release everything.

### The control plane is not in the object

`Compute` looks like it owns the machines. It doesn't. The machines are owned by a **daemon** — a separate control plane that `Compute` talks to over HTTP. With no `url`, that daemon runs inside your process against a local SQLite database, reached through ASGI with no socket involved. With `url=` or `SKYWARD_URL`, it's a daemon you started with `sky server start`. Nothing above the client can tell which it got.

The consequence is that machines outlive the process that asked for them:

```python
with sky.Compute(provider=sky.AWS(), nodes=8, name="training", delete_on_exit=False) as compute:
    train(data) >> compute

with sky.Compute.attached("training") as compute:  # tomorrow, another process
    evaluate() >> compute
```

`Compute.attached()` takes a name or id and no spec — the definition already lives in the daemon. Ownership is a **lease** the SDK renews in the background; when renewals stop, the daemon may reclaim the compute.

Internally, one asyncio event loop on a background daemon thread serves every call. That is what makes `task() > compute` a real future rather than a wrapper around a blocking call.

### What the daemon remembers

Your compute is a row. So is every node under it, every task you submit, and every attempt at that task. They live in one SQLite file at `~/.skyward/skyward.sqlite`, and that is not an implementation detail you can ignore — it is what the whole model rests on.

Because the state is on disk rather than in your process, the daemon can be restarted, killed, or replaced while your machines keep running. It reconnects by reading back what it needs: the provider binding it created, the SSH key it provisioned with, the address of every node. Anything genuinely live — the SSH connections, the forwarded ports — is rebuilt from those rows rather than recovered.

That is also why the definition and the observation are separate. What you asked for is `spec`, and only you write it. What actually exists is `status`, and only the daemon writes it. There is no operation to poll: the distance between the two is the work still pending.

Two consequences reach your code. `Compute.attached()` works because the definition already exists without you. And a task keeps its identity across retries — a retry is a new *execution* of the same task, never a new task — so the `Future` you're holding stays valid even when the machine under it is replaced.

You never have to touch any of this to use Skyward. When you do want to look, the store is a plain SQLite file and the daemon's only interface is HTTP:

```console
$ sqlite3 ~/.skyward/skyward.sqlite 'select id, name, status_state from computes'
$ http GET :17590/v1/computes
```

See [Persistence](persistence.md) for the tables and [HTTP API](http-api.md) for the routes.

### Node specification

`nodes` accepts three shapes, and they mean different things:

```python
sky.Compute(provider=sky.AWS(), nodes=4)                              # exactly 4
sky.Compute(provider=sky.AWS(), nodes=(2, 8))                         # elastic, 2 to 8
sky.Compute(provider=sky.AWS(), nodes=sky.Nodes(initial=8, min=4))    # open at 8, live on 4
sky.Compute(provider=sky.AWS(), nodes=sky.Nodes(initial=4, min=2, max=16))
```

An integer fixes both bounds. A tuple is an elastic range: the daemon opens at the floor and scales inside the range based on queued and running task load. `sky.Nodes` separates the two ideas that a tuple conflates — `initial` is the size the pool opens at, `min` is how few nodes you are willing to live on, and `max` is the ceiling.

`initial` is a request made once, not a target the daemon holds you to. With `min` below it, work begins as soon as `min` nodes are ready and the rest join as they come up, which matters when one slow provider would otherwise hold up seven fast ones — and a machine the opening request never got is not bought again. Eight asked for, five up, `min=4`: the pool is ready, it stays at five, and it buys another machine only if it drops below four. If you want eight to be self-healing, `min=8` says so.

A collective plugin freezes the world of a distributed job, so a compute using one cannot be resized while that definition is active.

## Operators

Most distributed computing frameworks express execution through configuration files, job submission APIs, or method calls like `pool.submit(fn, args)`. Skyward uses Python's operator overloading to create a small vocabulary where the syntax communicates intent. The expression `train(10) >> compute` reads as "send this computation to the compute" — and that's exactly what it does.

This works because `@sky.function` returns a `Pending` value rather than a result. Operators are defined on `Pending`, `Group`, and `Streaming` (via `__rshift__`, `__matmul__`, `__and__`, `__gt__`), and each triggers a different dispatch strategy. Dispatch serializes the function and arguments with cloudpickle, compresses with lz4, submits the task to the daemon, waits, and deserializes the result.

| Expression | Semantics | Return type |
|---|---|---|
| `task() >> compute` | Execute on one node (round-robin) | `T` |
| `task() @ compute` | Broadcast to all nodes | `list[T]` |
| `task1() & task2() >> compute` | Parallel execution | `tuple[T1, T2]` |
| `task() > compute` | Async, non-blocking | `Future[T]` |
| `streaming() >> compute` | Items as the node produces them | `Iterator[T]` |

### `>>` — execute on one node

The most common operation. Sends the computation to a single node and blocks until the result is available:

```python
result = train(10) >> compute
```

The target node is chosen round-robin. Ten tasks sent with `>>` distribute evenly across available nodes. This is the right operator for independent tasks that don't need to run everywhere — hyperparameter trials, inference on different inputs, any embarrassingly parallel workload.

### `@` — broadcast to all nodes

Sends the same computation to every node, and returns one result per node:

```python
with sky.Compute(provider=sky.AWS(), nodes=4) as compute:
    results = initialize_model(config) @ compute  # list of 4 results
```

Broadcast is the foundation for distributed training. Combined with `sky.shard()` inside the function, each node receives the full arguments but operates on its own partition. The function body is identical across nodes — the differentiation happens at runtime from `sky.instance_info()`.

Broadcast pins each execution to the rank it was admitted with, which is what lets rank-dependent code inside the function be meaningful.

### `&` — parallel composition

Combines multiple *different* computations into a group that executes in parallel. Results come back as a tuple with full type inference:

```python
a, b, c = (preprocess() & train() & evaluate()) >> compute
```

The distinction from broadcast matters: `@` runs the *same* function on all nodes, `&` runs *different* functions concurrently. Each computation may go to a different node, and the group blocks until all complete. Types are preserved individually — if `preprocess` returns `DataFrame`, `train` returns `Model`, and `evaluate` returns `float`, the destructured result is `tuple[DataFrame, Model, float]`.

### `>` — async dispatch

Like `>>`, but returns a `Future[T]` immediately instead of blocking, so remote computation overlaps with local work:

```python
future = train(10) > compute
# ... do local work while the remote computation runs ...
result = future.result()  # blocks only when you need the result
```

The `Future` follows the `concurrent.futures` protocol, so it works with `as_completed()`, `wait()`, and existing executor code.

### `gather()` — dynamic parallelism

`&` works when the number of parallel tasks is known at write time. When it isn't — you're iterating over a dataset, a list of configurations, any dynamic collection — `gather` groups an arbitrary number of computations:

```python
tasks = [process(chunk) for chunk in chunks]
results = sky.gather(*tasks) >> compute
```

`gather` produces the same `Group` type that `&` does, so dispatch behaviour is identical. The difference is syntactic: `&` for a fixed set of typed computations, `gather` for a dynamic collection.

With `stream=True`, results are yielded as they complete rather than after all of them finish — useful when tasks have variable duration and you want to start processing early:

```python
for result in sky.gather(*tasks, stream=True) >> compute:
    save(result)
```

Streaming preserves the original order by default. With `ordered=False`, results arrive in completion order — faster overall, but you lose the positional correspondence.

`compute.map(fn, items)` is the shorthand for the common case: one function over a collection, results in order.

## Image

Remote nodes start as bare cloud instances — a fresh OS with no Python, no libraries, no knowledge of your project. `Image` describes the environment that should exist on each node before any computation runs. It's declarative: you state what you need, and the daemon generates an idempotent bootstrap script that provisions it.

```python
image = sky.Image(
    python="3.13",
    pip=["torch", "numpy", "transformers"],
    apt=["ffmpeg", "libsndfile1"],
    env={"KERAS_BACKEND": "jax"},
    includes=["./my_module/"],
)

with sky.Compute(provider=sky.AWS(), image=image) as compute:
    ...
```

Each field maps to a phase of the bootstrap:

- `base` — the container image to start from. `sky.DockerImage` is a `str` subclass whose value *is* the tag, so `base=sky.DockerImage.pytorch("2.8")` flows straight through.
- `python` — the Python version to install. Defaults to matching your local version. Nodes use `uv` as the package manager, so installation is fast.
- `pip` — Python packages installed into the node's virtual environment.
- `pip_indexes` — extra or replacement package indexes, as `sky.PipIndex` values.
- `apt` — system packages installed before Python setup.
- `env` — environment variables set before your function executes.
- `includes` — local directories synced to the nodes. This is how your own code reaches remote machines without being published as a package. It's packed client-side into a blob, and the spec carries only its hash, so the same code is uploaded once no matter how many nodes read it.
- `excludes` — glob patterns skipped during that sync (`["__pycache__", "*.pyc"]`).
- `metrics` — which `sky.metrics.*` samplers the node runs in the background.

Because `Image` is frozen, two computes built from the same image produce the same environment — same Python version, same packages, same system dependencies. This is reproducibility without writing a Dockerfile: the environment specification lives in your Python code, versioned alongside your experiments.

`skyward` controls how Skyward itself reaches the nodes. The default detects whether you're running from an editable install and, if so, ships your local source instead of installing from PyPI — so changes to Skyward's own code appear on remote machines immediately during development.

## Runtime context

Skyward operates in two worlds. The **client side** is your machine — where `Compute` lives, where operators dispatch, where results come back. The **node side** is the remote machine, where your `@sky.function` body actually executes. Separate processes, separate machines.

Inside the function you're in the node world. You have the machine's resources — GPUs, local disk, network — but no reference to the compute object or your client's memory. What you do have is `sky.instance_info()`: this node's view of the topology.

```python
@sky.function
def distributed_task(data):
    info = sky.instance_info()
    print(f"rank {info.rank} of {info.nodes}")

    if info.is_head:
        coordinate_others()
    return process(data)
```

`Info` is a frozen dataclass read from environment variables the daemon sets before starting the worker. It carries `node`, `compute`, `rank`, `peers`, `worker`, and `workers_per_node`, plus derived values: `nodes`, `total_workers`, `global_worker_index`, `host`, `head`, `head_addr`, `head_port`, `job_id`, and `is_head`. Off a node it raises `NotOnANodeError`.

A compute has no head node — every node is symmetric, and the client reaches each one directly. `head` and `is_head` exist by convention for the training libraries that need a rendezvous address, and they mean rank zero, nothing more.

This is the same mechanism the plugins use. The `Torch` plugin reads `instance_info()` to set `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` before calling `init_process_group`. You don't need a plugin to reach it — `instance_info()` is available inside any `@sky.function` body.

The module behind these runtime names imports the standard library and nothing else, which is what keeps httpx off a machine whose only job is a training loop.

### Data sharding

A common pattern is to send the *same function* to all nodes and have each operate on a *different slice*. `sky.shard()` automates it: it reads this node's rank and returns only the portion belonging to it.

```python
@sky.function
def process(full_dataset):
    local_data = sky.shard(full_dataset)
    return analyze(local_data)


with sky.Compute(provider=sky.AWS(), nodes=4) as compute:
    results = process(dataset) @ compute  # each node gets 1/4
```

The function receives the *full* dataset as an argument — serialization is paid once — but each node only processes its shard.

The split is contiguous: rank `r` of `n` gets `data[len*r//n : len*(r+1)//n]`. With eight nodes and three items, three nodes get work and five get an empty sequence; that's a valid split, not an error.

Sharding is type-preserving. Lists produce lists, tuples produce tuples, NumPy arrays produce arrays, PyTorch tensors produce tensors — so you can shard a tensor and hand it straight to a model.

When sharding several sequences, the same positions are taken from each, so paired data stays aligned:

```python
@sky.function
def train(x_full, y_full):
    x, y = sky.shard(x_full, y_full, shuffle=True)
    # x[i] still corresponds to y[i]
    return fit(x, y)
```

`shuffle=True` permutes before splitting. The seed defaults to the compute id, so every node of one compute shuffles identically and two computes don't — pass `seed=` to pin it. `drop_last=True` discards the remainder so every shard has the same length, which is what a training step with fixed batch dimensions usually wants.

`node=` and `total_nodes=` override the topology, for testing or for sharding against a compute you aren't part of.

### Output policy

Four nodes printing the same progress bar is four times the noise. `sky.stdout(only=...)`, `sky.stderr(only=...)`, and `sky.silent` narrow which nodes' output makes the trip back. `only` takes a rank, a tuple of ranks, `"head"`, or a predicate on `Info`:

```python
@sky.function
@sky.stdout(only="head")
def train(data):
    for epoch in range(100):
        print(f"epoch {epoch}")  # only rank zero's output comes back
```

The filtering happens on the node, where the output is written, so what is silenced is never shipped rather than shipped and dropped.

## Streaming

The operators above all share a shape: serialize, ship, execute, serialize the result, ship it back. The full result materializes on the node before anything crosses the network. For most workloads that's fine. But some computations produce results incrementally — a training loop yielding metrics every epoch, a pipeline emitting rows, a search finding matches progressively. Waiting for the whole result wastes time and memory.

`@sky.stream` is the generator counterpart. It produces a `Streaming[T]`, and dispatching it returns a synchronous iterator whose elements arrive as the node yields them:

```python
@sky.stream
def generate_samples(n: int):
    for i in range(n):
        yield expensive_sample(i)


with sky.Compute(provider=sky.AWS()) as compute:
    for sample in generate_samples(1000) >> compute:
        save(sample)
```

The request that consumes the stream is also what starts the execution, so a node never produces items with no one reading them. A stream is not replayable and not resumable: it is a live connection to a running generator, not a stored result.

## Choosing hardware

### Providers

A provider account is a frozen value: which cloud, and what it takes to log in.

```python
aws = sky.AWS(name="production", region="us-east-1")
runpod = sky.RunPod(name="experiments", bid_multiplier=1.3)
```

`name` is the account alias, defaulting to the provider kind, so two accounts of the same kind coexist by having two names. Credentials are resolved in your process — from what you passed, then from the environment or credential file — and the daemon stores them on a provider row. The daemon never reads your environment.

See [Providers](providers.md) for every account's fields, and [Choosing the best provider](choosing-a-provider.md) for which to reach for.

### Accelerators

Every provider spells its hardware differently. One canonical catalog normalizes them, so `sky.accelerators.H100()` and a provider's `NVIDIA H100 80GB SXM5` resolve to the same accelerator and VRAM:

```python
sky.accelerators.A100()
sky.accelerators.H100(count=4)   # 4 per node
sky.accelerators.RTX_4090()
```

`count` is accelerators per node and is independent of `nodes`, the machine count. See [Accelerators](accelerators.md).

### Allocation strategies

`allocation` decides how offers are bought:

```python
sky.Compute(provider=sky.AWS(), allocation="spot_if_available")  # default
sky.Compute(provider=sky.AWS(), allocation="spot")               # cheaper, interruptible
sky.Compute(provider=sky.AWS(), allocation="on_demand")          # guaranteed, expensive
sky.Compute(provider=sky.AWS(), allocation="cheapest")           # whichever costs least
```

Spot instances are 50-90% cheaper. The trade-off is that the provider can reclaim them. Skyward detects the interruption and provisions a replacement, but your workflow has to tolerate a restart — which in practice means checkpointing.

### Multi-spec selection

When one provider isn't enough, pass one `Spec` per alternative:

```python
with sky.Compute(
    sky.Spec(provider=sky.Verda(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.AWS(), accelerator=sky.accelerators.H100(), max_hourly_cost=5.0),
    nodes=4,
    selection="cheapest",
) as compute:
    train(data) >> compute
```

`Spec` carries `provider`, `accelerator`, `cpus`, `memory_gb`, `region`, `disk_gb`, `architecture`, and `max_hourly_cost` — everything that describes *one placement choice*. `nodes`, `allocation`, `image`, `plugins`, `executor`, `options`, `ports`, and `volumes` belong to `Compute`, because they apply after a placement has been picked.

`selection="cheapest"` (the default) ranks every matching offer across all specs into one list by price. `selection="first"` respects the order you wrote. Either way, if provisioning fails on the chosen offer the daemon moves to the next one, so an out-of-stock provider becomes a fallback rather than a failure.

Both `allocation` and `selection` travel on the spec to the daemon. They are wire fields, not client behaviour.

## Executors and concurrency

`Executor` picks what runs your tasks inside a node:

```python
sky.Compute(provider=sky.AWS(), executor=sky.Executor(type="thread", concurrency=4))
```

- `"thread"` — the default. Shares the worker's address space, so the distributed collections reach the cluster with nothing in between.
- `"process"` — a subprocess per task, which is what a task holding the GIL or leaking state wants. `reuse=False` spends a fresh subprocess per task.
- `"loky"` — a reusable process pool that also restarts a worker that died.

`concurrency` is how many tasks run at once. `buffer` is the slack above it: that many more tasks are admitted and their payloads made ready, so a slot that frees finds the next one in hand rather than a round trip away. It's also the queue depth the daemon reads as backpressure when deciding to grow an elastic compute.

`sky.Options(...)` carries the operational knobs — SSH timeouts, worker timeout, autoscaling, health probes. Most travel to the daemon; `ready_timeout` and `shutdown_timeout` stay client-side, because they govern how long *this* process waits for its own compute.

## Next steps

- **[Getting started](getting-started.md)** — installation, credentials, and a first remote function
- **[Architecture](architecture.md)** — the daemon, persistence, reconciliation, and the node runtime
- **[Providers](providers.md)** — provider accounts and offer catalogs
- **[Distributed training](distributed-training.md)** — multi-node framework setup
- **[Distributed collections](distributed-collections.md)** — shared state across nodes
- **[Plugins](plugins/index.md)** — extending what a node does
