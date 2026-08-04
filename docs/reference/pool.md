# Compute and task dispatch

The client uses `Compute` for both an embedded control plane and a remote daemon. A `Compute` can receive one provider descriptor or several `Spec` alternatives. The alternatives are evaluated against the daemon's cached provider offers.

```python
import skyward as sky

@sky.function
def train(batch):
    return model(batch)

with sky.Compute(
    sky.Spec(sky.AWS(region="us-east-1"), accelerator="A100"),
    sky.Spec(sky.VastAI(), accelerator="A100", max_hourly_cost=2.0),
    nodes=2,
    allocation="spot_if_available",
    selection="cheapest",
    executor=sky.Executor(type="process", concurrency=2),
) as compute:
    result = train(batch) >> compute
```

`provider=...` is shorthand for one `Spec`. Pass either `provider` or positional specs, not both. `Spec` describes hardware requirements; node count, allocation, selection, image, executor, options, ports, volumes, and lifecycle settings belong to `Compute`.

## Lifecycle

Entering a `with Compute(...)` block registers provider accounts, creates or attaches the compute resource, waits for readiness, and starts the client-side lease. Leaving the block deletes the compute by default. Set `delete_on_exit=False` to keep it alive, then reconnect with `Compute.attached(ref)`.

With no `url`, the client uses an embedded daemon. With `url` or `SKYWARD_URL`, it uses the remote daemon. Both paths use the same control-plane API.

## Dispatch

`@sky.function` creates an inert `Pending` call. It runs only when dispatched:

| Expression | Result |
|---|---|
| `call() >> compute` | Run on one node and return the value |
| `call() @ compute` | Run on every node and return a list |
| `call() > compute` | Start asynchronously and return a `Future` |
| `a() & b() >> compute` | Run a `Group` and return results in submission order |
| `sky.gather(a(), b(), stream=True) >> compute` | Return an iterator as results arrive |
| `@sky.stream` and `stream_call() >> compute` | Yield items from a remote generator |

Inside a `with Compute(...)` block, `>> sky` uses the active compute. The explicit `compute` target is required outside that context.

`Compute.map(fn, items)` submits one pending call per item and returns results in input order. `Compute.current_nodes()` reports the number of ready nodes.

## Specifications and runtime options

`Spec` accepts `provider`, `accelerator`, `cpus`, `memory_gb`, `region`, `disk_gb`, `architecture`, and `max_hourly_cost`.

`Options` accepts provisioning and worker timeouts, retry settings, health checks, autoscaling settings, and the `cluster` capability flag. `ready_timeout` and `shutdown_timeout` control how long the current client waits for its compute.

`Executor` supports `thread`, `process`, and `loky`. `concurrency` sets the number of task slots per node, and `buffer` sets how many additional tasks can be admitted ahead of those slots. `reuse=False` is valid only for the `process` executor.

## Reference

::: skyward.Compute

::: skyward.Compute.attached

::: skyward.Spec

::: skyward.Options

::: skyward.Executor

::: skyward.Nodes

::: skyward.Image

::: skyward.Port

::: skyward.Volume

::: skyward.function

::: skyward.stream

::: skyward.Pending

::: skyward.Group

::: skyward.Streaming

::: skyward.gather
