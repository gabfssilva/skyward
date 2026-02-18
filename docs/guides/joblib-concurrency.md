# Joblib Concurrency

joblib's `Parallel` is the standard way to parallelize work in Python — scikit-learn, NLTK, and many other libraries use it internally. By default, it runs tasks across local threads or processes. Skyward's `JoblibPool` replaces the backend with a distributed one: `n_jobs=-1` sends tasks to cloud instances instead of local cores. No code changes needed beyond the context manager — existing `Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)` patterns work as-is.

## Defining Tasks

Any regular Python function works with joblib — no `@sky.compute` decorator needed:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:10:13"
```

The function doesn't need to be decorated because joblib handles its own serialization. `JoblibPool` intercepts joblib's task batches, wraps them as `@compute` calls internally, and dispatches them to the cluster.

## Distributed Execution with `JoblibPool`

Wrap your `Parallel` call inside a `JoblibPool` context manager:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:17:27"
```

When you enter the `JoblibPool` block, Skyward provisions the instances and registers a custom joblib backend. Every `Parallel(n_jobs=-1)` call inside the block distributes tasks across the cluster. The `concurrency` parameter controls how many tasks each node runs simultaneously — with 10 nodes and `concurrency=10`, you get 100 effective workers.

`JoblibPool` is a thin wrapper around `ComputePool` that manages backend registration. When you exit the block, the instances are terminated and the default joblib backend is restored.

## Measuring Throughput

Compare actual time against the theoretical ideal:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:29:36"
```

With 2000 tasks, 100 effective workers, and 5 seconds per task, the ideal time is `2000 / 100 * 5 = 100s`. Efficiency measures how close you get to that ideal — the ratio of ideal time to actual time.

## Real-World Results

Running with 10 `t4g.micro` instances (1GB RAM, 2 vCPUs) on AWS:

```
Tasks: 2000 | Nodes: 10 | Concurrency: 10
Effective workers: 100
Total time: 102.57s
Throughput: 19.50 tasks/s
Ideal time (2000 tasks / 100 workers * 5s): 100s
Efficiency: 97.5%
```

97.5% efficiency — nearly perfect linear scaling. The overhead comes from serialization, network round-trips, and scheduling. Skyward communicates with each worker via an SSH tunnel to a lightweight [Casty](https://github.com/gabfssilva/casty) actor system running on the node, using raw TCP over asyncio. The minimal protocol overhead — no HTTP, no REST, no message broker — is what makes near-ideal throughput possible even on the smallest instances.

This also illustrates the cost model: 10 `t4g.micro` instances at ~$0.008/hour each costs $0.08/hour total. The same 2000 tasks running locally at 1 task/second would take ~2.8 hours. The cluster finishes in under 2 minutes for a fraction of a cent.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/10_joblib_concurrency.py
```

---

**What you learned:**

- **`JoblibPool`** replaces joblib's backend with a distributed one — `n_jobs=-1` uses all cloud workers.
- **No `@sky.compute` needed** — joblib handles serialization; `JoblibPool` wraps batches internally.
- **Effective workers = nodes x concurrency** — both parameters multiply throughput.
- **Near-linear scaling** — 97.5% efficiency with minimal protocol overhead (SSH + Casty actors, raw TCP).
- **Standard joblib API** — `Parallel`, `delayed` work unchanged inside the context manager.
