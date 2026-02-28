# Joblib concurrency

joblib's `Parallel` is the standard way to parallelize work in Python — scikit-learn, NLTK, and many other libraries use it internally. By default, it runs tasks across local threads or processes. Skyward's `joblib` plugin replaces the backend with a distributed one: `n_jobs=-1` sends tasks to cloud instances instead of local cores. No code changes needed beyond the pool configuration — existing `Parallel(n_jobs=-1)(delayed(fn)(x) for x in data)` patterns work as-is.

## Defining tasks

Any regular Python function works with joblib — the plugin handles serialization and dispatch internally:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:10:13"
```

joblib handles its own serialization. The `joblib` plugin intercepts joblib's task batches, wraps them internally, and dispatches them to the cluster.

## Distributed execution with the Joblib plugin

Wrap your `Parallel` call inside a `ComputePool` with the `joblib` plugin:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:17:27"
```

When you enter the pool block, Skyward provisions the instances and the `joblib` plugin registers a custom joblib backend. Every `Parallel(n_jobs=-1)` call inside the block distributes tasks across the cluster. The `worker` parameter accepts a `Worker` dataclass that controls per-node execution — `Worker(concurrency=10)` means each node runs 10 tasks simultaneously. With 10 nodes and `concurrency=10`, you get 100 effective workers.

When you exit the block, the instances are terminated and the default joblib backend is restored.

## Measuring throughput

Compare actual time against the theoretical ideal:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:29:36"
```

With 2000 tasks, 100 effective workers, and 5 seconds per task, the ideal time is `2000 / 100 * 5 = 100s`. Efficiency measures how close you get to that ideal — the ratio of ideal time to actual time.

## Real-world results

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

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/10_joblib_concurrency.py
```

---

**What you learned:**

- **`plugins=[sky.plugins.joblib()]`** replaces joblib's backend with a distributed one — `n_jobs=-1` uses all cloud workers.
- **Plain functions** — joblib handles serialization; the plugin wraps batches internally.
- **Effective workers = nodes x worker concurrency** — both parameters multiply throughput.
- **Near-linear scaling** — 97.5% efficiency with minimal protocol overhead (SSH + Casty actors, raw TCP).
- **Standard joblib API** — `Parallel`, `delayed` work unchanged inside the context manager.
