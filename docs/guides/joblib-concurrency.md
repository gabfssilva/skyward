# Joblib Concurrency

In this guide you'll distribute **joblib tasks across cloud nodes**. You'll learn how `JoblibPool` replaces joblib's default backend with a distributed one — no code changes needed beyond the context manager.

## Defining Tasks

Any regular Python function works with joblib — no `@sky.compute` needed:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:10:13"
```

## Distributed Execution with JoblibPool

Wrap your `Parallel` call inside a `JoblibPool` context manager:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:17:27"
```

`JoblibPool` intercepts joblib's `n_jobs=-1` and distributes tasks across all cloud nodes. Each node runs `concurrency` tasks in parallel — 10 nodes x 10 concurrent = 100 effective workers.

## Measuring Throughput

Compare actual time against the ideal parallel time:

```python
--8<-- "examples/guides/10_joblib_concurrency.py:29:36"
```

With 2000 tasks, 100 effective workers, and 5s per task, the ideal time is 100s. Efficiency measures how close you get to that ideal.

## Real-World Results

Running with 10 `t4g.micro` nodes (1GB RAM, 2 vCPUs) on AWS:

```
Tasks: 2000 | Nodes: 10 | Concurrency: 10
Effective workers: 100
Total time: 102.57s
Throughput: 19.50 tasks/s
Ideal time (2000 tasks / 100 workers * 5s): 100s
Efficiency: 97.5%
```

97.5% efficiency — nearly perfect linear scaling. Skyward communicates with the cluster via SSH tunnels to a lightweight [Casty](https://github.com/gabfssilva/casty) actor system running on each node, using raw TCP over asyncio. The minimal protocol overhead is what makes this near-ideal throughput possible even on the smallest instances.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/10_joblib_concurrency.py
```

---

**What you learned:**

- **`JoblibPool`** distributes joblib work across cloud instances.
- **`n_jobs=-1`** uses all available distributed workers.
- **Standard joblib API** — `Parallel`, `delayed`, unchanged.
- **`concurrency`** controls parallelism per node, multiplying throughput.
