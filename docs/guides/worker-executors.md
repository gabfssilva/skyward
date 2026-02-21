# Worker Executors

Every node in a Skyward pool runs a worker process that receives tasks and executes them. The `Worker` dataclass controls how that execution happens — specifically, whether tasks run as **separate OS processes** or as **threads** inside the worker. This choice determines whether CPU-bound Python code can use all available cores or is limited by the GIL.

## A CPU-Bound Task

Consider a tight numerical loop — pure Python, no C extensions, no I/O:

```python
--8<-- "examples/guides/14_worker_executors.py:10:21"
```

This is the worst case for the GIL: the Python interpreter never releases the lock, so only one thread can make progress at a time. On a 2-vCPU machine with `concurrency=2`, a thread-based executor will show ~50% CPU utilization — one core active, one idle.

## Process Executor (Default)

The process executor runs each task in a separate OS process via `ProcessPoolExecutor`. Each process has its own Python interpreter and its own GIL, so CPU-bound work saturates all available cores:

```python
--8<-- "examples/guides/14_worker_executors.py:28:38"
```

On a 2-vCPU instance with `concurrency=2`, this achieves ~100% CPU utilization — both cores fully active. The `executor="process"` is the default, so `Worker(concurrency=2)` is equivalent.

## Thread Executor

The thread executor runs tasks as threads inside the worker process. All threads share the same memory space and the same GIL:

```python
--8<-- "examples/guides/14_worker_executors.py:41:51"
```

For the same CPU-bound task on a 2-vCPU machine, this tops out at ~50% CPU — the GIL serializes execution. However, threads are the right choice for I/O-bound work (network calls, disk reads) where tasks spend most of their time waiting rather than computing.

## Trade-offs

| | Process (`"process"`) | Thread (`"thread"`) |
|---|---|---|
| **GIL** | Bypassed — each process has its own interpreter | Shared — one thread runs at a time |
| **CPU-bound** | Full core utilization | Limited to ~1 core |
| **I/O-bound** | Works, but heavier per-task overhead | Lightweight, ideal for I/O waits |
| **Distributed collections** | Available (via IPC bridge) | Available (shared memory with worker) |
| **Task isolation** | Full — crash in one task doesn't affect others | Shared — exceptions propagate normally |
| **Serialization** | Task args and results cross process boundary (pickle) | No extra serialization |

## When to Use Each

**Use process (default)** for:

- Numerical computation, data transformation, training loops
- Any workload where Python code dominates CPU time
- Tasks that benefit from crash isolation

**Use thread** for:

- I/O-bound tasks (API calls, database queries, file processing)
- C extension heavy workloads that release the GIL (NumPy, PyTorch inference)

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/14_worker_executors.py
```

---

**What you learned:**

- **`Worker(executor="process")`** (default) runs tasks in separate OS processes — bypasses the GIL, full CPU utilization for compute-heavy work.
- **`Worker(executor="thread")`** runs tasks as threads — lightweight, shares memory, but GIL-limited for CPU-bound code.
- **`concurrency`** controls task slots per node — total parallelism = `nodes * concurrency`.
- **Distributed collections** work with both executors — the process executor uses an IPC bridge to proxy operations to the parent worker.
- **Choose based on workload**: process for CPU-bound, thread for I/O-bound.
