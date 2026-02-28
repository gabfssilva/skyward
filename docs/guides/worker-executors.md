# Worker executors

Every node in a Skyward pool runs a worker process that receives tasks and executes them. The `Worker` dataclass controls how that execution happens — specifically, whether tasks run as **separate OS processes** or as **threads** inside the worker. This choice determines whether CPU-bound Python code can use all available cores or is limited by the GIL.

## A CPU-bound task

Consider a tight numerical loop — pure Python, no C extensions, no I/O:

```python
--8<-- "examples/guides/14_worker_executors.py:10:21"
```

This is the worst case for the GIL: the Python interpreter never releases the lock, so only one thread can make progress at a time. On a 2-vCPU machine with `concurrency=2`, a thread-based executor will show ~50% CPU utilization — one core active, one idle.

## Thread executor (default)

The thread executor runs tasks as threads inside the worker process. All threads share the same memory space and the same GIL:

```python
--8<-- "examples/guides/14_worker_executors.py:28:40"
```

Threads are lightweight, support streaming (generator functions and iterator parameters), and work seamlessly with distributed collections. For most workloads — I/O-bound tasks, C extension heavy code (NumPy, PyTorch), and mixed workloads — the thread executor is the right choice. The `executor="thread"` is the default, so `Worker(concurrency=2)` is equivalent.

## Process executor

The process executor runs each task in a separate OS process via `ProcessPoolExecutor`. Each process has its own Python interpreter and its own GIL, so CPU-bound work saturates all available cores:

```python
--8<-- "examples/guides/14_worker_executors.py:42:53"
```

On a 2-vCPU instance with `concurrency=2`, this achieves ~100% CPU utilization — both cores fully active. Use this explicitly for pure-Python CPU-bound workloads where bypassing the GIL matters.

## Trade-offs

| | Process (`"process"`) | Thread (`"thread"`) |
|---|---|---|
| **GIL** | Bypassed — each process has its own interpreter | Shared — one thread runs at a time |
| **CPU-bound** | Full core utilization | Limited to ~1 core |
| **I/O-bound** | Works, but heavier per-task overhead | Lightweight, ideal for I/O waits |
| **Distributed collections** | Available (via IPC bridge) | Available (shared memory with worker) |
| **Task isolation** | Full — crash in one task doesn't affect others | Shared — exceptions propagate normally |
| **Serialization** | Task args and results cross process boundary (pickle) | No extra serialization |

## When to use each

**Use thread (default)** for:

- I/O-bound tasks (API calls, database queries, file processing)
- C extension heavy workloads that release the GIL (NumPy, PyTorch inference)
- Streaming tasks (generator functions, iterator parameters)
- Most ML training workloads (PyTorch, JAX, Keras release the GIL)

**Use process** for:

- Pure-Python CPU-bound computation (tight loops, data transformation)
- Any workload where Python code dominates CPU time
- Tasks that benefit from crash isolation

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/14_worker_executors.py
```

---

**What you learned:**

- **`Worker(executor="thread")`** (default) runs tasks as threads — lightweight, supports streaming, shares memory, but GIL-limited for pure-Python CPU-bound code.
- **`Worker(executor="process")`** runs tasks in separate OS processes — bypasses the GIL, full CPU utilization for compute-heavy work.
- **`concurrency`** controls task slots per node — total parallelism = `nodes * concurrency`.
- **Distributed collections** work with both executors — the process executor uses an IPC bridge to proxy operations to the parent worker.
- **Choose based on workload**: process for CPU-bound, thread for I/O-bound.
