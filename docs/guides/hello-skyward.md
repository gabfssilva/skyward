# Hello, Skyward!

This guide walks you through running your first function on a remote cloud instance. By the end, you'll understand the three core ideas in Skyward: compute functions, pools, and the `>>` operator — and what happens behind the scenes when you combine them.

## The Compute Function

Any Python function can run on the cloud. The only change is adding the `@sky.compute` decorator:

```python
--8<-- "examples/guides/01_hello_skyward.py:6:19"
```

This decorator doesn't execute anything. Calling `add(2, 3)` no longer returns `5` — it returns a `PendingCompute[int]`, a frozen description of the computation. The arguments are captured, the function is recorded, but nothing runs. This is **lazy computation**: you're building a description of work, not performing it. The actual execution — serializing the function, sending it to a remote machine, running it there — doesn't happen until you dispatch the computation with an operator like `>>`.

This design means `PendingCompute` is a value you can pass around, compose with other computations, or store for later. It's also what makes remote execution possible: because the computation is a data structure rather than a running process, it can be serialized with cloudpickle, sent over the network, and executed on a different machine.

## The Pool

A `ComputePool` is a context manager that provisions cloud infrastructure for the duration of your work:

```python
--8<-- "examples/guides/01_hello_skyward.py:23:30"
```

When you enter the `with` block, Skyward asks the provider (AWS, in this case) to launch an instance, waits for it to boot, opens an SSH tunnel, installs dependencies via an idempotent bootstrap script, and starts a worker process. When you exit the block — whether normally or through an exception — the instance is terminated and all infrastructure is torn down.

This is **ephemeral compute**: the machine exists only for the duration of your work. There are no instances to forget about, no environments that drift, no idle costs accumulating overnight. The pool's lifetime is the job's lifetime.

## Dispatching with `>>`

The `>>` operator is what connects a lazy computation to a pool:

```python
result = add(2, 3) >> pool
```

This single expression triggers the full execution pipeline. The pool serializes the function and its arguments using cloudpickle (compressed with zlib), sends the payload to the remote worker over the SSH tunnel, the worker deserializes and executes `add(2, 3)`, and the result — `5` — is serialized back and returned to your local process. From your perspective, it looks like a normal function call that happens to return from a remote machine.

The generic type flows through the entire chain: `add(2, 3)` produces `PendingCompute[int]`, and `>> pool` returns `int`. Your type checker sees the correct types whether the function runs locally or on a cloud GPU.

## Local Execution

During development, you often want to test a compute function without provisioning any infrastructure. Every `@sky.compute` function exposes the original, unwrapped version via `.local`:

```python
result = add.local(2, 3)  # executes immediately, returns 5
```

This bypasses the lazy computation entirely — no `PendingCompute`, no serialization, no pool required. It's useful for unit testing, debugging, and local profiling.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/01_hello_skyward.py
```

---

**What you learned:**

- **`@sky.compute`** transforms a function into a lazy `PendingCompute` — calling it captures the computation without executing it.
- **`ComputePool`** provisions cloud instances on enter and tears them down on exit — ephemeral, scoped infrastructure.
- **`>>`** dispatches a computation to the pool: serialize, send, execute remotely, return the result.
- **`.local`** bypasses remote execution for testing and debugging.
