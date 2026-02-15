# Parallel Execution

In this guide you'll run **multiple computations concurrently** on a single remote instance. You'll learn two ways to parallelize: `gather()` for dynamic collections and the `&` operator for type-safe composition.

## Compute Functions

Define the functions you want to run remotely:

```python
--8<-- "examples/guides/02_parallel_execution.py:6:24"
```

Each function is independent — they can run in any order, on the same instance, concurrently.

## Parallel with gather()

Use `gather()` when you have a dynamic number of computations:

```python
--8<-- "examples/guides/02_parallel_execution.py:31:33"
```

`gather()` collects multiple `@sky.compute` calls into a single parallel batch. All calls execute concurrently on the remote instance and return as a tuple.

## Type-Safe Parallel with &

Use `&` when you have a fixed number of computations and want full type inference:

```python
--8<-- "examples/guides/02_parallel_execution.py:35:37"
```

The `&` operator preserves the return types — `a` is `int` and `b` is `int`, inferred from `multiply`'s return type.

## Mixing Different Computations

You can chain up to 8 different computations with `&`:

```python
--8<-- "examples/guides/02_parallel_execution.py:39:41"
```

Each variable gets the correct type from its respective function.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/02_parallel_execution.py
```

---

**What you learned:**

- **`gather()`** runs a dynamic number of computations in parallel.
- **`&` operator** chains computations with full type inference.
- Both patterns use **`>>`** to send the batch to a pool.
- You can **mix different functions** in a single parallel batch.
