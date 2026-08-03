# Parallel execution

A single `>>` sends one computation to one node. But many workloads consist of multiple independent tasks — processing chunks of data, running different model configurations, evaluating several inputs. Skyward provides two ways to parallelize: `gather()` for dynamic collections and `&` for type-safe composition of a fixed set. Both dispatch all tasks concurrently and block until the results are ready.

## Compute functions

Define the functions you want to run remotely. Each one is independent — they can execute in any order, on the same or different nodes:

```python
--8<-- "guides/02_parallel_execution.py:8:26"
```

## Parallel with `gather()`

When you have a dynamic number of tasks — iterating over a list of chunks, a set of configurations, or any collection whose size isn't known at write time — use `gather()`:

```python
--8<-- "guides/02_parallel_execution.py:33:35"
```

`gather()` collects multiple `Pending` values into a `Group`. When dispatched with `>>`, all tasks execute concurrently on the pool's nodes (distributed via round-robin) and the results come back as a list. The pool handles serialization, dispatch, and collection — you just express which tasks should run in parallel.

## Type-safe parallel with `&`

When the number of parallel tasks is fixed and you want full type inference, use the `&` operator:

```python
--8<-- "guides/02_parallel_execution.py:37:39"
```

The `&` operator builds a `Group` from a fixed set of `Pending` values. Here, `a` and `b` are both `int` because `multiply` returns `int`. Groups with different return types are allowed, but their common result type is `object`.

## Mixing different computations

Since `&` preserves types per-position, you can compose completely different functions in a single parallel batch:

```python
--8<-- "guides/02_parallel_execution.py:41:43"
```

Each computation may go to a different node (round-robin scheduling), and the group blocks until all of them complete. The destructured values are returned in submission order.

The distinction from broadcast (`@`) is important: `@` runs the *same* function on *all* nodes, while `&` runs *different* functions concurrently. Use `@` when every node should do the same work; use `&` when you have distinct, independent tasks.

## Streaming results

By default, `gather()` waits for all tasks to finish before returning a list. With `stream=True`, it returns an iterator that yields results incrementally:

```python
--8<-- "guides/02_parallel_execution.py:45:50"
```

Streaming changes the return type from a list to an iterator. The default `ordered=True` preserves submission order. Pass `ordered=False` to yield results in **completion order**, with the fastest tasks first.

If you need results in the original submission order even while streaming, pass `ordered=True` (the default). Skyward will buffer internally and yield in order, though this means you won't see a result until all preceding tasks have also completed.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/02_parallel_execution.py
```

---

**What you learned:**

- **`gather()`** collects a dynamic number of computations into a parallel batch — dispatch with `>>`, results as a tuple.
- **`&` operator** composes a fixed set of computations with full type inference per position.
- **`stream=True`** yields results as they complete instead of waiting for all — useful for variable-duration tasks.
- **`@` vs `&`** — broadcast runs the same function on all nodes; `&` runs different functions concurrently.
