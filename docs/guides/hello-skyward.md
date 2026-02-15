# Hello, Skyward!

In this guide you'll run your first function on a **remote cloud instance**. You'll learn the three core ideas in Skyward: compute functions, pools, and the `>>` operator.

## The Compute Function

Any Python function can run on the cloud — just add the `@sky.compute` decorator:

```python
--8<-- "examples/guides/01_hello_skyward.py:6:19"
```

The function runs locally until you send it to a pool. Nothing special happens at decoration time — it just marks the function for remote execution.

## Running on the Cloud

Create a `ComputePool` and use `>>` to send work to it:

```python
--8<-- "examples/guides/01_hello_skyward.py:23:30"
```

`ComputePool` provisions a cloud instance. The `>>` operator serializes the function call, sends it to the remote instance, executes it, and returns the result — all transparently.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/01_hello_skyward.py
```

---

**What you learned:**

- **`@sky.compute`** marks a function for remote execution.
- **`ComputePool`** provisions cloud instances.
- **`>>`** sends a computation to the pool and returns the result.
