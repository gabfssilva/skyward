# Streaming

Most of the time, `@sky.compute` functions run remotely and return a single result — the entire return value is serialized, sent back over the network, and handed to the caller at once. This works well for most workloads, but some patterns don't fit: a function that produces millions of rows can't materialize them all in memory before sending; a training loop that yields metrics every epoch shouldn't wait until the last epoch to report; a pipeline that feeds data into a remote model shouldn't serialize the entire dataset upfront.

Streaming solves this. A `@sky.compute` function that uses `yield` instead of `return` becomes a **streaming computation** — results flow back to the caller one at a time as they're produced, and the caller consumes them as a regular Python iterator. No special API, no callback registration, no async boilerplate. You write a generator, dispatch it with `>>`, and iterate.

## Output streaming

The simplest form: the remote function `yield`s values instead of returning a single result. On the client side, `>>` returns an iterator instead of a value.

```python
--8<-- "examples/guides/13_streaming.py:20:26"
```

Dispatching this function returns something you can iterate over immediately — results arrive as the remote function produces them, not after it finishes:

```python
--8<-- "examples/guides/13_streaming.py:61:62"
```

Under the hood, the worker detects that the function is a generator and creates a Casty `stream_producer` actor. As the generator yields values, they're pushed into the stream with backpressure — if the client consumes slowly, the producer pauses. On the client side, the stream is wrapped in a synchronous iterator (`_SyncSource`) that bridges the async Casty protocol to Python's `__iter__`/`__next__`. The SSH tunnel carries the stream elements as individual messages, so each value crosses the network as soon as it's produced.

This means **time-to-first-result** scales with your function's first `yield`, not with the total computation time. A function that yields a progress update every epoch gives you live feedback from the first epoch onward.

## Input streaming

The inverse pattern: instead of streaming results *out*, you stream data *in*. Annotate a parameter with `Iterator[T]`, and Skyward streams the argument to the worker incrementally instead of serializing it all at once.

```python
--8<-- "examples/guides/13_streaming.py:32:40"
```

On the client side, pass any iterable — the elements are sent to the worker as a stream:

```python
--8<-- "examples/guides/13_streaming.py:66:69"
```

The detection is based on the type annotation: Skyward inspects the function's type hints and identifies parameters annotated as `Iterator[T]`. When it finds one, it replaces the argument with a Casty stream — spawning a `stream_producer` on the client side, pumping elements from the local iterator in a background thread, and giving the worker a `_SyncSource` that consumes the stream as a regular `for x in data` loop.

This is useful when the input data is large or lazily produced. Instead of serializing a 10GB dataset into a single cloudpickle blob, you can pass a generator that reads from disk chunk by chunk — each chunk crosses the network as a stream element, and the worker processes it as it arrives. Memory usage stays flat on both sides.

## Bidirectional streaming

Combine both: a function that takes an `Iterator[T]` input and `yield`s results. Data flows in, transformed results flow out, and neither side materializes the full dataset:

```python
--8<-- "examples/guides/13_streaming.py:43:51"
```

```python
--8<-- "examples/guides/13_streaming.py:73:75"
```

The client feeds values into the input stream, the worker consumes them one at a time, and each computed result is yielded back through the output stream. This is the streaming equivalent of a Unix pipe — data flows through the remote function without buffering the entire input or output.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/13_streaming.py
```

---

**What you learned:**

- **Output streaming** — `@sky.compute` generators yield results incrementally; `>>` returns a synchronous iterator on the client side.
- **Input streaming** — Parameters annotated as `Iterator[T]` are streamed to the worker instead of serialized whole.
- **Bidirectional** — Combine both: stream data in with `Iterator[T]`, yield results out with `yield`. Neither side buffers the full dataset.
- **Backpressure** — Casty's stream protocol pauses the producer if the consumer falls behind, preventing unbounded memory growth.
- **Zero API overhead** — No special classes or protocols. Write a generator, annotate with `Iterator`, and the streaming machinery activates automatically.
