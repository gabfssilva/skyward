# Streaming

Most of the time, `@sky.function` functions run remotely and return a single result — the entire return value is serialized, sent back over the network, and handed to the caller at once. This works well for most workloads, but some patterns don't fit: a function that produces millions of rows can't materialize them all in memory before sending; a training loop that yields metrics every epoch shouldn't wait until the last epoch to report; a pipeline that feeds data into a remote model shouldn't serialize the entire dataset upfront.

Streaming solves this. A generator decorated with `@sky.stream` becomes a **streaming computation** — results flow back to the caller one at a time as they're produced, and the caller consumes them as a regular Python iterator. Input streaming remains a `@sky.function` call with an `Iterator[T]` parameter.

## Output streaming

The simplest form: the remote function `yield`s values instead of returning a single result. On the client side, `>>` returns an iterator instead of a value.

```python
--8<-- "guides/13_streaming.py:20:26"
```

Dispatching this function returns a `Streaming` computation that produces an iterator — results arrive as the remote function yields them, not after it finishes:

```python
--8<-- "guides/13_streaming.py:61:62"
```

Under the hood, the stream is a pull. The worker holds the generator, and the daemon asks it for the next item each time the caller has somewhere to put one; each value crosses the SSH tunnel as its own Casty message, as soon as it's produced. Nothing is produced before it is asked for — if the caller consumes slowly, the generator waits — and a caller that stops reading closes the generator on the node.

This means **time-to-first-result** scales with your function's first `yield`, not with the total computation time. A function that yields a progress update every epoch gives you live feedback from the first epoch onward.

## Input streaming

The inverse pattern: instead of streaming results *out*, you stream data *in*. Annotate a parameter with `Iterator[T]`, and Skyward streams the argument to the worker incrementally instead of serializing it all at once.

```python
--8<-- "guides/13_streaming.py:32:40"
```

On the client side, pass any iterable — the elements are sent to the worker as a stream:

```python
--8<-- "guides/13_streaming.py:66:69"
```

The detection is based on the type annotation: Skyward inspects the function's type hints and identifies parameters annotated as `Iterator[T]`. When it finds one, it replaces the argument with a Casty stream — spawning a `stream_producer` on the client side, pumping elements from the local iterator in a background thread, and giving the worker a `_SyncSource` that consumes the stream as a regular `for x in data` loop.

This is useful when the input data is large or lazily produced. Instead of serializing a 10GB dataset into a single cloudpickle blob, you can pass a generator that reads from disk chunk by chunk — each chunk crosses the network as a stream element, and the worker processes it as it arrives. Memory usage stays flat on both sides.

## Bidirectional streaming

Combine both: a function that takes an `Iterator[T]` input and `yield`s results. Data flows in, transformed results flow out, and neither side materializes the full dataset:

```python
--8<-- "guides/13_streaming.py:43:51"
```

```python
--8<-- "guides/13_streaming.py:73:75"
```

The client feeds values into the input stream, the worker consumes them one at a time, and each computed result is yielded back through the output stream. This is the streaming equivalent of a Unix pipe — data flows through the remote function without buffering the entire input or output.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/13_streaming.py
```

---

**What you learned:**

- **Output streaming** — `@sky.stream` generators yield results incrementally; `>>` returns a synchronous iterator on the client side.
- **Input streaming** — Parameters annotated as `Iterator[T]` are streamed to the worker instead of serialized whole.
- **Bidirectional** — Combine both: stream data in with `Iterator[T]`, yield results out with `yield`. Neither side buffers the full dataset.
- **Backpressure** — The generator is pulled one item per request, so a consumer that falls behind pauses the producer, preventing unbounded memory growth.
- **Explicit output API** — Use `@sky.stream` for generator functions and `@sky.function` for functions that consume streamed input.
