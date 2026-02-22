# Why asyncio

Skyward orchestrates cloud machines — it provisions instances, installs dependencies, opens SSH tunnels, starts workers, routes tasks, and tears everything down when the job is done. But what does this orchestration actually *do* on your laptop?

The answer is networking. HTTP requests to cloud APIs. SSH connections to remote machines. TCP tunnels carrying actor messages. Polling loops waiting for instances to boot. The client never trains a model, never processes a dataset, never runs a computation. It coordinates. And the nature of that coordination — many concurrent I/O operations, zero CPU-bound work — is what makes asyncio the right foundation.

## Blocking vs non-blocking

When your code makes a network call — say, asking AWS to launch an instance — the response takes time. The machine needs to be allocated, the hypervisor needs to start it, the OS needs to boot. That might take 30 seconds. With a **blocking** call, the thread that made the request sits idle for those 30 seconds, doing nothing, unable to handle anything else.

```python
# Blocking — this thread is frozen until AWS responds
instance = ec2.run_instances(...)  # 30 seconds of waiting
```

A **non-blocking** call lets the thread move on to other work while the response is in transit. When the response arrives, the thread picks it back up. The distinction is irrelevant if you only have one thing to do — but when you're launching 8 instances, opening 8 SSH connections, and polling 8 cloud APIs simultaneously, the difference is fundamental.

## Threads vs async

Both threads and asyncio solve the same problem: doing multiple things concurrently. They differ in how.

**Threads** are managed by the operating system. Each thread has its own stack (typically ~8MB of memory), and the OS decides when to switch between them. This means threads work automatically — you don't need to think about yielding control — but the OS context switch is expensive (microseconds per switch, kernel involvement), and each thread consumes real memory. A hundred threads might use 800MB of stack space alone. Python's GIL adds another constraint: only one thread can execute Python bytecode at a time, so threads don't help with CPU-bound work anyway — they only help when threads are *waiting*, not *computing*.

**Asyncio** uses a single thread running an event loop. Concurrent tasks are coroutines — functions that explicitly yield control at `await` points. There's no OS involvement in switching between tasks; the event loop simply runs the next ready coroutine when the current one awaits. A coroutine's overhead is a few hundred bytes (no dedicated stack), and switching between coroutines costs nanoseconds, not microseconds. Ten thousand concurrent coroutines are cheap. The trade-off is that you must write `async/await` code — the concurrency is cooperative, not preemptive.

For CPU-bound work — number crunching, data transformation, model training — neither threads nor asyncio help much in Python (that's what processes are for). But for I/O-bound work with many concurrent operations — which is exactly what an orchestration client does — asyncio is strictly more efficient: lower memory, faster switching, no OS overhead.

## What the client actually does

Here is every category of work the Skyward client performs on your machine. All of it is I/O.

### Cloud API calls

Every provider interaction is an HTTP request. Querying available instance types, checking spot pricing, launching instances, polling their status, terminating them, tearing down infrastructure — these are all REST or GraphQL calls over HTTPS:

- **AWS** — EC2 API via `aioboto3` (async). Fleet creation, instance polling, spot capacity queries.
- **GCP** — Compute Engine API. Instance templates, bulk insert, firewall rules, machine type lookups.
- **RunPod** — GraphQL API for pod deployment, GPU type listing, SSH key management.
- **VastAI** — HTTP marketplace API for offer search, instance creation, status polling.
- **Verda** — OAuth2-authenticated REST API with automatic token refresh.

During the offer selection phase alone, Skyward may query multiple providers in parallel to compare pricing and availability — all concurrent HTTP requests.

### SSH

Once instances are running, the client opens SSH connections via `asyncssh`:

- **Connection establishment** with automatic retry (instances take time to accept SSH after boot).
- **Remote command execution** — transferring and running the bootstrap script, monitoring its output line by line.
- **File transfer** — syncing local code directories (the `includes` in your `Image`) to the remote machine.

Each node in the pool gets its own persistent SSH connection. With 8 nodes, that's 8 concurrent SSH sessions, each streaming bootstrap output in real time.

### TCP tunnels

Each SSH connection also establishes a **local port forward** — a TCP tunnel from a random local port to the remote machine's port 25520, where the Casty worker actor system listens. This tunnel stays open for the lifetime of the pool, carrying all task payloads and results as actor messages.

### Actor messaging

Task dispatch — the `>>`, `@`, `&` operators — translates to actor messages sent over these TCP tunnels. When you write `train(10) >> pool`, the function and arguments are serialized (cloudpickle + lz4), sent as a Casty message through the SSH tunnel to the remote worker, executed there, and the result flows back through the same path. The client's role is purely routing: serialize, send, wait for response, deserialize.

### What's not here

Computation. The client never executes your `@sky.compute` functions — that happens on the remote workers. The only CPU work on your laptop is serialization (cloudpickle + lz4 compression), which takes microseconds per task. Everything else is waiting for network responses.

## Why this matters at scale

Consider a pool with 100 nodes. The client simultaneously:

- Maintains **100 SSH connections** (each with keepalive heartbeats)
- Runs **100 TCP tunnels** (port forwards carrying actor messages)
- Polls **100 cloud API endpoints** during provisioning
- Routes tasks to **100 workers** through the actor hierarchy
- Streams bootstrap output from **100 instances** in parallel

With threads, each of these concurrent activities needs its own thread — at minimum 100 threads just for SSH, plus more for API polling and task routing. That's several hundred threads, each consuming ~8MB of stack space, each requiring OS context switches. With asyncio, all of it runs on a single event loop in a single thread. A hundred concurrent coroutines use a fraction of the memory of a hundred threads, and switching between them costs nanoseconds instead of microseconds — no kernel involvement, no context save/restore overhead.

The [Casty](https://gabfssilva.github.io/casty/) actor framework maps naturally onto this: each actor is a coroutine, message passing is `await`-based, and the `pipe_to_self` pattern bridges async operations (like SSH commands or HTTP calls) into actor messages without blocking the event loop.

This is also why the actor model fits the orchestration layer. Each node progresses through its own state machine — `idle → waiting → active` — at its own pace. Node 0 might finish bootstrapping while node 73 is still booting. With asyncio + actors, each node's lifecycle is an independent coroutine responding to messages, and the event loop interleaves them automatically. No thread coordination, no locks, no shared mutable state.

## The synchronous API

Despite being fully asynchronous internally, Skyward exposes a **synchronous API**. You write normal, blocking Python:

```python
with sky.ComputePool(provider=sky.AWS(), nodes=4) as pool:
    result = train(10) >> pool  # blocks until result is ready
```

The bridge is simple. When you enter the `ComputePool` context manager, Skyward starts a background daemon thread running an asyncio event loop. Every public method — `>>`, `@`, `>`, `gather` — calls `asyncio.run_coroutine_threadsafe()` to submit work to that event loop and blocks the calling thread until the result is ready.

This gives you the best of both worlds: the efficiency of async orchestration underneath, with the simplicity of synchronous code on top. You don't need to write `async/await` in your application code. You don't need to manage an event loop. The concurrency is an implementation detail of the runtime — invisible unless you want to understand it.

## Further reading

- [Architecture](architecture.md) — The actor hierarchy and how the cluster forms
- [Core Concepts](concepts.md) — Pool lifecycle, operators, and the orchestration model
- [Casty Documentation](https://gabfssilva.github.io/casty/) — The actor framework powering the runtime
