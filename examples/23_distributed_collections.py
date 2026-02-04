"""Distributed Collections Example.

Demonstrates Skyward's distributed data structures:

    sky.dict()     - Shared cache across workers
    sky.counter()  - Distributed progress tracking
    sky.list()     - Accumulate results from all nodes
    sky.set()      - Deduplicate processed items
    sky.barrier()  - Synchronize workers at checkpoints
    sky.lock()     - Mutual exclusion for critical sections
    sky.queue()    - Work queue for dynamic task distribution

These structures are backed by Ray Actors and provide:
- Automatic get-or-create by name
- Sync interface (magic methods block)
- Async interface (*_async methods)
- Configurable consistency (eventual/strong)
"""

import skyward as sky


# =============================================================================
# Example 1: Shared Cache with Counter Progress
# =============================================================================


@sky.compute
def process_with_cache(items: list[str]) -> dict:
    """Process items using shared cache to avoid duplicate work."""
    cache = sky.dict("embeddings")
    progress = sky.counter("processed")
    info = sky.instance_info()

    results = []
    cache_hits = 0
    cache_misses = 0

    for item in sky.shard(items):
        if item in cache:
            # Cache hit - reuse result
            result = cache[item]
            cache_hits += 1
        else:
            # Cache miss - compute and store
            result = f"computed:{item}:{info.node}"
            cache[item] = result
            cache_misses += 1

        results.append(result)
        progress.increment()

    return {
        "node": info.node,
        "processed": len(results),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }


# =============================================================================
# Example 2: Result Accumulation with Deduplication
# =============================================================================


@sky.compute
def accumulate_results(batch_id: int) -> dict:
    """Accumulate unique results across workers."""
    results = sky.list("all_results")
    seen = sky.set("seen_batches")
    info = sky.instance_info()

    # Check if already processed (idempotent)
    batch_key = f"batch:{batch_id}:node:{info.node}"
    if batch_key in seen:
        return {"node": info.node, "status": "skipped", "batch_id": batch_id}

    # Mark as processed
    seen.add(batch_key)

    # Simulate work
    result = {"batch_id": batch_id, "node": info.node, "value": batch_id * 10 + info.node}
    results.append(result)

    return {"node": info.node, "status": "processed", "batch_id": batch_id}


# =============================================================================
# Example 3: Synchronized Training with Barrier
# =============================================================================


@sky.compute
def synchronized_epoch(epoch: int) -> dict:
    """Simulate synchronized distributed training."""
    sync = sky.barrier("epoch_sync", n=sky.instance_info().total_nodes)
    progress = sky.counter("training_steps")
    info = sky.instance_info()

    # Phase 1: Local training
    local_loss = 1.0 / (epoch + 1) + info.node * 0.01
    progress.increment()

    # Synchronize before aggregation
    sync.wait()

    # Phase 2: Only head aggregates (after all workers done)
    if info.is_head:
        sync.reset()  # Reset for next epoch

    return {
        "node": info.node,
        "epoch": epoch,
        "loss": local_loss,
        "is_head": info.is_head,
    }


# =============================================================================
# Example 4: Critical Section with Lock
# =============================================================================


@sky.compute
def safe_update_checkpoint(step: int) -> dict:
    """Update shared checkpoint safely using lock."""
    lock = sky.lock("checkpoint_lock")
    state = sky.dict("checkpoint")
    info = sky.instance_info()

    # Critical section - only one worker at a time
    with lock:
        current = state.get("best_step", 0)
        current_loss = state.get("best_loss", float("inf"))

        # Simulate checking if we have better results
        my_loss = 1.0 / (step + 1)
        if my_loss < current_loss:
            state["best_step"] = step
            state["best_loss"] = my_loss
            state["saved_by"] = info.node
            updated = True
        else:
            updated = False

    return {
        "node": info.node,
        "step": step,
        "my_loss": my_loss,
        "updated_checkpoint": updated,
    }


# =============================================================================
# Example 5: Dynamic Work Queue
# =============================================================================


@sky.compute
def worker_from_queue() -> dict:
    """Workers pull tasks from shared queue."""
    queue = sky.queue("tasks")
    results = sky.list("queue_results")
    info = sky.instance_info()

    processed = []

    # Process tasks until queue is empty
    while True:
        task = queue.get(timeout=0.5)
        if task is None:
            break

        # Process task
        result = {"task": task, "worker": info.node, "result": task * 2}
        results.append(result)
        processed.append(task)

    return {
        "node": info.node,
        "tasks_processed": len(processed),
        "tasks": processed,
    }


@sky.compute
def producer_fill_queue(tasks: list[int]) -> dict:
    """Head node fills the work queue."""
    queue = sky.queue("tasks")
    info = sky.instance_info()

    if info.is_head:
        for task in tasks:
            queue.put(task)
        return {"node": info.node, "role": "producer", "tasks_added": len(tasks)}

    return {"node": info.node, "role": "worker", "tasks_added": 0}


# =============================================================================
# Helper functions to read distributed state from within the cluster
# =============================================================================


@sky.compute
def read_list(name: str) -> list:
    """Read a distributed list."""
    return list(sky.list(name))


@sky.compute
def read_dict(name: str) -> dict:
    """Read a distributed dict."""
    d = sky.dict(name)
    return dict(d.items())


# =============================================================================
# Main
# =============================================================================


@sky.pool(
    provider=sky.AWS(),
    nodes=4,
    vcpus=2,
    memory_gb=8,
    architecture="arm64",
    allocation="spot-if-available",
    image=sky.Image(skyward_source="local"),
)
def main():
    # -----------------------------------------------------------------
    # Example 1: Shared Cache
    # -----------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Shared Cache with Progress")
    print("=" * 60)

    # Some items will be processed by multiple nodes (cache hits)
    items = [f"item_{i}" for i in range(100)]
    cache_results = process_with_cache(items) @ sky

    total_hits = sum(r["cache_hits"] for r in cache_results)
    total_misses = sum(r["cache_misses"] for r in cache_results)
    print(f"  Total items: {len(items)}")
    print(f"  Cache hits: {total_hits}, misses: {total_misses}")
    for r in cache_results:
        print(f"    Node {r['node']}: {r['processed']} items, {r['cache_hits']} hits")

    # -----------------------------------------------------------------
    # Example 2: Result Accumulation
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Result Accumulation with Deduplication")
    print("=" * 60)

    # Each node processes a batch
    for batch in range(3):
        accumulate_results(batch) @ sky

    # Read accumulated results (from within the cluster)
    all_results = read_list("all_results") >> sky
    print(f"  Total accumulated results: {len(all_results)}")
    for r in all_results[:5]:  # Show first 5
        print(f"    Batch {r['batch_id']} from node {r['node']}: value={r['value']}")

    # -----------------------------------------------------------------
    # Example 3: Synchronized Training
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Synchronized Training with Barrier")
    print("=" * 60)

    for epoch in range(3):
        epoch_results = synchronized_epoch(epoch) @ sky
        losses = [r["loss"] for r in epoch_results]
        avg_loss = sum(losses) / len(losses)
        print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}")

    # -----------------------------------------------------------------
    # Example 4: Safe Checkpoint Update
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Critical Section with Lock")
    print("=" * 60)

    for step in range(5):
        update_results = safe_update_checkpoint(step) @ sky
        updated = [r for r in update_results if r["updated_checkpoint"]]
        if updated:
            print(f"  Step {step}: checkpoint updated by node {updated[0]['node']}")

    checkpoint = read_dict("checkpoint") >> sky
    best_loss = checkpoint.get("best_loss")
    loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
    print(f"  Final checkpoint: step={checkpoint.get('best_step')}, "
          f"loss={loss_str}, "
          f"saved_by=node_{checkpoint.get('saved_by')}")

    # -----------------------------------------------------------------
    # Example 5: Work Queue
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: Dynamic Work Queue")
    print("=" * 60)

    # Producer fills queue
    tasks = list(range(20))
    producer_fill_queue(tasks) @ sky

    # Workers consume from queue
    worker_results = worker_from_queue() @ sky

    for r in worker_results:
        print(f"  Node {r['node']}: processed {r['tasks_processed']} tasks")

    queue_results = read_list("queue_results") >> sky
    print(f"  Total results in list: {len(queue_results)}")


if __name__ == "__main__":
    main()
