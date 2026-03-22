"""Standalone Workers — No Cluster Mode.

Runs 4 independent workers without Casty cluster formation.
Useful for providers without intra-node networking (e.g., RunPod)
where nodes can't reach each other on private IPs.

Each worker runs in isolation — tasks are dispatched via SSH tunnels
from the client. Distributed collections (sky.dict, sky.barrier, etc.)
are not available in this mode.

    ┌────────┐     SSH     ┌──────────┐
    │ Client │────────────▶│ Worker 0 │
    │        │────────┐    └──────────┘
    │        │───┐    │    ┌──────────┐
    │        │─┐ │    └───▶│ Worker 1 │
    └────────┘ │ │         └──────────┘
               │ │         ┌──────────┐
               │ └────────▶│ Worker 2 │
               │           └──────────┘
               │           ┌──────────┐
               └──────────▶│ Worker 3 │
                           └──────────┘
"""

import skyward as sky


@sky.function
def process_chunk(chunk_id: int) -> dict:
    """Simulate independent data processing on each node."""
    import hashlib
    from time import monotonic

    start = monotonic()

    result = 0
    for i in range(500_000):
        data = f"chunk-{chunk_id}-item-{i}".encode()
        result += int(hashlib.sha256(data).hexdigest()[:8], 16)

    elapsed = monotonic() - start
    info = sky.instance_info()
    return {
        "chunk_id": chunk_id,
        "node": info.node,
        "result": result,
        "elapsed": round(elapsed, 2),
    }


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.RunPod(),
        nodes=4,
        options=sky.Options(cluster=False),
    ) as compute:
        results = sky.gather(*(process_chunk(i) for i in range(8))) >> compute
        for r in results:
            print(
                f"Chunk {r['chunk_id']}: node={r['node']} "
                f"elapsed={r['elapsed']}s"
            )
