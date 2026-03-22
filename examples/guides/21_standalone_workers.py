"""Standalone Workers — independent nodes without cluster networking."""

import skyward as sky


# --8<-- [start:function]
@sky.function
def process_chunk(chunk_id: int) -> dict:
    """Process a data chunk independently on a single node."""
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
# --8<-- [end:function]


if __name__ == "__main__":
    # --8<-- [start:pool]
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
    # --8<-- [end:pool]
