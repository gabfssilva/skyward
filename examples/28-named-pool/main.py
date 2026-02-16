"""Named pool from TOML configuration.

Provisions a pool defined in skyward.toml and broadcasts a simple
numpy computation across nodes.

    cd examples/28-named-pool
    uv run python main.py
"""

import skyward as sky


@sky.compute
def matrix_stats(seed: int) -> dict:
    import numpy as np

    node = sky.instance_info().node
    rng = np.random.default_rng(seed + node)
    data = rng.standard_normal((1000, 50))

    return {
        "node": node,
        "mean": float(data.mean()),
        "std": float(data.std()),
        "shape": data.shape,
    }


if __name__ == "__main__":
    with sky.ComputePool.Named("demo") as pool:
        results = matrix_stats(42) @ pool

        for r in sorted(results, key=lambda r: r["node"]):
            print(
                f"  node {r['node']}: "
                f"shape={r['shape']}, "
                f"mean={r['mean']:.4f}, "
                f"std={r['std']:.4f}"
            )
