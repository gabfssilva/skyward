"""Simple multi-node cluster test.

Provisions 2 CPU-only nodes, broadcasts a function that depends on
local modules (lib.transform, lib.config) and a third-party dep (numpy).
Tests that serialization and transitive dependencies work across the cluster.

    cd examples/25-simple-cluster
    uv run python main.py
"""

import skyward as sky


@sky.compute
def run_experiment(config) -> dict:
    import numpy as np
    from lib import normalize, summarize

    rng = np.random.default_rng(config.seed + sky.instance_info().node)
    data = rng.standard_normal((config.n_samples, config.n_features))

    normed = normalize(data)
    result = summarize(normed, config)
    result["node"] = sky.instance_info().node

    return result


if __name__ == "__main__":
    from lib import ExperimentConfig

    config = ExperimentConfig(seed=42, n_samples=500, n_features=5)

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=2,
        logging=sky.LogConfig(level="TRACE"),
        image=sky.Image(
            pip=['numpy'],
            includes=['lib']
        )
    ) as pool:
        results = run_experiment(config) @ pool

        for r in sorted(results, key=lambda r: r["node"]):
            print(
                f"  node {r['node']}: "
                f"shape={r['shape']}, "
                f"mean={r['mean']:.4f}, "
                f"std={r['std']:.4f}"
            )
