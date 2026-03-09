"""Cross-accelerator matmul benchmark.

Runs a fixed JAX GPU matmul scenario across different provider/accelerator
targets and saves the results as JSON for easy comparison.

Usage:
    uv run python examples/38_accelerator_benchmark.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import skyward as sky

# T4-safe baseline scenario
MATRIX_SIZE = 4096
DTYPE = "float16"
WARMUP = 20
ITERS = 100

TARGETS: list[dict[str, object]] = [
    # AWS
    # {"name": "aws-t4g", "provider": sky.AWS(), "accelerator": sky.accelerators.T4G()},
    # {"name": "aws-t4", "provider": sky.AWS(), "accelerator": sky.accelerators.T4()},
    # {"name": "aws-l4", "provider": sky.AWS(), "accelerator": sky.accelerators.L4()},
    # {"name": "aws-l40s", "provider": sky.AWS(), "accelerator": sky.accelerators.L40S()},
    # Verda
    # {"name": "verda-a100", "provider": sky.Verda(), "accelerator": sky.accelerators.A100()},
    # {"name": "verda-h100", "provider": sky.Verda(), "accelerator": sky.accelerators.H100()},
    # {"name": "verda-h200", "provider": sky.Verda(), "accelerator": sky.accelerators.H200()},
    # {"name": "verda-b200", "provider": sky.Verda(), "accelerator": sky.accelerators.B200()},
    # TensorDock
    # {
    #     "name": "tensordock-rtx-3090",
    #     "provider": sky.TensorDock(),
    #     "accelerator": sky.accelerators.RTX_3090(),
    # },
    # {
    #     "name": "tensordock-rtx-4090",
    #     "provider": sky.TensorDock(),
    #     "accelerator": sky.accelerators.RTX_4090(),
    # },
    # {
    #     "name": "tensordock-rtx-5090",
    #     "provider": sky.TensorDock(),
    #     "accelerator": sky.accelerators.RTX_5090(),
    # },
    # Hyperstack
    {
        "name": "hyperstack-rtx-a4000",
        "provider": sky.Hyperstack(),
        "accelerator": sky.accelerators.RTX_A4000(),
    },
]


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = math.ceil((p / 100.0) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


@sky.function
def run_matmul_benchmark(size: int, dtype: str, warmup: int, iters: int) -> dict[str, Any]:
    import time

    import jax
    import jax.numpy as jnp

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError("JAX GPU device not available on remote instance")
    gpu = gpu_devices[0]

    supported = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    if dtype not in supported:
        raise ValueError(f"Unsupported dtype: {dtype}")

    jax_dtype = supported[dtype]

    key_a, key_b = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key_a, (size, size), dtype=jax_dtype)
    b = jax.random.normal(key_b, (size, size), dtype=jax_dtype)
    a = jax.device_put(a, gpu)
    b = jax.device_put(b, gpu)

    matmul = jax.jit(lambda x, y: x @ y)

    # First call triggers JIT compilation.
    matmul(a, b).block_until_ready()

    # Warmup to stabilize runtime measurements after compilation.
    for _ in range(warmup):
        matmul(a, b).block_until_ready()

    latencies_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        matmul(a, b).block_until_ready()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = mean(latencies_ms)
    # GEMM FLOPs for C = A @ B is approximately 2 * N^3.
    tflops = (2 * (size**3)) / ((avg_ms / 1000.0) * 1e12)

    info = sky.instance_info()

    return {
        "gpu_name": gpu.device_kind,
        "provider_node": info.node if info else -1,
        "framework": "jax",
        "matrix_size": size,
        "dtype": dtype,
        "warmup": warmup,
        "iters": iters,
        "latency_p50_ms": median(latencies_ms),
        "latency_p95_ms": _percentile(latencies_ms, 95),
        "latency_avg_ms": avg_ms,
        "tflops": tflops,
    }


def _benchmark_target(target: dict[str, object]) -> dict[str, Any]:
    name = str(target["name"])
    provider = target["provider"]
    accelerator = target["accelerator"]

    with sky.Compute(
        provider=provider,  # type: ignore[arg-type]
        accelerator=accelerator,  # type: ignore[arg-type]
        image=sky.Image(pip=["jax[cuda12]"]),
    ) as pool:
        result = run_matmul_benchmark(MATRIX_SIZE, DTYPE, WARMUP, ITERS) >> pool
        result["target"] = name
        return result


def _save_results(results: list[dict[str, Any]]) -> Path:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = out_dir / f"accelerator-benchmark-{timestamp}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    all_results: list[dict[str, Any]] = []

    print("Running accelerator benchmark")
    print(
        f"Scenario: {DTYPE} matmul {MATRIX_SIZE}x{MATRIX_SIZE}, "
        f"warmup={WARMUP}, iters={ITERS}"
    )

    for target in TARGETS:
        name = str(target["name"])
        print(f"\n>>> {name}")
        try:
            benchmark = _benchmark_target(target)
            all_results.append(benchmark)
            print(
                f"p50={benchmark['latency_p50_ms']:.2f} ms | "
                f"p95={benchmark['latency_p95_ms']:.2f} ms | "
                f"TFLOPS={benchmark['tflops']:.2f} | "
                f"{benchmark['gpu_name']}"
            )
        except Exception as exc:
            error = {"target": name, "error": str(exc)}
            all_results.append(error)
            print(f"FAILED: {exc}")

    output_path = _save_results(all_results)
    print(f"\nSaved results to: {output_path}")
