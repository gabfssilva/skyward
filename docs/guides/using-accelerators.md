# Using accelerators

Most ML workloads need GPUs. Skyward abstracts GPU selection across providers — you specify the hardware you need, and the provider finds the right instance type, resolves availability, and provisions it. This guide covers how to request specific accelerators, detect the hardware available at runtime, and understand the difference between string and typed accelerator specs.

## Requesting a GPU

Specify the accelerator when creating a pool:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:43:47"
```

You can pass a plain string (`"A100"`, `"T4"`, `"H100"`) or use the typed factory functions under `sky.accelerators`. The factory functions carry catalog metadata — VRAM size, CUDA compatibility, form factor — and provide IDE autocomplete:

```python
# String — simple, works everywhere
sky.ComputePool(provider=sky.AWS(), accelerator="A100")

# Factory function — type-safe, with catalog defaults
sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.A100())
sky.ComputePool(provider=sky.AWS(), accelerator=sky.accelerators.H100(count=4))
```

Both forms produce an `Accelerator` dataclass — a frozen specification with `name`, `memory`, `count`, and optional metadata. The factory functions look up defaults from an internal catalog, so `sky.accelerators.H100()` already knows it has 80GB of memory without you specifying it.

The translation from a logical accelerator name to a provider-specific resource isn't a simple string match. An "A100" on AWS is a `p4d.24xlarge`, on RunPod it's a pod with a specific `gpuTypeId`, on VastAI it's a marketplace offer filtered by GPU model. The catalog centralizes this complexity so that the same `Accelerator` spec resolves correctly on any provider that supports it.

## Detecting hardware at runtime

Inside a `@sky.compute` function, `instance_info()` tells you what hardware is available:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:6:9"
```

`InstanceInfo` includes the node index, cluster size, head status, and the number and type of accelerators. This is useful for conditional logic — running a GPU path when CUDA is available, falling back to CPU otherwise.

## GPU vs CPU benchmark

A matrix multiplication benchmark illustrates the GPU advantage. The function runs on the remote instance, where the accelerator is available:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:12:39"
```

The first `torch.matmul` on GPU is a warmup call — it triggers CUDA kernel compilation, which is a one-time cost. After warmup, GPU matmul on a 4096x4096 matrix is typically 20-50x faster than CPU. The exact speedup depends on the GPU model, matrix size, and data type (fp32 vs fp16).

Note that imports happen *inside* the function. This is intentional — the function runs on the remote worker, where `torch` is installed via the Image's `pip` field. Your local machine doesn't need torch installed.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/04_gpu_accelerators.py
```

---

**What you learned:**

- **`accelerator`** parameter requests specific GPU hardware — works as a string or typed factory function.
- **`sky.accelerators.*`** provides catalog-backed specs with VRAM, CUDA version, and provider-specific resolution.
- **`instance_info()`** detects hardware at runtime — node identity, accelerators, cluster metadata.
- **Imports inside `@sky.compute`** — remote dependencies don't need to be installed locally.
- **GPU warmup** — first CUDA kernel compilation is a one-time cost; benchmark after warmup for accurate numbers.
