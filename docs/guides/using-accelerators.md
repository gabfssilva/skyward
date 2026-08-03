# Using accelerators

Most ML workloads need GPUs. Skyward abstracts GPU selection across providers — you specify the hardware you need, and the provider finds the right instance type, resolves availability, and provisions it. This guide covers how to request specific accelerators, detect the hardware available at runtime, and understand the difference between string and typed accelerator specs.

## Requesting a GPU

Specify the accelerator when creating a pool:

```python
--8<-- "guides/04_gpu_accelerators.py:43:47"
```

Use the factory functions under `sky.accelerators`. They carry catalog metadata — VRAM size, CUDA compatibility, form factor — and provide IDE autocomplete:

```python
sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.A100())
sky.Compute(provider=sky.AWS(), accelerator=sky.accelerators.H100(count=4))
```

Each factory returns an `Accelerator` dataclass with a canonical `name` and an integer `count`. The shared catalog resolves provider offer names and VRAM; `sky.accelerators.H100()` is the compact way to request one H100.

The translation from a logical accelerator name to a provider-specific resource isn't a simple string match. An "A100" on AWS is a `p4d.24xlarge`, on RunPod it's a pod with a specific `gpuTypeId`, on VastAI it's a marketplace offer filtered by GPU model. The catalog centralizes this complexity so that the same `Accelerator` spec resolves correctly on any provider that supports it.

## Detecting hardware at runtime

Inside a `@sky.function` function, `instance_info()` tells you what hardware is available:

```python
--8<-- "guides/04_gpu_accelerators.py:6:9"
```

`Info` includes topology: the node id, `rank`, `nodes`, head status, and peer addresses. It does not describe installed hardware; inspect the framework runtime, such as `torch.cuda`, inside the function.

## GPU vs CPU benchmark

A matrix multiplication benchmark illustrates the GPU advantage. The function runs on the remote instance, where the accelerator is available:

```python
--8<-- "guides/04_gpu_accelerators.py:12:39"
```

The first `torch.matmul` on GPU is a warmup call — it triggers CUDA kernel compilation, which is a one-time cost. After warmup, GPU matmul on a 4096x4096 matrix is typically 20-50x faster than CPU. The exact speedup depends on the GPU model, matrix size, and data type (fp32 vs fp16).

Note that imports happen *inside* the function. This is intentional — the function runs on the remote worker, where `torch` is installed via the Image's `pip` field. Your local machine doesn't need torch installed.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/04_gpu_accelerators.py
```

---

**What you learned:**

- **`accelerator`** parameter requests specific GPU hardware via factory functions like `sky.accelerators.A100()`.
- **`sky.accelerators.*`** provides catalog-backed specs with VRAM, CUDA version, and provider-specific resolution.
- **`instance_info()`** provides node identity and cluster metadata; use the framework runtime to inspect hardware.
- **Imports inside `@sky.function`** — remote dependencies don't need to be installed locally.
- **GPU warmup** — first CUDA kernel compilation is a one-time cost; benchmark after warmup for accurate numbers.
