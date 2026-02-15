# Using Accelerators

In this guide you'll learn how to **request specific GPU types** and detect the hardware available on your remote instance.

## Requesting a GPU

Specify the accelerator when creating a pool:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:43:47"
```

Skyward supports named accelerators (`"T4"`, `"A100"`, `"H100"`) and typed ones (`sky.accelerators.L40S()`). The provider finds the cheapest available instance matching your request.

## Detecting Hardware

Use `instance_info()` to inspect the remote instance:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:6:9"
```

`InstanceInfo` includes the node ID, IP address, GPU name, and cluster metadata.

## GPU vs CPU Benchmark

A simple matrix multiplication shows the GPU advantage:

```python
--8<-- "examples/guides/04_gpu_accelerators.py:12:39"
```

The warmup call compiles CUDA kernels. After that, GPU matmul on a 4096x4096 matrix is typically 20-50x faster than CPU.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/04_gpu_accelerators.py
```

---

**What you learned:**

- **`accelerator`** parameter requests a specific GPU type.
- **`sky.accelerators.*`** provides typed accelerator specs.
- **`instance_info()`** returns hardware and cluster metadata.
- GPU warmup is needed for the first CUDA kernel compilation.
