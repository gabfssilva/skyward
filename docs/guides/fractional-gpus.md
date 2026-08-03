# Fractional GPUs

The v2 public accelerator API represents the requested accelerator count as an integer. `sky.accelerators.L4(count=0.5)` is not a valid v2 request, and `Info` does not expose an accelerator count at runtime.

Provider-specific fractional plans are not represented by a general fractional-count field. Request a catalog accelerator with an integer count and inspect the actual device inside the function:

```python
--8<-- "guides/20_fractional_gpus.py:1:4"
```

```python
--8<-- "guides/20_fractional_gpus.py:37:45"
```

The example uses `sky.accelerators.L4()` and reads VRAM through `torch.cuda.get_device_properties(0)`. `sky.Options(ready_timeout=600)` controls how long the client waits for readiness.

## Multiple workloads on one GPU

For user-managed partitioning, use the MIG plugin with a process executor:

```python
with sky.Compute(
    provider=sky.Verda(),
    accelerator=sky.accelerators.A100(),
    executor=sky.Executor(type="process", concurrency=2, reuse=True),
    plugins=[sky.plugins.Mig(profile="3g.40gb")],
) as compute:
    tasks = [train(1) for _ in range(2)]
    results = sky.gather(*tasks) >> compute
```

See [NVIDIA MIG](nvidia-mig.md) for the complete example. MIG uses a full accelerator and creates isolated partitions during bootstrap; it is different from a provider-managed fractional instance.

## Run the example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/20_fractional_gpus.py
```

---

**What you learned:**

- **Accelerator counts are integers** in the v2 public API.
- **`Info` describes topology**, not accelerator inventory.
- **Runtime VRAM** can be read through the framework, such as `torch.cuda`.
- **MIG** is the current documented way to divide a full GPU across process workers.
