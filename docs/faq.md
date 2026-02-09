# Frequently Asked Questions

## General

### What is Skyward?

Skyward is a Python library for ephemeral GPU compute. It lets you run ML training code on cloud GPUs without managing infrastructure. You define your training function with a decorator, and Skyward handles provisioning instances, installing dependencies, executing your code, and terminating resources automatically.

### How is Skyward different from SageMaker or Vertex AI?

| Aspect | Managed Platforms (SageMaker, Vertex AI) | Skyward |
|--------|------------------------------------------|---------|
| Definition | YAML/JSON configs, platform-specific SDKs | Pure Python with decorators |
| Vendor lock-in | High | Low (multi-cloud) |
| Framework support | Their SDK wrappers | Any Python code |
| Local testing | Limited | Full (`.local()` method) |
| Debugging | Logs only | Interactive (SSM, SSH) |
| Cost | Platform markup | Direct cloud pricing |

Managed platforms wrap your code in their abstractions. Skyward runs your code as-is — the only change is adding a `@compute` decorator.

### When should I use ephemeral compute?

Ephemeral compute is ideal for workloads that **start, run, and end**:

- Training runs (fine-tuning, pretraining)
- Hyperparameter sweeps (GridSearchCV, Optuna)
- Batch inference (embeddings generation, dataset processing)
- Distributed training (multi-GPU, multi-node)
- CI/CD for ML (testing training pipelines)
- Research experiments (quick iterations)

It's **not designed for**:

- Model serving (use inference endpoints)
- Real-time APIs (use Lambda, Cloud Run)
- Long-running services (use ECS, Kubernetes)

---

## Costs and Billing

### How does billing work?

Skyward uses your cloud provider directly — there's no Skyward fee. You pay only for the cloud resources you consume (EC2 instances, etc.) at standard cloud rates.

Cost tracking is built-in:
```
[cost] Running: $12.34/hr (4x A100 spot @ $3.08/hr each)
[cost] Elapsed: 2.5 hours, Total: $30.85
[cost] Final: 6.2 hours, $76.23 (saved $298.41 vs on-demand)
```

### How can I reduce costs?

1. **Use spot instances**: Save 60-90% with `allocation="always-spot"` or `allocation="spot-if-available"`
2. **Set timeouts**: Use `timeout=3600` to auto-terminate after 1 hour
3. **Right-size GPUs**: Use T4 for prototyping, A100/H100 only for production training
4. **Use MIG partitioning**: Run multiple experiments on a single GPU

```python
sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    allocation="always-spot",        # 60-90% cheaper
    timeout=3600,         # Auto-shutdown safety net
)
```

### What if I forget to terminate instances?

Skyward's context manager ensures cleanup:

```python
with sky.ComputePool(...) as pool:
    result = train() >> pool
# Instances terminated here, even if an exception occurs
```

Additionally, the `timeout` parameter sets a self-destruct timer on the instance itself. Even if your laptop dies or Python crashes, the instance will terminate after the timeout.

---

## Technical

### What Python versions are supported?

Python 3.13 or higher is required.

### Can I use custom Docker images?

Not directly. Skyward uses `Image` to declaratively specify your environment:

```python
sky.Image(
    python="3.13",
    pip=["torch", "transformers"],
    apt=["ffmpeg"],
    env={"CUDA_VISIBLE_DEVICES": "0"},
)
```

This generates an idempotent bootstrap script that installs dependencies on a fresh cloud instance.

### How does function serialization work?

Skyward uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) to serialize:
- Your decorated function
- Its arguments
- Its closure (captured variables)

The serialized payload is sent to the remote instance, deserialized, and executed. Results are serialized back.

**Limitations:**
- Functions must be serializable (no open file handles, database connections, etc.)
- Large arguments are sent over the network (consider S3 for large datasets)
- Closures capture values, not references

### How do I debug remote execution?

1. **Local testing**: Use `.local()` to run the function locally first
   ```python
   result = my_function.local(args)  # Runs locally, no cloud
   ```

2. **Remote logs**: stdout/stderr are streamed as `LogLine` events
   ```python
   def my_callback(event):
       if isinstance(event, sky.LogLine):
           print(f"[remote] {event.line}")

   sky.ComputePool(on_event=my_callback)
   ```

3. **SSH/SSM access**: AWS instances can be accessed via Session Manager for interactive debugging

---

## Providers

### Which provider should I choose?

| Use Case | Recommended Provider |
|----------|---------------------|
| GPU training | AWS (widest GPU selection) |
| H100/A100 availability | Verda (specialized GPU cloud) |
| Multi-cloud redundancy | Mix providers as needed |

### Can I use multiple providers in the same workflow?

Yes, with `MultiPool`:

```python
with sky.MultiPool(
    sky.ComputePool(provider=sky.AWS(), accelerator="A100"),
    sky.ComputePool(provider=sky.Verda(), accelerator="H100"),
) as (aws_pool, verda_pool):
    result_aws = train() >> aws_pool
    result_verda = train() >> verda_pool
```

---

## Distributed Training

### How does Skyward set up distributed training?

Skyward automatically configures:
- `MASTER_ADDR`, `MASTER_PORT` for PyTorch
- `JAX_COORDINATOR_ADDRESS` for JAX
- `TF_CONFIG` for TensorFlow
- Security groups for inter-node communication

Use integration decorators for framework-specific setup:

```python
@sky.integrations.torch(backend="nccl")
@sky.compute
def train():
    import torch.distributed as dist
    # dist.is_initialized() is True
```

### How do I partition data across nodes?

Use `shard()` for simple partitioning:

```python
@sky.compute
def train(x_full, y_full):
    x_local, y_local = sky.shard(x_full, y_full, shuffle=True, seed=42)
    # Each node gets its portion
```

Or `DistributedSampler` for PyTorch DataLoaders:

```python
sampler = sky.DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler)
```

---

## Related Topics

- [Getting Started](getting-started.md)
- [Troubleshooting](troubleshooting.md)
- [API Reference](api-reference.md)
