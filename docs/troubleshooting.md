# Troubleshooting

Common issues and solutions when using Skyward.

## Provisioning Issues

### "No instances available"

Your requested configuration isn't available in the region.

**Solutions:**
1. Try a different region
2. Use `allocation="spot-if-available"` to fallback to on-demand
3. Request a service quota increase from your cloud provider
4. Try a different accelerator type

```python
# Fallback to on-demand if spot unavailable
sky.ComputePool(
    provider=sky.AWS(),
    accelerator="A100",
    allocation="spot-if-available",  # Instead of "always"
)
```

### "Permission denied"

Your cloud credentials lack required permissions.

**AWS:** Check your IAM permissions. Skyward needs:
- `ec2:RunInstances`, `ec2:TerminateInstances`, `ec2:DescribeInstances`
- `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress`
- `iam:PassRole` (for instance profiles)
- `ssm:*` (for Session Manager connectivity)

**DigitalOcean:** Verify your token at https://cloud.digitalocean.com/account/api (read/write access needed)

**Verda:** Check your API key and account permissions

### "Bootstrap timeout"

The instance took too long to set up.

**Causes:**
- Large pip dependencies
- Slow network in certain regions
- Instance type limitations

**Solutions:**
1. Increase the timeout:
   ```python
   sky.ComputePool(provider=sky.AWS(), timeout=7200)  # 2 hours
   ```
2. Reduce dependencies in your `Image`
3. Try a different region with better network performance

### Connection Issues (AWS)

AWS uses SSM (Session Manager) by default for reliable connectivity.

**If you experience connection issues:**
1. Ensure your AWS account has SSM access enabled
2. Verify the instance has outbound internet access for SSM endpoint communication
3. Check that IAM permissions include `AmazonSSMManagedInstanceCore`
4. Ensure VPC has SSM endpoints or NAT gateway

---

## Distributed Training Issues

### NCCL Timeout

Multi-node training fails with NCCL timeout errors.

**Solutions:**
1. Increase NCCL timeout:
   ```python
   import os
   os.environ["NCCL_SOCKET_TIMEOUT"] = "600"
   os.environ["NCCL_DEBUG"] = "INFO"
   ```

2. Check security group allows:
   - TCP port 29500 (MASTER_PORT)
   - All high ports for NCCL (1024-65535)

3. Verify all nodes can communicate with each other

### Process Group Not Initialized

`torch.distributed.is_initialized()` returns False.

**Solutions:**
1. Use the `@sky.integrations.torch()` decorator:
   ```python
   @sky.integrations.torch(backend="nccl")
   @sky.compute
   def train():
       import torch.distributed as dist
       # dist.is_initialized() is now True
   ```

2. Check that you're using `@ pool` (broadcast) not `>> pool`:
   ```python
   # Correct: broadcast to all nodes
   results = train() @ pool

   # Incorrect: only runs on one node
   result = train() >> pool
   ```

3. Single-node fallback:
   ```python
   import torch.distributed as dist

   if not dist.is_initialized():
       # Single-node fallback
       model = MyModel()
   else:
       model = DDP(MyModel())
   ```

### Memory Issues

Training runs out of GPU memory.

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable()
   ```
3. Enable mixed precision (fp16):
   ```python
   # PyTorch
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       output = model(input)

   # HuggingFace
   TrainingArguments(fp16=True)
   ```
4. Use a larger GPU or add more nodes

---

## GPU Issues

### "No instances with accelerator X available"

The requested GPU isn't available in your region or provider.

**Solutions:**
1. Check GPU availability for your region:
   ```python
   for instance in sky.AWS().available_instances():
       if instance.accelerator:
           print(f"{instance.name}: {instance.accelerator}")
   ```
2. Try a different region
3. Request a service quota increase

### "CUDA out of memory"

GPU memory exhausted during computation.

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Use a larger GPU or MIG partition with more memory
4. Clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### "MIG partition failed"

MIG partitioning didn't work as expected.

**Solutions:**
1. Ensure GPU supports MIG (H100 and A100 only)
2. Check profile compatibility with GPU memory:
   - H100/A100-80GB: `1g.10gb`, `2g.20gb`, `3g.40gb`, etc.
   - A100-40GB: `1g.5gb`, `2g.10gb`, `3g.20gb`
3. Verify no other processes are using the GPU

---

## Provider-Specific Issues

### AWS: Quota Limits

AWS service quotas limit instance launches.

**Solutions:**
1. Request quota increase in AWS console
2. Try a different region (quotas are per-region)
3. Use different instance types

### DigitalOcean: Authentication Failed

**Solutions:**
1. Verify token at https://cloud.digitalocean.com/account/api
2. Check token permissions (read/write access needed)
3. Regenerate token if expired

### Verda: Region Not Available

**Solutions:**
1. Remove the `region` parameter to enable auto-discovery:
   ```python
   sky.ComputePool(provider=sky.Verda())  # Auto-discovers
   ```
2. Check your account's region access

---

## Debugging Tips

### Enable Verbose Logging

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

### Use Event Callbacks

Monitor what's happening during execution:

```python
def debug_callback(event):
    print(f"[{type(event).__name__}] {event}")

with sky.ComputePool(
    provider=sky.AWS(),
    on_event=debug_callback,
) as pool:
    result = my_function() >> pool
```

### Test Locally First

Always test your function locally before running remotely:

```python
@sky.compute
def my_function(data):
    # Your code
    return result

# Test locally
result = my_function.local(test_data)
print(result)

# Then run remotely
with sky.ComputePool(...) as pool:
    result = my_function(data) >> pool
```

### Check Instance Logs

For AWS with SSM, you can access instance logs:

```bash
aws ssm start-session --target i-0abc123def456
```

---

## Getting Help

If you're still stuck:

1. **Check the documentation**: [API Reference](api-reference.md), [Concepts](concepts.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/gabfssilva/skyward/issues)
3. **Open a new issue** with:
   - Skyward version
   - Python version
   - Full error message and stack trace
   - Minimal reproducible example

---

## Related Topics

- [Getting Started](getting-started.md)
- [FAQ](faq.md)
- [Providers](providers.md)
- [Accelerators](accelerators.md)
