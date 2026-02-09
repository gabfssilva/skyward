# For HPC Users

If you're coming from traditional HPC—SLURM, MPI, shared clusters—Skyward may solve your problem too.

The core model is different: instead of submitting jobs to a scheduler and waiting for allocation, you provision ephemeral clusters on demand. They exist for your workload and shut down when done.

## Concept Mapping

| HPC | Skyward |
|-----|---------|
| `sbatch` / `srun` | `@sky.pool()` |
| MPI rank | `instance_info().node` |
| MPI world size | `instance_info().total_nodes` |
| Rank 0 | `is_head` |
| Data decomposition | `shard()` |
| Module loads | `Image()` |

## What Carries Over

Distributed training patterns work the same way. You still have a head node coordinating workers, you still shard data across ranks, you still synchronize gradients. The difference is how you get there.

Skyward sets up the environment variables that frameworks expect—`MASTER_ADDR`, `RANK`, `WORLD_SIZE` for PyTorch, `JAX_COORDINATOR_ADDRESS` for JAX, `TF_CONFIG` for TensorFlow. The framework handles communication via NCCL or Gloo, same as on a traditional cluster.

```python
@sky.pool(provider=sky.VastAI(), accelerator="A100", nodes=4)
def main():
    train() @ sky  # runs on all 4 nodes
```

This is roughly equivalent to:

```bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
srun python train.py
```

## What Doesn't Carry Over

Skyward doesn't wrap MPI. If your code uses `mpi4py` or raw MPI calls, you'll need to set that up yourself or refactor to use framework-native collectives.

There's no job queue—clusters are provisioned immediately or fail. No waiting, but also no backlog management.

Interconnect is cloud networking, not InfiniBand. For most ML workloads this is fine—gradient synchronization is bursty, not sustained. Workloads that require tight coupling (large all-to-all, frequent small messages) will feel the difference.

Checkpoint/restart isn't built in. Spot instances get replaced automatically on preemption, but your training code needs to handle saving and resuming state.

## Trade-offs

| | HPC | Skyward |
|-|-----|---------|
| Queue wait | Minutes to days | None (subject to provider availability) |
| Latest GPUs | Depends on facility | Yes |
| Interconnect | InfiniBand | Cloud networking |
| Cost model | Allocation / grants | Pay-per-use |
| Setup | Modules, MPI config | Decorators |
| Fault tolerance | Checkpoint/restart | Spot replacement |
| Persistence | Shared filesystem | Ephemeral by default |

## When This Matters

If you're training models with PyTorch, JAX, or TensorFlow and your bottleneck is getting access to GPUs, Skyward removes the queue. If your code is MPI-native or needs InfiniBand bandwidth, traditional HPC is still the right tool.
