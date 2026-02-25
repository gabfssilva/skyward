# For HPC Users

If you're coming from traditional HPC — SLURM, MPI, shared filesystems, queue-based scheduling — Skyward solves the same fundamental problem through a different model. Instead of submitting jobs to a scheduler and waiting for allocation on a shared cluster, you provision ephemeral clusters on demand from commercial cloud providers. They exist for the duration of your workload and shut down when done.

The concepts map directly. What SLURM calls a "rank" is `instance_info().node`. What MPI calls `MPI_COMM_WORLD.Get_size()` is `instance_info().total_nodes`. Rank 0 is `is_head`. Data decomposition across ranks becomes `shard()`. Module loads and environment setup become the `Image()` specification. The distributed training patterns — head node coordinating workers, data sharded across ranks, gradients synchronized — are identical. The difference is how you get there.

## Environment Setup

In a traditional HPC environment, you'd configure the compute environment through module loads, environment variables in your batch script, and MPI launch commands. Skyward replaces all of this with the `Image` and plugins.

```python
import skyward as sky

with sky.ComputePool(provider=sky.VastAI(), accelerator=sky.accelerators.A100(), nodes=4) as pool:
    train() @ pool  # runs on all 4 nodes
```

This is roughly equivalent to:

```bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
srun python train.py
```

Skyward sets up the environment variables that frameworks expect — `MASTER_ADDR`, `RANK`, `WORLD_SIZE` for PyTorch, `JAX_COORDINATOR_ADDRESS` for JAX, `TF_CONFIG` for TensorFlow. The framework handles communication via NCCL or Gloo, same as on a traditional cluster. If you've written DDP training code for SLURM, the training loop is identical — only the launch mechanism changes.

## What Carries Over

Your distributed training patterns work unchanged. You still have a head node coordinating workers, you still shard data across ranks, you still synchronize gradients. The framework-level code — `DistributedDataParallel`, `all_reduce`, `DistributedSampler`, `jax.pmap` — is the same whether it runs on a SLURM cluster or a Skyward pool. If you've spent time learning these APIs, that knowledge transfers directly.

Data sharding patterns also carry over. `sky.shard()` does modulo striding across ranks, which is the same approach as `DistributedSampler` or manual MPI-based decomposition. If you're doing domain decomposition or custom data splitting, `instance_info()` gives you the same rank/world-size information you'd get from `MPI_Comm_rank()` and `MPI_Comm_size()`.

## What Doesn't Carry Over

Skyward doesn't wrap MPI. If your code uses `mpi4py` or raw MPI calls for inter-process communication, you'll need to either configure MPI yourself on the remote instances or refactor to use framework-native collectives (NCCL for PyTorch, XLA for JAX). For most ML workloads this isn't a limitation — DDP, JAX distributed, and Keras distribution strategies handle the collectives you need — but tightly-coupled numerical simulations that depend on MPI's point-to-point messaging or custom communicators won't work out of the box.

There's no job queue. SLURM manages a backlog of jobs, prioritizes them, and allocates resources as they become available. Skyward provisions immediately or fails — there's no waiting, but also no backlog management. If the cloud provider can't give you the instances you need right now, you get an error rather than a queue position.

Interconnect depends on the provider. Some cloud instances offer InfiniBand or equivalent — AWS EFA on p5 instances provides 3200 Gbps networking, and some providers offer bare-metal InfiniBand. Others use standard cloud networking. For most ML workloads the difference is negligible: gradient synchronization is bursty (short spikes during all-reduce between training steps), not sustained, so even moderate bandwidth handles it well. Workloads that require tight coupling — large all-to-all operations, frequent small messages, lattice QCD, molecular dynamics — should target providers and instance types with high-bandwidth interconnect.

Checkpoint/restart isn't built in. SLURM environments typically have a shared filesystem where checkpoints persist across job submissions. Skyward clusters are ephemeral — when the pool exits, the instances are gone. Your training code needs to save checkpoints to persistent storage (S3, GCS, NFS, or a mounted volume) and resume from them if interrupted. Spot instances get replaced automatically on preemption, but your code is responsible for saving and loading state.

## The Trade-Off

The core trade-off is **queue wait vs. cost**. On a shared HPC cluster, you don't pay per-hour, but you wait for allocation — sometimes minutes, sometimes days, depending on the cluster's utilization and your priority in the scheduler. With Skyward, you get immediate access to the latest GPUs (H100, H200, MI300X) from multiple providers, but you pay for the time you use. For teams where GPU access is the bottleneck — waiting in SLURM queues, competing for GPU time on shared clusters — eliminating the queue can dramatically accelerate iteration cycles. For teams with free access to well-provisioned clusters, the cost model may not be justified.

The other significant trade-off is **persistence vs. ephemerality**. HPC clusters have shared filesystems that persist between jobs — your data, code, and checkpoints are always there. Skyward clusters are ephemeral by design: provision, compute, tear down. This means you need to explicitly manage data movement (syncing code via `includes`, loading data from cloud storage, saving checkpoints externally), but it also means there are no environments to maintain, no software versions to keep in sync across nodes, and no idle costs.

## When This Matters

If you're training models with PyTorch, JAX, or TensorFlow and your bottleneck is getting access to accelerators — waiting in SLURM queues, competing for GPU time on shared clusters — Skyward removes the queue. Your training code stays almost identical; only the launch mechanism changes.

If your code is MPI-native and tightly coupled, or if you have free access to a well-provisioned cluster with InfiniBand, traditional HPC may still be the right tool. But cloud instances with EFA, InfiniBand, and the latest GPU generations are closing the gap fast.
