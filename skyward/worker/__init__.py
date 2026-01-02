"""Worker isolation module for Skyward.

This module provides worker-level resource isolation using:
- cgroups v2 for CPU/memory limits
- MIG (Multi-Instance GPU) for GPU partitioning
- Multiple RPyC servers per instance

API:
    accelerator="H100:G:W"
           │        │  │
           │        │  └── W = workers (default 1)
           │        └── G = GPUs totais (default 1)
           └── Tipo de GPU

Examples:
    "H100"       → 1 GPU, 1 worker
    "H100:8"     → 8 GPUs, 1 worker (data parallel)
    "H100:8:8"   → 8 GPUs, 8 workers (1 GPU each)
    "H100:8:16"  → 8 GPUs, 16 workers (MIG 0.5 each)
"""

from skyward.worker.config import ResourceLimits, WorkerConfig, generate_worker_configs
from skyward.worker.partition import PartitionStrategy, create_partition
from skyward.worker.pool import Worker, WorkerPool

__all__ = [
    # Config
    "ResourceLimits",
    "WorkerConfig",
    "generate_worker_configs",
    # Partition
    "PartitionStrategy",
    "create_partition",
    # Pool
    "Worker",
    "WorkerPool",
]
