"""Metrics configuration for Skyward v2.

Declarative metric definitions for collecting system and GPU metrics
on remote instances. Metrics are collected as background processes
and streamed as events via the event bus.

Usage:
    import skyward.v2 as sky

    # Default metrics (CPU, Memory, GPU)
    image = sky.Image()

    # Custom metrics with specific intervals
    image = sky.Image(
        metrics=[
            sky.metrics.CPU(interval=0.5),
            sky.metrics.GPU(),
            sky.metrics.Memory(),
            sky.metrics.Disk("/data"),
        ]
    )

    # Disable metrics collection
    image = sky.Image(metrics=None)

    # Custom command metric
    image = sky.Image(
        metrics=[
            sky.metrics.Custom(
                name="gpu_power",
                command="nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                interval=2.0,
            ),
        ]
    )

Available metrics:
    CPU(interval=2)          - CPU utilization percentage
    Memory(interval=2)       - Memory utilization percentage
    MemoryUsed(interval=2)   - Memory used in MB
    MemoryTotal()            - Total memory in MB
    GPU(index=None, interval=3)  - GPU utilization percentage
    GPUMemory(index=None, interval=3)  - GPU memory used in MB
    GPUMemoryTotal(index=None)  - Total GPU memory in MB
    GPUTemp(index=None, interval=3)  - GPU temperature in Celsius
    Disk(path="/", interval=5)  - Disk usage percentage
    NetworkRx(interface="eth0", interval=3)  - Bytes received
    NetworkTx(interface="eth0", interval=3)  - Bytes transmitted
    Custom(name, command, interval=3)  - Custom shell command
    Default(cpu_interval=2, memory_interval=2, gpu_interval=3)  - Default set
"""

from __future__ import annotations

# Re-export V1 metric specs (no SDK dependencies at import time)
from skyward.spec.metrics import (
    # Core type
    Metric,
    # CPU
    CPU,
    # Memory
    Memory,
    MemoryUsed,
    MemoryTotal,
    # GPU
    GPU,
    GPUMemory,
    GPUMemoryTotal,
    GPUTemp,
    # Disk
    Disk,
    # Network
    NetworkRx,
    NetworkTx,
    # Custom
    Custom,
    # Default set
    Default,
    # Type alias
    MetricsConfig,
)

__all__ = [
    # Core type
    "Metric",
    # CPU
    "CPU",
    # Memory
    "Memory",
    "MemoryUsed",
    "MemoryTotal",
    # GPU
    "GPU",
    "GPUMemory",
    "GPUMemoryTotal",
    "GPUTemp",
    # Disk
    "Disk",
    # Network
    "NetworkRx",
    "NetworkTx",
    # Custom
    "Custom",
    # Default
    "Default",
    # Type alias
    "MetricsConfig",
]
