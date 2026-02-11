"""Dynamic metrics collection configuration.

Define which metrics to collect and their sampling intervals.
Metrics are collected on remote instances and streamed back as events.

Example:
    import skyward as sky

    # Default metrics (CPU, Memory, GPU)
    pool = ComputePool(
        provider=AWS(),
        metrics=sky.metrics.Default(),
    )

    # Custom selection
    pool = ComputePool(
        metrics=[
            sky.metrics.CPU(interval=0.5),
            sky.metrics.GPU(interval=2.0),
            sky.metrics.Disk("/data"),
        ],
    )

    # Disable metrics collection
    pool = ComputePool(metrics=None)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metric:
    """Definition of a metric collector.

    Each metric runs as a background process on the remote instance,
    executing a shell command at the specified interval and emitting
    the result as a metric event.

    Attributes:
        name: Metric name (e.g., "cpu", "gpu_util_0"). Used as identifier in events.
        command: Shell command that outputs a numeric value to stdout.
        interval: Seconds between collections.
        multi: If True, command may output multiple lines (one per GPU, etc).
            Each line becomes a separate metric: name_0, name_1, etc.
    """

    name: str
    command: str
    interval: float = 2
    multi: bool = False


# =============================================================================
# CPU Metrics
# =============================================================================


def CPU(interval: float = 2) -> Metric:
    """CPU utilization percentage (0-100).

    Reads from /proc/stat to calculate overall CPU usage.

    Args:
        interval: Seconds between samples.

    Returns:
        Metric that emits "cpu" values.
    """
    return Metric(
        name="cpu",
        command="awk '/^cpu / {printf \"%.1f\", ($2+$4)*100/($2+$4+$5)}' /proc/stat",
        interval=interval,
    )


# =============================================================================
# Memory Metrics
# =============================================================================


def Memory(interval: float = 2) -> Metric:
    """Memory utilization percentage (0-100).

    Args:
        interval: Seconds between samples.

    Returns:
        Metric that emits "mem" values.
    """
    return Metric(
        name="mem",
        command="free | awk '/^Mem:/ {printf \"%.1f\", $3/$2*100}'",
        interval=interval,
    )


def MemoryUsed(interval: float = 2) -> Metric:
    """Memory used in megabytes.

    Args:
        interval: Seconds between samples.

    Returns:
        Metric that emits "mem_used_mb" values.
    """
    return Metric(
        name="mem_used_mb",
        command="free | awk '/^Mem:/ {printf \"%d\", $3/1024}'",
        interval=interval,
    )


def MemoryTotal() -> Metric:
    """Total memory in megabytes.

    Sampled infrequently since total memory doesn't change.

    Returns:
        Metric that emits "mem_total_mb" values every 60s.
    """
    return Metric(
        name="mem_total_mb",
        command="free | awk '/^Mem:/ {printf \"%d\", $2/1024}'",
        interval=60.0,
    )


# =============================================================================
# GPU Metrics
# =============================================================================


def _gpu_metric(
    name: str,
    query: str,
    index: int | None,
    interval: float,
) -> Metric:
    """Factory for GPU metrics with optional index.

    Internal helper that creates nvidia-smi based metrics.

    Args:
        name: Base metric name (e.g., "gpu_util", "gpu_mem_mb").
        query: nvidia-smi query field (e.g., "utilization.gpu", "memory.used").
        index: GPU index (0-based). If None, collects from ALL GPUs.
        interval: Seconds between samples.

    Returns:
        Metric configured for the specified GPU query.
    """
    if index is not None:
        return Metric(
            name=f"{name}_{index}",
            command=(
                f"nvidia-smi -i {index} --query-gpu={query} "
                f"--format=csv,noheader,nounits 2>/dev/null | tr -d ' '"
            ),
            interval=interval,
        )
    return Metric(
        name=name,
        command=(
            f"nvidia-smi --query-gpu={query} "
            f"--format=csv,noheader,nounits 2>/dev/null | tr -d ' '"
        ),
        interval=interval,
        multi=True,
    )


def GPU(index: int | None = None, interval: float = 3) -> Metric:
    """GPU utilization percentage (0-100).

    Uses nvidia-smi to query GPU utilization. Fails silently if no GPU.

    Args:
        index: GPU index (0-based). If None, collects from ALL GPUs,
            emitting gpu_util_0, gpu_util_1, etc.
        interval: Seconds between samples.

    Returns:
        Metric that emits "gpu_util" or "gpu_util_{index}" values.
    """
    return _gpu_metric("gpu_util", "utilization.gpu", index, interval)


def GPUMemory(index: int | None = None, interval: float = 3) -> Metric:
    """GPU memory used in megabytes.

    Args:
        index: GPU index (0-based). If None, collects from ALL GPUs.
        interval: Seconds between samples.

    Returns:
        Metric that emits "gpu_mem_mb" or "gpu_mem_mb_{index}" values.
    """
    return _gpu_metric("gpu_mem_mb", "memory.used", index, interval)


def GPUMemoryTotal(index: int | None = None) -> Metric:
    """Total GPU memory in megabytes.

    Sampled infrequently since total memory doesn't change.

    Args:
        index: GPU index (0-based). If None, collects from ALL GPUs.

    Returns:
        Metric that emits "gpu_mem_total_mb" values every 60s.
    """
    return _gpu_metric("gpu_mem_total_mb", "memory.total", index, 60.0)


def GPUTemp(index: int | None = None, interval: float = 3) -> Metric:
    """GPU temperature in Celsius.

    Args:
        index: GPU index (0-based). If None, collects from ALL GPUs.
        interval: Seconds between samples.

    Returns:
        Metric that emits "gpu_temp" or "gpu_temp_{index}" values.
    """
    return _gpu_metric("gpu_temp", "temperature.gpu", index, interval)


# =============================================================================
# Disk Metrics
# =============================================================================


def Disk(path: str = "/", interval: float = 5.0) -> Metric:
    """Disk usage percentage for a path.

    Args:
        path: Filesystem path to check (e.g., "/", "/data").
        interval: Seconds between samples. Default 5.0 since disk
            usage changes slowly.

    Returns:
        Metric that emits "disk_{path}" values.
    """
    safe_name = path.replace("/", "_").strip("_") or "root"
    return Metric(
        name=f"disk_{safe_name}",
        command=f"df {path} 2>/dev/null | tail -1 | awk '{{print $5}}' | tr -d '%'",
        interval=interval,
    )


# =============================================================================
# Network Metrics
# =============================================================================


def NetworkRx(interface: str = "eth0", interval: float = 3) -> Metric:
    """Network bytes received (cumulative).

    Args:
        interface: Network interface name.
        interval: Seconds between samples.

    Returns:
        Metric that emits "net_rx_{interface}" values.
    """
    return Metric(
        name=f"net_rx_{interface}",
        command=f"cat /sys/class/net/{interface}/statistics/rx_bytes 2>/dev/null",
        interval=interval,
    )


def NetworkTx(interface: str = "eth0", interval: float = 3) -> Metric:
    """Network bytes transmitted (cumulative).

    Args:
        interface: Network interface name.
        interval: Seconds between samples.

    Returns:
        Metric that emits "net_tx_{interface}" values.
    """
    return Metric(
        name=f"net_tx_{interface}",
        command=f"cat /sys/class/net/{interface}/statistics/tx_bytes 2>/dev/null",
        interval=interval,
    )


# =============================================================================
# Custom Metrics
# =============================================================================


def Custom(name: str, command: str, interval: float = 3) -> Metric:
    """Create a custom metric from any shell command.

    The command should output a single numeric value to stdout.
    If the command fails or outputs nothing, the metric is not emitted.

    Args:
        name: Metric identifier (alphanumeric + underscore recommended).
        command: Shell command that outputs a numeric value.
        interval: Seconds between samples.

    Returns:
        Custom Metric.

    Example:
        # Count open file descriptors
        Custom(
            name="open_fds",
            command="ls /proc/self/fd | wc -l",
            interval=5.0,
        )

        # GPU power draw in watts
        Custom(
            name="gpu_power_watts",
            command="nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1",
            interval=2.0,
        )
    """
    return Metric(name=name, command=command, interval=interval)


# =============================================================================
# Default Configuration
# =============================================================================


def Default(
    *,
    cpu_interval: float = 2,
    memory_interval: float = 2,
    gpu_interval: float = 3,
) -> tuple[Metric, ...]:
    """Default metrics set: CPU, Memory, and GPU (if available).

    Provides a sensible default for most use cases. GPU metrics are
    collected but gracefully ignored if nvidia-smi is not available.

    Args:
        cpu_interval: Interval for CPU metrics.
        memory_interval: Interval for memory metrics.
        gpu_interval: Interval for GPU metrics.

    Returns:
        Tuple of default metrics.

    Example:
        # Use defaults
        pool = ComputePool(metrics=sky.metrics.Default())

        # Custom intervals
        pool = ComputePool(
            metrics=sky.metrics.Default(cpu_interval=0.5, gpu_interval=2.0)
        )
    """
    return (
        CPU(cpu_interval),
        Memory(memory_interval),
        MemoryUsed(memory_interval),
        MemoryTotal(),
        GPU(interval=gpu_interval),
        GPUMemory(interval=gpu_interval),
        GPUMemoryTotal(),
        GPUTemp(interval=gpu_interval),
    )


# =============================================================================
# Type Aliases
# =============================================================================

type MetricsConfig = tuple[Metric, ...] | list[Metric] | None
"""Type for metrics configuration in ComputePool.

- tuple[Metric, ...] or list[Metric]: Specific metrics to collect
- None: Disable metrics collection entirely
"""
