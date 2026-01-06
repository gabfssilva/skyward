"""Modular metrics collectors for server-side streaming.

This module provides lightweight metrics collection using:
- procfs for CPU and memory (no external dependencies)
- pynvml for NVIDIA GPUs (~1ms per collection)
- rocm-smi for AMD GPUs (subprocess fallback)
- sysfs/neuron-monitor for AWS Trainium/Inferentia
- hl-smi for Habana Gaudi accelerators

The MetricsStream class orchestrates collection and provides a generator
for efficient streaming via RPyC.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import astuple, dataclass, field
from typing import Any, Protocol


class AcceleratorCollector(Protocol):
    """Protocol for accelerator metrics collectors."""

    def is_available(self) -> bool:
        """Check if this accelerator type is present."""
        ...

    def collect(self) -> dict[str, float | None]:
        """Collect accelerator metrics."""
        ...


# =============================================================================
# System Metrics (procfs)
# =============================================================================


@dataclass(frozen=True, slots=True)
class CPUState:
    """CPU time snapshot for delta calculation."""

    user: int
    nice: int
    system: int
    idle: int
    iowait: int
    irq: int
    softirq: int
    steal: int


def read_cpu_state() -> CPUState:
    """Read CPU state from /proc/stat.

    Returns aggregate CPU times across all cores.
    """
    with open("/proc/stat") as f:
        parts = f.readline().split()
    return CPUState(
        user=int(parts[1]),
        nice=int(parts[2]),
        system=int(parts[3]),
        idle=int(parts[4]),
        iowait=int(parts[5]),
        irq=int(parts[6]),
        softirq=int(parts[7]),
        steal=int(parts[8]) if len(parts) > 8 else 0,
    )


def cpu_percent(prev: CPUState, curr: CPUState) -> float:
    """Calculate CPU usage percentage from delta between two snapshots."""
    prev_idle = prev.idle + prev.iowait
    curr_idle = curr.idle + curr.iowait
    prev_total = sum(astuple(prev))
    curr_total = sum(astuple(curr))

    delta_idle = curr_idle - prev_idle
    delta_total = curr_total - prev_total

    if delta_total == 0:
        return 0.0
    return 100.0 * (1.0 - delta_idle / delta_total)


def read_memory() -> tuple[float, float, float]:
    """Read memory from /proc/meminfo.

    Returns:
        Tuple of (percent_used, used_mb, total_mb).
    """
    with open("/proc/meminfo") as f:
        mem = {}
        for line in f:
            parts = line.split()
            mem[parts[0].rstrip(":")] = int(parts[1])

    total = mem["MemTotal"]
    avail = mem.get("MemAvailable", mem["MemFree"])
    used = total - avail

    return (
        100.0 * used / total if total else 0.0,
        used / 1024,
        total / 1024,
    )


# =============================================================================
# Accelerator Collectors
# =============================================================================


class NvidiaCollector:
    """NVIDIA GPU metrics via pynvml (~1ms per collection)."""

    __slots__ = ("_initialized", "_device_count")

    def __init__(self) -> None:
        self._initialized = False
        self._device_count = 0

    def is_available(self) -> bool:
        """Check if NVIDIA GPUs are available via pynvml."""
        if self._initialized:
            return self._device_count > 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._initialized = True
            return self._device_count > 0
        except Exception:
            self._initialized = True
            return False

    def collect(self) -> dict[str, float | None]:
        """Collect GPU metrics aggregated across all devices."""
        if not self.is_available():
            return {}

        import pynvml

        total_util = 0.0
        total_mem_used = 0
        total_mem_total = 0
        max_temp = 0.0

        for i in range(self._device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            total_util += util.gpu
            total_mem_used += mem.used
            total_mem_total += mem.total
            max_temp = max(max_temp, temp)

        return {
            "gpu_utilization": total_util / self._device_count,
            "gpu_memory_used_mb": total_mem_used / (1024 * 1024),
            "gpu_memory_total_mb": total_mem_total / (1024 * 1024),
            "gpu_temperature": max_temp,
        }


class AMDCollector:
    """AMD GPU metrics via rocm-smi subprocess."""

    __slots__ = ("_available",)

    def __init__(self) -> None:
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if AMD GPUs are available via rocm-smi."""
        if self._available is not None:
            return self._available
        try:
            import subprocess

            r = subprocess.run(
                ["rocm-smi", "-L"],
                capture_output=True,
                timeout=5,
            )
            self._available = r.returncode == 0
        except Exception:
            self._available = False
        return self._available

    def collect(self) -> dict[str, float | None]:
        """Collect AMD GPU metrics via rocm-smi."""
        if not self.is_available():
            return {}

        import subprocess

        try:
            # Get utilization
            r = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode != 0:
                return {}

            import json

            data = json.loads(r.stdout)

            # Parse JSON output (structure depends on rocm-smi version)
            total_util = 0.0
            total_mem_used = 0.0
            total_mem_total = 0.0
            max_temp = 0.0
            gpu_count = 0

            for key, gpu_data in data.items():
                if not key.startswith("card"):
                    continue
                gpu_count += 1

                # GPU utilization (percentage)
                if "GPU use (%)" in gpu_data:
                    total_util += float(gpu_data["GPU use (%)"])

                # Memory (varies by version)
                if "GPU memory use (%)" in gpu_data:
                    mem_pct = float(gpu_data["GPU memory use (%)"])
                    # Estimate from percentage if absolute not available
                    total_mem_used += mem_pct

                # Temperature
                if "Temperature (Sensor edge) (C)" in gpu_data:
                    temp = float(gpu_data["Temperature (Sensor edge) (C)"])
                    max_temp = max(max_temp, temp)

            if gpu_count == 0:
                return {}

            return {
                "gpu_utilization": total_util / gpu_count,
                "gpu_memory_used_mb": total_mem_used,  # May need adjustment
                "gpu_memory_total_mb": total_mem_total,
                "gpu_temperature": max_temp,
            }

        except Exception:
            return {}


class NeuronCollector:
    """AWS Trainium/Inferentia metrics via sysfs and neuron-monitor."""

    __slots__ = ("_available", "_device_count")

    def __init__(self) -> None:
        self._available: bool | None = None
        self._device_count = 0

    def is_available(self) -> bool:
        """Check if Neuron devices are available via sysfs."""
        if self._available is not None:
            return self._available
        try:
            from pathlib import Path

            devices = list(Path("/sys/devices/virtual/neuron_device").glob("neuron*"))
            self._device_count = len(devices)
            self._available = self._device_count > 0
        except Exception:
            self._available = False
        return self._available

    def collect(self) -> dict[str, float | None]:
        """Collect Neuron accelerator metrics."""
        if not self.is_available():
            return {}

        import subprocess

        try:
            # Use neuron-monitor for metrics (JSON output)
            r = subprocess.run(
                ["neuron-monitor", "-c", "1", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode != 0:
                return {}

            import json

            data = json.loads(r.stdout)

            # Parse neuron-monitor JSON structure
            total_util = 0.0
            total_mem_used = 0.0
            total_mem_total = 0.0
            core_count = 0

            for instance in data.get("neuron_runtime_data", []):
                for nc in instance.get("neuron_cores", []):
                    core_count += 1
                    total_util += nc.get("utilization", {}).get("total", 0)
                    mem = nc.get("memory", {})
                    total_mem_used += mem.get("used_bytes", 0)
                    total_mem_total += mem.get("total_bytes", 0)

            if core_count == 0:
                return {}

            return {
                "gpu_utilization": total_util / core_count,
                "gpu_memory_used_mb": total_mem_used / (1024 * 1024),
                "gpu_memory_total_mb": total_mem_total / (1024 * 1024),
                "gpu_temperature": None,  # Neuron doesn't expose temperature
            }

        except Exception:
            return {}


class HabanaCollector:
    """Habana Gaudi accelerator metrics via hl-smi."""

    __slots__ = ("_available",)

    def __init__(self) -> None:
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if Habana Gaudi accelerators are available."""
        if self._available is not None:
            return self._available
        try:
            import subprocess

            r = subprocess.run(
                ["hl-smi", "-L"],
                capture_output=True,
                timeout=5,
            )
            self._available = r.returncode == 0
        except Exception:
            self._available = False
        return self._available

    def collect(self) -> dict[str, float | None]:
        """Collect Habana Gaudi metrics via hl-smi."""
        if not self.is_available():
            return {}

        import subprocess

        try:
            # hl-smi supports JSON output
            r = subprocess.run(
                [
                    "hl-smi",
                    "--query-aip=utilization.aip,memory.used,memory.total,temperature.aip",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode != 0:
                return {}

            total_util = 0.0
            total_mem_used = 0.0
            total_mem_total = 0.0
            max_temp = 0.0
            device_count = 0

            for line in r.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    device_count += 1
                    total_util += float(parts[0])
                    total_mem_used += float(parts[1])
                    total_mem_total += float(parts[2])
                    max_temp = max(max_temp, float(parts[3]))

            if device_count == 0:
                return {}

            return {
                "gpu_utilization": total_util / device_count,
                "gpu_memory_used_mb": total_mem_used,
                "gpu_memory_total_mb": total_mem_total,
                "gpu_temperature": max_temp,
            }

        except Exception:
            return {}


# =============================================================================
# MetricsStream (Orchestrator)
# =============================================================================


@dataclass
class MetricsStream:
    """Server-side metrics streaming orchestrator.

    Manages CPU state for delta calculation and auto-detects the appropriate
    accelerator collector based on available hardware.

    Thread-safe: multiple connections can safely call collect() concurrently.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cpu_state: CPUState | None = field(default=None, init=False)
    _accelerator: AcceleratorCollector | None = field(default=None, init=False)
    _accelerator_type: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._detect_accelerator()
        # Prime CPU state for first delta calculation
        self._cpu_state = read_cpu_state()

    def _detect_accelerator(self) -> None:
        """Auto-detect available accelerator (priority: NVIDIA > AMD > Neuron > Habana)."""
        collectors = [
            ("nvidia", NvidiaCollector()),
            ("amd", AMDCollector()),
            ("neuron", NeuronCollector()),
            ("habana", HabanaCollector()),
        ]

        for name, collector in collectors:
            if collector.is_available():
                self._accelerator = collector
                self._accelerator_type = name
                return

    @property
    def accelerator_type(self) -> str | None:
        """Return detected accelerator type, or None if no accelerator."""
        return self._accelerator_type

    def collect(self) -> dict[str, Any]:
        """Collect all metrics (system + accelerator).

        Returns dict with:
            - cpu_percent: float
            - memory_percent: float
            - memory_used_mb: float
            - memory_total_mb: float
            - gpu_utilization: float | None
            - gpu_memory_used_mb: float | None
            - gpu_memory_total_mb: float | None
            - gpu_temperature: float | None
            - _timing_*: timing diagnostics (temporary)
        """
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # CPU requires delta calculation (thread-safe)
        with self._lock:
            curr = read_cpu_state()
            cpu = cpu_percent(self._cpu_state, curr) if self._cpu_state else 0.0
            self._cpu_state = curr
        timings["cpu_ms"] = (time.perf_counter() - t0) * 1000

        # Memory (stateless read)
        t1 = time.perf_counter()
        mem_pct, mem_used, mem_total = read_memory()
        timings["memory_ms"] = (time.perf_counter() - t1) * 1000

        metrics: dict[str, Any] = {
            "cpu_percent": cpu,
            "memory_percent": mem_pct,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
        }

        # Accelerator metrics (if available)
        if self._accelerator:
            t2 = time.perf_counter()
            accel_metrics = self._accelerator.collect()
            timings["accelerator_ms"] = (time.perf_counter() - t2) * 1000
            metrics.update(accel_metrics)

        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        metrics["_timings"] = timings

        return metrics

    def stream(self, interval: float = 0.2) -> Iterator[dict[str, Any]]:
        """Generate metrics at specified interval (non-blocking yield).

        Each call creates an independent collector thread that continuously
        collects metrics in the background. This ensures that even if the
        yield blocks (waiting for client to consume), metrics are always fresh.

        Args:
            interval: Time between collection samples in seconds.

        Yields:
            Dict of metrics (see collect() for structure).
        """
        # Per-stream state (not shared between calls)
        stop = threading.Event()
        latest: dict[str, Any] | None = None
        latest_lock = threading.Lock()
        new_data = threading.Event()

        def collector_loop() -> None:
            nonlocal latest
            while not stop.is_set():
                metrics = self.collect()  # Thread-safe (uses _lock internally)
                with latest_lock:
                    latest = metrics
                new_data.set()
                stop.wait(timeout=interval)

        # Start collector thread
        collector = threading.Thread(target=collector_loop, daemon=True)
        collector.start()

        try:
            while True:
                new_data.wait()
                new_data.clear()
                with latest_lock:
                    metrics = latest
                if metrics:
                    yield metrics  # May block on network, but collector continues
        finally:
            stop.set()
            collector.join(timeout=1)
