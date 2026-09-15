"""Metric builders — what a node measures about itself, for ``Image(metrics=...)``.

``Default()`` names every reading the node's own collector takes, all of them served
by the one loop that reads them together. Every other builder returns one
``MetricSpec`` — a shell command the node runs on its own period: run it, emit the
value if it is a bare number, sleep, repeat. An image that names no metrics gets
``Default()``; one that names any gets exactly those, so keeping the defaults beside
a metric of one's own is saying so::

    import skyward as sky

    image = sky.Image(metrics=[*sky.metrics.Default(), sky.metrics.Custom("loss", "cat /tmp/loss")])
    image = sky.Image(metrics=[sky.metrics.GPU(index=0), sky.metrics.GPU(index=1)])

GPU builders read every device at once and report a single aggregate per node —
utilisation and temperature averaged, memory summed — because a collector reads one
value per sample; ``nvidia-smi`` printing a line per GPU would fail the numeric gate
and emit nothing. Pass ``index`` to read one device instead. Per-device series
(``gpu_util_0``, ``gpu_util_1``, …) without an explicit index are not emitted.
"""

from skyward.shared.schemas import READINGS, MetricSpec, Reading

_AVG = "awk '{s+=$1;n++} END{if(n)printf \"%.1f\",s/n}'"
_SUM = "awk '{s+=$1} END{if(NR)printf \"%d\",s}'"


def CPU(interval: float = 2) -> MetricSpec:
    """CPU utilisation percentage (0-100), from ``top``.

    Parameters
    ----------
    interval : float
        Seconds between samples.
    """
    return MetricSpec(
        name="cpu",
        command="top -bn2 -d0.1 | awk '/^%Cpu/{c=100-$8} END{printf \"%.1f\",c}'",
        interval=interval,
    )


def Memory(interval: float = 2) -> MetricSpec:
    """Memory utilisation percentage (0-100)."""
    return MetricSpec(
        name="mem",
        command="free | awk '/^Mem:/ {printf \"%.1f\", $3/$2*100}'",
        interval=interval,
    )


def MemoryUsed(interval: float = 2) -> MetricSpec:
    """Memory used, in megabytes."""
    return MetricSpec(
        name="mem_used_mb",
        command="free | awk '/^Mem:/ {printf \"%d\", $3/1024}'",
        interval=interval,
    )


def MemoryTotal() -> MetricSpec:
    """Total memory, in megabytes — sampled every 60s since it does not change."""
    return MetricSpec(
        name="mem_total_mb",
        command="free | awk '/^Mem:/ {printf \"%d\", $2/1024}'",
        interval=60.0,
    )


def _gpu(name: str, query: str, aggregate: str, index: int | None, interval: float) -> MetricSpec:
    """One ``nvidia-smi`` gauge, aggregated to a single value across the devices it reads."""
    selector = f"-i {index} " if index is not None else ""
    suffix = f"_{index}" if index is not None else ""
    command = f"nvidia-smi {selector}--query-gpu={query} --format=csv,noheader,nounits 2>/dev/null | {aggregate}"
    return MetricSpec(name=f"{name}{suffix}", command=command, interval=interval)


def GPU(index: int | None = None, interval: float = 3) -> MetricSpec:
    """GPU utilisation percentage (0-100), averaged across devices (or one ``index``)."""
    return _gpu("gpu_util", "utilization.gpu", _AVG, index, interval)


def GPUMemory(index: int | None = None, interval: float = 3) -> MetricSpec:
    """GPU memory used, in megabytes, summed across devices (or one ``index``)."""
    return _gpu("gpu_mem_mb", "memory.used", _SUM, index, interval)


def GPUMemoryTotal(index: int | None = None) -> MetricSpec:
    """Total GPU memory, in megabytes — sampled every 60s since it does not change."""
    return _gpu("gpu_mem_total_mb", "memory.total", _SUM, index, 60.0)


def GPUTemp(index: int | None = None, interval: float = 3) -> MetricSpec:
    """GPU temperature, in Celsius, averaged across devices (or one ``index``)."""
    return _gpu("gpu_temp", "temperature.gpu", _AVG, index, interval)


def Disk(path: str = "/", interval: float = 5.0) -> MetricSpec:
    """Disk usage percentage for ``path``, from ``df``."""
    safe = path.replace("/", "_").strip("_") or "root"
    return MetricSpec(
        name=f"disk_{safe}",
        command=f"df {path} 2>/dev/null | tail -1 | awk '{{print $5}}' | tr -d '%'",
        interval=interval,
    )


def NetworkRx(interface: str = "eth0", interval: float = 3) -> MetricSpec:
    """Network bytes received on ``interface`` (cumulative)."""
    return MetricSpec(
        name=f"net_rx_{interface}",
        command=f"cat /sys/class/net/{interface}/statistics/rx_bytes 2>/dev/null",
        interval=interval,
    )


def NetworkTx(interface: str = "eth0", interval: float = 3) -> MetricSpec:
    """Network bytes transmitted on ``interface`` (cumulative)."""
    return MetricSpec(
        name=f"net_tx_{interface}",
        command=f"cat /sys/class/net/{interface}/statistics/tx_bytes 2>/dev/null",
        interval=interval,
    )


def Custom(name: str, command: str, interval: float = 3) -> MetricSpec:
    """A metric from any shell command that prints a single number.

    Parameters
    ----------
    name : str
        Metric identifier (alphanumeric and underscore).
    command : str
        Shell command that prints one numeric value; anything else is dropped.
    interval : float
        Seconds between samples.
    """
    return MetricSpec(name=name, command=command, interval=interval)


def Default() -> tuple[Reading, ...]:
    """Every reading the node's own collector takes — what an image that names no metrics gets.

    CPU, memory, network and disk everywhere; GPU utilisation, memory, temperature and
    power where ``nvidia-smi`` answers. See :data:`skyward.shared.schemas.Reading` for
    what each one measures and how often.
    """
    return READINGS


__all__ = [
    "CPU",
    "Memory",
    "MemoryUsed",
    "MemoryTotal",
    "GPU",
    "GPUMemory",
    "GPUMemoryTotal",
    "GPUTemp",
    "Disk",
    "NetworkRx",
    "NetworkTx",
    "Custom",
    "Default",
]
