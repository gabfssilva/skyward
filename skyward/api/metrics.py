"""Metrics type definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metric:
    """Definition of a metric collector.

    Each metric runs as a background process on the remote instance,
    executing a shell command at the specified interval and emitting
    the result as a metric event.

    Attributes
    ----------
    name
        Metric name (e.g., "cpu", "gpu_util_0"). Used as identifier in events.
    command
        Shell command that outputs a numeric value to stdout.
    interval
        Seconds between collections.
    multi
        If True, command may output multiple lines (one per GPU, etc).
        Each line becomes a separate metric: name_0, name_1, etc.
    """

    name: str
    command: str
    interval: float = 2
    multi: bool = False


type MetricsConfig = tuple[Metric, ...] | list[Metric] | None
"""Type for metrics configuration in ComputePool.

- tuple[Metric, ...] or list[Metric]: Specific metrics to collect
- None: Disable metrics collection entirely
"""
