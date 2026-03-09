"""Metrics type definitions for remote instance monitoring.

Defines the ``Metric`` dataclass that configures background metric
collectors on remote workers, and the ``MetricsConfig`` type alias
used by ``Image`` to control which system metrics are gathered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metric:
    """Definition of a background metric collector on remote workers.

    Each metric runs as a background process on the remote instance,
    executing a shell command at the specified interval and streaming
    the result back as a metric event.

    Parameters
    ----------
    name
        Metric identifier (e.g., ``"cpu"``, ``"gpu_util"``).  Used as
        the key in metric events and console display.
    command
        Shell command that outputs a numeric value to stdout.
        Executed repeatedly at each collection interval.
    interval
        Seconds between consecutive collections.  Default ``2``.
    multi
        If ``True``, the command may output multiple lines (one per
        GPU, core, etc.).  Each line becomes a separate metric with
        a numeric suffix: ``name_0``, ``name_1``, etc.
    """

    name: str
    command: str
    interval: float = 2
    multi: bool = False


type MetricsConfig = tuple[Metric, ...] | list[Metric] | None
"""Metrics configuration for ``Image``.

- ``tuple[Metric, ...]`` or ``list[Metric]`` — specific metrics to collect.
- ``None`` — disable metrics collection entirely.
"""
