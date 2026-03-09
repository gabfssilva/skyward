"""Logging configuration for Skyward.

This module provides structured logging via stdlib + rich, integrated with ComputePool.
Logging is disabled by default and enabled when a pool is created with logging=True
or a LogConfig instance.

Example:
    from skyward import ComputePool, LogConfig

    # Simple: enable with defaults
    with ComputePool(nodes=4, logging=True) as pool:
        pool.compute(fn, *args)

    # Custom: configure level and file output
    with ComputePool(
        nodes=4,
        logging=LogConfig(level="DEBUG", file="skyward.log"),
    ) as pool:
        pool.compute(fn, *args)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from skyward.api.logging import LogConfig as LogConfig
from skyward.api.logging import LogLevel as LogLevel
from skyward.observability.logger import logger

_CONTEXT_KEYS = (
    "actor", "component", "integration", "provider",
    "cluster_id", "node_id", "instance_id", "collection", "name",
)


def _format_context(record: logging.LogRecord) -> str:
    extras: dict[str, object] = getattr(record, "extras", {})
    parts = [f"{k}={extras[k]}" for k in _CONTEXT_KEYS if k in extras]
    return f" [{' '.join(parts)}]" if parts else ""


def _patcher(record: logging.LogRecord) -> None:
    record._ctx = _format_context(record)  # type: ignore[attr-defined]


def setup_logging(config: LogConfig) -> list[int]:
    """Configure logging and return handler IDs for cleanup."""
    logger.remove()
    logger.enable("skyward")
    logger.configure(patcher=_patcher)

    handler_ids: list[int] = []

    if config.console:
        hid = logger.add(
            sys.stderr,
            level=config.level,
            filter="skyward",
        )
        handler_ids.append(hid)

    if config.file:
        Path(config.file).parent.mkdir(parents=True, exist_ok=True)
        hid = logger.add(
            config.file,
            level="DEBUG",
            rotation=config.rotation,
            retention=config.retention,
            compression="gzip",
        )
        handler_ids.append(hid)

    return handler_ids


def teardown_logging(handler_ids: list[int]) -> None:
    """Remove handlers and disable logging."""
    for hid in handler_ids:
        logger.remove(hid)
    logger.disable("skyward")
