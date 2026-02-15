"""Logging configuration for Skyward.

This module provides structured logging via loguru, integrated with ComputePool.
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

import sys
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

# Disable by default (library behavior)
# logger.disable("skyward")

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"]

_CONTEXT_KEYS = (
    "actor", "component", "integration", "provider",
    "cluster_id", "node_id", "instance_id", "collection", "name",
)


def _format_context(record: Any) -> str:
    extra = record.get("extra", {})
    parts = [f"{k}={extra[k]}" for k in _CONTEXT_KEYS if k in extra]
    return f" [{' '.join(parts)}]" if parts else ""


CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    "<dim>{extra[_ctx]}</dim> - "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line}{extra[_ctx]} - {message}"
)


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Logging configuration for ComputePool.

    Attributes:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        file: Path to log file. Defaults to .skyward/skyward.log.
        console: Whether to log to stderr. Defaults to True.
        rotation: File rotation policy (e.g., "50 MB", "1 day"). Defaults to "50 MB".
        retention: Number of old log files to keep. Defaults to 10.
    """

    level: LogLevel = "INFO"
    file: str = ".skyward/skyward.log"
    console: bool = False
    rotation: str = "50 MB"
    retention: int = 10



def _setup_logging(config: LogConfig) -> list[int]:
    """Configure logging and return handler IDs for cleanup.

    Args:
        config: Logging configuration.

    Returns:
        List of handler IDs that were added (for later removal).
    """
    from pathlib import Path

    # Remove default handler (ID=0) that logs to stderr without filter
    logger.remove()



    logger.enable("skyward")
    handler_ids: list[int] = []

    logger.configure(patcher=lambda r: r["extra"].update(_ctx=_format_context(r)))

    # Console output
    if config.console:
        hid = logger.add(
            sys.stderr,
            level=config.level,
            format=CONSOLE_FORMAT,
            colorize=True,
            filter="skyward",
        )
        handler_ids.append(hid)

    # File output
    if config.file:
        # Ensure parent directory exists
        Path(config.file).parent.mkdir(parents=True, exist_ok=True)
        hid = logger.add(
            config.file,
            level="DEBUG",
            format=FILE_FORMAT,
            rotation=config.rotation,
            retention=config.retention,
            compression="zip",
            diagnose=False,
            enqueue=False,
        )
        handler_ids.append(hid)

    return handler_ids


def _teardown_logging(handler_ids: list[int]) -> None:
    """Remove handlers and disable logging.

    Args:
        handler_ids: List of handler IDs to remove.
    """
    for hid in handler_ids:
        logger.remove(hid)
    logger.disable("skyward")
