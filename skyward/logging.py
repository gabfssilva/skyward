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
from typing import Literal

from loguru import logger

# Disable by default (library behavior)
logger.disable("skyward")

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Detailed format for console (with colors)
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Format for file output (no colors)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Logging configuration for ComputePool.

    Attributes:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        file: Path to log file. If provided, logs are written to this file.
        console: Whether to log to stderr. Defaults to True.
        rotation: File rotation policy (e.g., "50 MB", "1 day"). Defaults to "50 MB".
        retention: Number of old log files to keep. Defaults to 10.
    """

    level: LogLevel = "INFO"
    file: str | None = None
    console: bool = True
    rotation: str = "50 MB"
    retention: int = 10


def _setup_logging(config: LogConfig) -> list[int]:
    """Configure logging and return handler IDs for cleanup.

    Args:
        config: Logging configuration.

    Returns:
        List of handler IDs that were added (for later removal).
    """
    logger.enable("skyward")
    handler_ids: list[int] = []

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
        hid = logger.add(
            config.file,
            level="DEBUG",  # File always captures everything
            format=FILE_FORMAT,
            rotation=config.rotation,
            retention=config.retention,
            compression="zip",
            diagnose=False,  # Don't expose credentials in tracebacks
            enqueue=True,  # Thread-safe for multiprocessing
            filter="skyward",
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
