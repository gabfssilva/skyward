"""Async-compatible audit decorator for observability.

Provides entry/exit logging with timing for both sync and async functions.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, overload

from loguru import logger


def _truncate(s: str, max_len: int = 200) -> str:
    """Truncate string with ellipsis if too long."""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


@overload
def audit[F: Callable[..., Any]](fn: F) -> F: ...


@overload
def audit[F: Callable[..., Any]](
    fn: None = None,
    *,
    operation: str | None = None,
    args: bool = False,
    result: bool = False,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"] = "DEBUG",
    max_repr: int = 200,
) -> Callable[[F], F]: ...


def audit[F: Callable[..., Any]](
    fn: F | None = None,
    *,
    operation: str | None = None,
    args: bool = False,
    result: bool = False,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"] = "DEBUG",
    max_repr: int = 200,
) -> F | Callable[[F], F]:
    """Decorator for logging entry/exit with timing.

    Works with both sync and async functions.

    Logs:
    - → entry (with args if enabled)
    - ← exit with duration (with result if enabled)
    - ✗ exception with traceback

    Args:
        fn: Function to decorate (for bare @audit usage).
        operation: Custom operation name (defaults to function name).
        args: Log function arguments on entry.
        result: Log function result on exit.
        level: Log level for entry/exit messages.
        max_repr: Max length for repr strings (truncated with ...).

    Usage:
        @audit
        async def fetch_data(): ...

        @audit(operation="Provision", args=True)
        async def provision_instance(spec): ...

        @audit(result=True, level="INFO")
        def compute_value(x, y): ...
    """

    def decorator(func: F) -> F:
        op = operation or func.__name__
        sig = inspect.signature(func)
        is_async = asyncio.iscoroutinefunction(func)

        def _format_args(*a: Any, **kw: Any) -> str:
            if not args:
                return ""
            try:
                bound = sig.bind(*a, **kw)
                bound.apply_defaults()
                formatted = ", ".join(
                    f"{k}={_truncate(repr(v), max_repr)}"
                    for k, v in bound.arguments.items()
                )
                return f"({formatted})"
            except Exception:
                return "(...)"

        def _format_result(r: Any) -> str:
            if not result:
                return ""
            return f" → {_truncate(repr(r), max_repr)}"

        if is_async:

            @wraps(func)
            async def async_wrapper(*a: Any, **kw: Any) -> Any:
                args_str = _format_args(*a, **kw)
                logger.opt(depth=1).log(level, f"→ {op}{args_str}")
                start = time.perf_counter()

                try:
                    r = await func(*a, **kw)
                except Exception:
                    elapsed = time.perf_counter() - start
                    logger.opt(depth=1).exception(f"✗ {op} [{elapsed:.2f}s]")
                    raise

                elapsed = time.perf_counter() - start
                result_str = _format_result(r)
                logger.opt(depth=1).log(level, f"← {op} [{elapsed:.2f}s]{result_str}")
                return r

            return async_wrapper  # type: ignore[return-value]

        else:

            @wraps(func)
            def sync_wrapper(*a: Any, **kw: Any) -> Any:
                args_str = _format_args(*a, **kw)
                logger.opt(depth=1).log(level, f"→ {op}{args_str}")
                start = time.perf_counter()

                try:
                    r = func(*a, **kw)
                except Exception:
                    elapsed = time.perf_counter() - start
                    logger.opt(depth=1).exception(f"✗ {op} [{elapsed:.2f}s]")
                    raise

                elapsed = time.perf_counter() - start
                result_str = _format_result(r)
                logger.opt(depth=1).log(level, f"← {op} [{elapsed:.2f}s]{result_str}")
                return r

            return sync_wrapper  # type: ignore[return-value]

    # Support both @audit and @audit(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = ["audit"]
