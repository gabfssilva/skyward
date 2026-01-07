"""Decorators for error handling and observability."""

import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from loguru import logger

from skyward.callback import emit
from skyward.events import Error


def audit[F: Callable[..., Any]](
    operation: str | None = None,
    *,
    args: bool = False,
    result: bool = False,
    emit_error: bool = True,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "TRACE"] = "INFO",
) -> Callable[[F], F]:
    """Decorator for logging entry/exit, timing, and optional error emission.

    Unified decorator combining logging with error event emission:
    - → logs entry (with args if enabled)
    - ← logs exit with duration (with result if enabled)
    - ✗ logs exception WITH TRACEBACK + emits Error event

    Args:
        operation: Custom operation name (defaults to function name).
        args: Log function arguments on entry.
        result: Log function result on exit.
        emit_error: Emit Error event on exception (default True).
        level: Log level for entry/exit messages.

    Usage:
        @audit("Provisioning")                    # Basic
        @audit("Fetch", args=True)                # With arguments
        @audit("Compute", result=True)            # With result
        @audit("Debug", emit_error=False)         # Without error event
        @audit(level="DEBUG")                     # Debug level logging
    """

    def decorator(func: F) -> F:
        op = operation or func.__name__
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*a: Any, **kw: Any) -> Any:
            start = time.time()

            # Build entry message
            if args:
                bound = sig.bind(*a, **kw)
                bound.apply_defaults()
                formatted = ", ".join(f"{k}={v!r}" for k, v in bound.arguments.items())
                msg = f"{op}({formatted})"
            else:
                msg = op

            logger.opt(depth=1).log(level, f"→ {msg}")

            try:
                r = func(*a, **kw)
            except Exception as e:
                elapsed = f"{time.time() - start:.2f}s"
                logger.opt(depth=1).exception(f"✗ {msg} [{elapsed}]")
                if emit_error:
                    emit(Error(message=f"{op} failed: {e}"))
                raise

            # Build exit message
            elapsed = f"{time.time() - start:.2f}s"
            result_str = f" → {r!r}" if result else ""
            logger.opt(depth=1).log(level, f"← {msg} [{elapsed}]{result_str}")
            return r

        return wrapper

    return decorator
