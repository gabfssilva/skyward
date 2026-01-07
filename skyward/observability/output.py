"""Output control for distributed execution.

Provides decorators to control stdout/stderr emission based on worker rank/node,
essential for SPMD training where only the coordinator should emit logs.

Example:
    import skyward as sky
    from skyward import is_head

    @sky.stdout(only="head")
    @sky.compute
    def train():
        print("This only prints from head node")
        ...

    # Or with predicates:
    @sky.stdout(only=is_head)
    @sky.stderr(only=lambda i: i.node == 0)
    @sky.compute
    def train():
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import wraps
from io import StringIO
from typing import TYPE_CHECKING, Literal, TextIO

if TYPE_CHECKING:
    from skyward.cluster.info import InstanceInfo

# =============================================================================
# Type Aliases
# =============================================================================

type OutputPredicate = Callable[["InstanceInfo"], bool]
"""Predicate that determines if a worker should emit output."""

type OutputSpec = OutputPredicate | Literal["head"]
"""Either a predicate or 'head' shortcut for the most common pattern."""

# =============================================================================
# Predicate Helpers
# =============================================================================

def is_head(info: InstanceInfo) -> bool:
    """True if this is the head worker (global_worker_index == 0)."""
    return info.is_head

# =============================================================================
# Output Control Decorators
# =============================================================================


def _resolve_predicate(spec: OutputSpec) -> OutputPredicate:
    """Convert an OutputSpec to a predicate function."""
    match spec:
        case "head":
            return is_head
        case _ if callable(spec):
            return spec
        case _:
            raise ValueError(f"Invalid output spec: {spec!r}")


def stdout[**P, R](
    only: OutputSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Control stdout emission in distributed execution.

    Silences stdout for workers that don't match the predicate.
    stderr is NOT affected - errors from any worker are always visible.

    Args:
        only: Predicate or "head" shortcut. Workers matching this emit stdout.
            - "head": Only head worker (global_worker_index == 0)
            - Callable[[InstanceInfo], bool]: Custom predicate

    Returns:
        Decorator that wraps the function with stdout control.

    Example:
        import skyward as sky
        from skyward import is_head

        # String shortcut (most common)
        @sky.stdout(only="head")
        @sky.compute
        def train():
            print("Only head prints this")

        # Predicate helper
        @sky.stdout(only=is_head)
        @sky.compute
        def train():
            print("Only head prints this")

        # Custom predicate
        @sky.stdout(only=lambda i: i.node < 2)
        @sky.compute
        def train():
            print("Only nodes 0 and 1 print this")
    """
    predicate = _resolve_predicate(only)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster.info import instance_info

            info = instance_info()

            # Outside pool context or predicate matches: emit normally
            if info is None or predicate(info):
                return fn(*args, **kwargs)

            # Silence stdout
            with redirect_stdout(StringIO()):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def stderr[**P, R](
    only: OutputSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Control stderr emission in distributed execution.

    Silences stderr for workers that don't match the predicate.
    Use with caution - silencing errors can hide problems.

    Args:
        only: Predicate or "head" shortcut. Workers matching this emit stderr.
            - "head": Only head worker (global_worker_index == 0)
            - Callable[[InstanceInfo], bool]: Custom predicate

    Returns:
        Decorator that wraps the function with stderr control.

    Example:
        import skyward as sky

        # Silence stderr from non-head workers
        @sky.stderr(only="head")
        @sky.compute
        def train():
            import warnings
            warnings.warn("Only head emits this warning")

        # Combine with stdout control
        @sky.stdout(only="head")
        @sky.stderr(only="head")
        @sky.compute
        def train():
            print("stdout from head only")
            sys.stderr.write("stderr from head only")
    """
    predicate = _resolve_predicate(only)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            from skyward.cluster.info import instance_info

            info = instance_info()

            # Outside pool context or predicate matches: emit normally
            if info is None or predicate(info):
                return fn(*args, **kwargs)

            # Silence stderr
            with redirect_stderr(StringIO()):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def silent[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Silence both stdout and stderr completely.

    Useful for functions that should never emit output regardless of rank.

    Example:
        @sky.silent
        @sky.compute
        def background_task():
            # No output at all
            ...
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return fn(*args, **kwargs)

    return wrapper


# =============================================================================
# Callback Writer (existing)
# =============================================================================


class CallbackWriter(TextIO):
    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buffer = StringIO()

    def write(self, s: str) -> int:
        self._callback(s)
        return self._buffer.write(s)

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    # Required TextIO methods
    def read(self, n: int = -1) -> str:
        return self._buffer.read(n)

    def readline(self, limit: int = -1) -> str:
        return self._buffer.readline(limit)

    def flush(self) -> None:
        self._buffer.flush()

    def close(self) -> None:
        self._buffer.close()

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True


@contextmanager
def redirect_output(
    callback: Callable[[str], None],
) -> Generator[tuple[CallbackWriter, CallbackWriter]]:
    out = CallbackWriter(callback)
    err = CallbackWriter(callback)
    with redirect_stdout(out), redirect_stderr(err):  # type: ignore[type-var]
        yield out, err
