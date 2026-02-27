"""Subprocess-side state for plugin around_process lifecycle.

Tracks which around_process context managers have been entered
so each is entered exactly once per subprocess. Mirrors state.py
but lives in loky subprocesses rather than the main worker process.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.plugins.plugin import ProcessLifecycle

_active_contexts: dict[str, AbstractContextManager[None]] = {}


def is_setup(name: str) -> bool:
    """Check if a plugin's around_process has been entered."""
    return name in _active_contexts


def ensure_around_process(
    name: str,
    factory: ProcessLifecycle,
    info: object,
) -> None:
    """Enter the around_process context manager if not already active.

    Idempotent: calling multiple times with the same name is a no-op
    after the first successful entry.
    """
    if name in _active_contexts:
        return
    cm = factory(info)  # type: ignore[arg-type]
    cm.__enter__()
    _active_contexts[name] = cm


def cleanup() -> None:
    """Exit all around_process contexts in reverse order."""
    for ctx in reversed(list(_active_contexts.values())):
        ctx.__exit__(None, None, None)
    _active_contexts.clear()


def reset() -> None:
    """Reset state without calling __exit__ (for testing)."""
    _active_contexts.clear()
