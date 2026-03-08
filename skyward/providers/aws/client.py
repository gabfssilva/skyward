"""AWS client factories with dependency injection.

Provides typed client factories that can be injected into components.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

type Client[T] = Callable[[], AbstractAsyncContextManager[T]]
"""Factory that returns an async context manager for a client."""


class EC2ClientFactory:
    """Wrapper for EC2 client factory."""

    def __init__(self, factory: Callable[[], AbstractAsyncContextManager[Any]]) -> None:
        self._factory = factory

    def __call__(self) -> AbstractAsyncContextManager[Any]:
        return self._factory()
