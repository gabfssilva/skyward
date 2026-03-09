"""App stub — type-checking interface for application context."""

from __future__ import annotations

from typing import Any


class App:
    """Application context manager for console lifecycle and spy wiring.

    Provide a Rich adaptive console and optional spy actor for
    observing pool events. Usually not needed directly — ``ComputePool``
    manages its own ``App`` internally.

    Parameters
    ----------
    console
        Whether to enable Rich console output. Default ``True``.

    Examples
    --------
    >>> with sky.App(console=True):
    ...     with sky.ComputePool(...) as pool:
    ...         result = train(data) >> pool
    """

    console: bool

    def __init__(self, console: bool = True) -> None: ...

    def __enter__(self) -> App: ...

    def __exit__(self, *args: Any) -> None: ...
