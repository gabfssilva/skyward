"""App stub — type-checking interface for the application context.

The ``App`` context manager manages the Rich adaptive console and
optional spy actor for observing pool events.  Usually not needed
directly — ``Session`` and ``Compute`` manage their own ``App``
internally.
"""

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
    ...     with sky.ComputePool(...) as compute:
    ...         result = train(data) >> compute
    """

    console: bool

    def __init__(self, console: bool = True) -> None: ...

    def __enter__(self) -> App: ...

    def __exit__(self, *args: Any) -> None: ...
