"""Compute stub — type-checking interface for single-pool convenience.

``Compute`` is the primary entry point for most users.  It creates a
``Session``, provisions a single pool, and yields it as a context
manager.  Accepts either explicit ``Spec`` objects, flat keyword
arguments that map to ``Spec`` fields, or a named pool from
``skyward.toml``.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Unpack, overload

if TYPE_CHECKING:
    from skyward.api.pool import Pool
    from skyward.api.spec import Options, Spec, SpecKwargs

@overload
@contextmanager
def Compute(
    *specs: Spec,
    options: Options = ...,  # type: ignore[assignment]
) -> Generator[Pool, None, None]:
    pass

@overload
@contextmanager
def Compute(
    *,
    name: str,
    options: Options = ...,  # type: ignore[assignment]
) -> Generator[Pool, None, None]:
    pass

@overload
@contextmanager
def Compute(
    options: Options = ...,  # type: ignore[assignment]
    **kwargs: Unpack[SpecKwargs],
) -> Generator[Pool, None, None]:
    pass

@contextmanager
def Compute(
    *specs: Spec,
    name: str | None = None,
    options: Options = ...,  # type: ignore[assignment]
    **kwargs: Unpack[SpecKwargs],
) -> Generator[Pool, None, None]:
    """Single-pool convenience — creates a Session, provisions one pool, yields it.

    Three calling conventions (mutually exclusive):

    - **Named pool**: ``sky.Compute(name="train")`` — loads from
      ``skyward.toml``.
    - **Flat kwargs** (most common): pass ``Spec`` fields directly as
      keyword arguments.  A single ``Spec`` is assembled automatically.
    - **Explicit specs**: pass one or more ``Spec`` objects positionally
      for multi-provider fallback or advanced configuration.

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects.  For multi-provider fallback,
        pass multiple specs and control selection via
        ``options.selection``.  Mutually exclusive with **kwargs and name.
    name
        Pool name from ``skyward.toml``. Mutually exclusive with
        specs and kwargs.
    options
        Operational tuning (timeouts, retries, autoscaling, worker
        config, console, logging, shutdown timeout).  Defaults are
        sensible for most workloads.
    **kwargs
        Flat keyword arguments matching ``Spec`` fields (e.g.,
        ``provider``, ``accelerator``, ``nodes``).  Assembled into a
        single ``Spec`` when no positional specs are given.

    Yields
    ------
    Pool
        A fully provisioned pool ready for task dispatch via operators
        (``>>``, ``@``, ``>``) or methods (``run``, ``broadcast``, ``map``).

    Raises
    ------
    ValueError
        If incompatible calling conventions are mixed.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool

    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ... ) as pool:
    ...     result = train(data) >> pool

    >>> with sky.Compute(name="train") as pool:
    ...     result = train(data) >> pool
    """
    ...  # type: ignore[misc]
