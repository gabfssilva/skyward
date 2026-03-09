"""Compute stub — type-checking interface for single-pool convenience."""

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
    options: Options = ...,  # type: ignore[assignment]
    **kwargs: Unpack[SpecKwargs],
) -> Generator[Pool, None, None]:
    pass

@contextmanager
def Compute(
    *specs: Spec,
    options: Options = ...,  # type: ignore[assignment]
    **kwargs: Unpack[SpecKwargs],
) -> Generator[Pool, None, None]:
    """Single-pool convenience: creates a Session and provisions one pool.

    Accepts either explicit ``Spec`` objects or flat keyword arguments
    that are assembled into a single ``Spec`` automatically.

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects. For multi-provider fallback,
        pass multiple specs. Mutually exclusive with flat kwargs.
    options
        Operational tuning (timeouts, retries, autoscaling, session
        settings). Defaults are sensible for most workloads.
    **kwargs
        Flat keyword arguments matching ``Spec`` fields. Assembled
        into a single ``Spec`` when no positional specs are given.

    Yields
    ------
    Pool
        A fully provisioned pool ready for task dispatch.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool

    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ... ) as pool:
    ...     result = train(data) >> pool
    """
    ...  # type: ignore[misc]
    yield  # type: ignore[misc]
