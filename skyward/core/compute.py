"""Single-pool convenience context manager.

``Compute`` is syntactic sugar that creates a ``Session``, provisions
one pool via ``session.compute()``, and tears everything down on exit.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Unpack

from skyward.core.spec import Options, Spec, SpecKwargs

if TYPE_CHECKING:
    from skyward.core.pool import ComputePool

_DEFAULT_OPTIONS = Options()


@contextmanager
def Compute(
    *specs: Spec,
    options: Options = _DEFAULT_OPTIONS,
    **kwargs: Unpack[SpecKwargs],
) -> Generator[ComputePool, None, None]:
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
    ComputePool
        A fully provisioned pool ready for task dispatch.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as compute:
    ...     result = train(data) >> compute

    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ... ) as compute:
    ...     result = train(data) >> compute
    """
    from skyward.core.context import _active_pool
    from skyward.core.session import Session

    if specs and kwargs:
        raise ValueError("Cannot mix positional Spec objects with flat keyword arguments")
    if not specs and not kwargs:
        raise ValueError("Either Spec objects or keyword arguments (provider, ...) must be provided")

    built_specs = specs if specs else (Spec(**kwargs),)

    with Session(
        console=options.console,
        logging=options.logging,
        shutdown_timeout=options.shutdown_timeout,
    ) as session:
        pool = session.compute(*built_specs, options=options)
        token = _active_pool.set(pool)
        try:
            yield pool
        finally:
            _active_pool.reset(token)
