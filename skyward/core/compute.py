"""Single-pool convenience context manager.

``Compute`` is syntactic sugar that creates a ``Session``, provisions
one pool via ``session.compute()``, and tears everything down on exit.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Unpack

from skyward.core.errors import NoOffersError, ProvisioningError
from skyward.core.spec import Options, Spec, SpecKwargs

if TYPE_CHECKING:
    from skyward.api.pool import Pool

_DEFAULT_OPTIONS = Options()


@contextmanager
def Compute(
    *specs: Spec,
    name: str | None = None,
    options: Options = _DEFAULT_OPTIONS,
    **kwargs: Unpack[SpecKwargs],
) -> Generator[Pool, None, None]:
    """Single-pool convenience: creates a Session and provisions one pool.

    Three calling conventions (mutually exclusive):

    - **Named pool**: ``sky.Compute(name="train")`` — loads from
      ``skyward.toml``.
    - **Flat kwargs**: ``sky.Compute(provider=sky.AWS(), ...)`` —
      assembles a single ``Spec`` from keyword arguments.
    - **Explicit specs**: ``sky.Compute(Spec(...), Spec(...))`` —
      multi-provider fallback.

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects. For multi-provider fallback,
        pass multiple specs. Mutually exclusive with flat kwargs.
    name
        Pool name from ``skyward.toml``. Mutually exclusive with
        specs and kwargs.
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

    >>> with sky.Compute(name="train") as pool:
    ...     result = train(data) >> pool
    """
    from skyward.core.context import _active_pool

    if name is not None:
        if specs or kwargs:
            raise ValueError("Cannot mix 'name' with specs or keyword arguments")
        from skyward.core.pool import ComputePool

        pool = ComputePool.Named(name)
        with pool:
            token = _active_pool.set(pool)
            try:
                yield pool
            finally:
                _active_pool.reset(token)
        return

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
        try:
            pool = session.compute(*built_specs, options=options)
        except NoOffersError as e:
            from rich.console import Console as RichConsole

            from skyward.actors.console.view import _print_no_offers_error

            _print_no_offers_error(RichConsole(stderr=True), e)
            raise SystemExit(1) from None
        except ProvisioningError as e:
            if not options.console:
                import sys

                sys.stderr.write(f"\nProvisioning failed: {e.reason}\n")
            raise SystemExit(1) from None
        token = _active_pool.set(pool)
        try:
            yield pool
        finally:
            _active_pool.reset(token)
