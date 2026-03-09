"""Compute stub — type-checking interface for single-pool convenience.

``Compute`` is the primary entry point for most users.  It creates a
``Session``, provisions a single pool, and yields it as a context
manager.  Accepts either explicit ``Spec`` objects or flat keyword
arguments that map to ``Spec`` fields.
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
    """Single-pool convenience — creates a Session, provisions one pool, yields it.

    This is the recommended entry point for most workloads.  It handles
    the full lifecycle: session creation, event loop startup, offer
    selection, instance provisioning, environment bootstrap, worker
    readiness, and teardown — all behind a single ``with`` block.

    Accepts two mutually exclusive calling conventions:

    - **Flat kwargs** (most common): pass ``Spec`` fields directly as
      keyword arguments.  A single ``Spec`` is assembled automatically.
    - **Explicit specs**: pass one or more ``Spec`` objects positionally
      for multi-provider fallback or advanced configuration.

    Lifecycle
    ---------
    On entry:

    1. Validates arguments (flat kwargs and positional specs are mutually
       exclusive).
    2. Creates a fresh ``Session`` with settings drawn from *options*.
    3. Enters the session (spins up event loop, actor system).
    4. Calls ``session.compute(*specs, options=options)`` to provision
       the pool.
    5. Sets the pool in the ``_active_pool`` context variable so that
       the ``sky`` singleton resolves to it.
    6. Yields the ready pool.

    On exit:

    1. Resets the ``_active_pool`` context variable.
    2. Exits the session, which gracefully stops the pool, tears down
       the actor system, and joins the background thread.

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects.  For multi-provider fallback,
        pass multiple specs and control selection via
        ``options.selection``.  Mutually exclusive with **kwargs.
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
        If both positional specs and flat kwargs are provided, or if
        neither is provided.
    RuntimeError
        If provisioning fails (no offers, quota exceeded, SSH timeout,
        bootstrap failure).

    Examples
    --------
    Flat kwargs — single provider, single spec:

    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool

    Multi-provider fallback — cheapest across VastAI and AWS:

    >>> with sky.Compute(
    ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
    ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
    ...     options=sky.Options(selection="cheapest"),
    ... ) as pool:
    ...     result = train(data) >> pool

    Using the ``sky`` singleton instead of the pool variable:

    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4):
    ...     result = train(data) >> sky

    Elastic autoscaling between 2 and 8 nodes:

    >>> with sky.Compute(
    ...     provider=sky.AWS(), accelerator="A100", nodes=(2, 8),
    ... ) as pool:
    ...     result = train(data) >> pool

    Notes
    -----
    - Each ``Compute()`` call creates a **fresh** ``Session``.  There is
      no session reuse across ``Compute`` contexts.
    - Exiting the ``with`` block destroys **both** the pool and the
      session — all cloud instances are terminated.
    - For multi-pool workflows, use ``Session`` directly and call
      ``session.compute()`` multiple times.

    See Also
    --------
    Session : Long-lived infrastructure owner for multi-pool orchestration.
    Spec : Hardware and environment specification.
    Options : Operational tuning knobs.
    """
    ...  # type: ignore[misc]
