"""Session stub — type-checking interface for multi-pool orchestration.

The ``Session`` class is the long-lived infrastructure context that
owns the asyncio event loop, the Casty actor system, and one or more
compute pools.  This module provides the type stub; the real
implementation lives in ``skyward.core.session``.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Unpack, overload

if TYPE_CHECKING:
    from skyward.api.logging import LogConfig
    from skyward.api.pool import Pool
    from skyward.api.spec import Options, Spec, SpecKwargs


class Session:
    """Long-lived infrastructure owner for one or more compute pools.

    A ``Session`` manages the background asyncio event loop (running in a
    daemon thread), the Casty actor system, and a top-level session actor
    that coordinates pool lifecycles.  Multiple pools created through the
    same session share these resources.

    Use ``Session`` directly when you need **multi-pool orchestration** —
    for example, a training pool and a separate preprocessing pool, or
    progressive provisioning across providers.  For the common single-pool
    case, prefer the simpler ``sky.Compute()`` context manager, which
    creates and tears down a ``Session`` automatically.

    Lifecycle
    ---------
    On ``__enter__``:

    1. Initializes logging (if enabled).
    2. Creates a fresh ``asyncio`` event loop in a background daemon thread.
    3. Starts the Casty actor system and spawns the session actor.
    4. Optionally attaches the Rich console spy for live monitoring.
    5. Registers itself in the ``_active_session`` context variable.

    On ``__exit__``:

    1. Sends ``StopSession`` to the session actor, which gracefully stops
       all pools and replies with ``SessionStopped``.
    2. Tears down the actor system.
    3. Stops and joins the background event loop thread (10 s join timeout).
    4. Cleans up logging handlers.

    If shutdown exceeds *shutdown_timeout*, a warning is logged and cleanup
    proceeds without waiting.  Exceptions during teardown are logged but
    **not** re-raised, so user exceptions propagate cleanly.

    Parameters
    ----------
    console
        Enable the Rich adaptive console spy for live pool monitoring.
        Adds a spy actor that observes all pool events and renders them
        as a Rich Live display.  Default ``True``.
    logging
        Logging configuration.  ``True`` uses sensible defaults
        (``LogConfig(console=False)``), ``False`` disables logging
        entirely, or pass a ``LogConfig`` instance for fine-grained
        control.  Default ``True``.
    shutdown_timeout
        Maximum seconds to wait for a graceful shutdown of the session
        actor and actor system before forcing cleanup.  Default ``120.0``.

    Examples
    --------
    Single pool within a session:

    >>> with sky.Session() as session:
    ...     pool = session.compute(
    ...         sky.Spec(provider=sky.AWS(), accelerator="A100", nodes=4),
    ...     )
    ...     result = train(data) >> pool

    Multiple pools sharing the same session infrastructure:

    >>> with sky.Session() as session:
    ...     preprocess = session.compute(
    ...         sky.Spec(provider=sky.AWS(), accelerator="T4", nodes=2),
    ...         name="preprocess",
    ...     )
    ...     train_pool = session.compute(
    ...         sky.Spec(provider=sky.AWS(), accelerator="A100", nodes=8),
    ...         name="train",
    ...     )
    ...     data = prepare(raw) >> preprocess
    ...     model = train(data) >> train_pool

    Notes
    -----
    - All pools share the same event loop and actor system.  If the
      session actor itself crashes, all pools become non-functional.
    - The background thread is a daemon — if the main thread crashes,
      the session does not get to shut down gracefully.
    - ``_active_session`` is a ``ContextVar``, so nested sessions in
      different coroutines or threads are isolated correctly.

    See Also
    --------
    Compute : Single-pool convenience context manager.
    Pool : The protocol returned by ``session.compute()``.
    """

    def __init__(
        self,
        *,
        console: bool = True,
        logging: LogConfig | bool = True,
        shutdown_timeout: float = 120.0,
    ) -> None: ...

    def __enter__(self) -> Session: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    @property
    def is_active(self) -> bool:
        """Whether the session is entered and the actor system is running.

        Returns ``False`` before ``__enter__`` completes, after
        ``__exit__`` begins teardown, or if startup failed.

        Returns
        -------
        bool
            ``True`` when the session can create new pools.
        """
        ...

    @overload
    def compute(
        self,
        *specs: Spec,
        name: str | None = ...,
        options: Options = ...,
    ) -> Pool: ...

    @overload
    def compute(
        self,
        *,
        name: str | None = ...,
        options: Options = ...,
        **kwargs: Unpack[SpecKwargs],
    ) -> Pool: ...

    def compute(  # pyright: ignore[reportInconsistentOverload]
        self,
        *specs: Spec,
        name: str | None = None,
        options: Options = ...,  # type: ignore[assignment]
        **kwargs: Unpack[SpecKwargs],
    ) -> Pool:
        """Provision a compute pool within this session.

        Two modes:

        - **Single provider** — pass ``provider=``, ``nodes=``, etc.
        - **Multi-spec fallback** — pass positional ``Spec(...)`` args.

        Resolves hardware offers from the provider(s), selects the best
        match, provisions cloud instances, bootstraps the environment
        (Python, packages, includes), and starts remote workers.  The
        returned pool is fully ready for task dispatch.

        All pools created through the same session share the event loop
        and actor system.  Each pool is independent — one pool failing
        does not affect others.

        Parameters
        ----------
        *specs
            One or more ``Spec`` objects defining hardware, environment,
            and provider.  For multi-provider fallback, pass multiple
            specs and control selection via ``options.selection``.
        name
            Human-readable pool name.  Auto-generated as ``"pool-<n>"``
            when ``None``.  Must be unique within the session.
        options
            Operational tuning (timeouts, retries, autoscaling, worker
            config, console).  Defaults are sensible for most workloads.
        **kwargs
            Flat keyword arguments matching ``Spec`` fields. Assembled
            into a single ``Spec`` when no positional specs are given.

        Returns
        -------
        Pool
            A fully provisioned pool ready for task dispatch via
            operators (``>>``, ``@``, ``>``) or methods.

        Raises
        ------
        RuntimeError
            If the session is not active (not entered or already exited),
            or if provisioning fails (no offers, quota exceeded, SSH
            timeout, bootstrap failure).
        ValueError
            If no specs are provided, or both specs and kwargs given.

        Examples
        --------
        >>> pool = session.compute(provider=sky.AWS(), accelerator="A100", nodes=4)
        >>> result = train(data) >> pool

        >>> pool = session.compute(
        ...     sky.Spec(provider=sky.VastAI(), accelerator="A100"),
        ...     sky.Spec(provider=sky.AWS(), accelerator="A100"),
        ...     options=sky.Options(selection="cheapest"),
        ... )
        >>> result = train(data) >> pool
        """
        ...
