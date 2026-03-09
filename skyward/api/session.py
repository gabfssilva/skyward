"""Session stub — type-checking interface for multi-pool orchestration."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.logging import LogConfig
    from skyward.api.pool import Pool
    from skyward.api.spec import Options, Spec


class Session:
    """Infrastructure owner for one or more compute pools.

    A Session is the long-lived infrastructure context that can host one or
    more compute pools.  It manages the asyncio event loop running in a
    background daemon thread, the Casty actor system, and the top-level
    session actor.

    Parameters
    ----------
    console
        Enable the Rich adaptive console spy.  Wiring is deferred to
        Task 10; the flag is stored for future use.
    logging
        Logging configuration.  ``True`` uses sensible defaults,
        ``False`` disables logging, or pass a ``LogConfig`` instance.
    shutdown_timeout
        Maximum seconds to wait for a graceful shutdown of the session
        actor and actor system.

    Examples
    --------
    >>> with Session() as session:
    ...     pool = session.compute(
    ...         sky.Spec(provider=sky.AWS(), accelerator="A100", nodes=4),
    ...     )
    ...     result = train(data) >> pool
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
        """True when the session is entered and the actor system is running."""
        ...

    def compute(
        self,
        *specs: Spec,
        name: str | None = None,
        options: Options = ...,  # type: ignore[assignment]
    ) -> Pool:
        """Provision a compute pool within this session.

        Parameters
        ----------
        *specs
            One or more ``Spec`` objects defining hardware, environment,
            and provider. For multi-provider fallback, pass multiple specs.
        name
            Pool name.  Auto-generated as ``pool-<n>`` when ``None``.
        options
            Operational tuning (timeouts, retries, autoscaling).
            Defaults are sensible for most workloads.

        Returns
        -------
        Pool
            A fully provisioned pool ready for task dispatch.

        Raises
        ------
        RuntimeError
            When the session is not active or provisioning fails.
        ValueError
            When no specs are provided.
        """
        ...
