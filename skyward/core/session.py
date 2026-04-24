"""Session — owns the event loop, actor system, and session actor.

A Session is the long-lived infrastructure context that can host one or
more compute pools.  It manages the asyncio event loop running in a
background daemon thread, the Casty actor system, and the top-level
session actor.

    with Session() as session:
        ...  # pools created here share the same actor system
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from contextvars import Token
from types import TracebackType
from typing import TYPE_CHECKING, Any, Unpack, overload

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig

from skyward.api.spec import ConsoleMode
from skyward.observability.logger import logger
from skyward.observability.logging import LogConfig, setup_logging, teardown_logging

from .context import _active_session
from .loop import check_fd_budget, cleanup_loop, run_loop, run_sync
from .offers import PoolConfig, select_offers
from .spec import (
    DEFAULT_BOOTSTRAP_TIMEOUT,
    DEFAULT_MAX_PROVISION_ATTEMPTS,
    DEFAULT_PROVISION_RETRY_DELAY,
    DEFAULT_PROVISION_TIMEOUT,
    DEFAULT_SSH_RETRY_INTERVAL,
    DEFAULT_SSH_TIMEOUT,
    Options,
    Spec,
    SpecKwargs,
    Worker,
)

_DEFAULT_OPTIONS = Options()


def _resolve[T: (int, float)](user: T | None, provider: T | None, default: T) -> T:
    if user is not None:
        return user
    if provider is not None:
        return provider
    return default


if TYPE_CHECKING:
    from skyward.api.projection import SessionProjection
    from skyward.core.pool import ComputePool


class Session:
    """Infrastructure owner for one or more compute pools.

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
    """

    def __init__(
        self,
        *,
        console: bool | ConsoleMode = True,
        logging: LogConfig | bool = True,
        shutdown_timeout: float = 120.0,
        projection: SessionProjection | None = None,
    ) -> None:
        from skyward.api.projection import SessionProjection as _Proj

        self._console: bool | ConsoleMode = console
        self._logging = logging
        self._shutdown_timeout = shutdown_timeout
        self._projection = projection or _Proj()
        self._unsubscribe: Callable[[], None] | None = None

        self._log_handler_ids: list[int] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._system: ActorSystem | None = None
        self._session_ref: ActorRef[Any] | None = None
        self._active: bool = False
        self._context_token: Token[Session | None] | None = None
        self._pools: dict[str, Any] = {}
        self._pending_pool_refs: dict[str, ActorRef[Any]] = {}
        self._original_sigint: Any = None

    def __enter__(self) -> Session:
        """Start the session infrastructure."""
        if self._logging:
            match self._logging:
                case True:
                    log_config = LogConfig(console=False)
                case _:
                    log_config = self._logging
            self._log_handler_ids = setup_logging(log_config)

        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(
            target=lambda: run_loop(loop),
            daemon=True,
            name="skyward-session-loop",
        )
        self._loop_thread.start()

        try:
            run_sync(loop, self._start_async())
            self._active = True
            self._context_token = _active_session.set(self)
            logger.info("Session started")
        except Exception as e:
            logger.exception("Error starting session: {err}", err=e)
            self._cleanup()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the session and release all resources."""
        interrupted = exc_type is KeyboardInterrupt

        if self._context_token is not None:
            with suppress(ValueError):
                _active_session.reset(self._context_token)
            self._context_token = None

        try:
            if self._active and self._loop is not None:
                if interrupted:
                    sys.stderr.write(
                        "\nInterrupted. Shutting down gracefully… "
                        "(press Ctrl+C again to force exit)\n",
                    )
                    self._defer_interrupts()
                run_sync(
                    self._loop,
                    self._stop_async(),
                    timeout=self._shutdown_timeout,
                )
        except TimeoutError:
            logger.warning(
                "Session stop timed out after {t}s, forcing cleanup",
                t=self._shutdown_timeout,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted during shutdown, forcing cleanup")
        except Exception as e:
            logger.warning("Error stopping session: {err}", err=e)
        finally:
            self._active = False
            self._cleanup()
            self._restore_interrupts()
            logger.info("Session stopped")

            if self._log_handler_ids:
                teardown_logging(self._log_handler_ids)

    async def _start_async(self) -> None:
        """Create actor system and spawn the session actor."""
        from skyward.actors.console import (
            EventReceived,
            LogReceived,
            ViewUpdated,
            resolve_console,
        )
        from skyward.actors.projection import projection_actor
        from skyward.actors.session.actor import session_actor

        self._system = ActorSystem(
            "skyward",
            config=CastyConfig(suppress_dead_letters_on_shutdown=True),
        )
        await self._system.__aenter__()

        if factory := resolve_console(self._console):
            console_ref = self._system.spawn(factory(), "console")
            self._unsubscribe = self._projection.subscribe(
                on_change=lambda _old, new: console_ref.tell(ViewUpdated(view=new)),
                on_log=lambda log: console_ref.tell(LogReceived(log=log)),
                on_event=lambda ev: console_ref.tell(EventReceived(event=ev)),
            )

        proj_ref = self._system.spawn(
            projection_actor(self._projection), "projection",
        )
        session_behavior = Behaviors.spy(
            session_actor(), proj_ref, spy_children=True,
        )
        self._session_ref = self._system.spawn(session_behavior, "session")

    async def _stop_async(self) -> None:
        """Stop all pools, then the session actor, then the actor system."""
        await self._stop_all_pools()

        if self._session_ref is not None and self._system is not None:
            from skyward.actors.session.messages import StopSession

            await self._system.ask(
                self._session_ref,
                lambda reply_to: StopSession(reply_to=reply_to),
                timeout=self._shutdown_timeout,
            )

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

        if self._system is not None:
            await self._system.__aexit__(None, None, None)
            self._system = None

    async def _stop_all_pools(self) -> None:
        """Send StopPool to every tracked pool and await termination."""
        for name, pool in self._pools.items():
            if not pool.is_active:
                continue
            try:
                await pool._stop_pool_actor()
            except Exception as e:
                logger.warning(
                    "Error stopping pool {name}: {err}", name=name, err=e,
                )

        if self._pending_pool_refs and self._system is not None:
            from skyward.actors.pool.messages import StopPool

            for name, ref in self._pending_pool_refs.items():
                try:
                    await self._system.ask(
                        ref,
                        lambda reply_to: StopPool(reply_to=reply_to),
                        timeout=self._shutdown_timeout,
                    )
                except Exception as e:
                    logger.warning(
                        "Error stopping pending pool {name}: {err}",
                        name=name, err=e,
                    )
            self._pending_pool_refs.clear()

    def _defer_interrupts(self) -> None:
        """Replace SIGINT handler so a second Ctrl+C force-exits."""
        try:
            self._original_sigint = signal.getsignal(signal.SIGINT)
        except ValueError:
            return

        def _force_exit(_signum: int, _frame: Any) -> None:
            sys.stderr.write("\nForced exit.\n")
            os._exit(1)

        with suppress(ValueError):
            signal.signal(signal.SIGINT, _force_exit)

    def _restore_interrupts(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_sigint is None:
            return
        with suppress(ValueError):
            signal.signal(signal.SIGINT, self._original_sigint)
        self._original_sigint = None

    def _cleanup(self) -> None:
        """Stop the event loop and join the background thread."""
        cleanup_loop(self._loop, self._loop_thread)
        self._loop = None
        self._loop_thread = None

    @property
    def is_active(self) -> bool:
        """True when the session is entered and the actor system is running."""
        return self._active

    @property
    def projection(self) -> SessionProjection:
        """The session projection accumulating domain events."""
        return self._projection

    @overload
    def compute(
        self,
        *specs: Spec,
        name: str | None = ...,
        options: Options = ...,
    ) -> ComputePool: ...

    @overload
    def compute(
        self,
        *,
        name: str | None = ...,
        options: Options = ...,
        **kwargs: Unpack[SpecKwargs],
    ) -> ComputePool: ...

    def compute(  # pyright: ignore[reportInconsistentOverload]
        self,
        *specs: Spec,
        name: str | None = None,
        options: Options = _DEFAULT_OPTIONS,
        **kwargs: Unpack[SpecKwargs],
    ) -> ComputePool:
        """Provision a compute pool within this session.

        Two modes:

        - **Single provider** — pass ``provider=``, ``nodes=``, etc.
        - **Multi-spec fallback** — pass positional ``Spec(...)`` args.

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
        **kwargs
            Flat keyword arguments matching ``Spec`` fields. Assembled
            into a single ``Spec`` when no positional specs are given.

        Returns
        -------
        ComputePool
            A fully provisioned pool ready for task dispatch.

        Raises
        ------
        RuntimeError
            When the session is not active or provisioning fails.
        ValueError
            When no specs are provided, or both specs and kwargs given.
        """
        if not self._active:
            raise RuntimeError("Session is not active")
        if specs and kwargs:
            raise ValueError("Cannot mix positional Spec objects with flat keyword arguments")
        if not specs and not kwargs:
            raise ValueError("Either Spec objects or keyword arguments (provider, ...) must be provided")

        if not specs:
            specs = (Spec(**kwargs),)

        pool_name = name or f"pool-{len(self._pools)}"
        built_specs = list(specs)

        first_spec = built_specs[0]
        effective_worker = options.worker or Worker()

        provider_opts = first_spec.provider.default_options()

        provision_timeout = float(_resolve(
            options.provision_timeout,
            provider_opts.provision_timeout if provider_opts else None,
            DEFAULT_PROVISION_TIMEOUT,
        ))
        ssh_timeout = float(_resolve(
            options.ssh_timeout,
            provider_opts.ssh_timeout if provider_opts else None,
            DEFAULT_SSH_TIMEOUT,
        ))
        bootstrap_timeout = float(_resolve(
            options.bootstrap_timeout,
            provider_opts.bootstrap_timeout if provider_opts else None,
            DEFAULT_BOOTSTRAP_TIMEOUT,
        ))

        pool_config = PoolConfig(
            image=first_spec.image,
            worker=effective_worker,
            provision_timeout=provision_timeout,
            ssh_timeout=ssh_timeout,
            bootstrap_timeout=bootstrap_timeout,
            ssh_retry_interval=_resolve(
                options.ssh_retry_interval,
                provider_opts.ssh_retry_interval if provider_opts else None,
                DEFAULT_SSH_RETRY_INTERVAL,
            ),
            provision_retry_delay=_resolve(
                options.provision_retry_delay,
                provider_opts.provision_retry_delay if provider_opts else None,
                DEFAULT_PROVISION_RETRY_DELAY,
            ),
            max_provision_attempts=_resolve(
                options.max_provision_attempts,
                provider_opts.max_provision_attempts if provider_opts else None,
                DEFAULT_MAX_PROVISION_ATTEMPTS,
            ),
            volumes=tuple(first_spec.volumes),
            autoscale_cooldown=options.autoscale_cooldown,
            autoscale_idle_timeout=options.autoscale_idle_timeout,
            reconcile_tick_interval=options.reconcile_tick_interval,
            plugins=tuple(first_spec.plugins),
            cluster=options.cluster,
            retry_on_interruption=options.retry_on_interruption,
        )

        envelope = float(provision_timeout + ssh_timeout + bootstrap_timeout + 30)
        pool_ref, spec, cid, cluster, instances = self._spawn_pool(
            built_specs, pool_config, pool_name, envelope,
        )

        from .pool import ComputePool as _ComputePool

        pool = _ComputePool._from_session(
            session=self,
            pool_ref=pool_ref,
            spec=spec,
            specs=tuple(built_specs),
            plugins=tuple(first_spec.plugins),
            cluster_id=cid,
            cluster=cluster,
            instances=instances,
            image=first_spec.image,
            worker=effective_worker,
            default_compute_timeout=options.default_compute_timeout,
        )
        self._pools[pool_name] = pool
        return pool

    def _spawn_pool(
        self,
        built_specs: list[Spec],
        pool_config: PoolConfig,
        pool_name: str,
        provision_timeout: float,
    ) -> tuple[ActorRef[Any], Any, str, Any, tuple[Any, ...]]:
        """Select offers, create pool actor, start provisioning, and wait.

        The pool actor is registered in ``_pending_pool_refs`` before the
        blocking wait so that ``_stop_all_pools`` can terminate it on
        interrupt.

        Parameters
        ----------
        built_specs
            Resolved spec list for offer selection.
        pool_config
            Pool configuration (image, worker, scaling, etc.).
        pool_name
            Logical name for this pool.
        provision_timeout
            Maximum seconds to wait for provisioning.

        Returns
        -------
        tuple
            ``(pool_ref, spec, cluster_id, cluster, instances)``

        Raises
        ------
        RuntimeError
            When provisioning fails.
        """
        loop = self._get_loop()
        system = self._system
        session_ref = self._session_ref
        assert system is not None and session_ref is not None

        offers, provider_config, cloud_provider, spec = run_sync(
            loop, select_offers(built_specs, pool_config),
        )

        check_fd_budget(spec.nodes.max or spec.nodes.desired)

        pool_ref = self._create_pool_actor(pool_name)

        from skyward.core.errors import ProvisioningError

        try:
            started = self._start_pool(
                pool_ref, spec, provider_config, cloud_provider,
                offers, provision_timeout,
            )
        except ProvisioningError:
            self._pending_pool_refs.pop(pool_name, None)
            raise
        self._pending_pool_refs.pop(pool_name, None)

        return pool_ref, spec, started.cluster_id, started.cluster, started.instances

    def _create_pool_actor(self, pool_name: str) -> ActorRef[Any]:
        """Ask the session actor to spawn a pool actor and return its ref.

        The ref is immediately tracked in ``_pending_pool_refs`` so
        ``_stop_all_pools`` can reach it during shutdown.
        """
        from skyward.actors.session.messages import CreatePool, PoolCreated

        loop = self._get_loop()
        system = self._system
        session_ref = self._session_ref
        assert system is not None and session_ref is not None

        result: PoolCreated = run_sync(
            loop,
            system.ask(
                session_ref,
                lambda reply_to: CreatePool(name=pool_name, reply_to=reply_to),
                timeout=30,
            ),
        )
        self._pending_pool_refs[pool_name] = result.pool_ref
        return result.pool_ref

    def _start_pool(
        self,
        pool_ref: ActorRef[Any],
        spec: Any,
        provider_config: Any,
        provider: Any,
        offers: tuple[Any, ...],
        timeout: float,
    ) -> Any:
        """Send StartPool to the pool actor and block until ready or failed."""
        from skyward.actors.pool.messages import PoolStarted, ProvisionFailed, StartPool
        from skyward.core.errors import ProvisioningError

        loop = self._get_loop()
        system = self._system
        assert system is not None

        result: PoolStarted | ProvisionFailed = run_sync(
            loop,
            system.ask(
                pool_ref,
                lambda reply_to: StartPool(
                    spec=spec,
                    provider_config=provider_config,
                    provider=provider,
                    offers=offers,
                    reply_to=reply_to,  # type: ignore[arg-type]
                ),
                timeout=timeout,
            ),
        )

        match result:
            case ProvisionFailed(reason=reason):
                raise ProvisioningError(pool_name="", reason=reason)
            case PoolStarted() as started:
                return started

        raise RuntimeError(f"Unexpected result from pool actor: {result}")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the running event loop or raise."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")
        return self._loop
