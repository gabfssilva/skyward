"""Session — owns the event loop, console consumer, and pool objects.

A Session is the long-lived infrastructure context that can host one or
more compute pools.  It manages the asyncio event loop running in a
background daemon thread, the console consumer, and the pool objects.

    with Session() as session:
        ...  # pools created here share the same event loop
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


def _resolve[T: (int, float, bool)](user: T | None, provider: T | None, default: T) -> T:
    if user is not None:
        return user
    if provider is not None:
        return provider
    return default


if TYPE_CHECKING:
    from skyward.actors.console import ConsoleConsumer
    from skyward.actors.pool.pool import Pool, PoolStarted
    from skyward.api.projection import SessionProjection
    from skyward.core.pool import ComputePool


class Session:
    """Infrastructure owner for one or more compute pools.

    Parameters
    ----------
    console
        Console mode (``True`` → rich when TTY, ``False`` → silent, or a
        ``ConsoleMode`` literal).
    logging
        Logging configuration.  ``True`` uses sensible defaults,
        ``False`` disables logging, or pass a ``LogConfig`` instance.
    shutdown_timeout
        Maximum seconds to wait for a graceful shutdown of the pools.
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
        self._console_consumer: ConsoleConsumer | None = None
        self._active: bool = False
        self._context_token: Token[Session | None] | None = None
        self._pools: dict[str, Any] = {}
        self._pending_pools: dict[str, Pool] = {}
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
        """Wire the console consumer to the projection."""
        from skyward.actors.console import (
            ConsoleConsumer,
            EventReceived,
            LogReceived,
            ViewUpdated,
            resolve_console,
        )

        if factory := resolve_console(self._console):
            consumer = ConsoleConsumer(factory(), asyncio.get_running_loop())
            consumer.start()
            self._console_consumer = consumer
            self._unsubscribe = self._projection.subscribe(
                on_change=lambda _old, new: consumer.tell(ViewUpdated(view=new)),
                on_log=lambda log: consumer.tell(LogReceived(log=log)),
                on_event=lambda ev: consumer.tell(EventReceived(event=ev)),
            )

    async def _stop_async(self) -> None:
        """Stop all pools, then the console consumer."""
        await self._stop_all_pools()

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

        if self._console_consumer is not None:
            await self._console_consumer.stop()
            self._console_consumer = None

    async def _stop_all_pools(self) -> None:
        """Stop every tracked pool, including pending ones."""
        for name, pool in self._pools.items():
            if not pool.is_active:
                continue
            try:
                await pool._stop_pool()
            except Exception as e:
                logger.warning(
                    "Error stopping pool {name}: {err}", name=name, err=e,
                )

        for name, pool_obj in list(self._pending_pools.items()):
            try:
                await pool_obj.stop()
            except Exception as e:
                logger.warning(
                    "Error stopping pending pool {name}: {err}",
                    name=name, err=e,
                )
        self._pending_pools.clear()

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
        """True when the session is entered and the event loop is running."""
        return self._active

    @property
    def projection(self) -> SessionProjection:
        """The session projection accumulating domain events."""
        return self._projection

    def stop_pool(self, name: str) -> bool:
        """Stop a pool by name, whether ready or still provisioning.

        Looks the pool up in two places:

        - ``self._pools`` — fully started pools (``ComputePool.__exit__``
          path).
        - ``self._pending_pools`` — pools whose ``start`` hasn't finished
          yet (mid-provisioning). Stopping tears down any cloud instances
          they have already created.

        Returns
        -------
        bool
            ``True`` when the pool was found (in either map) and the stop
            was issued, ``False`` when no pool with that name exists.
        """
        if not self._active or self._loop is None:
            return False

        pool = self._pools.pop(name, None)
        if pool is not None:
            try:
                run_sync(
                    self._loop,
                    pool._stop_pool(),
                    timeout=self._shutdown_timeout,
                )
            except Exception as e:
                logger.warning(
                    "Error stopping pool {name}: {err}", name=name, err=e,
                )
            self._pending_pools.pop(name, None)
            return True

        pool_obj = self._pending_pools.pop(name, None)
        if pool_obj is not None:
            try:
                run_sync(
                    self._loop, pool_obj.stop(), timeout=self._shutdown_timeout,
                )
            except Exception as e:
                logger.warning(
                    "Error stopping pending pool {name}: {err}",
                    name=name, err=e,
                )
            return True

        return False

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
            cluster=_resolve(
                options.cluster,
                provider_opts.cluster if provider_opts else None,
                True,
            ),
            retry_on_interruption=options.retry_on_interruption,
            health_checker=options.health_checker,
            ports=tuple(first_spec.ports),
        )

        envelope = float(provision_timeout + ssh_timeout + bootstrap_timeout + 30)
        pool_obj, spec, cid, cluster, instances = self._spawn_pool(
            built_specs, pool_config, pool_name, envelope,
        )

        from .pool import ComputePool as _ComputePool

        pool = _ComputePool._from_session(
            session=self,
            pool=pool_obj,
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

    def adopt(
        self,
        *,
        name: str,
        provider_config: Any,
        cluster: Any,
        instances: tuple[Any, ...],
        node_ids: tuple[int, ...],
        timeout: float = 600.0,
    ) -> ComputePool:
        """Re-adopt a previously-running pool from persisted live instances.

        Recreates the provider from *provider_config*, then asks the pool
        to ``recover`` — each node adopts its instance, skipping
        bootstrap/worker-launch (coordinates are refreshed via one
        ``get_instance``) — and returns a ``ComputePool`` bound to this
        session.

        Parameters
        ----------
        name
            Server-side pool name to restore.
        provider_config
            The persisted ``ProviderConfig`` used to recreate the provider.
        cluster
            The persisted cluster (its ``spec`` is reused).
        instances
            Live instances to re-adopt, aligned with *node_ids*.
        node_ids
            Persisted ranks, parallel to *instances* (head = rank 0).
        timeout
            Maximum seconds to wait for re-adoption.

        Returns
        -------
        ComputePool
            A pool bound to this session over the re-adopted nodes.

        Raises
        ------
        ProvisioningError
            If re-adoption fails (e.g. the instances are gone).
        """
        if not self._active:
            raise RuntimeError("Session is not active")

        loop = self._get_loop()
        provider = run_sync(loop, provider_config.create_provider())
        spec = cluster.spec
        pool_obj = self._create_pool(name)
        try:
            started: PoolStarted = run_sync(
                loop,
                pool_obj.recover(
                    spec, provider, cluster, instances, node_ids=node_ids,
                ),
                timeout=timeout,
            )
        finally:
            self._pending_pools.pop(name, None)

        from .pool import ComputePool as _ComputePool

        pool = _ComputePool._from_session(
            session=self,
            pool=pool_obj,
            spec=spec,
            specs=(),
            plugins=tuple(getattr(spec, "plugins", ())),
            cluster_id=started.cluster_id,
            cluster=started.cluster,
            instances=started.instances,
            image=spec.image,
            worker=spec.worker,
            default_compute_timeout=300.0,
        )
        self._pools[name] = pool
        return pool

    def discard(self, *, provider_config: Any, cluster: Any) -> None:
        """Tear down a cluster's instances — cleanup for a failed reattach.

        Best-effort: recreates the provider and calls ``teardown`` so a
        reattach that finds its instances unreachable does not leak them.
        """
        loop = self._get_loop()
        provider = run_sync(loop, provider_config.create_provider())
        run_sync(loop, provider.teardown(cluster))

    def _spawn_pool(
        self,
        built_specs: list[Spec],
        pool_config: PoolConfig,
        pool_name: str,
        provision_timeout: float,
    ) -> tuple[Pool, Any, str, Any, tuple[Any, ...]]:
        """Select offers, create the pool object, provision, and wait.

        The pool is registered in ``_pending_pools`` before the blocking
        wait so that ``_stop_all_pools`` can terminate it on interrupt.

        Returns
        -------
        tuple
            ``(pool, spec, cluster_id, cluster, instances)``

        Raises
        ------
        ProvisioningError
            When provisioning fails.
        """
        loop = self._get_loop()

        from skyward.core.errors import NoOffersError

        try:
            offers, provider_config, cloud_provider, spec = run_sync(
                loop, select_offers(built_specs, pool_config),
            )
        except NoOffersError as e:
            from skyward.api.events import Pool

            self._projection.handle(Pool.NoOffers(
                pool_name=pool_name,
                specs=tuple(
                    (s.provider, s.accelerator, s.allocation) for s in e.specs
                ),
            ))
            raise

        check_fd_budget(spec.nodes.max or spec.nodes.desired)

        pool_obj = self._create_pool(pool_name)

        from skyward.core.errors import ProvisioningError

        try:
            started: PoolStarted = run_sync(
                loop,
                pool_obj.start(spec, cloud_provider, offers),
                timeout=provision_timeout,
            )
        except ProvisioningError:
            self._pending_pools.pop(pool_name, None)
            raise
        self._pending_pools.pop(pool_name, None)

        return pool_obj, spec, started.cluster_id, started.cluster, started.instances

    def _create_pool(self, pool_name: str) -> Pool:
        """Create a pool object and track it for interrupt-time shutdown."""
        from skyward.actors.pool.pool import Pool

        pool_obj = Pool(pool_name=pool_name, emit=self._projection.handle)
        self._pending_pools[pool_name] = pool_obj
        return pool_obj

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the running event loop or raise."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")
        return self._loop
