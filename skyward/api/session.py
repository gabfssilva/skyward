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
import threading
from contextvars import Token
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from casty import ActorRef, ActorSystem, Behaviors, CastyConfig

from skyward.observability.logger import logger
from skyward.observability.logging import LogConfig, setup_logging, teardown_logging

from .context import _active_session
from .loop import check_fd_budget, cleanup_loop, run_loop, run_sync
from .offers import PoolConfig, select_offers
from .spec import DEFAULT_IMAGE, Image, SelectionStrategy, Spec, Volume, Worker

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.api.pool import ComputePool
    from skyward.api.provider import ProviderConfig
    from skyward.plugins.plugin import Plugin


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
        console: bool = True,
        logging: LogConfig | bool = True,
        shutdown_timeout: float = 120.0,
    ) -> None:
        self._console = console
        self._logging = logging
        self._shutdown_timeout = shutdown_timeout

        self._log_handler_ids: list[int] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._system: ActorSystem | None = None
        self._session_ref: ActorRef[Any] | None = None
        self._active: bool = False
        self._context_token: Token[Session | None] | None = None
        self._pools: dict[str, Any] = {}

    def __enter__(self) -> Session:
        """Start the session infrastructure."""
        if self._logging:
            match self._logging:
                case True:
                    log_config = LogConfig(console=False)
                case _:
                    log_config = LogConfig(
                        level=self._logging.level,
                        file=self._logging.file,
                        console=False,
                        rotation=self._logging.rotation,
                        retention=self._logging.retention,
                    )
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
        try:
            if self._context_token is not None:
                _active_session.reset(self._context_token)
                self._context_token = None

            if self._active and self._loop is not None:
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
        except Exception as e:
            logger.warning("Error stopping session: {err}", err=e)
        finally:
            self._active = False
            self._cleanup()
            logger.info("Session stopped")

            if self._log_handler_ids:
                teardown_logging(self._log_handler_ids)

    async def _start_async(self) -> None:
        """Create actor system and spawn the session actor."""
        from skyward.actors.session.actor import session_actor

        self._system = ActorSystem(
            "skyward",
            config=CastyConfig(suppress_dead_letters_on_shutdown=True),
        )
        await self._system.__aenter__()

        session_behavior = session_actor()

        if self._console:
            from skyward.actors.console import console_actor

            spy_ref = self._system.spawn(console_actor(), "console")
            session_behavior = Behaviors.spy(
                session_behavior, spy_ref, spy_children=True,
            )

        self._session_ref = self._system.spawn(session_behavior, "session")

    async def _stop_async(self) -> None:
        """Ask the session actor to stop and tear down the actor system."""
        if self._session_ref is not None and self._system is not None:
            from skyward.actors.session.messages import StopSession

            await self._system.ask(
                self._session_ref,
                lambda reply_to: StopSession(reply_to=reply_to),
                timeout=self._shutdown_timeout,
            )

        if self._system is not None:
            await self._system.__aexit__(None, None, None)
            self._system = None

    def _cleanup(self) -> None:
        """Stop the event loop and join the background thread."""
        cleanup_loop(self._loop, self._loop_thread)
        self._loop = None
        self._loop_thread = None

    @property
    def is_active(self) -> bool:
        """True when the session is entered and the actor system is running."""
        return self._active

    def compute(
        self,
        *specs: Spec,
        name: str | None = None,
        provider: ProviderConfig | None = None,
        nodes: int | tuple[int, int] = 1,
        accelerator: Accelerator | None = None,
        vcpus: float | None = None,
        memory_gb: float | None = None,
        architecture: Literal["x86_64", "arm64"] | None = None,
        allocation: Literal["spot", "on-demand", "spot-if-available"] = "spot-if-available",
        selection: SelectionStrategy = "cheapest",
        image: Image = DEFAULT_IMAGE,
        ttl: int = 600,
        worker: Worker | None = None,
        max_hourly_cost: float | None = None,
        default_compute_timeout: float = 300.0,
        provision_timeout: int = 300,
        ssh_timeout: int = 300,
        ssh_retry_interval: int = 2,
        provision_retry_delay: float = 5.0,
        max_provision_attempts: int = 3,
        volumes: list[Volume] | tuple[Volume, ...] = (),
        autoscale_cooldown: float = 30.0,
        autoscale_idle_timeout: float = 60.0,
        reconcile_tick_interval: float = 15.0,
        plugins: list[Plugin] | tuple[Plugin, ...] = (),
    ) -> ComputePool:
        """Provision a compute pool within this session.

        Parameters
        ----------
        *specs
            One or more ``Spec`` objects for multi-provider fallback.
            Mutually exclusive with the ``provider`` keyword argument.
        name
            Pool name.  Auto-generated as ``pool-<n>`` when ``None``.
        provider
            Cloud provider configuration (e.g. ``sky.AWS()``).
            Mutually exclusive with positional ``specs``.
        nodes
            Fixed node count or ``(min, max)`` for autoscaling.
        accelerator
            GPU type (e.g. ``"A100"``).  ``None`` for CPU-only.
        vcpus
            Minimum vCPUs per node.
        memory_gb
            Minimum RAM in GB per node.
        architecture
            CPU architecture filter.
        allocation
            Instance lifecycle strategy.
        selection
            Multi-spec selection strategy.
        image
            Environment specification.
        ttl
            Auto-shutdown timeout in seconds.
        worker
            Worker configuration (concurrency, executor).
        max_hourly_cost
            Cost cap per node per hour in USD.
        default_compute_timeout
            Default timeout in seconds for submitted tasks.
        provision_timeout
            Maximum seconds to wait for provisioning.
        ssh_timeout
            SSH connection timeout in seconds.
        ssh_retry_interval
            Seconds between SSH retry attempts.
        provision_retry_delay
            Seconds between provision retry attempts.
        max_provision_attempts
            Maximum number of provision attempts.
        volumes
            S3/GCS volumes to mount on workers.
        autoscale_cooldown
            Seconds between autoscaling decisions.
        autoscale_idle_timeout
            Seconds of idle before autoscaler scales down.
        reconcile_tick_interval
            Seconds between reconciler ticks.
        plugins
            Composable plugins to apply to the pool.

        Returns
        -------
        ComputePool
            A fully provisioned pool ready for task dispatch.

        Raises
        ------
        RuntimeError
            When the session is not active or provisioning fails.
        ValueError
            When argument validation fails.
        """
        if not self._active:
            raise RuntimeError("Session is not active")

        pool_name = name or f"pool-{len(self._pools)}"

        if specs and provider is not None:
            raise ValueError("Cannot specify both positional Spec args and 'provider'")
        if not specs and provider is None:
            raise ValueError("Either Spec args or 'provider' must be provided")

        scaling: tuple[int, int] | None = None
        if specs:
            built_specs = list(specs)
        else:
            assert provider is not None
            match nodes:
                case (min_n, max_n):
                    scaling = (min_n, max_n)
                case _:
                    pass
            built_specs = [Spec(
                provider=provider,
                accelerator=accelerator,
                nodes=nodes,
                vcpus=vcpus,
                memory_gb=memory_gb,
                architecture=architecture,
                allocation=allocation,
                region=getattr(provider, "region", None),
                max_hourly_cost=max_hourly_cost,
                ttl=ttl,
            )]

        effective_worker = worker or Worker()
        pool_config = PoolConfig(
            image=image,
            worker=effective_worker,
            scaling=scaling,
            ssh_timeout=ssh_timeout,
            ssh_retry_interval=ssh_retry_interval,
            provision_retry_delay=provision_retry_delay,
            max_provision_attempts=max_provision_attempts,
            volumes=tuple(volumes),
            autoscale_cooldown=autoscale_cooldown,
            autoscale_idle_timeout=autoscale_idle_timeout,
            reconcile_tick_interval=reconcile_tick_interval,
            plugins=tuple(plugins),
        )

        pool_ref, spec, cid, cluster, instances = self._spawn_pool(
            built_specs, pool_config, pool_name, float(provision_timeout),
        )

        from .pool import ComputePool as _ComputePool

        pool = _ComputePool._from_session(
            session=self,
            pool_ref=pool_ref,
            spec=spec,
            specs=tuple(built_specs),
            plugins=tuple(plugins),
            cluster_id=cid,
            cluster=cluster,
            instances=instances,
            image=image,
            worker=effective_worker,
            default_compute_timeout=default_compute_timeout,
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
        """Select offers and ask the session actor to spawn a pool.

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
        from skyward.actors.session.messages import PoolSpawned, PoolSpawnFailed, SpawnPool

        loop = self._get_loop()
        system = self._system
        session_ref = self._session_ref
        assert system is not None and session_ref is not None

        offers, provider_config, cloud_provider, spec = run_sync(
            loop, select_offers(built_specs, pool_config),
        )

        check_fd_budget(spec.nodes)

        def _spawn_factory(
            reply_to: ActorRef[PoolSpawned | PoolSpawnFailed],
        ) -> SpawnPool:
            return SpawnPool(
                name=pool_name,
                spec=spec,
                provider_config=provider_config,
                provider=cloud_provider,
                offers=offers,
                provision_timeout=provision_timeout,
                reply_to=reply_to,
            )

        result: PoolSpawned | PoolSpawnFailed = run_sync(
            loop,
            system.ask(session_ref, _spawn_factory, timeout=provision_timeout),
        )

        match result:
            case PoolSpawnFailed(name=n, reason=reason):
                raise RuntimeError(f"Pool '{n}' provisioning failed: {reason}")
            case PoolSpawned(
                pool_ref=pool_ref,
                cluster_id=cid,
                instances=instances,
                cluster=cluster,
            ):
                return pool_ref, spec, cid, cluster, instances

        raise RuntimeError(f"Unexpected result from session actor: {result}")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the running event loop or raise."""
        if self._loop is None:
            raise RuntimeError("Event loop not running")
        return self._loop
