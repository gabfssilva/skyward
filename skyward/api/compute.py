"""Single-pool convenience context manager.

``Compute`` is syntactic sugar that creates a ``Session``, provisions
one pool via ``session.compute()``, and tears everything down on exit.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from skyward.api.spec import DEFAULT_IMAGE, Image, SelectionStrategy, Spec, Volume, Worker

if TYPE_CHECKING:
    from skyward.accelerators import Accelerator
    from skyward.api.pool import ComputePool
    from skyward.api.provider import ProviderConfig
    from skyward.observability.logging import LogConfig
    from skyward.plugins.plugin import Plugin


@contextmanager
def Compute(
    *specs: Spec,
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
    logging: LogConfig | bool = True,
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
    shutdown_timeout: float = 120.0,
    console: bool = True,
) -> Generator[ComputePool, None, None]:
    """Single-pool convenience: creates a Session and provisions one pool.

    Equivalent to::

        with sky.Session(console=console, logging=logging) as session:
            pool = session.compute(provider=..., accelerator=..., nodes=...)
            yield pool

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects for multi-provider fallback.
        Mutually exclusive with the ``provider`` keyword argument.
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
    logging
        Logging configuration.  ``True`` uses sensible defaults,
        ``False`` disables logging, or pass a ``LogConfig`` instance.
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
    shutdown_timeout
        Maximum seconds to wait for a graceful shutdown of the session.
    console
        Enable the Rich adaptive console spy.

    Yields
    ------
    ComputePool
        A fully provisioned pool ready for task dispatch.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), accelerator="A100", nodes=4) as pool:
    ...     result = train(data) >> pool
    """
    from skyward.api.context import _active_pool
    from skyward.api.session import Session

    with Session(
        console=console,
        logging=logging,
        shutdown_timeout=shutdown_timeout,
    ) as session:
        pool = session.compute(
            *specs,
            provider=provider,
            nodes=nodes,
            accelerator=accelerator,
            vcpus=vcpus,
            memory_gb=memory_gb,
            architecture=architecture,
            allocation=allocation,
            selection=selection,
            image=image,
            ttl=ttl,
            worker=worker,
            max_hourly_cost=max_hourly_cost,
            default_compute_timeout=default_compute_timeout,
            provision_timeout=provision_timeout,
            ssh_timeout=ssh_timeout,
            ssh_retry_interval=ssh_retry_interval,
            provision_retry_delay=provision_retry_delay,
            max_provision_attempts=max_provision_attempts,
            volumes=volumes,
            autoscale_cooldown=autoscale_cooldown,
            autoscale_idle_timeout=autoscale_idle_timeout,
            reconcile_tick_interval=reconcile_tick_interval,
            plugins=plugins,
        )
        token = _active_pool.set(pool)
        try:
            yield pool
        finally:
            _active_pool.reset(token)
