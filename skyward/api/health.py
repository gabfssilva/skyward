"""Periodic remote health probe for compute pool nodes.

Defines :class:`HealthChecker` (configuration) and :func:`hc_loop`
(top-level sync generator shipped to the worker over cloudpickle).

The probe runs entirely on the remote node: a single
``execute_with_streaming`` dispatch yields one tuple per ``interval``;
the node actor consumes the stream and replaces the node after
``consecutive_failures`` consecutive negative results.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skyward.api.runtime import InstanceInfo


type HealthCheckResult = bool | str
"""Outcome of a single health probe.

- ``True`` -- node is healthy.
- ``False`` -- node is unhealthy with no reason.
- non-empty ``str`` -- node is unhealthy, the string is the reason.
"""

type HealthCheckFn = Callable[["InstanceInfo"], HealthCheckResult]
"""Predicate evaluated remotely on each tick."""

type HealthYield = tuple[Literal["ok", "fail"], str | None]
"""Wire format yielded by :func:`hc_loop` per tick."""


__all__ = ["HealthChecker", "HealthCheckFn", "HealthCheckResult", "HealthYield", "hc_loop"]


@dataclass(frozen=True, slots=True)
class HealthChecker:
    """Periodic remote health probe.

    Attach to :class:`Options` to enable per-node liveness checks. The
    user-supplied ``fn`` runs on the remote worker once per ``interval``;
    ``consecutive_failures`` failures in a row trigger node replacement
    via the existing ``NodeExhausted`` -> reconciler path.

    Setting a health checker also gates pool readiness: a node is not
    considered active and does not receive tasks until ``fn`` first
    returns ``True``. Failures during warm-up count toward
    ``consecutive_failures`` exactly like in steady state.

    Parameters
    ----------
    fn
        Predicate evaluated on the node. Receives :class:`InstanceInfo`
        and returns ``True``, ``False``, or a string. Exceptions and
        timeouts count as failures.
    interval
        Seconds between checks. Default ``30.0``.
    timeout
        Maximum seconds for a single ``fn`` invocation. On timeout the
        check counts as a failure. Default ``15.0``.
    consecutive_failures
        Number of consecutive failures required to trigger replacement.
        Default ``3``.
    initial_delay
        Seconds to wait after the node becomes active before the first
        check. Default ``0.0`` -- ``around_app`` already gates worker
        readiness.

    Examples
    --------
    >>> import skyward as sky
    >>> with sky.Compute(
    ...     provider=sky.AWS(),
    ...     nodes=4,
    ...     options=sky.Options(
    ...         health_checker=sky.HealthChecker(
    ...             fn=lambda info: torch.cuda.is_available(),
    ...             interval=30.0,
    ...             consecutive_failures=3,
    ...         ),
    ...     ),
    ... ) as pool:
    ...     ...
    """

    fn: HealthCheckFn
    interval: float = 30.0
    timeout: float = 15.0
    consecutive_failures: int = 3
    initial_delay: float = 0.0

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be > 0, got {self.interval}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.consecutive_failures < 1:
            raise ValueError(
                f"consecutive_failures must be >= 1, got {self.consecutive_failures}",
            )
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")


def hc_loop(
    fn: HealthCheckFn,
    interval: float,
    timeout: float,
    initial_delay: float,
) -> Generator[HealthYield, None, None]:
    """Run the user health check forever, yielding one tuple per tick.

    Top-level so cloudpickle can serialize a ``functools.partial`` of it.
    Executes on the remote worker as a sync generator drained by
    :mod:`skyward.infra.worker`.

    Each iteration runs ``fn`` in a one-shot ``ThreadPoolExecutor`` with
    ``future.result(timeout=...)`` to enforce the per-check timeout. On
    timeout the executor is shut down with ``wait=False`` -- a hung user
    function leaks a thread, but that's bounded: after
    ``consecutive_failures`` timeouts the actor replaces the node.

    Parameters
    ----------
    fn
        User predicate.
    interval
        Sleep between checks (seconds).
    timeout
        Per-check timeout (seconds).
    initial_delay
        Settle time before the first check (seconds).

    Yields
    ------
    tuple[Literal["ok", "fail"], str | None]
        ``("ok", None)`` on healthy; ``("fail", reason)`` otherwise where
        ``reason`` is the user string, ``repr(exc)`` on exception, or
        ``"timeout after Xs"`` on timeout.
    """
    from skyward.api.runtime import instance_info

    info = instance_info()
    if initial_delay > 0:
        time.sleep(initial_delay)
    while True:
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            future = ex.submit(fn, info)
            try:
                result = future.result(timeout=timeout)
            except _FutureTimeout:
                yield ("fail", f"timeout after {timeout}s")
                continue
            except Exception as e:
                yield ("fail", repr(e))
                continue
            match result:
                case True:
                    yield ("ok", None)
                case str() as reason if reason:
                    yield ("fail", reason)
                case _:
                    yield ("fail", None)
        finally:
            ex.shutdown(wait=False)
        time.sleep(interval)
