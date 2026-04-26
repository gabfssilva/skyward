"""End-to-end tests for the periodic remote health probe.

These tests use ``sky.Container`` (Docker-based) to exercise the full
streaming health-check path: ``execute_with_streaming``-equivalent
dispatch on the node actor, generator drained on the worker, async
iterator consumed inside the actor, and (for the negative case) the
``Preempted -> NodeExhausted -> reconciler`` replacement loop.
"""

from __future__ import annotations

import time

import pytest

import skyward as sky

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("health")]


@sky.function
def noop() -> int:
    return 42


class TestHealthChecker:
    def test_healthy_node_keeps_running(self) -> None:
        """Healthy ticks must not trigger replacement; tasks still dispatch."""
        with sky.Compute(
            provider=sky.Container(),
            nodes=1,
            options=sky.Options(
                health_checker=sky.HealthChecker(
                    fn=lambda _info: True, interval=1.0, timeout=2.0,
                    consecutive_failures=2, initial_delay=0.0,
                ),
            ),
        ) as compute:
            time.sleep(5.0)
            assert noop() >> compute == 42

    def test_warming_gates_pool_ready(self) -> None:
        """Pool entry must block until the first healthy check.

        Spins up two pools in the same session: one with an
        always-healthy probe, one whose probe fails twice before passing
        on the third tick. The warming pool must take at least the two
        extra intervals longer to become operational, proving that
        ``NodeActivated`` is delayed by the warming gate.
        """

        def make_warming_check() -> object:
            counter = {"n": 0}

            def fn(_info: object) -> bool:
                counter["n"] += 1
                return counter["n"] >= 3

            return fn

        with sky.Session(console=False) as session:
            t0 = time.monotonic()
            session.compute(
                provider=sky.Container(),
                nodes=1, name="healthy",
                options=sky.Options(
                    health_checker=sky.HealthChecker(
                        fn=lambda _info: True, interval=1.0, timeout=2.0,
                        consecutive_failures=5, initial_delay=0.0,
                    ),
                ),
            )
            t_healthy = time.monotonic() - t0

            t0 = time.monotonic()
            session.compute(
                provider=sky.Container(),
                nodes=1, name="warming",
                options=sky.Options(
                    health_checker=sky.HealthChecker(
                        fn=make_warming_check(),  # type: ignore[arg-type]
                        interval=1.0, timeout=2.0,
                        consecutive_failures=5, initial_delay=0.0,
                    ),
                ),
            )
            t_warming = time.monotonic() - t0

        assert t_warming >= t_healthy + 1.5, (
            f"warming pool ready in {t_warming:.2f}s; healthy pool in "
            f"{t_healthy:.2f}s; expected >= 1.5s extra for warming"
        )

    def test_post_warming_failure_triggers_preemption(self) -> None:
        """A node that warms up then fails must preempt.

        ``fn`` returns ``True`` once (clears warming, activates node)
        then fails forever. Subscribes to the session projection and
        asserts ``Node.Preempted`` fires with a "health check" reason --
        proves the full
        ``warming -> active -> fail x N -> Preempted -> NodeExhausted``
        chain runs.
        """
        from skyward.api.events import Node
        from skyward.core.session import Session as _CoreSession

        def make_fn() -> object:
            state = {"n": 0}

            def fn(_info: object) -> bool | str:
                state["n"] += 1
                if state["n"] == 1:
                    return True
                return "synthetic failure"

            return fn

        seen_preempted: list[Node.Preempted] = []

        with _CoreSession(console=False) as session:
            session.projection.subscribe(
                on_event=lambda ev: (
                    seen_preempted.append(ev)
                    if isinstance(ev, Node.Preempted) else None
                ),
            )
            session.compute(
                provider=sky.Container(),
                nodes=(1, 2),
                options=sky.Options(
                    health_checker=sky.HealthChecker(
                        fn=make_fn(),  # type: ignore[arg-type]
                        interval=1.0, timeout=2.0,
                        consecutive_failures=2, initial_delay=0.0,
                    ),
                    reconcile_tick_interval=2.0,
                ),
            )
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not seen_preempted:
                time.sleep(0.5)

        assert seen_preempted, "Node.Preempted event never fired"
        assert any(
            "health check" in ev.reason for ev in seen_preempted
        ), f"no preemption reason mentions health check; got {[ev.reason for ev in seen_preempted]}"
