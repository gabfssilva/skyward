"""Tests for ``log_console_actor``.

Focused on the interaction between ``Node.Bootstrap.Output`` and
``Log.Emitted`` — the projection re-dispatches post-Ready bootstrap
output as ``Log.Emitted`` (see ``skyward/api/projection.py``), and the
log console must not emit the same line twice.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType

import pytest
from casty import ActorSystem

from skyward.actors.console import (
    EventReceived,
    LogReceived,
    ViewUpdated,
    log_console_actor,
)
from skyward.api.events import Log, Node
from skyward.api.views import (
    BootstrapView,
    NodeStatus,
    NodeView,
    PoolPhase,
    PoolView,
    ScalingView,
    SessionView,
    TasksView,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _view(*, bootstrapping: bool) -> SessionView:
    if bootstrapping:
        node = NodeView(
            node_id=0,
            status=NodeStatus.BOOTSTRAPPING,
            bootstrap=BootstrapView(
                phases=("apt",), completed=frozenset(), active="apt", output="",
            ),
        )
    else:
        node = NodeView(node_id=0, status=NodeStatus.READY, bootstrap=None)
    pool = PoolView(
        name="pool-1",
        phase=PoolPhase.BOOTSTRAP if bootstrapping else PoolPhase.READY,
        tasks=TasksView(),
        scaling=ScalingView(desired=1),
        total_nodes=1,
        nodes=MappingProxyType({0: node}),
    )
    return SessionView(pools=MappingProxyType({"pool-1": pool}))


class TestLogConsoleStdoutDuplication:
    def test_post_ready_stdout_emitted_once(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Post-Ready bootstrap output arrives as both Bootstrap.Output
        (via ``_event_subs``) and Log.Emitted (via projection's recursive
        handling when ``node.bootstrap is None``). The log console must
        deduplicate.
        """

        async def run() -> None:
            async with ActorSystem("test-log-dup") as system:
                ref = system.spawn(log_console_actor(), "log-console")
                ref.tell(ViewUpdated(view=_view(bootstrapping=False)))
                await asyncio.sleep(0.05)

                ref.tell(EventReceived(event=Node.Bootstrap.Output(
                    pool_name="pool-1", node_id=0, output="hello from remote",
                )))
                ref.tell(EventReceived(event=Log.Emitted(
                    pool_name="pool-1", node_id=0, message="hello from remote",
                )))
                ref.tell(LogReceived(log=Log.Emitted(
                    pool_name="pool-1", node_id=0, message="hello from remote",
                )))
                await asyncio.sleep(0.1)

        asyncio.run(run())

        _, err = capfd.readouterr()
        occurrences = err.count("hello from remote")
        assert occurrences == 1, f"expected one emit, got {occurrences}:\n{err}"

    def test_bootstrap_output_during_bootstrap_still_emits(
        self, capfd: pytest.CaptureFixture[str],
    ) -> None:
        """While the node is bootstrapping, the projection does not
        convert the event to ``Log.Emitted`` — so the log console's
        ``Bootstrap.Output`` handler is the only path and must emit.
        """

        async def run() -> None:
            async with ActorSystem("test-log-bootstrap") as system:
                ref = system.spawn(log_console_actor(), "log-console")
                ref.tell(ViewUpdated(view=_view(bootstrapping=True)))
                await asyncio.sleep(0.05)

                ref.tell(EventReceived(event=Node.Bootstrap.Output(
                    pool_name="pool-1", node_id=0, output="apt installing",
                )))
                await asyncio.sleep(0.1)

        asyncio.run(run())

        _, err = capfd.readouterr()
        assert err.count("apt installing") == 1, err
