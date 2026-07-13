"""Tests for ``LogConsole``.

Focused on the interaction between ``Node.Bootstrap.Output`` and
``Log.Emitted`` — the projection re-dispatches post-Ready bootstrap
output as ``Log.Emitted`` (see ``skyward/api/projection.py``), and the
log console must not emit the same line twice.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from skyward.console import (
    EventReceived,
    LogConsole,
    LogReceived,
    ViewUpdated,
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

        console = LogConsole()
        console.handle(ViewUpdated(view=_view(bootstrapping=False)))
        console.handle(EventReceived(event=Node.Bootstrap.Output(
            pool_name="pool-1", node_id=0, output="hello from remote",
        )))
        console.handle(EventReceived(event=Log.Emitted(
            pool_name="pool-1", node_id=0, message="hello from remote",
        )))
        console.handle(LogReceived(log=Log.Emitted(
            pool_name="pool-1", node_id=0, message="hello from remote",
        )))

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

        console = LogConsole()
        console.handle(ViewUpdated(view=_view(bootstrapping=True)))
        console.handle(EventReceived(event=Node.Bootstrap.Output(
            pool_name="pool-1", node_id=0, output="apt installing",
        )))

        _, err = capfd.readouterr()
        assert err.count("apt installing") == 1, err
