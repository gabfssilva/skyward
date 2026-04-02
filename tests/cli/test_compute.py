"""Tests for sky compute commands."""

from __future__ import annotations

import json
from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import pytest

from skyward.cli import app
from skyward.daemon.protocol import (
    DaemonError,
    PoolList,
    PoolShutdown,
    PoolSummary,
    PoolViewResponse,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _make_summary(**overrides: object) -> PoolSummary:
    defaults = {
        "name": "train",
        "phase": "READY",
        "nodes_ready": 2,
        "nodes_total": 4,
        "tasks_done": 10,
        "tasks_running": 3,
        "started_at": 1000.0,
    }
    return PoolSummary(**{**defaults, **overrides})


def _mock_request(response: object) -> AsyncMock:
    return AsyncMock(return_value=response)


class TestComputeRegistered:
    def test_compute_subcommand_visible_in_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit, match="0"):
            app(["--help"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "Manage running compute pools" in out

    def test_compute_app_in_all(self) -> None:
        from skyward.cli import __all__

        assert "compute_app" in __all__


class TestListPools:
    def test_list_json_with_pools(self, capsys: pytest.CaptureFixture[str]) -> None:
        summaries = (_make_summary(), _make_summary(name="infer", phase="PROVISIONING", tasks_done=0))
        mock = _mock_request(PoolList(pools=summaries))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "list", "--json"], exit_on_error=False)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2
        assert data[0]["name"] == "train"
        assert data[1]["name"] == "infer"
        assert data[0]["phase"] == "READY"
        assert data[0]["nodes_ready"] == 2
        assert data[0]["tasks_done"] == 10

    def test_list_empty_pools(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(PoolList(pools=()))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "list"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "No pools running" in out

    def test_list_empty_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(PoolList(pools=()))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "list", "--json"], exit_on_error=False)

        out = capsys.readouterr().out
        assert json.loads(out) == []

    def test_list_rich_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        summaries = (_make_summary(),)
        mock = _mock_request(PoolList(pools=summaries))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "list"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "train" in out
        assert "ready" in out

    def test_default_command_is_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(PoolList(pools=()))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "No pools running" in out


class TestViewPool:
    def test_view_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        view = PoolView(
            name="train",
            phase=PoolPhase.READY,
            total_nodes=2,
            nodes=MappingProxyType({
                0: NodeView(node_id=0, status=NodeStatus.READY),
                1: NodeView(node_id=1, status=NodeStatus.READY),
            }),
            tasks=TasksView(done=5, running=1),
            scaling=ScalingView(desired=2),
        )
        mock = _mock_request(PoolViewResponse(view=view))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "view", "train", "--json"], exit_on_error=False)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["name"] == "train"
        assert data["phase"] == "READY"
        assert data["total_nodes"] == 2
        assert "0" in data["nodes"]

    def test_view_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(DaemonError(error="Pool 'nope' not found"))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "view", "nope"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "not found" in out

    def test_view_rich(self, capsys: pytest.CaptureFixture[str]) -> None:
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        view = PoolView(
            name="train",
            phase=PoolPhase.READY,
            total_nodes=1,
            nodes=MappingProxyType({
                0: NodeView(node_id=0, status=NodeStatus.READY),
            }),
            tasks=TasksView(),
            scaling=ScalingView(desired=1),
        )
        mock = _mock_request(PoolViewResponse(view=view))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "view", "train"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "train" in out
        assert "ready" in out


class TestShowTasks:
    def test_tasks_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        from skyward.api.views import (
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        view = PoolView(
            name="train",
            phase=PoolPhase.READY,
            total_nodes=1,
            tasks=TasksView(done=5, failed=1, running=2, throughput=3.5),
            scaling=ScalingView(desired=1),
        )
        mock = _mock_request(PoolViewResponse(view=view))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "tasks", "train", "--json"], exit_on_error=False)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["done"] == 5
        assert data["failed"] == 1
        assert data["throughput"] == 3.5

    def test_tasks_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(DaemonError(error="Pool 'x' not found"))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "tasks", "x"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "not found" in out


class TestShowStats:
    def test_stats_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        from skyward.api.views import (
            NodeStatus,
            NodeView,
            PoolPhase,
            PoolView,
            ScalingView,
            TasksView,
        )

        view = PoolView(
            name="train",
            phase=PoolPhase.READY,
            total_nodes=1,
            nodes=MappingProxyType({
                0: NodeView(
                    node_id=0,
                    status=NodeStatus.READY,
                    metrics=MappingProxyType({"gpu_util": 87.5, "mem_used": 12.3}),
                ),
            }),
            tasks=TasksView(),
            scaling=ScalingView(desired=1),
        )
        mock = _mock_request(PoolViewResponse(view=view))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "stats", "train", "--json"], exit_on_error=False)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["0"]["gpu_util"] == 87.5
        assert data["0"]["mem_used"] == 12.3


class TestStopPool:
    def test_stop_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(PoolShutdown())

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "stop", "train"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "stopped" in out

    def test_stop_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock = _mock_request(DaemonError(error="Pool 'nope' not found"))

        with patch("skyward.cli.compute._daemon_request", mock), pytest.raises(SystemExit, match="0"):
            app(["compute", "stop", "nope"], exit_on_error=False)

        out = capsys.readouterr().out
        assert "not found" in out
