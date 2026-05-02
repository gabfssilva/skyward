"""Tests for the live view renderer in ``sky compute view``."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _frames(*items):
    """Build an async iterator yielding the given (event_type, payload) tuples."""

    async def gen():
        for item in items:
            yield item

    return gen()


@pytest.fixture
def fake_iter_sse(monkeypatch):
    """Replace iter_sse with one that yields a canned list of frames."""

    def install(*frames):
        async def fake(_url, _name):
            for f in frames:
                yield f

        monkeypatch.setattr("skyward.cli._view.iter_sse", fake)

    return install


def _snapshot(name="p", phase="PROVISIONING", nodes=None):
    """Build a fake snapshot in the JSON shape ``pool_view_to_json`` emits.

    ``nodes`` is a list of ``{node_id, status, bootstrap}`` dicts. The
    helper reshapes them into the ``{node_id_str: node_dict}`` mapping the
    real wire format uses.
    """
    nodes_list = nodes or []
    node_map = {
        str(n["node_id"]): {"node_id": n["node_id"], "instance": None, **n}
        for n in nodes_list
    }
    return {
        "name": name,
        "phase": phase,
        "total_nodes": len(nodes_list),
        "nodes": node_map,
        "tasks": {"queued": 0, "running": 0, "done": 0, "failed": 0, "first_task_at": 0.0},
        "scaling": {"desired": 0, "pending": 0, "draining": 0, "reconciler_state": "watching"},
        "instances": [],
        "started_at": 0.0,
        "ready_at": 0.0,
    }


# ── --json mode ──────────────────────────────────────────────────


def test_ndjson_emits_one_line_per_frame(fake_iter_sse, capsys):
    from skyward.cli._view import render_ndjson

    snap = _snapshot(nodes=[{"node_id": 0, "status": "READY", "bootstrap": None}])
    fake_iter_sse(
        ("snapshot", snap),
        ("Pool.PhaseChanged", {"type": "Pool.PhaseChanged", "fields": {"pool_name": "p", "phase": "READY"}}),
        ("done", {"status": "ready"}),
    )

    code = asyncio.run(render_ndjson("http://x", "p"))
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert len(out) == 3

    import json as _json
    first = _json.loads(out[0])
    assert first["event"] == "snapshot"
    assert first["data"]["phase"] == "PROVISIONING"
    last = _json.loads(out[2])
    assert last["event"] == "done"
    assert last["data"]["status"] == "ready"


def test_ndjson_returns_1_on_failed(fake_iter_sse, capsys):
    from skyward.cli._view import render_ndjson

    fake_iter_sse(
        ("snapshot", _snapshot()),
        ("done", {"status": "failed", "error": "boom"}),
    )
    code = asyncio.run(render_ndjson("http://x", "p"))
    assert code == 1


# ── --once mode ──────────────────────────────────────────────────


def test_once_renders_snapshot_and_exits(fake_iter_sse):
    from skyward.cli._view import render_once

    snap = _snapshot(
        phase="BOOTSTRAP",
        nodes=[
            {"node_id": 0, "status": "READY", "bootstrap": None},
            {"node_id": 1, "status": "BOOTSTRAPPING", "bootstrap": {
                "phases": ["apt", "pip"],
                "completed": ["apt"],
                "active": "pip",
                "output": "installing torch",
            }},
        ],
    )
    fake_iter_sse(
        ("snapshot", snap),
        ("Pool.PhaseChanged", {"type": "Pool.PhaseChanged", "fields": {"pool_name": "p", "phase": "READY"}}),
    )

    # Just assert it runs without error and returns 0; the output goes to a
    # Rich Console writing to stderr, which capsys can't easily inspect for
    # rendered glyphs. Behavior parity is exercised through the snapshot
    # reconstruction unit tests in test_wire.
    assert asyncio.run(render_once("http://x", "p")) == 0


def test_once_handles_null_snapshot(fake_iter_sse):
    from skyward.cli._view import render_once

    fake_iter_sse(("snapshot", None))
    assert asyncio.run(render_once("http://x", "p")) == 0


# ── default mode (Rich Live) ─────────────────────────────────────


def test_live_returns_0_on_ready_done(fake_iter_sse):
    from skyward.cli._view import render_live

    fake_iter_sse(
        ("snapshot", _snapshot(nodes=[{"node_id": 0, "status": "READY", "bootstrap": None}])),
        ("Pool.PhaseChanged", {"type": "Pool.PhaseChanged", "fields": {"pool_name": "p", "phase": "READY"}}),
        ("done", {"status": "ready"}),
    )
    assert asyncio.run(render_live("http://x", "p")) == 0


def test_live_returns_1_on_failed_done(fake_iter_sse):
    from skyward.cli._view import render_live

    fake_iter_sse(
        ("snapshot", _snapshot()),
        ("Pool.ProvisionFailed", {
            "type": "Pool.ProvisionFailed",
            "fields": {"pool_name": "p", "reason": "no offers"},
        }),
        ("done", {"status": "failed", "error": "no offers"}),
    )
    assert asyncio.run(render_live("http://x", "p")) == 1


# ── log mode ─────────────────────────────────────────────────────


def test_log_emits_plain_lines_for_lifecycle_events(fake_iter_sse, capsys):
    from skyward.cli._view import render_log

    fake_iter_sse(
        ("snapshot", _snapshot()),
        ("Pool.Provisioning", {
            "type": "Pool.Provisioning",
            "fields": {"pool_name": "p", "total_nodes": 1, "started_at": 0.0},
        }),
        ("Node.Ready", {
            "type": "Node.Ready",
            "fields": {"pool_name": "p", "node_id": 0},
        }),
        ("done", {"status": "ready"}),
    )
    code = asyncio.run(render_log("http://x", "p"))
    assert code == 0
    err = capsys.readouterr().err
    assert "provisioning 1 nodes" in err
    assert "ready" in err


def test_log_returns_1_on_failed_done(fake_iter_sse, capsys):
    from skyward.cli._view import render_log

    fake_iter_sse(
        ("snapshot", _snapshot()),
        ("done", {"status": "failed", "error": "no offers"}),
    )
    assert asyncio.run(render_log("http://x", "p")) == 1
    assert "failed: no offers" in capsys.readouterr().err


# ── render() dispatcher ──────────────────────────────────────────


def test_render_dispatches_to_json(fake_iter_sse, capsys):
    from skyward.cli._view import render

    fake_iter_sse(("snapshot", _snapshot()), ("done", {"status": "ready"}))
    code = asyncio.run(render("http://x", "p", mode="json"))
    assert code == 0
    assert capsys.readouterr().out  # NDJSON written


def test_render_dispatches_to_once(fake_iter_sse, capsys):
    from skyward.cli._view import render

    fake_iter_sse(("snapshot", _snapshot()))
    code = asyncio.run(render("http://x", "p", mode="once"))
    assert code == 0


# ── Event dispatch ───────────────────────────────────────────────


def test_dispatch_event_runs_for_each_known_type():
    """Smoke test: every event handled by ``_dispatch_event`` in the live
    renderer produces output without raising."""
    from rich.console import Console

    from skyward.actors.console.state import _State
    from skyward.api.events import Error, Node, Pool, Task
    from skyward.cli._view import _dispatch_event

    console = Console()
    state = _State(total_nodes=2)

    events = [
        Pool.ProvisionFailed(pool_name="p", reason="no offers"),
        Pool.Stopped(pool_name="p"),
        Node.Ready(pool_name="p", node_id=0),
        Node.Lost(pool_name="p", node_id=0, reason="preempted"),
        Node.ConnectionFailed(pool_name="p", error="timeout"),
        Node.Preempted(pool_name="p", reason="capacity"),
        Node.WorkerFailed(pool_name="p", error="boom"),
        Node.Bootstrap.Failed(pool_name="p", node_id=0, phase="apt", error="bad"),
        Task.Queued(pool_name="p", task_id="t1", name="train", kind="run"),
        Task.Queued(pool_name="p", task_id="t2", name="train", kind="broadcast"),
        Task.Completed(pool_name="p", task_id="t1", node_id=0, elapsed=1.2),
        Task.Failed(pool_name="p", task_id="t1", node_id=0, error="boom"),
        Error.Occurred(pool_name="p", message="oops"),
    ]
    for ev in events:
        _dispatch_event(console, ev, state)  # must not raise


def test_event_from_json_roundtrip():
    """Every event the dispatcher knows about must round-trip through the wire."""
    from skyward.api.events import Pool
    from skyward.server.wire import event_from_json, event_to_json

    original = Pool.PhaseChanged(pool_name="p", phase="BOOTSTRAP")
    payload = event_to_json(original)
    reconstructed = event_from_json(payload)
    assert reconstructed == original


def test_node_connected_roundtrip_with_real_instance():
    """Node.Connected.instance must survive the wire so the client projection
    can transition the node from WAITING → SSH AND the renderer can use the
    real instance id, region, accelerator, pricing for badges."""
    from skyward.accelerators import A100
    from skyward.api.events import Node
    from skyward.api.model import Instance, InstanceType, Offer
    from skyward.server.wire import event_from_json, event_to_json

    instance = Instance(
        id="i-abc",
        status="ready",
        offer=Offer(
            id="o-1",
            instance_type=InstanceType(
                name="p4d",
                accelerator=A100(),
                vcpus=8,
                memory_gb=32,
                architecture="x86_64",
                specific=None,
            ),
            spot_price=None,
            on_demand_price=1.0,
            billing_unit="hour",
            specific=None,
        ),
        ip="1.2.3.4",
        region="us-east-1",
    )
    original = Node.Connected(pool_name="p", node_id=0, instance=instance)
    payload = event_to_json(original)

    reconstructed = event_from_json(payload)
    assert reconstructed is not None
    assert reconstructed.pool_name == "p"
    assert reconstructed.node_id == 0
    assert isinstance(reconstructed.instance, Instance)
    assert reconstructed.instance.id == "i-abc"
    assert reconstructed.instance.region == "us-east-1"
    assert reconstructed.instance.offer.instance_type.name == "p4d"
    assert reconstructed.instance.offer.instance_type.accelerator.name == "A100"


def test_pool_provisioned_roundtrip_keeps_event_intact():
    from skyward.api.events import Pool
    from skyward.server.wire import event_from_json, event_to_json

    original = Pool.Provisioned(pool_name="p", cluster=None, instances=())
    payload = event_to_json(original)
    reconstructed = event_from_json(payload)
    assert reconstructed is not None
    assert reconstructed.pool_name == "p"
    assert reconstructed.cluster is None
    assert reconstructed.instances == ()


def test_event_from_json_unknown_returns_none():
    from skyward.server.wire import event_from_json

    assert event_from_json({"type": "Bogus.Event", "fields": {}}) is None
    assert event_from_json({"type": "Pool.NotARealEvent", "fields": {}}) is None
    assert event_from_json({"fields": {}}) is None  # missing type
