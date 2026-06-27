"""Tests for the HTTP server routes.

Skipped when the ``[server]`` extra (starlette + uvicorn) is not installed.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]

pytest.importorskip("starlette")


@pytest.fixture
def state():
    from skyward.server.state import ServerState

    class _FakeSession:
        def __init__(self):
            self._stopped: list[str] = []

        def stop_pool(self, name: str) -> bool:
            self._stopped.append(name)
            # If the test set up a fake pool whose teardown should run, call it
            pool = getattr(self, "_pool_for_stop", None)
            if pool is not None and hasattr(pool, "__exit__"):
                pool.__exit__(None, None, None)
            return True

    return ServerState(session=_FakeSession())


@pytest.fixture
def request_factory(state):
    from starlette.requests import Request

    class _FakeApp:
        def __init__(self, server_state):
            class _AppState:
                pass

            self.state = _AppState()
            self.state.server_state = server_state

    app = _FakeApp(state)

    def make(method: str, path: str, *, query: bytes = b"", body: bytes = b"", path_params: dict | None = None, headers: list | None = None) -> Request:
        scope = {
            "type": "http",
            "method": method,
            "headers": headers or [],
            "path": path,
            "raw_path": path.encode(),
            "query_string": query,
            "app": app,
            "path_params": path_params or {},
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return Request(scope, receive)

    return make


class _FakePool:
    def __init__(self):
        self.exited = False

    def current_nodes(self):
        return 1

    @property
    def concurrency(self):
        return 1

    @property
    def is_active(self):
        return True

    def __exit__(self, *_):
        self.exited = True
        return False


async def test_health_returns_payload(request_factory):
    from skyward.server.routes import health

    response = await health(request_factory("GET", "/health"))
    assert response.status_code == 200
    import json as _json

    body = _json.loads(response.body)
    assert body["status"] == "ok"
    assert "version" in body
    assert body["pools"] == 0
    assert body["executions"] == 0


async def test_shutdown_kills_pid_and_returns_202(monkeypatch, request_factory):
    from skyward.server import routes

    seen: dict = {}

    def fake_kill(pid, sig):  # noqa: ANN001
        seen["pid"] = pid
        seen["sig"] = sig

    monkeypatch.setattr("os.kill", fake_kill)

    response = await routes.shutdown(request_factory("POST", "/shutdown"))
    assert response.status_code == 202
    import os
    import signal

    assert seen["pid"] == os.getpid()
    assert seen["sig"] == signal.SIGTERM


# ── non-blocking create ──────────────────────────────────────────


def _encoded_payload():
    """Cloudpickle+lz4 of a (specs, options) pair the route can decode."""
    from skyward.api.spec import Options, Spec
    from skyward.providers import VastAI
    from skyward.server.wire import encode

    return encode(((Spec(provider=VastAI()),), Options()))


async def test_create_returns_202_immediately(monkeypatch, state, request_factory):
    """The handler must not wait on the (potentially slow) compute() call."""
    from skyward.server import routes

    started = asyncio.Event()
    release = asyncio.Event()

    def slow_compute(*specs, **kwargs):
        started.set()
        # block until released — we assert the response came back before this returns
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not release.is_set():
            time.sleep(0.01)
        return _FakePool()

    state.session.compute = slow_compute

    request = request_factory("POST", "/compute", query=b"name=p1", body=_encoded_payload())
    response = await asyncio.wait_for(routes.create_pool(request), timeout=1.0)

    assert response.status_code == 202
    import json as _json

    body = _json.loads(response.body)
    assert body["name"] == "p1"
    assert body["status"] == "creating"
    assert body["current_nodes"] == 0

    # cleanup: release the blocking compute call so the background task can finish
    release.set()
    entry = state.get_pool("p1")
    assert entry is not None
    assert entry.task is not None
    await entry.task


async def test_create_eventually_ready(state, request_factory):
    from skyward.server import routes

    state.session.compute = lambda *specs, **kw: _FakePool()
    request = request_factory("POST", "/compute", query=b"name=p2", body=_encoded_payload())
    response = await routes.create_pool(request)
    assert response.status_code == 202

    entry = state.get_pool("p2")
    assert entry is not None
    await entry.task

    assert entry.status == "ready"
    assert entry.pool is not None


async def test_create_failure_marks_failed(state, request_factory):
    from skyward.server import routes

    def boom(*specs, **kw):
        raise RuntimeError("provision failed")

    state.session.compute = boom
    request = request_factory("POST", "/compute", query=b"name=p3", body=_encoded_payload())
    await routes.create_pool(request)

    entry = state.get_pool("p3")
    await entry.task
    assert entry.status == "failed"
    assert "provision failed" in entry.error


async def test_create_rejects_duplicate_name(state, request_factory):
    from skyward.server import routes

    state.session.compute = lambda *s, **kw: _FakePool()
    req1 = request_factory("POST", "/compute", query=b"name=dup", body=_encoded_payload())
    await routes.create_pool(req1)

    req2 = request_factory("POST", "/compute", query=b"name=dup", body=_encoded_payload())
    response = await routes.create_pool(req2)
    assert response.status_code == 409


async def test_delete_during_creating_signals_stop_pool(state, request_factory):
    from skyward.server import routes

    release = asyncio.Event()

    def slow_compute(*s, **kw):
        import time
        while not release.is_set():
            time.sleep(0.01)
        return _FakePool()

    # As soon as stop_pool is called, release the blocking compute so the
    # background task can wind down within the delete handler's timeout.
    original_stop_pool = state.session.stop_pool

    def stop_pool_releases(name: str) -> bool:
        release.set()
        return original_stop_pool(name)

    state.session.stop_pool = stop_pool_releases
    state.session.compute = slow_compute

    create_req = request_factory("POST", "/compute", query=b"name=cx", body=_encoded_payload())
    await routes.create_pool(create_req)
    entry = state.get_pool("cx")
    assert entry.status == "creating"

    delete_req = request_factory("DELETE", "/compute/cx", path_params={"name": "cx"})
    response = await routes.delete_pool(delete_req)
    assert response.status_code == 204
    assert state.get_pool("cx") is None
    assert "cx" in state.session._stopped


async def test_delete_ready_pool_signals_stop_pool(state, request_factory):
    from skyward.server import routes

    pool = _FakePool()
    state.session.compute = lambda *s, **kw: pool
    state.session._pool_for_stop = pool  # _FakeSession.stop_pool will __exit__ this
    create_req = request_factory("POST", "/compute", query=b"name=rd", body=_encoded_payload())
    await routes.create_pool(create_req)
    entry = state.get_pool("rd")
    await entry.task
    assert entry.status == "ready"

    delete_req = request_factory("DELETE", "/compute/rd", path_params={"name": "rd"})
    response = await routes.delete_pool(delete_req)
    assert response.status_code == 204
    # The route delegates teardown to session.stop_pool — the fake invokes __exit__
    assert pool.exited is True
    assert "rd" in state.session._stopped
    assert state.get_pool("rd") is None


async def test_submit_rejects_when_not_ready(state, request_factory):
    from skyward.server import routes

    release = asyncio.Event()

    def slow_compute(*s, **kw):
        import time
        while not release.is_set():
            time.sleep(0.01)
        return _FakePool()

    state.session.compute = slow_compute
    create_req = request_factory("POST", "/compute", query=b"name=nr", body=_encoded_payload())
    await routes.create_pool(create_req)

    submit_req = request_factory(
        "POST", "/compute/nr/executions",
        path_params={"name": "nr"},
        body=b"any",
    )
    response = await routes.submit_execution(submit_req)
    assert response.status_code == 409
    import json as _json
    body = _json.loads(response.body)
    assert body["status"] == "creating"
    release.set()
    entry = state.get_pool("nr")
    if entry is not None and entry.task is not None:
        await entry.task


# ── SSE event stream ─────────────────────────────────────────────


@pytest.fixture
def sse_state(state):
    """A state whose session has a real SessionProjection."""
    from skyward.api.projection import SessionProjection

    state.session.projection = SessionProjection()
    return state


async def _drain_sse(response):
    """Collect all SSE bytes from a StreamingResponse into a list of frames."""
    chunks = []

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    async def receive():
        return {"type": "http.disconnect"}

    await response({"type": "http"}, receive, send)
    raw = b"".join(chunks).decode()
    # split on blank line; strip trailing empties
    return [b for b in raw.split("\n\n") if b.strip()]


def _parse_frame(frame: str) -> tuple[str, dict | None]:
    """Parse one SSE frame into (event_type, payload_dict)."""
    import json as _json
    event_type = ""
    data = ""
    for line in frame.splitlines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data += line[len("data:"):].strip()
    if not data:
        return event_type, None
    return event_type, _json.loads(data)


async def test_events_404_when_pool_missing(sse_state, request_factory):
    from skyward.server import routes

    response = await routes.stream_events(
        request_factory("GET", "/compute/missing/events", path_params={"name": "missing"})
    )
    assert response.status_code == 404


async def test_events_starts_with_snapshot(sse_state, request_factory):
    from skyward.server import routes

    sse_state.register_creating("p1", task=asyncio.create_task(asyncio.sleep(0)))
    response = await routes.stream_events(
        request_factory("GET", "/compute/p1/events", path_params={"name": "p1"})
    )
    frames = await _drain_sse(response)
    assert frames, "expected at least the snapshot frame"
    event_type, payload = _parse_frame(frames[0])
    assert event_type == "snapshot"
    # No PoolView yet — projection has no events for p1, so payload is None
    assert payload is None
    await sse_state.get_pool("p1").task


async def test_events_filters_by_pool_name(sse_state, request_factory):
    from skyward.api.events import Pool
    from skyward.server import routes

    sse_state.register_creating("target", task=asyncio.create_task(asyncio.sleep(0.5)))

    response_coro = routes.stream_events(
        request_factory("GET", "/compute/target/events", path_params={"name": "target"})
    )
    response = await response_coro

    chunks: list[bytes] = []
    disconnect = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if len(chunks) >= 3:
                disconnect.set()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    runner = asyncio.create_task(response({"type": "http"}, receive, send))
    await asyncio.sleep(0.05)  # let snapshot frame go out

    # Fire two events: one matching, one not
    sse_state.session.projection.handle(Pool.PhaseChanged(pool_name="target", phase="SSH"))
    sse_state.session.projection.handle(Pool.PhaseChanged(pool_name="other", phase="READY"))
    sse_state.session.projection.handle(Pool.PhaseChanged(pool_name="target", phase="BOOTSTRAP"))

    await asyncio.wait_for(runner, timeout=2.0)
    raw = b"".join(chunks).decode()
    frames = [b for b in raw.split("\n\n") if b.strip() and not b.strip().startswith(":")]
    types = [_parse_frame(f)[0] for f in frames]
    assert types[0] == "snapshot"
    assert "Pool.PhaseChanged" in types
    # No frame for the "other" pool
    payloads = [
        _parse_frame(f)[1]
        for f in frames
        if _parse_frame(f)[0] == "Pool.PhaseChanged"
    ]
    pool_names = [p["fields"]["pool_name"] for p in payloads if p]
    assert all(n == "target" for n in pool_names)
    await sse_state.get_pool("target").task


async def test_events_closes_on_failed(sse_state, request_factory):
    from skyward.api.events import Pool
    from skyward.server import routes

    task = asyncio.create_task(asyncio.sleep(0.5))
    sse_state.register_creating("bad", task=task)
    response = await routes.stream_events(
        request_factory("GET", "/compute/bad/events", path_params={"name": "bad"})
    )

    chunks: list[bytes] = []
    done_event = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if b"event: done" in message.get("body", b""):
                done_event.set()

    async def receive():
        await done_event.wait()
        return {"type": "http.disconnect"}

    runner = asyncio.create_task(response({"type": "http"}, receive, send))
    await asyncio.sleep(0.05)
    sse_state.set_failed("bad", "boom")
    sse_state.session.projection.handle(Pool.ProvisionFailed(pool_name="bad", reason="boom"))

    await asyncio.wait_for(runner, timeout=2.0)
    raw = b"".join(chunks).decode()
    assert "event: done" in raw
    task.cancel()


async def test_events_disconnect_unsubscribes(sse_state, request_factory):
    from skyward.server import routes

    task = asyncio.create_task(asyncio.sleep(0.5))
    sse_state.register_creating("x", task=task)
    proj = sse_state.session.projection
    initial_subs = len(proj._event_subs)

    response = await routes.stream_events(
        request_factory("GET", "/compute/x/events", path_params={"name": "x"})
    )

    async def send(message):
        return

    async def receive():
        return {"type": "http.disconnect"}

    await response({"type": "http"}, receive, send)
    # After the generator's finally runs, callback is removed
    assert len(proj._event_subs) == initial_subs
    task.cancel()


async def test_get_includes_status(state, request_factory):
    from skyward.server import routes

    state.session.compute = lambda *s, **kw: _FakePool()
    create_req = request_factory("POST", "/compute", query=b"name=gs", body=_encoded_payload())
    await routes.create_pool(create_req)
    entry = state.get_pool("gs")
    await entry.task

    get_req = request_factory("GET", "/compute/gs", path_params={"name": "gs"})
    response = await routes.get_pool(get_req)
    assert response.status_code == 200
    import json as _json
    body = _json.loads(response.body)
    assert body["status"] == "ready"
    assert body["current_nodes"] == 1


# ── node listing (W0) ────────────────────────────────────────────


def _node_snapshot(node_id: int, instance_id: str):
    from skyward.actors.snapshot import NodeSnapshot, NodeStatus

    return NodeSnapshot(
        node_id=node_id, instance_id=instance_id, status=NodeStatus.READY,
    )


def _instance(instance_id: str, ip: str, *, ssh_password: str | None = None):
    from skyward.api.model import Instance, InstanceType, Offer

    itype = InstanceType(
        name="t", accelerator=None, vcpus=1, memory_gb=1,
        architecture="x86_64", specific=None,
    )
    offer = Offer(
        id="o", instance_type=itype, spot_price=1.0, on_demand_price=2.0,
        billing_unit="hour", specific=None,
    )
    return Instance(
        id=instance_id, status="ready", offer=offer, ip=ip,
        private_ip="10.0.0." + instance_id[-1], ssh_port=2222,
        ssh_password=ssh_password,
    )


def _two_node_snapshot():
    """Snapshot with rank 0 (head) + rank 1, both ready with instances."""
    from skyward.actors.snapshot import (
        PoolPhase,
        PoolSnapshot,
        ScalingSnapshot,
        TaskCounters,
    )
    from skyward.api.model import Cluster
    from skyward.api.spec import PoolSpec

    inst0 = _instance("i-0", "1.1.1.1")
    inst1 = _instance("i-1", "2.2.2.2", ssh_password="secret")
    cluster = Cluster(
        id="c", status="ready", spec=PoolSpec.__new__(PoolSpec),
        offer=inst0.offer, ssh_key_path="/tmp/does-not-exist-key",
        ssh_user="ubuntu", use_sudo=False, shutdown_command="",
        specific=None, instances=(inst0, inst1),
    )
    return PoolSnapshot(
        name="n", phase=PoolPhase.READY,
        nodes=(_node_snapshot(1, "i-1"), _node_snapshot(0, "i-0")),
        tasks=TaskCounters(), scaling=ScalingSnapshot(),
        cluster=cluster, instances=(inst0, inst1),
    )


class _NodesPool(_FakePool):
    def snapshot(self):
        return _two_node_snapshot()


class _CapturePool(_FakePool):
    """Captures the target passed to run_async for node-targeting tests."""

    def __init__(self):
        super().__init__()
        self.captured: dict = {}

    def run_async(self, pending, *, task_id=None, target=None):
        self.captured["target"] = target
        fut: Future = Future()
        fut.set_result({"exit": 0})
        return fut


async def test_list_nodes_404_when_missing(state, request_factory):
    from skyward.server import routes

    req = request_factory("GET", "/compute/none/nodes", path_params={"name": "none"})
    response = await routes.list_nodes(req)
    assert response.status_code == 404


async def test_list_nodes_409_when_not_ready(state, request_factory):
    from skyward.server import routes

    release = asyncio.Event()

    def slow_compute(*s, **kw):
        import time
        while not release.is_set():
            time.sleep(0.01)
        return _NodesPool()

    state.session.compute = slow_compute
    await routes.create_pool(
        request_factory("POST", "/compute", query=b"name=nn", body=_encoded_payload()),
    )
    req = request_factory("GET", "/compute/nn/nodes", path_params={"name": "nn"})
    response = await routes.list_nodes(req)
    assert response.status_code == 409
    import json as _json
    assert _json.loads(response.body)["status"] == "creating"
    release.set()
    entry = state.get_pool("nn")
    if entry is not None and entry.task is not None:
        await entry.task


async def test_list_nodes_returns_rank_ordered(state, request_factory):
    from skyward.server import routes

    state.session.compute = lambda *s, **kw: _NodesPool()
    await routes.create_pool(
        request_factory("POST", "/compute", query=b"name=ln", body=_encoded_payload()),
    )
    entry = state.get_pool("ln")
    await entry.task

    req = request_factory("GET", "/compute/ln/nodes", path_params={"name": "ln"})
    response = await routes.list_nodes(req)
    assert response.status_code == 200
    import json as _json
    body = _json.loads(response.body)
    ranks = [n["rank"] for n in body["nodes"]]
    assert ranks == [0, 1]
    head, second = body["nodes"]
    assert head["is_head"] is True
    assert head["ip"] == "1.1.1.1"
    assert head["ssh_user"] == "ubuntu"
    assert head["ssh_port"] == 2222
    assert head["has_password"] is False
    assert head["key_exists_on_server"] is False
    assert second["is_head"] is False
    assert second["has_password"] is True
    assert "secret" not in response.body.decode()


# ── node-targeted execution (W2) ─────────────────────────────────


async def _ready_capture_pool(state, request_factory, name: str) -> _CapturePool:
    from skyward.server import routes

    pool = _CapturePool()
    state.session.compute = lambda *s, **kw: pool
    await routes.create_pool(
        request_factory("POST", "/compute", query=f"name={name}".encode(), body=_encoded_payload()),
    )
    await state.get_pool(name).task
    return pool


async def test_submit_run_node_head_threads_target(state, request_factory):
    from skyward.server import routes
    from skyward.server.wire import encode

    pool = await _ready_capture_pool(state, request_factory, "tgh")
    req = request_factory(
        "POST", "/compute/tgh/executions", query=b"mode=run&node=head",
        path_params={"name": "tgh"}, body=encode({"x": 1}),
    )
    resp = await routes.submit_execution(req)
    assert resp.status_code == 202
    assert pool.captured["target"] == "head"


async def test_submit_run_node_int_threads_target(state, request_factory):
    from skyward.server import routes
    from skyward.server.wire import encode

    pool = await _ready_capture_pool(state, request_factory, "tgi")
    req = request_factory(
        "POST", "/compute/tgi/executions", query=b"mode=run&node=2",
        path_params={"name": "tgi"}, body=encode({"x": 1}),
    )
    resp = await routes.submit_execution(req)
    assert resp.status_code == 202
    assert pool.captured["target"] == 2


async def test_submit_run_node_invalid_400(state, request_factory):
    from skyward.server import routes
    from skyward.server.wire import encode

    await _ready_capture_pool(state, request_factory, "tgx")
    req = request_factory(
        "POST", "/compute/tgx/executions", query=b"mode=run&node=abc",
        path_params={"name": "tgx"}, body=encode({"x": 1}),
    )
    resp = await routes.submit_execution(req)
    assert resp.status_code == 400


# ── file operations (W3) ─────────────────────────────────────────


def _file_pool():
    from skyward.actors.messages import NodeFileResult
    from skyward.core.pool import ComputePool

    class _FilePool(ComputePool):
        def __init__(self):
            self._store: dict[str, bytes] = {}

        def current_nodes(self):
            return 1

        @property
        def concurrency(self):
            return 1

        @property
        def is_active(self):
            return True

        def __exit__(self, *_):
            return False

        def ls(self, path, node="head", timeout=30.0):
            return (NodeFileResult(node_id=0, success=True, listing=f"total 0\n{path}"),)

        def rm(self, path, node="all", timeout=30.0):
            return (NodeFileResult(node_id=0, success=True),)

        def upload_file(self, content, remote, node="all", timeout=120.0):
            self._store[remote] = content
            return (NodeFileResult(node_id=0, success=True),)

        def download_file(self, remote, node="head", timeout=120.0):
            if remote in self._store:
                return (NodeFileResult(node_id=0, success=True, content=self._store[remote]),)
            return (NodeFileResult(node_id=0, success=False, error="not found"),)

    return _FilePool()


async def _ready_file_pool(state, request_factory, name: str):
    from skyward.server import routes

    pool = _file_pool()
    state.session.compute = lambda *s, **kw: pool
    await routes.create_pool(
        request_factory("POST", "/compute", query=f"name={name}".encode(), body=_encoded_payload()),
    )
    await state.get_pool(name).task
    return pool


async def test_files_404_when_missing(state, request_factory):
    from skyward.server import routes

    req = request_factory("GET", "/compute/none/files", query=b"path=/x", path_params={"name": "none"})
    resp = await routes.list_files(req)
    assert resp.status_code == 404


async def test_files_path_required_400(state, request_factory):
    from skyward.server import routes

    await _ready_file_pool(state, request_factory, "fp0")
    req = request_factory("GET", "/compute/fp0/files", path_params={"name": "fp0"})
    resp = await routes.list_files(req)
    assert resp.status_code == 400


async def test_ls_returns_results(state, request_factory):
    from skyward.server import routes

    await _ready_file_pool(state, request_factory, "fp1")
    req = request_factory("GET", "/compute/fp1/files", query=b"path=/opt&node=head", path_params={"name": "fp1"})
    resp = await routes.list_files(req)
    assert resp.status_code == 200
    import json as _json
    body = _json.loads(resp.body)
    assert body["results"][0]["success"] is True
    assert "/opt" in body["results"][0]["listing"]


async def test_upload_then_download_round_trips(state, request_factory):
    from skyward.server import routes

    await _ready_file_pool(state, request_factory, "fp2")
    payload = b"\x00\x01binary\xff"
    put = request_factory(
        "PUT", "/compute/fp2/files", query=b"path=/tmp/x.bin&node=all",
        path_params={"name": "fp2"}, body=payload,
    )
    assert (await routes.upload_file(put)).status_code == 200

    get = request_factory(
        "GET", "/compute/fp2/files/content", query=b"path=/tmp/x.bin&node=head",
        path_params={"name": "fp2"},
    )
    resp = await routes.download_file(get)
    assert resp.status_code == 200
    assert resp.body == payload


async def test_download_all_rejected_422(state, request_factory):
    from skyward.server import routes

    await _ready_file_pool(state, request_factory, "fp3")
    get = request_factory(
        "GET", "/compute/fp3/files/content", query=b"path=/tmp/x&node=all",
        path_params={"name": "fp3"},
    )
    assert (await routes.download_file(get)).status_code == 422


async def test_download_missing_file_404(state, request_factory):
    from skyward.server import routes

    await _ready_file_pool(state, request_factory, "fp4")
    get = request_factory(
        "GET", "/compute/fp4/files/content", query=b"path=/nope&node=head",
        path_params={"name": "fp4"},
    )
    assert (await routes.download_file(get)).status_code == 404


# ── log export (W6) ──────────────────────────────────────────────


async def test_get_pool_log_404_when_no_history(state, request_factory):
    from skyward.server import routes

    req = request_factory("GET", "/compute/nohist/log", path_params={"name": "nohist"})
    assert (await routes.get_pool_log(req)).status_code == 404


async def test_get_pool_log_returns_events(state, request_factory):
    from skyward.api.events import Log
    from skyward.server import routes

    for i in range(3):
        state.history.append(Log.Emitted(pool_name="lg", node_id=0, message=f"m{i}", task_id="t"))
    req = request_factory("GET", "/compute/lg/log", path_params={"name": "lg"})
    resp = await routes.get_pool_log(req)
    assert resp.status_code == 200
    import json as _json
    body = _json.loads(resp.body)
    assert body["count"] == 3
    assert body["events"][0]["fields"]["message"] == "m0"


async def test_get_pool_log_limit(state, request_factory):
    from skyward.api.events import Log
    from skyward.server import routes

    for i in range(5):
        state.history.append(Log.Emitted(pool_name="lq", node_id=0, message=f"m{i}", task_id="t"))
    req = request_factory("GET", "/compute/lq/log", query=b"limit=2", path_params={"name": "lq"})
    body = __import__("json").loads((await routes.get_pool_log(req)).body)
    assert body["count"] == 2
    assert body["events"][-1]["fields"]["message"] == "m4"


async def test_submit_captures_source_header(state, request_factory):
    import base64

    from skyward.server import routes
    from skyward.server.wire import encode

    await _ready_capture_pool(state, request_factory, "src")
    source = "print('hello')\nx = 1\n"
    header = base64.b64encode(source.encode()).decode("ascii")
    req = request_factory(
        "POST", "/compute/src/executions", query=b"mode=run",
        path_params={"name": "src"}, body=encode({"x": 1}),
        headers=[(b"x-skyward-source", header.encode())],
    )
    resp = await routes.submit_execution(req)
    assert resp.status_code == 202
    eid = __import__("json").loads(resp.body)["id"]
    assert state.history.sources("src")[eid] == source
