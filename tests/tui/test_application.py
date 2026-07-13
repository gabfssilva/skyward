"""Tests for sky.Application — the dashboard runner (Phase 2).

The dashboard and cloud provisioning are stubbed: a fake ``Session`` carries
a real ``SessionProjection`` and a fake pool, and a fake dashboard runs the
``on_ready``/``call_from_thread(exit)`` handshake without a real terminal.
This exercises the threading/orchestration without a TTY or cloud.
"""

from __future__ import annotations

import threading

import pytest

import skyward as sky
from skyward.api.events import Pool
from skyward.api.projection import SessionProjection
from skyward.core.application import Application

pytestmark = pytest.mark.unit


class _FakePool:
    name = "app"


class _FakeSession:
    """Stands in for ``skyward.core.session.Session``."""

    instances: list[_FakeSession] = []

    def __init__(self, *, console: str, logging: object, shutdown_timeout: float) -> None:
        self.console = console
        self.entered = False
        self.exited = False
        self._projection = SessionProjection()
        self.compute_calls: list[tuple] = []
        _FakeSession.instances.append(self)

    def __enter__(self) -> _FakeSession:
        self.entered = True
        return self

    @property
    def projection(self) -> SessionProjection:
        return self._projection

    def compute(self, *specs: object, name: str | None = None, options: object = None) -> _FakePool:
        self.compute_calls.append((specs, name))
        # Feed an event so the dashboard's projection has something to show.
        self._projection.handle(Pool.Provisioning(pool_name="app", total_nodes=1, started_at=1000.0))
        return _FakePool()

    def __exit__(self, *_exc: object) -> None:
        self.exited = True


class _FakeTUI:
    """Stands in for ``SkywardTUI``: drives the ready/exit handshake."""

    instances: list[_FakeTUI] = []

    def __init__(self, source: object, *, start_dark: bool = False, on_ready=None) -> None:  # noqa: ANN001
        self.source = source
        self._on_ready = on_ready
        self._exit = threading.Event()
        self.ran = False
        _FakeTUI.instances.append(self)

    def run(self) -> None:
        self.ran = True
        if self._on_ready is not None:
            self._on_ready()
        self._exit.wait(timeout=5)

    def exit(self, *_a: object) -> None:
        self._exit.set()

    def call_from_thread(self, fn, *a: object) -> object:  # noqa: ANN001
        return fn(*a)


@pytest.fixture(autouse=True)
def _patch_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSession.instances.clear()
    _FakeTUI.instances.clear()
    monkeypatch.setattr("skyward.core.session.Session", _FakeSession)
    monkeypatch.setattr("skyward.tui.app.SkywardTUI", _FakeTUI)
    # Force the dashboard branch even though pytest has no TTY.
    monkeypatch.setenv("SKYWARD_CONSOLE_FORCE_TTY", "1")


# ── spec/options resolution ─────────────────────────────────────


def test_resolve_kwargs_builds_single_spec() -> None:
    app = Application(provider=sky.VastAI(), nodes=2)
    specs, _ = app._resolve()
    assert len(specs) == 1
    assert specs[0].nodes == 2


def test_resolve_rejects_mixing_name_and_kwargs() -> None:
    app = Application(name="train", provider=sky.AWS())
    with pytest.raises(ValueError, match="Cannot mix 'name'"):
        app._resolve()


def test_resolve_rejects_empty() -> None:
    app = Application()  # type: ignore[call-arg]  # intentionally invalid: asserts the runtime guard
    with pytest.raises(ValueError, match="Either Spec objects or keyword"):
        app._resolve()


# ── dashboard orchestration (stubbed) ───────────────────────────


def test_run_executes_workload_and_returns_result() -> None:
    app = Application(provider=sky.VastAI(), nodes=1)
    seen: dict[str, object] = {}

    def workload(pool: object) -> str:
        seen["pool"] = pool
        return "trained"

    result = app.run(workload)

    assert result == "trained"
    assert isinstance(seen["pool"], _FakePool)
    session = _FakeSession.instances[-1]
    assert session.console == "silent"  # dashboard replaces the console
    assert session.entered and session.exited  # provisioned and torn down
    assert session.compute_calls  # workload provisioned a pool
    assert _FakeTUI.instances[-1].ran  # the dashboard owned the main thread


def test_run_propagates_workload_errors_after_teardown() -> None:
    app = Application(provider=sky.VastAI(), nodes=1)

    def workload(_pool: object) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        app.run(workload)

    session = _FakeSession.instances[-1]
    assert session.exited  # teardown still happened


# ── @sky.app decorator ──────────────────────────────────────────


def test_app_decorator_injects_pool_and_forwards_args() -> None:
    @sky.app(provider=sky.VastAI(), nodes=1)
    def main(pool: object, factor: int) -> str:
        return f"{type(pool).__name__}*{factor}"

    result = main(7)

    assert result == "_FakePool*7"
    assert _FakeTUI.instances[-1].ran  # ran under the dashboard
    assert _FakeSession.instances[-1].console == "silent"


def test_app_decorator_preserves_name() -> None:
    @sky.app(provider=sky.VastAI(), nodes=1)
    def train_entry(pool: object) -> None:
        return None

    assert train_entry.__name__ == "train_entry"


def test_run_headless_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop the forced-TTY override -> headless branch (no dashboard).
    monkeypatch.delenv("SKYWARD_CONSOLE_FORCE_TTY", raising=False)

    app = Application(provider=sky.VastAI(), nodes=1)
    result = app.run(lambda pool: "ok")

    assert result == "ok"
    assert not _FakeTUI.instances  # no dashboard spawned without a TTY
    session = _FakeSession.instances[-1]
    assert session.console == "log"  # headless uses the log console
    assert session.entered and session.exited
