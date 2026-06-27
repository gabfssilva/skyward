"""Tests for the client-side current-session store and resolution."""

from __future__ import annotations

import pytest

from skyward.cli import _session_store as ss

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
def session_file(tmp_path, monkeypatch):
    f = tmp_path / "current-session"
    monkeypatch.setattr(ss, "SESSION_FILE", f)
    return f


def test_round_trip(session_file):
    assert ss.read_current_session() is None
    ss.write_current_session("alpha")
    assert ss.read_current_session() == "alpha"
    ss.clear_current_session()
    assert ss.read_current_session() is None
    ss.clear_current_session()  # idempotent


def test_resolve_explicit_wins(session_file, monkeypatch):
    monkeypatch.setattr(ss, "live_sessions", lambda url: ["x", "y"])
    assert ss.resolve_session("z", None) == "z"


def test_resolve_persisted_when_live(session_file, monkeypatch):
    ss.write_current_session("beta")
    monkeypatch.setattr(ss, "live_sessions", lambda url: ["beta", "other"])
    assert ss.resolve_session(None, None) == "beta"


def test_resolve_persisted_stale_falls_through(session_file, monkeypatch):
    ss.write_current_session("gone")
    monkeypatch.setattr(ss, "live_sessions", lambda url: ["only"])
    assert ss.resolve_session(None, None) == "only"
    assert ss.read_current_session() == "only"


def test_resolve_single_persists(session_file, monkeypatch):
    monkeypatch.setattr(ss, "live_sessions", lambda url: ["solo"])
    assert ss.resolve_session(None, None) == "solo"
    assert ss.read_current_session() == "solo"


def test_resolve_zero_errors(session_file, monkeypatch):
    monkeypatch.setattr(ss, "live_sessions", lambda url: [])
    with pytest.raises(SystemExit):
        ss.resolve_session(None, None)


def test_resolve_many_errors(session_file, monkeypatch):
    monkeypatch.setattr(ss, "live_sessions", lambda url: ["a", "b"])
    with pytest.raises(SystemExit):
        ss.resolve_session(None, None)
