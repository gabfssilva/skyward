"""Tests for the on-disk session reattach handle store."""

from __future__ import annotations

import pytest

from skyward.core.session_store import (
    NodeHandle,
    SessionHandle,
    list_handles,
    pack_payload,
    read_handle,
    remove_handle,
    unpack_payload,
    write_handle,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _handle(name: str = "s", payload: bytes = b"data") -> SessionHandle:
    return SessionHandle(
        version=1, name=name, created_at="2026-06-16", cluster_id="c-1",
        prebaked=False, head_node_id=0,
        nodes=(
            NodeHandle(
                node_id=0, instance_id="i-0", ip="1.2.3.4", private_ip="10.0.0.1",
                ssh_port=22, ssh_user="ubuntu", ssh_key_path="/k", ssh_password=None,
            ),
            NodeHandle(
                node_id=1, instance_id="i-1", ip="5.6.7.8", private_ip=None,
                ssh_port=2222, ssh_user="root", ssh_key_path="/k", ssh_password="pw",
            ),
        ),
        payload=payload,
    )


def test_write_read_round_trip(tmp_path):
    h = _handle()
    write_handle(h, sessions_dir=tmp_path)
    assert read_handle("s", sessions_dir=tmp_path) == h


def test_atomic_write_leaves_no_tmp(tmp_path):
    write_handle(_handle(), sessions_dir=tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_pack_unpack_round_trips_arbitrary_objects():
    cfg = {"region": "us-east-1"}
    cluster = {"specific": [1, 2, 3], "nested": {"k": "v"}}
    cfg2, cluster2 = unpack_payload(pack_payload(cfg, cluster))
    assert cfg2 == cfg
    assert cluster2 == cluster


def test_read_missing_returns_none(tmp_path):
    assert read_handle("nope", sessions_dir=tmp_path) is None


def test_read_corrupt_returns_none(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    assert read_handle("bad", sessions_dir=tmp_path) is None


def test_read_version_mismatch_returns_none(tmp_path):
    write_handle(_handle(), sessions_dir=tmp_path)
    f = tmp_path / "s.json"
    f.write_text(f.read_text().replace('"version": 1', '"version": 99'))
    assert read_handle("s", sessions_dir=tmp_path) is None


def test_list_handles_skips_bad_files(tmp_path):
    write_handle(_handle("a"), sessions_dir=tmp_path)
    write_handle(_handle("b"), sessions_dir=tmp_path)
    (tmp_path / "broken.json").write_text("garbage")
    names = {h.name for h in list_handles(sessions_dir=tmp_path)}
    assert names == {"a", "b"}


def test_remove_handle(tmp_path):
    write_handle(_handle(), sessions_dir=tmp_path)
    remove_handle("s", sessions_dir=tmp_path)
    assert read_handle("s", sessions_dir=tmp_path) is None
    remove_handle("s", sessions_dir=tmp_path)  # idempotent
