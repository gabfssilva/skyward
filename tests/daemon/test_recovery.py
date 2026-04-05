from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import pytest
from casty import InMemoryJournal, PersistedEvent

from skyward.daemon.server import DaemonServer
from skyward.daemon.state import PoolRegistered


def _short_sock(tmp_path: Path) -> Path:
    """Return a short socket path to avoid AF_UNIX length limits."""
    import tempfile
    short = Path(tempfile.mkdtemp()) / "d.sock"
    return short


@dataclass(frozen=True, slots=True)
class _FakeCluster:
    id: str
    ssh_key_path: str
    ssh_user: str = "ubuntu"
    instances: tuple[object, ...] = ()
    use_sudo: bool = False
    shutdown_command: str = "shutdown -h now"
    prebaked: bool = False


@dataclass(frozen=True, slots=True)
class _FakeInstance:
    id: str
    status: str
    ip: str | None = None
    ssh_port: int = 22
    spot: bool = False


@dataclass(frozen=True, slots=True)
class _FakeNodes:
    desired: int = 2
    max: int | None = None
    min: int | None = None


@dataclass(frozen=True, slots=True)
class _FakeSpec:
    nodes: _FakeNodes
    ttl: int = 600
    ssh_timeout: float = 30.0
    ssh_retry_interval: float = 5.0
    provision_timeout: float = 60.0
    bootstrap_timeout: float = 120.0


def _make_registered_event(
    tmp_path: Path, *, name: str = "train",
    instance_ids: tuple[str, ...] = ("i-1", "i-2"),
) -> PoolRegistered:
    """Build a PoolRegistered event with serialized recovery data."""
    cluster = _FakeCluster(id="c-123", ssh_key_path=str(tmp_path / "ssh_key"))
    spec = _FakeSpec(nodes=_FakeNodes(desired=len(instance_ids)))
    config = MagicMock()

    return PoolRegistered(
        pool_name=name,
        cluster_id="c-123",
        instance_ids=instance_ids,
        provider_name="aws",
        cluster_bytes=cloudpickle.dumps(cluster),
        spec_bytes=cloudpickle.dumps(spec),
        provider_config_bytes=cloudpickle.dumps(config),
    )


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_recovers_pool_entries_from_journal(self, tmp_path: Path) -> None:
        """After restart, daemon reads journal and knows which pools existed."""
        journal = InMemoryJournal()
        event = _make_registered_event(tmp_path)
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)
        server._recover_pool = AsyncMock(return_value=True)

        async with server:
            server._recover_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_skips_when_ssh_key_missing(self, tmp_path: Path) -> None:
        """No SSH key on disk -> terminate instances, remove from journal."""
        journal = InMemoryJournal()
        event = _make_registered_event(tmp_path)
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        mock_provider = MagicMock()
        mock_provider.terminate = AsyncMock()
        mock_provider.teardown = AsyncMock()

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        with patch.object(server, "_create_provider", new_callable=AsyncMock, return_value=mock_provider):
            async with server:
                assert "train" not in server._pools

    @pytest.mark.asyncio
    async def test_recovery_succeeds_all_instances_alive(self, tmp_path: Path) -> None:
        """All instances alive -> pool recovered and available."""
        journal = InMemoryJournal()
        event = _make_registered_event(tmp_path)
        (tmp_path / "ssh_key").touch()
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        mock_provider = MagicMock()
        mock_provider.get_instance = AsyncMock(
            side_effect=lambda cluster, iid: (cluster, _FakeInstance(
                id=iid, status="provisioned", ip="10.0.0.1",
            )),
        )

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        with patch.object(server, "_create_provider", new_callable=AsyncMock, return_value=mock_provider), \
             patch.object(server, "_recover_pool_via_session", new_callable=AsyncMock) as mock_session:
            mock_session.return_value = MagicMock()
            async with server:
                mock_session.assert_called_once()
                assert "train" in server._pools

    @pytest.mark.asyncio
    async def test_recovery_terminates_when_below_min_nodes(self, tmp_path: Path) -> None:
        """Not enough alive instances -> terminate survivors, clean up."""
        journal = InMemoryJournal()
        event = _make_registered_event(tmp_path, instance_ids=("i-1", "i-2"))
        (tmp_path / "ssh_key").touch()
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        mock_provider = MagicMock()
        mock_provider.get_instance = AsyncMock(
            side_effect=lambda cluster, iid: (
                cluster,
                _FakeInstance(id=iid, status="provisioned", ip="10.0.0.1")
                if iid == "i-1"
                else None,
            ),
        )
        mock_provider.terminate = AsyncMock()
        mock_provider.teardown = AsyncMock()

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        with patch.object(server, "_create_provider", new_callable=AsyncMock, return_value=mock_provider):
            async with server:
                mock_provider.terminate.assert_called_once()
                assert "train" not in server._pools

    @pytest.mark.asyncio
    async def test_recovery_handles_deserialization_error(self, tmp_path: Path) -> None:
        """Corrupt cluster_bytes -> skip pool, log error, don't crash."""
        journal = InMemoryJournal()
        event = PoolRegistered(
            pool_name="train",
            cluster_id="c-bad",
            instance_ids=("i-1",),
            provider_name="aws",
            cluster_bytes=b"not-valid-pickle",
            spec_bytes=b"not-valid-pickle",
            provider_config_bytes=b"not-valid-pickle",
        )
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        async with server:
            assert "train" not in server._pools

    @pytest.mark.asyncio
    async def test_recovery_skips_legacy_entries_without_bytes(self, tmp_path: Path) -> None:
        """Old journal entries without recovery bytes -> skip gracefully."""
        journal = InMemoryJournal()
        event = PoolRegistered(
            pool_name="old-pool",
            cluster_id="c-old",
            instance_ids=("i-1",),
        )
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        async with server:
            assert "old-pool" not in server._pools

    @pytest.mark.asyncio
    async def test_recovery_starts_ttl_for_recovered_pools(self, tmp_path: Path) -> None:
        """Recovered pool has no clients -> idle TTL starts immediately."""
        journal = InMemoryJournal()
        event = _make_registered_event(tmp_path, instance_ids=("i-1",))
        (tmp_path / "ssh_key").touch()
        await journal.persist("daemon-state", [PersistedEvent(sequence_nr=1, event=event, timestamp=0.0)])

        mock_provider = MagicMock()
        mock_provider.get_instance = AsyncMock(
            side_effect=lambda cluster, iid: (cluster, _FakeInstance(
                id=iid, status="provisioned", ip="10.0.0.1",
            )),
        )

        sock = _short_sock(tmp_path)
        server = DaemonServer(socket_path=sock, journal=journal)

        with patch.object(server, "_create_provider", new_callable=AsyncMock, return_value=mock_provider), \
             patch.object(server, "_recover_pool_via_session", new_callable=AsyncMock) as mock_session:
            mock_pool = MagicMock()
            mock_session.return_value = mock_pool
            async with server:
                assert "train" in server._ttl_tasks
