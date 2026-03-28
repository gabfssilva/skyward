import pytest
from casty import ActorSystem, InMemoryJournal

from skyward.daemon.state import (
    DaemonState,
    PoolRegistered,
    PoolRemoved,
    ClientJoined,
    ClientLeft,
    RegisterPool,
    RemovePool,
    AddClient,
    RemoveClient,
    GetState,
    daemon_state_actor,
    apply_event,
)


class TestDaemonStateEvents:
    def test_apply_pool_registered(self) -> None:
        state = DaemonState()
        event = PoolRegistered(
            pool_name="train",
            cluster_id="c-123",
            instance_ids=("i-1", "i-2"),
            project_dir="/home/user/project",
        )
        new = apply_event(state, event)
        assert "train" in new.pools
        assert new.pools["train"].cluster_id == "c-123"
        assert new.pools["train"].instance_ids == ("i-1", "i-2")

    def test_apply_pool_removed(self) -> None:
        state = DaemonState()
        state = apply_event(state, PoolRegistered(
            pool_name="train", cluster_id="c-1",
            instance_ids=("i-1",), project_dir="/tmp",
        ))
        state = apply_event(state, PoolRemoved(pool_name="train"))
        assert "train" not in state.pools

    def test_apply_client_joined(self) -> None:
        state = DaemonState()
        state = apply_event(state, PoolRegistered(
            pool_name="train", cluster_id="c-1",
            instance_ids=(), project_dir="/tmp",
        ))
        state = apply_event(state, ClientJoined(pool_name="train", client_id="abc"))
        assert "abc" in state.pools["train"].clients

    def test_apply_client_left(self) -> None:
        state = DaemonState()
        state = apply_event(state, PoolRegistered(
            pool_name="train", cluster_id="c-1",
            instance_ids=(), project_dir="/tmp",
        ))
        state = apply_event(state, ClientJoined(pool_name="train", client_id="abc"))
        state = apply_event(state, ClientLeft(pool_name="train", client_id="abc"))
        assert "abc" not in state.pools["train"].clients


    def test_apply_pool_registered_with_recovery_data(self) -> None:
        state = DaemonState()
        event = PoolRegistered(
            pool_name="train",
            cluster_id="c-123",
            instance_ids=("i-1", "i-2"),
            project_dir="/home/user/project",
            provider_name="aws",
            cluster_bytes=b"pickled-cluster",
            spec_bytes=b"pickled-spec",
            provider_config_bytes=b"pickled-config",
        )
        new = apply_event(state, event)
        entry = new.pools["train"]
        assert entry.provider_name == "aws"
        assert entry.cluster_bytes == b"pickled-cluster"
        assert entry.spec_bytes == b"pickled-spec"
        assert entry.provider_config_bytes == b"pickled-config"

    def test_pool_registered_recovery_fields_default_empty(self) -> None:
        """Backwards compatibility: old events without recovery fields."""
        event = PoolRegistered(
            pool_name="train",
            cluster_id="c-1",
            instance_ids=("i-1",),
            project_dir="/tmp",
        )
        assert event.provider_name == ""
        assert event.cluster_bytes == b""


class TestDaemonStateActor:
    @pytest.mark.asyncio
    async def test_register_and_query(self) -> None:
        journal = InMemoryJournal()
        async with ActorSystem("test") as system:
            ref = system.spawn(
                daemon_state_actor("daemon-1", journal), "daemon-state",
            )
            await system.ask(
                ref,
                lambda r: RegisterPool(
                    pool_name="train", cluster_id="c-1",
                    instance_ids=("i-1",), project_dir="/tmp",
                    reply_to=r,
                ),
                timeout=2.0,
            )
            state: DaemonState = await system.ask(
                ref, lambda r: GetState(reply_to=r), timeout=2.0,
            )
            assert "train" in state.pools

    @pytest.mark.asyncio
    async def test_recovery_from_journal(self) -> None:
        """State recovers after actor system restart."""
        journal = InMemoryJournal()

        # Phase 1: register a pool
        async with ActorSystem("test") as system:
            ref = system.spawn(
                daemon_state_actor("daemon-1", journal), "state",
            )
            await system.ask(
                ref,
                lambda r: RegisterPool(
                    pool_name="train", cluster_id="c-1",
                    instance_ids=("i-1", "i-2"), project_dir="/tmp",
                    reply_to=r,
                ),
                timeout=2.0,
            )

        # Phase 2: new system, same journal -- state should recover
        async with ActorSystem("test") as system:
            ref = system.spawn(
                daemon_state_actor("daemon-1", journal), "state",
            )
            state: DaemonState = await system.ask(
                ref, lambda r: GetState(reply_to=r), timeout=2.0,
            )
            assert "train" in state.pools
            assert state.pools["train"].instance_ids == ("i-1", "i-2")
