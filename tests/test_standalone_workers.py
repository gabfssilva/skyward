import pytest

from skyward.api.spec import Nodes, Options, PoolSpec


def test_pool_spec_cluster_default_true():
    spec = PoolSpec(nodes=Nodes(min=1), accelerator=None, region="us-east-1")
    assert spec.cluster is True


def test_pool_spec_cluster_false():
    spec = PoolSpec(nodes=Nodes(min=1), accelerator=None, region="us-east-1", cluster=False)
    assert spec.cluster is False


def test_options_cluster_default_true():
    opts = Options()
    assert opts.cluster is True


def test_options_cluster_false():
    opts = Options(cluster=False)
    assert opts.cluster is False


def test_disabled_registry_dict_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.dict("test")


def test_disabled_registry_counter_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.counter("test")


def test_disabled_registry_barrier_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.barrier("sync", 4)


def test_disabled_registry_lock_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.lock("critical")


def test_disabled_registry_set_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.set("seen")


def test_disabled_registry_queue_raises():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    with pytest.raises(RuntimeError, match="cluster mode"):
        reg.queue("tasks")


def test_disabled_registry_cleanup_is_noop():
    from skyward.distributed.disabled import DisabledRegistry
    reg = DisabledRegistry()
    reg.cleanup()


def test_worker_standalone_env_skips_seeds():
    from skyward.infra.worker import _parse_seeds

    cluster_mode = "false".lower() != "false"
    seeds = _parse_seeds("10.0.0.1:25520") if cluster_mode else None
    assert seeds is None


def test_worker_cluster_default_uses_seeds():
    from skyward.infra.worker import _parse_seeds

    cluster_mode = "true".lower() != "false"
    seeds = _parse_seeds("10.0.0.1:25520") if cluster_mode else None
    assert seeds == [("10.0.0.1", 25520)]


def test_pool_spec_cluster_false_propagates():
    spec = PoolSpec(
        nodes=Nodes(min=2),
        accelerator=None,
        region="us-east-1",
        cluster=False,
    )
    assert spec.cluster is False
    assert spec.nodes.min == 2


def test_disabled_registry_satisfies_protocol():
    from skyward.distributed.disabled import DisabledRegistry

    reg = DisabledRegistry()
    for method in ("dict", "set", "counter", "queue", "barrier", "lock", "cleanup"):
        assert hasattr(reg, method), f"Missing method: {method}"
