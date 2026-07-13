"""Tests for the ``ports`` configuration field and the public ``Port`` type."""
import skyward as sky
from skyward.api.spec import PoolSpec, Port


def test_port_defaults() -> None:
    p = Port(remote=8080)
    assert p.remote == 8080
    assert p.local == 0
    assert p.route == "round_robin"


def test_spec_retains_ports() -> None:
    spec = sky.Spec(provider=None, ports=[Port(remote=8080, local=8080)])  # type: ignore[arg-type]
    assert spec.ports == [Port(remote=8080, local=8080, route="round_robin")]


def test_pool_spec_ports_default_empty() -> None:
    assert PoolSpec(nodes=sky.Nodes(desired=1), accelerator=None, region="").ports == ()


def test_public_port_is_exported() -> None:
    assert sky.Port is Port
