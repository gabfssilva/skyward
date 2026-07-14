"""What the user's arguments turn into before anything is provisioned.

Everything here is translation: an accelerator name, a node range, a provider's
credentials. No control plane is started — the point is that the spec the daemon
would receive is the spec the user described.
"""

import pytest

import skyward2 as skyward
from skyward2 import Compute
from skyward2.protocol.schemas import NodeBounds


def spec_of(pool: Compute):
    return pool._spec  # noqa: SLF001


def test_a_provider_becomes_a_single_spec():
    pool = skyward.Compute(provider=skyward.Container(), cpus=2, memory_gb=4)
    spec = spec_of(pool)

    assert len(spec.specs) == 1
    assert spec.specs[0].provider.kind == "container"
    assert (spec.specs[0].cpus, spec.specs[0].memory_gb) == (2, 4)
    assert spec.nodes == NodeBounds(desired=1)


def test_specs_are_alternatives_and_each_provider_is_registered():
    pool = skyward.Compute(
        skyward.Spec(skyward.AWS(access_key_id="a", secret_access_key="b"), accelerator="A100"),
        skyward.Spec(skyward.VastAI(api_key="k"), accelerator=skyward.accelerators.A100(count=2)),
        selection="first",
    )
    spec = spec_of(pool)

    assert [wanted.provider.kind for wanted in spec.specs] == ["aws", "vastai"]
    assert [wanted.accelerator for wanted in spec.specs] == ["a100", "a100"], "raw names normalize like offers do"
    assert [wanted.accelerator_count for wanted in spec.specs] == [1, 2]
    assert spec.selection == "first"
    assert set(pool._providers) == {"aws", "vastai"}  # noqa: SLF001


def test_credentials_come_from_the_environment_when_not_given(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "from-the-env")

    assert skyward.RunPod().credentials == {"api_key": "from-the-env"}
    assert skyward.RunPod(api_key="explicit").credentials == {"api_key": "explicit"}


@pytest.mark.parametrize(
    ("nodes", "bounds"),
    [
        (4, NodeBounds(desired=4)),
        ((2, 8), NodeBounds(desired=8, min=2, max=8)),
        (skyward.Nodes(desired=8, min=4), NodeBounds(desired=8, min=4)),
    ],
)
def test_nodes_can_be_a_count_a_range_or_a_bound(nodes: int, bounds: NodeBounds):
    assert spec_of(skyward.Compute(provider=skyward.Container(), nodes=nodes)).nodes == bounds


def test_a_pool_needs_a_provider_or_specs_but_not_both():
    with pytest.raises(ValueError, match="not both"):
        skyward.Compute(skyward.Spec(skyward.Container()), provider=skyward.Container())

    with pytest.raises(ValueError, match="at least one spec"):
        skyward.Compute()


def test_an_unknown_accelerator_is_an_error_at_the_call_site():
    with pytest.raises(AttributeError, match="unknown accelerator"):
        skyward.accelerators.H999()


def test_a_timeout_rides_on_the_pending_call():
    @skyward.function(timeout=600)
    def slow() -> int:
        return 1

    assert slow().timeout == 600
    assert slow().with_timeout(30).timeout == 30
