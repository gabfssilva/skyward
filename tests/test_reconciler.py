"""The bounds a compute may size itself within — and the two zeros that must survive.

``min=0`` (scale to zero) and ``desired=0`` (lazy start) are real values, not the
absence of one; the clamp must read them as themselves rather than fall through to
``desired``. This is the regression that pinned an elastic pool at its ceiling.
"""

from skyward.application.reconciler import bounds
from skyward.protocol.schemas import ComputeSpec, NodeBounds, PluginRef, ProviderRef, Spec

SPECS = (Spec(provider=ProviderRef(kind="fake")),)


def spec(nodes: NodeBounds, plugins: tuple[PluginRef, ...] = ()) -> ComputeSpec:
    return ComputeSpec(specs=SPECS, nodes=nodes, plugins=plugins)


def test_a_fixed_count_collapses_the_clamp_onto_itself():
    assert bounds(spec(NodeBounds(desired=4))) == (4, 4)


def test_an_elastic_pool_keeps_its_min_and_max():
    assert bounds(spec(NodeBounds(desired=8, min=4, max=8))) == (4, 8)


def test_scale_to_zero_keeps_a_min_of_zero():
    assert bounds(spec(NodeBounds(desired=8, min=0, max=8))) == (0, 8)


def test_lazy_start_keeps_a_desired_of_zero():
    assert bounds(spec(NodeBounds(desired=0, min=0, max=8))) == (0, 8)


def test_a_collective_freezes_the_pool_whatever_it_asked_for():
    frozen = spec(NodeBounds(desired=4, min=2, max=8), plugins=(PluginRef(kind="torch"),))
    assert bounds(frozen) == (4, 4)
