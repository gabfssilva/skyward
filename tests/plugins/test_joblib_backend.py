"""SkywardBackend.effective_n_jobs — pool capacity across every NodeSpec form."""

from __future__ import annotations

import pytest

import skyward as sky
from skyward.api.spec import Nodes

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class _Spec:
    def __init__(self, nodes):
        self.nodes = nodes


class _Pool:
    def __init__(self, nodes, concurrency=10):
        self._specs = (_Spec(nodes),)
        self.concurrency = concurrency


def _backend(nodes, concurrency=10):
    # Touch the lazy factory first: importing skyward.plugins.joblib directly
    # binds the *module* onto skyward.plugins, shadowing sky.plugins.joblib()
    # for the rest of the process.
    sky.plugins.joblib
    from skyward.plugins.joblib import SkywardBackend

    return SkywardBackend(_Pool(nodes, concurrency))


class TestEffectiveNJobs:
    @pytest.mark.parametrize(
        ("nodes", "expected"),
        [
            (10, 100),                              # fixed count
            ((2, 8), 80),                           # elastic — the ceiling, not the start
            (Nodes(desired=8), 80),                 # Nodes, no autoscaling
            (Nodes(desired=8, min=4), 80),          # partial readiness
            (Nodes(desired=4, min=2, max=16), 160), # elastic + early start
            (Nodes(desired=0, max=8), 80),          # lazy start
        ],
    )
    def test_capacity_is_reachable_nodes_times_concurrency(self, nodes, expected) -> None:
        assert _backend(nodes).effective_n_jobs(-1) == expected

    def test_zero_requested_stays_zero(self) -> None:
        assert _backend(4).effective_n_jobs(0) == 0
