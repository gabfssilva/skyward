# pyright: reportOptionalMemberAccess=false, reportUnusedExpression=false
"""Distributed collections must survive the death of a node.

Casty v2 collections replicate across ``min(3, num_nodes)`` nodes with
quorum writes, so on a 3-node pool killing one container must neither
lose the counter's state nor fence subsequent writes.
"""
from __future__ import annotations

import subprocess
import time

import pytest

import skyward as sky

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("distributed-resilience"),
]

_PREFIX = "skyward-dist-res"


def _kill_one_worker_container() -> str:
    """Docker-kill one running container of this pool (not the first one)."""
    out = subprocess.run(
        ["docker", "ps", "--filter", f"name={_PREFIX}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert len(out) >= 3, f"expected 3 containers, got {out}"
    victim = sorted(out)[-1]
    subprocess.run(["docker", "kill", victim], capture_output=True, check=True)
    return victim


class TestCounterSurvivesNodeDeath:
    def test_counter_survives_node_death(self) -> None:
        with sky.Compute(
            provider=sky.Container(network="skyward", container_prefix=_PREFIX),
            nodes=3,
            vcpus=0.5,
            memory_gb=0.5,
            options=sky.Options(console=False),
        ) as pool:

            @sky.function
            def increment() -> bool:
                sky.counter("resilient_counter").increment(1)
                return True

            @sky.function
            def read() -> int:
                return sky.counter("resilient_counter").value

            @sky.function
            def increment_and_read() -> int:
                c = sky.counter("resilient_counter")
                c.increment(1)
                return c.value

            assert increment() @ pool == [True, True, True]
            assert read() >> pool == 3

            _kill_one_worker_container()
            time.sleep(5.0)

            # State survived and writes are not fenced: quorum (2/3) holds.
            # >= 4 rather than == 4: an increment lost to a retried dispatch
            # may have landed before the retry, and retries re-execute.
            deadline = time.monotonic() + 120.0
            while True:
                try:
                    assert increment_and_read() >> pool >= 4
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(2.0)
