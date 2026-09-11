"""A node's health check, driven directly.

The check runs in a thread and may outlast its timeout; the generator reports every
interval that passes while it runs as a failure, without starting a second copy.
"""

import threading

import pytest

from skyward.worker import worker
from skyward.worker.api import Info

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def on_a_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYWARD_NODE", "nod_test")
    monkeypatch.setenv("SKYWARD_COMPUTE", "cmp_test")
    monkeypatch.setenv("SKYWARD_RANK", "0")
    monkeypatch.setenv("SKYWARD_PEERS", "10.0.0.1")


def describe_health() -> None:
    def describe_a_check_slower_than_its_timeout() -> None:
        async def it_is_invoked_once_and_every_interval_yields_a_timeout() -> None:
            calls: list[Info] = []
            release = threading.Event()

            def slow(info: Info) -> bool:
                calls.append(info)
                release.wait(5)
                return True

            checks = worker.health(slow, interval=0.02, timeout=0.05, initial_delay=0)
            try:
                results = [await anext(checks) for _ in range(4)]
                assert results == [(False, "timeout after 0.05s")] * 4
                assert len(calls) == 1
            finally:
                release.set()
                await checks.aclose()

        async def it_yields_healthy_on_the_round_after_it_returns() -> None:
            calls: list[Info] = []
            release = threading.Event()

            def slow(info: Info) -> bool:
                calls.append(info)
                release.wait(5)
                return True

            checks = worker.health(slow, interval=0.02, timeout=0.05, initial_delay=0)
            try:
                assert await anext(checks) == (False, "timeout after 0.05s")
                assert await anext(checks) == (False, "timeout after 0.05s")
                release.set()
                assert await anext(checks) == (True, None)
                assert len(calls) == 1
            finally:
                release.set()
                await checks.aclose()

    def describe_a_check_that_raises() -> None:
        async def it_yields_the_exception_and_the_next_round_starts_a_new_check() -> None:
            calls: list[Info] = []
            failure = ValueError("disk full")

            def flaky(info: Info) -> bool:
                calls.append(info)
                if len(calls) == 1:
                    raise failure
                return True

            checks = worker.health(flaky, interval=0.01, timeout=1.0, initial_delay=0)
            try:
                assert await anext(checks) == (False, repr(failure))
                assert await anext(checks) == (True, None)
                assert len(calls) == 2
            finally:
                await checks.aclose()
