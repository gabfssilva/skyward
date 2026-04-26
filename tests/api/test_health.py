"""Unit tests for the periodic remote health probe generator."""

from __future__ import annotations

import time
from itertools import islice
from typing import Any

import pytest

from skyward.api import health as health_mod
from skyward.api.health import HealthChecker, hc_loop

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
def fake_info(monkeypatch: pytest.MonkeyPatch) -> object:
    """Replace ``instance_info`` so ``hc_loop`` can run outside a worker."""
    sentinel = object()
    monkeypatch.setattr(health_mod, "instance_info", lambda: sentinel, raising=False)

    from skyward.api import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "instance_info", lambda: sentinel)
    return sentinel


def _take(gen, n: int) -> list:  # type: ignore[no-untyped-def]
    return list(islice(gen, n))


class TestHealthCheckerValidation:
    def test_default_construction(self) -> None:
        hc = HealthChecker(fn=lambda _info: True)
        assert hc.interval == 30.0
        assert hc.timeout == 15.0
        assert hc.consecutive_failures == 3
        assert hc.initial_delay == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("interval", 0.0),
            ("interval", -1.0),
            ("timeout", 0.0),
            ("timeout", -5.0),
            ("consecutive_failures", 0),
            ("initial_delay", -0.1),
        ],
    )
    def test_invalid_values_raise(self, field: str, value: Any) -> None:
        with pytest.raises(ValueError):
            HealthChecker(fn=lambda _info: True, **{field: value})


class TestHcLoopOutcomes:
    def test_true_yields_ok(self, fake_info: object) -> None:
        gen = hc_loop(lambda _info: True, interval=0.001, timeout=1.0, initial_delay=0.0)
        assert _take(gen, 2) == [("ok", None), ("ok", None)]

    def test_false_yields_fail_no_reason(self, fake_info: object) -> None:
        gen = hc_loop(lambda _info: False, interval=0.001, timeout=1.0, initial_delay=0.0)
        assert _take(gen, 1) == [("fail", None)]

    def test_str_yields_fail_with_reason(self, fake_info: object) -> None:
        gen = hc_loop(
            lambda _info: "GPU unreachable", interval=0.001, timeout=1.0, initial_delay=0.0,
        )
        assert _take(gen, 1) == [("fail", "GPU unreachable")]

    def test_empty_str_yields_fail_no_reason(self, fake_info: object) -> None:
        gen = hc_loop(lambda _info: "", interval=0.001, timeout=1.0, initial_delay=0.0)
        assert _take(gen, 1) == [("fail", None)]

    def test_exception_yields_fail_with_repr(self, fake_info: object) -> None:
        def boom(_info: object) -> bool:
            raise RuntimeError("kaboom")

        gen = hc_loop(boom, interval=0.001, timeout=1.0, initial_delay=0.0)
        first = _take(gen, 1)
        assert len(first) == 1
        tag, reason = first[0]
        assert tag == "fail"
        assert reason is not None
        assert "RuntimeError" in reason
        assert "kaboom" in reason

    def test_timeout_yields_fail(self, fake_info: object) -> None:
        def slow(_info: object) -> bool:
            time.sleep(0.5)
            return True

        gen = hc_loop(slow, interval=0.001, timeout=0.05, initial_delay=0.0)
        first = _take(gen, 1)[0]
        tag, reason = first
        assert tag == "fail"
        assert reason is not None
        assert "timeout" in reason


class TestHcLoopBehaviour:
    def test_alternates_with_changing_outcomes(self, fake_info: object) -> None:
        calls = {"n": 0}

        def alternating(_info: object) -> bool:
            calls["n"] += 1
            return calls["n"] % 2 == 1

        gen = hc_loop(alternating, interval=0.001, timeout=1.0, initial_delay=0.0)
        assert _take(gen, 4) == [
            ("ok", None), ("fail", None), ("ok", None), ("fail", None),
        ]

    def test_initial_delay_blocks_first_yield(self, fake_info: object) -> None:
        gen = hc_loop(
            lambda _info: True, interval=0.001, timeout=1.0, initial_delay=0.05,
        )
        t0 = time.monotonic()
        next(gen)
        assert time.monotonic() - t0 >= 0.05
