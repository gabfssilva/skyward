"""The one channel the machine talks back on — the parsing half."""

from __future__ import annotations

import pytest

from skyward.worker.journal import Console, Metric, Phase, parse

pytestmark = pytest.mark.unit


def test_a_metric_line_parses_to_a_named_reading():
    assert parse('{"type":"metric","name":"gpu_util","value":87.5}') == Metric(name="gpu_util", value=87.5)


def test_an_integer_reading_is_still_a_float_reading():
    assert parse('{"type":"metric","name":"mem_used_mb","value":16384}') == Metric(name="mem_used_mb", value=16384.0)


def test_console_and_phase_still_parse():
    assert parse('{"type":"console","content":"epoch 3"}') == Console(content="epoch 3")
    assert parse('{"type":"phase","event":"completed","phase":"bootstrap"}') == Phase(event="completed", phase="bootstrap")


def test_a_half_written_line_is_nothing_to_fail_over():
    assert parse('{"type":"metric","name":"cp') is None
