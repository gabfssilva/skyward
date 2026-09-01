"""What the dashboard's gauges may aggregate.

The badges average what they collect, so what they collect decides whether the
number is a percentage or an accident — ``mem_used_mb`` averaged into the ``mem``
gauge is a node at 28900%.
"""

from types import MappingProxyType

import pytest

from skyward.core.widgets import _find_metrics

pytestmark = pytest.mark.local


def describe_collecting_a_gauge() -> None:
    def it_prefers_the_metric_itself() -> None:
        raw = MappingProxyType({"mem": 22.6, "mem_used_mb": 28900.0})

        assert _find_metrics(raw, "mem") == [22.6]

    def it_falls_back_to_the_per_device_shards() -> None:
        raw = MappingProxyType({"gpu_util_0": 90.0, "gpu_util_1": 70.0})

        assert _find_metrics(raw, "gpu_util") == [90.0, 70.0]

    def it_does_not_mistake_absolute_metrics_for_shards() -> None:
        raw = MappingProxyType({"mem_used_mb": 28900.0, "mem_total_mb": 128000.0})

        assert _find_metrics(raw, "mem") == []
