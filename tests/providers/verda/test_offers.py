"""Verda GPU model normalization → canonical accelerator catalog."""

from __future__ import annotations

import pytest

from skyward.accelerators.catalog import SPECS
from skyward.offers.feed import _normalize_gpu_name
from skyward.providers.verda.types import parse_gpu_model

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("8x H100 SXM5 80GB", "H100"),
        ("4x A100 80GB", "A100"),
        ("1x RTX A6000 48GB", "RTX A6000"),
        ("1x RTX 6000 Ada 48GB", "RTX 6000 Ada"),
        ("1x RTX PRO 6000 96GB", "RTX PRO 6000"),
        ("8x Tesla V100 32GB", "V100"),
    ],
)
def test_parse_gpu_model_matches_canonical_catalog(description: str, expected: str) -> None:
    parsed = parse_gpu_model(description)
    assert parsed == expected
    assert parsed is not None
    assert _normalize_gpu_name(parsed) in SPECS, (
        f"{parsed!r} from Verda is not a canonical catalog name — "
        "offers query for this accelerator will return zero results"
    )


def test_parse_gpu_model_returns_none_on_malformed() -> None:
    assert parse_gpu_model("") is None
    assert parse_gpu_model("garbage") is None
    assert parse_gpu_model("H100 80GB") is None  # missing count prefix
