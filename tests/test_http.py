from __future__ import annotations

import pytest

from skyward.infra.http import HttpError

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_http_error_str():
    err = HttpError(status=429, body="rate limited")
    assert str(err) == "HTTP 429: rate limited"
