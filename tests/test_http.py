from __future__ import annotations

import pytest

from skyward.infra.http import HttpError

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_http_error_str():
    err = HttpError(status=429, body="rate limited")
    assert str(err) == "HTTP 429: rate limited"


def test_http_error_is_frozen():
    err = HttpError(status=200, body="ok")
    with pytest.raises(AttributeError):
        err.status = 201  # type: ignore[misc]


def test_http_error_is_exception():
    err = HttpError(status=500, body="internal server error")
    assert isinstance(err, Exception)
    with pytest.raises(HttpError):
        raise err
