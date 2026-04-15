"""H2: basic ``ServerPool`` API contract (offline checks)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_server_pool_inactive_raises() -> None:
    import skyward as sky
    from skyward.server.driver.pool import ServerPool

    spec = MagicMock()
    spec.specs = ()
    pool = ServerPool("name", "/tmp/ignore.sock", spec)

    @sky.function
    def _noop() -> int:
        return 1

    with pytest.raises(RuntimeError):
        pool.run(_noop())


def test_server_pool_python_version_mismatch() -> None:
    from skyward.server.driver.pool import (
        PythonVersionMismatchError,
        ServerPool,
    )

    fake_spec = MagicMock()
    fake_image = MagicMock()
    fake_image.python = "2.7"
    fake_spec_inner = MagicMock()
    fake_spec_inner.image = fake_image
    fake_spec.specs = (fake_spec_inner,)

    pool = ServerPool("name", "/tmp/ignore.sock", fake_spec)
    with pytest.raises(PythonVersionMismatchError):
        pool._check_python_version()


def test_server_pool_operator_shape() -> None:
    from skyward.server.driver.pool import ServerPool

    assert hasattr(ServerPool, "__rshift__")
    assert hasattr(ServerPool, "__matmul__")
    assert hasattr(ServerPool, "__gt__")
