import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vscode.sidecar.executor import run_main


@pytest.fixture(autouse=True)
def _no_daemon():
    """Prevent all tests from touching the real daemon."""
    mock_pool = MagicMock()
    mock_pool.__enter__ = MagicMock(return_value=mock_pool)
    mock_pool.__exit__ = MagicMock(return_value=False)

    with patch("skyward.daemon.pool.DaemonPool", return_value=mock_pool):
        yield


def test_run_main_simple():
    code = '''
def main(x: int = 5):
    return x * 2

main.__sky_main__ = True
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = run_main(f.name, "main", {"x": 3}, "default")

    assert result == 6


def test_run_main_no_args():
    code = '''
def train():
    return 42

train.__sky_main__ = True
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        result = run_main(f.name, "train", {}, "default")

    assert result == 42


def test_run_main_function_not_found():
    code = '''
def other():
    pass

other.__sky_main__ = True
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        with pytest.raises(AttributeError, match="not found"):
            run_main(f.name, "missing", {}, "default")


def test_run_main_not_decorated():
    code = '''
def train():
    return 42
'''
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        with pytest.raises(ValueError, match="not decorated"):
            run_main(f.name, "train", {}, "default")


def test_run_main_bad_file():
    with pytest.raises((ImportError, FileNotFoundError)):
        run_main("/tmp/nonexistent_sky_test.py", "fn", {}, "default")
