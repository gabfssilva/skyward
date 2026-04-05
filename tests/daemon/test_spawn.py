from pathlib import Path

from skyward.daemon.spawn import is_daemon_running


class TestDaemonDetection:
    def test_not_running_when_no_socket(self, tmp_path: Path) -> None:
        assert not is_daemon_running(tmp_path / "nonexistent.sock")
