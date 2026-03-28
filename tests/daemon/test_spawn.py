from pathlib import Path

from skyward.daemon.spawn import is_daemon_running, daemon_socket_path


class TestDaemonDetection:
    def test_socket_path(self) -> None:
        path = daemon_socket_path()
        assert path == Path.home() / ".skyward" / "daemon.sock"

    def test_not_running_when_no_socket(self, tmp_path: Path) -> None:
        assert not is_daemon_running(tmp_path / "nonexistent.sock")
