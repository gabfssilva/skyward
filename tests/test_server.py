"""``sky server`` and the pid this machine recorded: what stop and restart do to it."""

from __future__ import annotations

import ipaddress
import json
import resource
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from skyward.cli import server
from skyward.server import daemon
from tests.conftest import serving

pytestmark = pytest.mark.local


class Recorded:
    """The daemon this machine recorded, as the commands are able to see it."""

    def __init__(self, process: int | None) -> None:
        self.process = process
        self.signalled: list[int] = []
        self.started = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(daemon, "pid", lambda: self.process)
        monkeypatch.setattr(daemon, "alive", lambda process: self.process == process)
        monkeypatch.setattr(daemon, "forget", self.forget)
        monkeypatch.setattr(server.os, "kill", self.kill)
        monkeypatch.setattr(server, "start", self.start)
        monkeypatch.setattr(server, "live", lambda target: False)

    def kill(self, process: int, _signal: int) -> None:
        self.signalled.append(process)
        self.process = None

    def forget(self) -> None:
        self.process = None

    def start(self, **_: object) -> None:
        self.started += 1


def describe_restarting_the_daemon() -> None:
    def it_stops_the_one_this_machine_started_before_it_starts_another(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(4242)
        recorded.install(monkeypatch)

        server.restart()

        assert recorded.signalled == [4242], "the pid was signalled, not asked"
        assert recorded.started == 1

    def it_starts_one_when_there_is_nothing_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(None)
        recorded.install(monkeypatch)

        server.restart()

        assert recorded.signalled == [] and recorded.started == 1

    def it_refuses_when_something_answers_that_this_machine_did_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(None)
        recorded.install(monkeypatch)
        monkeypatch.setattr(server, "live", lambda target: True)

        with pytest.raises(SystemExit) as refused:
            server.restart()

        assert "no pid" in str(refused.value)
        assert recorded.started == 0, "a daemon somebody else started is not restarted behind their back"


def describe_stopping_the_daemon() -> None:
    def it_says_there_is_nothing_to_stop_when_no_pid_was_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
        Recorded(None).install(monkeypatch)

        with pytest.raises(SystemExit) as refused:
            server.stop()

        assert "nothing to stop" in str(refused.value)

    def it_clears_a_pid_whose_process_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = Recorded(4242)
        recorded.install(monkeypatch)
        monkeypatch.setattr(daemon, "alive", lambda process: False)

        server.stop()

        assert recorded.signalled == [] and recorded.process is None


def describe_a_daemon_starting() -> None:
    @pytest.fixture
    def shell_limit() -> Iterator[int]:
        """The descriptor limit a macOS shell hands down, put back afterwards."""
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        yield 256
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    def it_raises_the_descriptor_limit_it_inherited(shell_limit: int) -> None:
        daemon.descriptors(4096)

        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 4096

    def it_never_lowers_a_limit_already_above_what_it_wants(shell_limit: int) -> None:
        daemon.descriptors(64)

        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == shell_limit


def loopback() -> str:
    """The name this machine gives its loopback interface: ``lo0`` on macOS, ``lo`` on Linux."""
    return next(name for name, addresses in daemon.interfaces().items() if "127.0.0.1" in addresses)


def external() -> tuple[str, str]:
    """An interface that reaches past this machine, and its first IPv4 address."""
    found = next(
        ((name, address) for name, addresses in daemon.interfaces().items() for address in addresses if not ipaddress.IPv4Address(address).is_loopback),
        None,
    )
    if found is None:
        pytest.skip("this machine has no interface beyond loopback")
    return found


def describe_the_interface() -> None:
    @pytest.fixture(autouse=True)
    def recorded_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(daemon, "RUNTIME_DIR", tmp_path)
        monkeypatch.setattr(daemon, "INTERFACE_FILE", tmp_path / "server.interface")

    def describe_setting_it() -> None:
        def it_records_an_interface_this_machine_has() -> None:
            server.set_interface(loopback())

            assert daemon.interface() == loopback()

        @pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0"])
        def it_takes_an_address_this_machine_holds_or_the_wildcard(address: str) -> None:
            server.set_interface(address)

            assert daemon.interface() == address

        @pytest.mark.parametrize("interface", ["nowhere0", "203.0.113.7"])
        def it_refuses_what_this_machine_does_not_have(interface: str) -> None:
            with pytest.raises(SystemExit, match="no interface or address"):
                server.set_interface(interface)

            assert daemon.interface() is None

    def describe_where_a_daemon_listens() -> None:
        def on_its_host_alone_without_one() -> None:
            assert daemon.listening("127.0.0.1") == ("127.0.0.1",)

        def on_the_addresses_the_interface_has_beside_its_host() -> None:
            name, address = external()
            daemon.choose(name)

            assert daemon.listening("127.0.0.1")[:2] == ("127.0.0.1", address), "the host comes first, since that is where this machine looks"

        def on_an_address_once_when_the_host_already_is_it() -> None:
            daemon.choose(loopback())

            assert daemon.listening("127.0.0.1") == ("127.0.0.1",)

        def on_the_wildcard_alone_since_it_covers_the_rest() -> None:
            daemon.choose("0.0.0.0")

            assert daemon.listening("127.0.0.1") == ("0.0.0.0",)


def describe_a_daemon_with_an_interface() -> None:
    @pytest.fixture
    def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A home of its own for the daemon, whose runtime directory is where the interface is read from."""
        home = tmp_path / "home"
        (home / ".skyward").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        return home

    def it_answers_on_the_interface_and_where_it_always_did(home: Path, tmp_path: Path) -> None:
        name, address = external()
        (home / ".skyward" / "server.interface").write_text(name)

        with serving(tmp_path / "skyward.sqlite", tmp_path / "daemon.log") as url:
            beside = f"http://{address}:{urlsplit(url).port}"

            assert httpx.get(f"{beside}/v1/health/live").json()["live"] is True
            assert httpx.get(f"{url}/v1/health/live").json()["live"] is True

    def it_starts_on_its_host_when_the_interface_cannot_be_listened_on(home: Path, tmp_path: Path) -> None:
        (home / ".skyward" / "server.interface").write_text("203.0.113.7")

        with serving(tmp_path / "skyward.sqlite", tmp_path / "daemon.log") as url:
            assert httpx.get(f"{url}/v1/health/live").json()["live"] is True

        logged = [json.loads(line)["message"] for line in (tmp_path / "logs" / "skyward.log").read_text().splitlines()]
        assert any("not listening on 203.0.113.7" in message for message in logged), "an address it could not take is said, not dropped"
