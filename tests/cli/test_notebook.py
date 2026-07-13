"""Tests for sky notebook install / remove."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import httpx
import pytest

from skyward.cli import app

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
def fake_kernelspec(monkeypatch):
    """Inject a fake ``skyward.notebook`` / ``.kernelspec`` into ``sys.modules``.

    The command body imports them lazily, so injecting before invoking the
    command is enough. Returns the recording namespace.
    """
    calls: dict[str, Any] = {"install": [], "remove": []}

    pkg = ModuleType("skyward.notebook")
    mod = ModuleType("skyward.notebook.kernelspec")

    def install_kernelspec(session: str) -> str:
        calls["install"].append(session)
        return f"skyward-{session}"

    def remove_kernelspec(session: str) -> None:
        calls["remove"].append(session)

    mod.KERNEL_NAME_PREFIX = "skyward-"
    mod.install_kernelspec = install_kernelspec
    mod.remove_kernelspec = remove_kernelspec
    pkg.kernelspec = mod

    monkeypatch.setitem(sys.modules, "skyward.notebook", pkg)
    monkeypatch.setitem(sys.modules, "skyward.notebook.kernelspec", mod)
    return calls


@pytest.fixture
def resolves(monkeypatch):
    def install(name: str) -> None:
        monkeypatch.setattr(
            "skyward.cli.notebook.resolve_session", lambda session, url: name
        )

    return install


def test_install_resolves_and_calls_kernelspec(fake_kernelspec, resolves):
    resolves("demo")
    with pytest.raises(SystemExit) as e:
        app(["notebook", "install"], exit_on_error=False)
    assert e.value.code == 0
    assert fake_kernelspec["install"] == ["demo"]


def test_install_explicit_session(fake_kernelspec, resolves):
    resolves("prod")
    with pytest.raises(SystemExit) as e:
        app(["notebook", "install", "--session", "prod"], exit_on_error=False)
    assert e.value.code == 0
    assert fake_kernelspec["install"] == ["prod"]


def test_remove_calls_kernelspec(fake_kernelspec, resolves):
    resolves("demo")
    with pytest.raises(SystemExit) as e:
        app(["notebook", "remove"], exit_on_error=False)
    assert e.value.code == 0
    assert fake_kernelspec["remove"] == ["demo"]


def test_unknown_action_exits_nonzero(fake_kernelspec, resolves):
    resolves("demo")
    with pytest.raises(SystemExit) as e:
        app(["notebook", "bogus"], exit_on_error=False)
    assert e.value.code == 1


def test_connect_error_exits(fake_kernelspec, monkeypatch, capsys):
    def boom(session, url):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr("skyward.cli.notebook.resolve_session", boom)
    with pytest.raises(SystemExit) as e:
        app(["notebook", "install"], exit_on_error=False)
    assert e.value.code == 1
    assert "Could not reach server" in capsys.readouterr().out


def test_import_cli_is_lazy():
    for name in ("jupyter_client", "skyward.notebook"):
        sys.modules.pop(name, None)
    import importlib

    import skyward.cli

    importlib.reload(skyward.cli)
    assert "jupyter_client" not in sys.modules
    assert "skyward.notebook" not in sys.modules
