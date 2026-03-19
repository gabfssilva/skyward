"""Tests for sky version command."""

import sys

import pytest

from skyward.cli import app


def test_version_prints_skyward_and_python(capsys):
    with pytest.raises(SystemExit, match="0"):
        app(["version"], exit_on_error=False)
    out = capsys.readouterr().out
    assert "skyward" in out
    assert sys.version.split()[0] in out


def test_help_shows_all_subcommands(capsys):
    with pytest.raises(SystemExit, match="0"):
        app(["--help"], exit_on_error=False)
    out = capsys.readouterr().out
    assert "offers" in out
    assert "providers" in out
    assert "config" in out
    assert "version" in out
