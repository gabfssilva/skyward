"""Tests for sky providers commands."""

import json

import pytest

from skyward.cli import app


class TestProvidersList:
    def test_lists_all_providers(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "list", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        provider_names = {row["Provider"] for row in data}
        assert "aws" in provider_names
        assert "vastai" in provider_names
        assert "gcp" in provider_names

    def test_detects_configured_provider(self, capsys, monkeypatch):
        monkeypatch.setenv("VAST_API_KEY", "test-key")
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "list", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        vastai = next(r for r in data if r["Provider"] == "vastai")
        assert vastai["Status"] == "ok"

    def test_unconfigured_shows_dash(self, capsys, monkeypatch, tmp_path):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.setitem(
            __import__("skyward.cli.providers", fromlist=["_FILE_AUTH"])._FILE_AUTH,
            "vastai",
            (tmp_path / "nonexistent", ""),
        )
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "list", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        vastai = next(r for r in data if r["Provider"] == "vastai")
        assert vastai["Status"] == "-"

    def test_rich_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "list"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "Provider" in out
        assert "aws" in out


class TestProvidersCheck:
    def test_check_unconfigured_provider(self, capsys, monkeypatch, tmp_path):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.setitem(
            __import__("skyward.cli.providers", fromlist=["_FILE_AUTH"])._FILE_AUTH,
            "vastai",
            (tmp_path / "nonexistent", ""),
        )
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "check", "vastai", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert data["provider"] == "vastai"
        assert data["checks"][0]["status"] != "ok"

    def test_check_unknown_provider(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "check", "nonexistent", "--json"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "nonexistent" in out

    def test_check_requires_name_or_all(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["providers", "check"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "Specify" in out or "provider" in out.lower()
