"""Tests for sky config commands."""

import json

import pytest

from skyward.cli import app


class TestConfigPath:
    def test_json_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "path", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert "global" in data
        assert "project" in data
        assert "skyward.toml" in data["project"]

    def test_rich_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "path"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "Global config" in out
        assert "Project config" in out


class TestConfigShow:
    def test_json_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "show", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)
        assert "providers" in data
        assert "pools" in data

    def test_pool_not_found(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "show", "--pool", "nonexistent"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "not found" in out

    def test_rich_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "show"], exit_on_error=False)


class TestConfigValidate:
    def test_validates_without_error(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "validate"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "Global config" in out

    def test_json_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["config", "validate", "--json"], exit_on_error=False)
        lines = capsys.readouterr().out.strip().splitlines()
        for line in lines:
            data = json.loads(line)
            assert "label" in data
            assert "status" in data

    def test_detects_valid_provider_ref(self, capsys, monkeypatch, tmp_path):
        toml_content = b"""
[providers.myaws]
type = "aws"

[pools.training]
provider = "myaws"
"""
        config_file = tmp_path / "skyward.toml"
        config_file.write_bytes(toml_content)
        monkeypatch.setattr("skyward.config.load_config", lambda **_kw: {
            "providers": {"myaws": {"type": "aws"}},
            "pools": {"training": {"provider": "myaws"}},
        })
        with pytest.raises(SystemExit, match="0"):
            app(["config", "validate", "--json"], exit_on_error=False)
        lines = capsys.readouterr().out.strip().splitlines()
        statuses = [json.loads(l) for l in lines]
        pool_check = next((s for s in statuses if "training" in s["label"]), None)
        assert pool_check is not None
        assert pool_check["status"] == "ok"

    def test_detects_invalid_provider_ref(self, capsys, monkeypatch):
        monkeypatch.setattr("skyward.config.load_config", lambda **_kw: {
            "providers": {},
            "pools": {"inference": {"provider": "lambda"}},
        })
        with pytest.raises(SystemExit, match="0"):
            app(["config", "validate", "--json"], exit_on_error=False)
        lines = capsys.readouterr().out.strip().splitlines()
        statuses = [json.loads(l) for l in lines]
        pool_check = next((s for s in statuses if "inference" in s["label"]), None)
        assert pool_check is not None
        assert pool_check["status"] == "fail"
