"""Tests for sky offers commands."""

import json
import sqlite3

import pytest

from skyward.cli import app
from skyward.offers.repository import OfferRepository, _SCHEMA


@pytest.fixture()
def _patch_repo(monkeypatch):
    """Patch _load_repo to return an in-memory repo with test data."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript(_SCHEMA)

    db.execute(
        "INSERT INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("a100", "A100", 80.0, "NVIDIA", "Ampere", "11.0", "12.0"),
    )
    db.execute(
        "INSERT INTO accelerators VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("h100", "H100", 80.0, "NVIDIA", "Hopper", "12.0", "12.0"),
    )
    db.execute(
        "INSERT INTO specs VALUES (?, ?, ?, ?, ?)",
        ("spec-a100", "a100", 48.0, 192.0, "x86_64"),
    )
    db.execute(
        "INSERT INTO specs VALUES (?, ?, ?, ?, ?)",
        ("spec-h100", "h100", 96.0, 320.0, "x86_64"),
    )
    db.execute(
        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("aws", "spec-a100", 1, "p4d.24xlarge", "us-east-1", 1.12, 4.10, "hour", None),
    )
    db.execute(
        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("vastai", "spec-a100", 1, "a100-80gb", "us-west", 0.85, None, "hour", None),
    )
    db.execute(
        "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("aws", "spec-h100", 8, "p5.48xlarge", "us-east-1", None, 32.77, "hour", None),
    )
    db.commit()

    repo = OfferRepository(db)
    monkeypatch.setattr("skyward.cli.offers._load_repo", lambda **_kw: repo)


@pytest.mark.usefixtures("_patch_repo")
class TestOffersList:
    def test_default_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list"], exit_on_error=False)
        out = capsys.readouterr().out
        assert "A100" in out
        assert "aws" in out

    def test_json_output(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 3
        assert all("Provider" in row for row in data)

    def test_filter_by_provider(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--provider", "vastai", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert all(row["Provider"] == "vastai" for row in data)

    def test_filter_by_accelerator(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--accelerator", "H100", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1
        assert data[0]["Accelerator"] == "8x H100"

    def test_filter_spot_only(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--spot", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert all(row["Spot"] != "-" for row in data)

    def test_sort_by_vram(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--sort", "vram", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 3

    def test_limit(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "list", "--limit", "1", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1


@pytest.mark.usefixtures("_patch_repo")
class TestOffersQuery:
    def test_raw_sql(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(
                ["offers", "query", "SELECT * FROM catalog WHERE provider = 'aws'", "--json"],
                exit_on_error=False,
            )
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert all(row["provider"] == "aws" for row in data)

    def test_empty_result(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(
                ["offers", "query", "SELECT * FROM catalog WHERE provider = 'nonexistent'", "--json"],
                exit_on_error=False,
            )
        data = json.loads(capsys.readouterr().out)
        assert data == []


@pytest.mark.usefixtures("_patch_repo")
class TestOffersSummary:
    def test_summary_all(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "summary", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert len(data) >= 2
        assert all("GPU" in row for row in data)

    def test_summary_filtered(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            app(["offers", "summary", "--accelerator", "A100", "--json"], exit_on_error=False)
        data = json.loads(capsys.readouterr().out)
        assert all(row["GPU"] == "A100" for row in data)
