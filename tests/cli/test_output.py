"""Tests for CLI output helpers."""

import json

import pytest

from skyward.cli._output import format_price, print_status, print_table


class TestPrintTable:
    def test_json_output(self, capsys):
        print_table(["Name", "Value"], [["a", 1], ["b", 2]], as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data == [{"Name": "a", "Value": 1}, {"Name": "b", "Value": 2}]

    def test_rich_table_renders(self, capsys):
        print_table(["Col"], [["val"]])
        out = capsys.readouterr().out
        assert "Col" in out
        assert "val" in out

    def test_empty_rows(self, capsys):
        print_table(["A", "B"], [], as_json=True)
        assert json.loads(capsys.readouterr().out) == []


class TestPrintStatus:
    def test_json_output(self, capsys):
        print_status("Config", "ok", "/path/to/file", as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data == {"label": "Config", "status": "ok", "detail": "/path/to/file"}

    def test_rich_output(self, capsys):
        print_status("Config", "ok", "found")
        out = capsys.readouterr().out
        assert "Config" in out


class TestFormatPrice:
    @pytest.mark.parametrize(
        ("price", "unit", "expected"),
        [
            (1.234, "hr", "$1.23/hr"),
            (0.5, "hr", "$0.50/hr"),
            (None, "hr", "-"),
            (2.0, "min", "$2.00/min"),
        ],
    )
    def test_format(self, price, unit, expected):
        assert format_price(price, unit) == expected
