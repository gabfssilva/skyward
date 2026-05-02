"""Round-trip tests for the script-exec PendingFunction builder."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _run_locally(script: Path, argv_extra: list[str] | None = None) -> dict:
    from skyward.cli._script import build_exec_pending

    pending = build_exec_pending(script, argv_extra or [])
    return pending.fn(*pending.args, **pending.kwargs)


def test_captures_stdout(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hello'); print('world')")
    result = _run_locally(script)
    assert result["stdout"] == "hello\nworld\n"
    assert result["stderr"] == ""
    assert result["exit"] == 0


def test_captures_stderr(tmp_path: Path) -> None:
    script = tmp_path / "warn.py"
    script.write_text("import sys; sys.stderr.write('boom\\n')")
    result = _run_locally(script)
    assert result["stdout"] == ""
    assert result["stderr"] == "boom\n"
    assert result["exit"] == 0


def test_propagates_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "exit.py"
    script.write_text("import sys; sys.exit(7)")
    result = _run_locally(script)
    assert result["exit"] == 7


def test_traceback_for_uncaught(tmp_path: Path) -> None:
    script = tmp_path / "bang.py"
    script.write_text("raise RuntimeError('nope')")
    result = _run_locally(script)
    assert result["exit"] == 1
    assert "RuntimeError: nope" in result["stderr"]


def test_argv_forwarded(tmp_path: Path) -> None:
    script = tmp_path / "argv.py"
    script.write_text("import sys; print(sys.argv)")
    result = _run_locally(script, ["a", "b"])
    assert str(script) in result["stdout"]
    assert "'a'" in result["stdout"]
    assert "'b'" in result["stdout"]


def test_dunder_main(tmp_path: Path) -> None:
    script = tmp_path / "main.py"
    script.write_text("if __name__ == '__main__': print('ok')")
    result = _run_locally(script)
    assert result["stdout"] == "ok\n"
