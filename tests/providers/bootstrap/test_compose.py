"""Tests for bootstrap script composition — emit atomicity and CR handling."""
from __future__ import annotations

import pytest

from skyward.providers.bootstrap.compose import make_header


class TestEmitAtomicity:
    """emit() must use flock to serialize concurrent writes."""

    def test_emit_uses_flock(self) -> None:
        header = make_header(metrics=None)
        assert "flock" in header
        assert "events.lock" in header

    def test_emit_flock_guards_printf(self) -> None:
        header = make_header(metrics=None)
        lines = header.splitlines()
        emit_body = _extract_function_body(lines, "emit")
        body_text = "\n".join(emit_body)
        # flock and printf must both be inside the emit body
        assert "flock" in body_text
        assert "printf" in body_text
        # flock must appear before printf (guards the write)
        flock_pos = body_text.index("flock")
        printf_pos = body_text.index("printf")
        assert flock_pos < printf_pos


class TestRunPhaseCarriageReturn:
    """run_phase() uses __CR__ sentinel to detect \\r for overwrite support."""

    def test_run_phase_uses_cr_sentinel(self) -> None:
        header = make_header(metrics=None)
        lines = header.splitlines()
        run_phase_body = _extract_function_body(lines, "run_phase")
        body_text = "\n".join(run_phase_body)
        assert "__CR__" in body_text, "run_phase must use __CR__ sentinel for \\r detection"

    def test_run_phase_filters_empty_lines(self) -> None:
        header = make_header(metrics=None)
        lines = header.splitlines()
        run_phase_body = _extract_function_body(lines, "run_phase")
        body_text = "\n".join(run_phase_body)
        assert '[ -n "$line" ]' in body_text or '-n "$line"' in body_text


class TestParseJsonlWithCarriageReturn:
    """_parse_jsonl_line handles console events with escaped \\r content."""

    def test_parse_console_with_escaped_cr(self) -> None:
        from skyward.infra.ssh import RawBootstrapConsole, _parse_jsonl_line

        line = r'{"type":"console","content":"\r 50%|█████     | 500/1000","stream":"stdout"}'
        event = _parse_jsonl_line(line)
        assert isinstance(event, RawBootstrapConsole)
        assert "50%" in event.content

    def test_parse_console_with_overwrite_flag(self) -> None:
        from skyward.infra.ssh import RawBootstrapConsole, _parse_jsonl_line

        line = '{"type":"console","content":"50%|█████","stream":"stdout","overwrite":true}'
        event = _parse_jsonl_line(line)
        assert isinstance(event, RawBootstrapConsole)
        assert event.overwrite is True

    def test_parse_console_overwrite_defaults_false(self) -> None:
        from skyward.infra.ssh import RawBootstrapConsole, _parse_jsonl_line

        line = '{"type":"console","content":"some log","stream":"stdout"}'
        event = _parse_jsonl_line(line)
        assert isinstance(event, RawBootstrapConsole)
        assert event.overwrite is False

    def test_parse_console_with_multiple_escaped_cr(self) -> None:
        from skyward.infra.ssh import RawBootstrapConsole, _parse_jsonl_line

        line = r'{"type":"console","content":"\r 10%|█         | 100/1000\r 20%|██        | 200/1000","stream":"stdout"}'
        event = _parse_jsonl_line(line)
        assert isinstance(event, RawBootstrapConsole)
        assert "10%" in event.content
        assert "20%" in event.content


# ── helpers ──────────────────────────────────────────────────────────


def _extract_function_body(lines: list[str], func_name: str) -> list[str]:
    """Extract the body lines of a bash function from the script."""
    body: list[str] = []
    depth = 0
    inside = False

    for line in lines:
        stripped = line.strip()
        if not inside and func_name + "()" in stripped:
            inside = True
            if "{" in stripped:
                depth += 1
            continue

        if inside:
            depth += stripped.count("{") - stripped.count("}")
            body.append(line)
            if depth <= 0:
                break

    return body
