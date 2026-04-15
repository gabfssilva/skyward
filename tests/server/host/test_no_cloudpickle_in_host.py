"""Invariant gate — the HTTP host tree must never import ``cloudpickle``.

Rationale: design §5.4 forbids the server process from deserializing user
code. Keeping the import out of ``skyward/server/host/`` makes that
guarantee structural instead of a convention.
"""

from __future__ import annotations

from pathlib import Path


def test_server_host_never_imports_cloudpickle() -> None:
    root = Path(__file__).resolve().parents[3] / "skyward" / "server" / "host"
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        if "cloudpickle" in py.read_text(encoding="utf-8"):
            offenders.append(str(py))
    assert not offenders, (
        "server/host must not import cloudpickle; offenders:\n"
        + "\n".join(offenders)
    )
