"""Build the pending function for ``sky install`` — remote ``uv add``.

The worker subprocess is launched via ``nohup env <SKYWARD_* only> …`` and
does not inherit the bootstrap PATH, so ``uv`` is resolved explicitly
(mirroring ``providers/common.py``). ``uv add`` (not ``uv pip install``)
targets ``/opt/skyward/.venv`` via the project and persists in
``pyproject.toml``.
"""

from __future__ import annotations

import skyward as sky
from skyward.api.function import PendingFunction


@sky.function
def _uv_add(packages: tuple[str, ...]) -> dict[str, int]:
    import shutil
    import subprocess
    import sys

    uv = shutil.which("uv") or "/root/.local/bin/uv"
    proc = subprocess.Popen(
        [uv, "add", *packages],
        cwd="/opt/skyward",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return {"exit": proc.wait()}


def build_install_pending(packages: tuple[str, ...]) -> PendingFunction[dict]:
    """Build the install task for already-resolved pip specifiers."""
    return _uv_add(packages)
