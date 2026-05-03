"""Build a ``PendingFunction`` that executes a script remotely.

The returned function ``exec``s the script source on the worker and
returns the exit code. stdout/stderr are not captured here — they go
through the worker's stdio writer, which prefixes each line with
``[task-id=<eid>]`` (see :mod:`skyward.infra.worker`) and reaches the
CLI as ``Log.Emitted`` events filterable by ``task_id``.
"""

from __future__ import annotations

from pathlib import Path

import skyward as sky
from skyward.api.function import PendingFunction


@sky.function
def _exec_script(content: str, argv: list[str], marker: str) -> dict:
    import sys
    import traceback

    saved_argv = sys.argv
    sys.argv = argv
    exit_code = 0
    try:
        try:
            exec(  # noqa: S102
                compile(content, marker, "exec"),
                {"__name__": "__main__", "__file__": marker},
            )
        except SystemExit as e:
            exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
    finally:
        sys.argv = saved_argv
    return {"exit": exit_code}


def build_exec_pending(script_path: Path, argv_extra: list[str]) -> PendingFunction[dict]:
    content = script_path.read_text()
    return _exec_script(content, [str(script_path), *argv_extra], str(script_path))
