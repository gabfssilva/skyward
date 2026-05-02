"""Build a ``PendingFunction`` that executes a script remotely.

The returned function ``exec``s the script source on the worker, capturing
stdout/stderr in memory and returning them alongside the exit code.

Real-time streaming is not implemented today; output is buffered and
returned at the end. See the project plan for the SSE follow-up.
"""

from __future__ import annotations

from pathlib import Path

import skyward as sky
from skyward.api.function import PendingFunction


@sky.function
def _exec_script(content: str, argv: list[str], marker: str) -> dict:
    import contextlib
    import io
    import sys
    import traceback

    out = io.StringIO()
    err = io.StringIO()
    saved_argv = sys.argv
    sys.argv = argv
    exit_code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(  # noqa: S102
                    compile(content, marker, "exec"),
                    {"__name__": "__main__", "__file__": marker},
                )
            except SystemExit as e:
                exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
            except BaseException:
                traceback.print_exc(file=err)
                exit_code = 1
    finally:
        sys.argv = saved_argv
    return {"stdout": out.getvalue(), "stderr": err.getvalue(), "exit": exit_code}


def build_exec_pending(script_path: Path, argv_extra: list[str]) -> PendingFunction[dict]:
    content = script_path.read_text()
    return _exec_script(content, [str(script_path), *argv_extra], str(script_path))
