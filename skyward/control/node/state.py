from __future__ import annotations

import sys
from typing import Final

type NodeId = int

_PYTHON_VERSION: Final = f"{sys.version_info.major}.{sys.version_info.minor}"


class PythonVersionMismatchError(RuntimeError):
    def __init__(self, local: str, remote: str) -> None:
        self.local = local
        self.remote = remote
        super().__init__(
            f"Python version mismatch: local={local}, remote={remote}. "
            f"Cloudpickle cannot safely serialize bytecode across versions. "
            f"Set Image(python='{local}') or use Image(python='auto')."
        )


def check_python_version(remote_version: str) -> None:
    if remote_version != _PYTHON_VERSION:
        raise PythonVersionMismatchError(local=_PYTHON_VERSION, remote=remote_version)
