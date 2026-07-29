"""Local code, packed for the workers.

The user names directories and files to carry along; this builds the tar.gz the
node unpacks into the remote environment's site-packages. It runs on the client,
because that is the only place the files are — with a remote daemon the machine
that provisions the node is not the machine the user is sitting at.
"""

from __future__ import annotations

import fnmatch
import io
import tarfile
from pathlib import Path

_DEFAULT_EXCLUDES = ("__pycache__", "*.pyc", "*.pyo", ".git", ".venv", "node_modules", "*.egg-info")


def tarball(includes: tuple[str, ...], excludes: tuple[str, ...] = ()) -> bytes:
    """A tar.gz of the given paths, resolved against the working directory.

    Directories are walked; a path that does not exist is skipped. ``excludes`` and
    a built-in set of noise patterns (``__pycache__``, ``.git``, ``.venv``, ...) drop
    any path whose components match, so a synced tree does not carry a virtualenv or
    a git history along with it.
    """
    root = Path.cwd()
    patterns = _DEFAULT_EXCLUDES + excludes

    def excluded(path: Path) -> bool:
        return any(fnmatch.fnmatch(part, pattern) for part in path.parts for pattern in patterns)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for include in includes:
            path = Path(include)
            absolute = path.is_absolute()
            source = path if absolute else root / include
            if not source.exists():
                continue
            if source.is_file():
                archive.add(str(source), arcname=source.name if absolute else include)
                continue
            base = source.parent if absolute else root
            for entry in source.rglob("*"):
                relative = entry.relative_to(base)
                if entry.is_file() and not excluded(relative):
                    archive.add(str(entry), arcname=str(relative))
    return buffer.getvalue()
