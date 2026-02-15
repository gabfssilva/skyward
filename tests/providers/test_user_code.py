"""Tests for user code sync utilities (build_user_code_tarball, upload_user_code)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from skyward.providers.common import build_user_code_tarball


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "__init__.py").write_text("from .core import main\n")
    (lib / "core.py").write_text("def main(): return 42\n")

    nested = lib / "sub"
    nested.mkdir()
    (nested / "__init__.py").write_text("")
    (nested / "helper.py").write_text("def help(): pass\n")

    pycache = lib / "__pycache__"
    pycache.mkdir()
    (pycache / "core.cpython-313.pyc").write_bytes(b"fake")

    (tmp_path / "utils.py").write_text("X = 1\n")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")

    return tmp_path


def _extract_names(tarball: bytes) -> set[str]:
    buf = io.BytesIO(tarball)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        return {m.name for m in tar.getmembers()}


class TestBuildUserCodeTarball:
    def test_includes_directory(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("lib",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert "lib/__init__.py" in names
        assert "lib/core.py" in names
        assert "lib/sub/helper.py" in names

    def test_excludes_pycache_by_default(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("lib",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert not any("__pycache__" in n for n in names)

    def test_includes_single_file(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("utils.py",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert "utils.py" in names
        assert len(names) == 1

    def test_custom_excludes(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("lib",),
            excludes=("sub",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert "lib/__init__.py" in names
        assert not any("sub" in n for n in names)

    def test_missing_include_skipped(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("nonexistent",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert len(names) == 0

    def test_multiple_includes(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=("lib", "utils.py"),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert "utils.py" in names
        assert "lib/core.py" in names

    def test_empty_includes_returns_empty_tarball(self, project_dir: Path) -> None:
        tarball = build_user_code_tarball(
            includes=(),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert len(names) == 0

    def test_glob_excludes_within_directory(self, project_dir: Path) -> None:
        (project_dir / "lib" / "notes.csv").write_text("x,y\n")
        tarball = build_user_code_tarball(
            includes=("lib",),
            excludes=("*.csv",),
            project_root=project_dir,
        )
        names = _extract_names(tarball)
        assert not any(n.endswith(".csv") for n in names)
        assert "lib/core.py" in names
