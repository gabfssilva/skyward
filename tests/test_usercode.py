"""The user-code tarball: what goes in, and what the excludes keep out."""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from skyward.core import usercode

pytestmark = pytest.mark.unit


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Path]:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "model.py").write_text("y = 2\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "model.pyc").write_text("junk")
    (tmp_path / "pkg" / "big.log").write_text("noise")
    (tmp_path / "solo.py").write_text("z = 3\n")

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(cwd)


def _names(blob: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        return set(archive.getnames())


def test_a_directory_is_walked_and_a_file_is_carried_as_named(project: Path):
    names = _names(usercode.tarball(("pkg", "solo.py")))
    assert "pkg/__init__.py" in names
    assert "pkg/model.py" in names
    assert "solo.py" in names


def test_default_excludes_drop_pycache(project: Path):
    names = _names(usercode.tarball(("pkg",)))
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_extra_excludes_are_honoured(project: Path):
    names = _names(usercode.tarball(("pkg",), excludes=("*.log",)))
    assert "pkg/model.py" in names
    assert not any(name.endswith(".log") for name in names)


def test_a_missing_include_is_skipped(project: Path):
    names = _names(usercode.tarball(("pkg", "does_not_exist.py")))
    assert "pkg/model.py" in names
