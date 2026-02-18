from __future__ import annotations

import io
import tarfile

import pytest

import skyward as sky
from skyward import ComputePool, Image
from skyward.providers.common import build_user_code_tarball

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("image")]


class TestPipPackages:
    def test_pip_package_available_on_worker(self, pip_pool):
        @sky.compute
        def check_requests():
            import requests

            return requests.__version__

        version = check_requests() >> pip_pool
        assert isinstance(version, str)
        assert len(version) > 0


class TestAptPackages:
    def test_apt_package_available_on_worker(self, apt_pool):
        @sky.compute
        def check_jq():
            import subprocess

            result = subprocess.run(
                ["jq", "--version"], capture_output=True, text=True
            )
            return result.stdout.strip()

        version = check_jq() >> apt_pool
        assert "jq" in version


class TestEnvVars:
    def test_env_vars_visible_on_worker(self, env_pool):
        @sky.compute
        def read_env():
            import os

            return os.environ.get("MY_TEST_VAR")

        value = read_env() >> env_pool
        assert value == "hello123"


class TestIncludes:
    @pytest.mark.xfail(
        reason="Image.includes only supports paths relative to CWD. "
        "Absolute paths outside the project root cause ValueError in "
        "build_user_code_tarball (file.relative_to(root) fails). "
        "The pool then hangs waiting for bootstrap that silently failed.",
        strict=False,
    )
    @pytest.mark.timeout(60)
    def test_local_module_importable_on_worker(self, tmp_path):
        module_dir = tmp_path / "my_test_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("VERSION = '0.1.0'\n")

        with ComputePool(
            provider=sky.Container(),
            nodes=1,
            image=Image(includes=[str(module_dir)]),
        ) as pool:
            @sky.compute
            def import_module():
                import my_test_module  # noqa: F811  # pyright: ignore[reportMissingImports]

                return my_test_module.VERSION

            result = import_module() >> pool
            assert result == "0.1.0"


class TestTarballBuilding:
    def test_includes_directory(self, tmp_path):
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "__init__.py").write_text("x = 1")

        tar_bytes = build_user_code_tarball(
            includes=("lib",), project_root=tmp_path,
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()

        assert any("__init__.py" in n for n in names)

    def test_excludes_pycache(self, tmp_path):
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "__init__.py").write_text("x = 1")
        cache = lib / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-313.pyc").write_text("compiled")

        tar_bytes = build_user_code_tarball(
            includes=("lib",), project_root=tmp_path,
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()

        assert not any("__pycache__" in n for n in names)

    def test_custom_excludes(self, tmp_path):
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "__init__.py").write_text("x = 1")
        (lib / "data.csv").write_text("a,b,c")

        tar_bytes = build_user_code_tarball(
            includes=("lib",), excludes=("*.csv",), project_root=tmp_path,
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()

        assert not any(".csv" in n for n in names)
