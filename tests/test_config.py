from pathlib import Path

import pytest

from skyward.api.pool import ComputePool
from skyward.config import _deep_merge, load_config, resolve_pool
from skyward.providers.aws.config import AWS
from skyward.providers.gcp.config import GCP
from skyward.providers.runpod.config import RunPod
from skyward.providers.vastai.config import VastAI
from skyward.providers.verda.config import Verda

pytestmark = [pytest.mark.xdist_group("unit")]


class TestDeepMerge:
    def test_shallow_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"providers": {"aws": {"region": "us-east-1", "ami": "abc"}}}
        override = {"providers": {"aws": {"region": "us-west-2"}}}
        result = _deep_merge(base, override)
        assert result == {"providers": {"aws": {"region": "us-west-2", "ami": "abc"}}}

    def test_override_adds_new_keys(self):
        base = {"pools": {"a": {"nodes": 1}}}
        override = {"pools": {"b": {"nodes": 2}}}
        result = _deep_merge(base, override)
        assert result == {"pools": {"a": {"nodes": 1}, "b": {"nodes": 2}}}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}


class TestLoadConfig:
    def test_project_only(self, tmp_path: Path):
        toml = tmp_path / "skyward.toml"
        toml.write_text('[pools.dev]\nnodes = 1\nprovider = "aws"\n')
        result = load_config(project_dir=tmp_path, global_path=tmp_path / "nonexistent.toml")
        assert result["pools"]["dev"]["nodes"] == 1

    def test_global_only(self, tmp_path: Path):
        global_toml = tmp_path / "defaults.toml"
        global_toml.write_text('[providers.my-aws]\ntype = "aws"\nregion = "us-west-2"\n')
        result = load_config(project_dir=tmp_path / "noproject", global_path=global_toml)
        assert result["providers"]["my-aws"]["region"] == "us-west-2"

    def test_merge_project_overrides_global(self, tmp_path: Path):
        global_toml = tmp_path / "defaults.toml"
        global_toml.write_text('[pools.train]\nnodes = 2\nprovider = "aws"\n')
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "skyward.toml").write_text('[pools.train]\nnodes = 8\nprovider = "aws"\n')
        result = load_config(project_dir=project_dir, global_path=global_toml)
        assert result["pools"]["train"]["nodes"] == 8

    def test_no_files_returns_empty_sections(self, tmp_path: Path):
        result = load_config(project_dir=tmp_path / "nope", global_path=tmp_path / "nope.toml")
        assert result == {"providers": {}, "pools": {}}


class TestResolvePool:
    def test_aws_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.my-aws]\n'
            'type = "aws"\n'
            'region = "us-west-2"\n'
            '\n'
            '[pools.train]\n'
            'provider = "my-aws"\n'
            'nodes = 4\n'
            'accelerator = "A100"\n'
            'allocation = "spot"\n'
        )
        pool = resolve_pool("train", project_dir=tmp_path)
        assert isinstance(pool, ComputePool)
        spec = pool._specs[0]
        assert isinstance(spec.provider, AWS)
        assert spec.provider.region == "us-west-2"
        assert spec.nodes == 4
        assert spec.accelerator == "A100"
        assert spec.allocation == "spot"

    def test_vastai_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.cheap]\n'
            'type = "vastai"\n'
            'geolocation = "US"\n'
            '\n'
            '[pools.infer]\n'
            'provider = "cheap"\n'
            'nodes = 1\n'
            'accelerator = "H100"\n'
        )
        pool = resolve_pool("infer", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, VastAI)
        assert spec.provider.geolocation == "US"
        assert spec.nodes == 1
        assert spec.accelerator == "H100"

    def test_pool_with_image(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            'accelerator = "T4"\n'
            '\n'
            '[pools.ml.image]\n'
            'python = "3.12"\n'
            'pip = ["torch", "numpy"]\n'
            'apt = ["git"]\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.image.python == "3.12"
        assert "torch" in pool.image.pip
        assert "git" in pool.image.apt

    def test_pool_with_image_env(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.image]\n'
            'pip = ["torch"]\n'
            '\n'
            '[pools.ml.image.env]\n'
            'HF_TOKEN = "abc"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.image.env == {"HF_TOKEN": "abc"}

    def test_unknown_pool_raises(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text("")
        with pytest.raises(KeyError, match="nope"):
            resolve_pool("nope", project_dir=tmp_path)

    def test_unknown_provider_ref_raises(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[pools.x]\nprovider = "missing"\nnodes = 1\n'
        )
        with pytest.raises(KeyError, match="missing"):
            resolve_pool("x", project_dir=tmp_path)

    def test_unknown_provider_type_raises(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.bad]\ntype = "fake"\n\n'
            '[pools.x]\nprovider = "bad"\nnodes = 1\n'
        )
        with pytest.raises(ValueError, match="fake"):
            resolve_pool("x", project_dir=tmp_path)

    def test_pool_defaults(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.minimal]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("minimal", project_dir=tmp_path)
        spec = pool._specs[0]
        assert spec.allocation == "spot-if-available"
        assert spec.ttl == 600

    def test_pool_with_worker(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.train]\nprovider = "a"\nnodes = 2\n\n'
            '[pools.train.worker]\nconcurrency = 4\nexecutor = "process"\n'
        )
        pool = resolve_pool("train", project_dir=tmp_path)
        assert pool.worker.concurrency == 4
        assert pool.worker.executor == "process"

    def test_pool_with_worker_concurrency_only(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.train]\nprovider = "a"\nnodes = 1\n\n'
            '[pools.train.worker]\nconcurrency = 8\n'
        )
        pool = resolve_pool("train", project_dir=tmp_path)
        assert pool.worker.concurrency == 8
        assert pool.worker.executor == "auto"

    def test_pool_with_timeouts(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.train]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            'default_compute_timeout = 600.0\n'
            'provision_timeout = 900\n'
            'ssh_timeout = 120\n'
            'ssh_retry_interval = 5\n'
            'provision_retry_delay = 15.0\n'
            'max_provision_attempts = 5\n'
        )
        pool = resolve_pool("train", project_dir=tmp_path)
        assert pool.default_compute_timeout == 600.0
        assert pool.provision_timeout == 900
        assert pool.ssh_timeout == 120
        assert pool.ssh_retry_interval == 5
        assert pool.provision_retry_delay == 15.0
        assert pool.max_provision_attempts == 5

    def test_pool_with_selection(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.train]\nprovider = "a"\nnodes = 1\nselection = "first"\n'
        )
        pool = resolve_pool("train", project_dir=tmp_path)
        assert pool.selection == "first"

    def test_runpod_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.rp]\n'
            'type = "runpod"\n'
            'container_disk_gb = 100\n'
            '\n'
            '[pools.gpu]\nprovider = "rp"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, RunPod)
        assert spec.provider.container_disk_gb == 100

    def test_verda_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.v]\ntype = "verda"\nregion = "FIN-01"\n\n'
            '[pools.gpu]\nprovider = "v"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        assert isinstance(pool._specs[0].provider, Verda)

    def test_gcp_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.g]\n'
            'type = "gcp"\n'
            'project = "my-project"\n'
            'zone = "us-central1-a"\n'
            '\n'
            '[pools.gpu]\n'
            'provider = "g"\n'
            'nodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, GCP)
        assert spec.provider.project == "my-project"
        assert spec.provider.zone == "us-central1-a"

    def test_global_provider_project_pool(self, tmp_path: Path):
        global_toml = tmp_path / "defaults.toml"
        global_toml.write_text('[providers.shared]\ntype = "aws"\nregion = "eu-west-1"\n')
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "skyward.toml").write_text(
            '[pools.train]\nprovider = "shared"\nnodes = 2\n'
        )
        pool = resolve_pool("train", project_dir=project_dir, global_path=global_toml)
        spec = pool._specs[0]
        assert isinstance(spec.provider, AWS)
        assert spec.provider.region == "eu-west-1"
        assert spec.nodes == 2


class TestComputePoolNamed:
    def test_named_returns_pool(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\nregion = "us-east-1"\n\n'
            '[pools.dev]\nprovider = "a"\nnodes = 2\naccelerator = "T4"\n'
        )
        monkeypatch.chdir(tmp_path)
        pool = ComputePool.Named("dev")
        assert isinstance(pool, ComputePool)
        spec = pool._specs[0]
        assert spec.nodes == 2
        assert spec.accelerator == "T4"
