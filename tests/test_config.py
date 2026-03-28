from pathlib import Path

import pytest

from skyward.accelerators import Accelerator
from skyward.api.logging import LogConfig
from skyward.api.metrics import Metric
from skyward.api.spec import Nodes, Worker
from skyward.config import (
    _build_nodes,
    _build_plugins,
    _deep_merge,
    load_config,
    resolve_pool,
)
from skyward.core.pool import ComputePool
from skyward.providers.aws.config import AWS
from skyward.providers.gcp.config import GCP
from skyward.providers.hyperstack.config import Hyperstack
from skyward.providers.jarvislabs.config import JarvisLabs
from skyward.providers.runpod.config import RunPod
from skyward.providers.scaleway.config import Scaleway
from skyward.providers.tensordock.config import TensorDock
from skyward.providers.vastai.config import VastAI
from skyward.providers.verda.config import Verda
from skyward.providers.vultr.config import Vultr

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


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
        assert spec.accelerator == Accelerator.from_name("A100")
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
        assert spec.accelerator == Accelerator.from_name("H100")

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
        assert spec.ttl == 1200

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

    def test_hyperstack_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.hs]\n'
            'type = "hyperstack"\n'
            'region = "CANADA-1"\n'
            '\n'
            '[pools.gpu]\nprovider = "hs"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, Hyperstack)
        assert spec.provider.region == "CANADA-1"

    def test_scaleway_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.scw]\n'
            'type = "scaleway"\n'
            'zone = "fr-par-2"\n'
            '\n'
            '[pools.gpu]\nprovider = "scw"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, Scaleway)
        assert spec.provider.zone == "fr-par-2"

    def test_scaleway_pool_auto_zone(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.scw]\n'
            'type = "scaleway"\n'
            '\n'
            '[pools.gpu]\nprovider = "scw"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, Scaleway)
        assert spec.provider.zone is None

    def test_vultr_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.v]\n'
            'type = "vultr"\n'
            'region = "ord"\n'
            '\n'
            '[pools.gpu]\nprovider = "v"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, Vultr)
        assert spec.provider.region == "ord"

    def test_vultr_bare_metal_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.v]\n'
            'type = "vultr"\n'
            'mode = "bare-metal"\n'
            'region = "ewr"\n'
            '\n'
            '[pools.gpu]\nprovider = "v"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        spec = pool._specs[0]
        assert isinstance(spec.provider, Vultr)
        assert spec.provider.mode == "bare-metal"

    def test_jarvislabs_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.jl]\n'
            'type = "jarvislabs"\n'
            'region = "EU1"\n'
            'storage_gb = 100\n'
            '\n'
            '[pools.gpu]\nprovider = "jl"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        assert isinstance(pool, ComputePool)
        spec = pool._specs[0]
        assert isinstance(spec.provider, JarvisLabs)
        assert spec.provider.region == "EU1"
        assert spec.provider.storage_gb == 100

    def test_tensordock_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.td]\n'
            'type = "tensordock"\n'
            'location = "us"\n'
            'storage_gb = 200\n'
            '\n'
            '[pools.gpu]\nprovider = "td"\nnodes = 1\n'
        )
        pool = resolve_pool("gpu", project_dir=tmp_path)
        assert isinstance(pool, ComputePool)
        spec = pool._specs[0]
        assert isinstance(spec.provider, TensorDock)
        assert spec.provider.location == "us"
        assert spec.provider.storage_gb == 200

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


class TestResolvePoolWithVolumes:
    def test_pool_with_volumes(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[[pools.ml.volumes]]\n'
            'bucket = "my-data"\n'
            'mount = "/data"\n'
            'read_only = true\n'
            '\n'
            '[[pools.ml.volumes]]\n'
            'bucket = "my-ckpts"\n'
            'mount = "/checkpoints"\n'
            'prefix = "exp-1/"\n'
            'read_only = false\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool.volumes) == 2
        assert pool.volumes[0].bucket == "my-data"
        assert pool.volumes[0].mount == "/data"
        assert pool.volumes[0].read_only is True
        assert pool.volumes[1].prefix == "exp-1/"
        assert pool.volumes[1].read_only is False

    def test_pool_without_volumes_has_empty_tuple(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\n\n'
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.volumes == ()


class TestResolvePoolNamed:
    def test_named_returns_pool(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            '[providers.a]\ntype = "aws"\nregion = "us-east-1"\n\n'
            '[pools.dev]\nprovider = "a"\nnodes = 2\naccelerator = "T4"\n'
        )
        pool = resolve_pool("dev", project_dir=tmp_path)
        assert isinstance(pool, ComputePool)
        spec = pool._specs[0]
        assert spec.nodes == 2
        assert spec.accelerator == Accelerator.from_name("T4")


_PROVIDER_PREAMBLE = '[providers.a]\ntype = "aws"\n\n'


class TestBuildNodes:
    def test_int(self):
        assert _build_nodes(4) == 4

    def test_list_pair(self):
        assert _build_nodes([2, 8]) == (2, 8)

    def test_dict_full(self):
        result = _build_nodes({"min": 4, "desired": 2, "max": 16})
        assert isinstance(result, Nodes)
        assert result.min == 4
        assert result.desired == 2
        assert result.max == 16

    def test_dict_min_only(self):
        result = _build_nodes({"min": 8})
        assert isinstance(result, Nodes)
        assert result.min == 8
        assert result.max is None
        assert result.desired is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid nodes"):
            _build_nodes("bad")  # type: ignore[arg-type]


class TestBuildPlugins:
    def test_torch_defaults(self):
        plugins = _build_plugins({"torch": True})
        assert len(plugins) == 1
        assert plugins[0].name == "torch"

    def test_torch_with_params(self):
        plugins = _build_plugins({"torch": {"backend": "nccl", "cuda": "cu121"}})
        assert len(plugins) == 1
        assert plugins[0].name == "torch"

    def test_multiple_plugins(self):
        plugins = _build_plugins({"torch": True, "keras": {"backend": "torch"}})
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"torch", "keras"}

    def test_unknown_plugin_raises(self):
        with pytest.raises(ValueError, match="Unknown plugin 'nonexistent'"):
            _build_plugins({"nonexistent": True})

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError, match="Invalid config for plugin"):
            _build_plugins({"torch": 42})


class TestResolvePoolWithWorker:
    def test_worker_from_toml(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.worker]\n'
            'concurrency = 4\n'
            'executor = "process"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.worker.concurrency == 4
        assert pool.worker.resolved_executor == "process"

    def test_worker_default(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert isinstance(pool.worker, Worker)
        assert pool.worker.concurrency == 1


class TestResolvePoolWithNodes:
    def test_nodes_int(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 4\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool._specs[0].nodes == 4

    def test_nodes_tuple(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = [2, 8]\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool._specs[0].nodes == (2, 8)

    def test_nodes_table(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            '\n'
            '[pools.ml.nodes]\n'
            'min = 4\n'
            'desired = 2\n'
            'max = 16\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        nodes = pool._specs[0].nodes
        assert isinstance(nodes, Nodes)
        assert nodes.min == 4
        assert nodes.desired == 2
        assert nodes.max == 16

    def test_nodes_table_min_only(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            '\n'
            '[pools.ml.nodes]\n'
            'min = 8\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        nodes = pool._specs[0].nodes
        assert isinstance(nodes, Nodes)
        assert nodes.min == 8


class TestResolvePoolWithLogging:
    def test_logging_true(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\nlogging = true\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.logging is True

    def test_logging_false(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\nlogging = false\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.logging is False

    def test_logging_table(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.logging]\n'
            'level = "DEBUG"\n'
            'rotation = "100 MB"\n'
            'retention = 5\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert isinstance(pool.logging, LogConfig)
        assert pool.logging.level == "DEBUG"
        assert pool.logging.rotation == "100 MB"
        assert pool.logging.retention == 5

    def test_logging_default(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.logging is True


class TestResolvePoolWithPlugins:
    def test_torch_plugin(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.plugins.torch]\n'
            'backend = "nccl"\n'
            'cuda = "cu121"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool._plugins) == 1
        assert pool._plugins[0].name == "torch"

    def test_torch_defaults(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.plugins]\n'
            'torch = true\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool._plugins) == 1
        assert pool._plugins[0].name == "torch"

    def test_multiple_plugins(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.plugins]\n'
            'jax = true\n'
            '\n'
            '[pools.ml.plugins.keras]\n'
            'backend = "jax"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool._plugins) == 2
        names = {p.name for p in pool._plugins}
        assert names == {"jax", "keras"}

    def test_mig_plugin(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.plugins.mig]\n'
            'profile = "3g.40gb"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool._plugins) == 1
        assert pool._plugins[0].name == "mig"

    def test_unknown_plugin_raises(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.plugins]\n'
            'nonexistent = true\n'
        )
        with pytest.raises(ValueError, match="Unknown plugin 'nonexistent'"):
            resolve_pool("ml", project_dir=tmp_path)

    def test_no_plugins_default(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool._plugins == ()


class TestResolvePoolWithMetrics:
    def test_metrics_disabled(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[pools.ml.image]\n'
            'metrics = false\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.image.metrics is None

    def test_metrics_custom(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[[pools.ml.image.metrics]]\n'
            'name = "gpu_util"\n'
            'command = "nvidia-smi"\n'
            'interval = 5\n'
            'multi = true\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool.image.metrics) == 1
        m = pool.image.metrics[0]
        assert isinstance(m, Metric)
        assert m.name == "gpu_util"
        assert m.command == "nvidia-smi"
        assert m.interval == 5
        assert m.multi is True

    def test_metrics_default(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.image.metrics is not None
        assert len(pool.image.metrics) > 0


class TestResolvePoolWithStorage:
    def test_volume_with_storage(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            '\n'
            '[[pools.ml.volumes]]\n'
            'bucket = "my-data"\n'
            'mount = "/data"\n'
            '\n'
            '[pools.ml.volumes.storage]\n'
            'endpoint = "https://s3.us-west-2.amazonaws.com"\n'
            'access_key = "AKIATEST"\n'
            'secret_key = "secret123"\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert len(pool.volumes) == 1
        assert pool.volumes[0].storage is not None
        assert pool.volumes[0].storage.endpoint == "https://s3.us-west-2.amazonaws.com"
        assert pool.volumes[0].storage.access_key == "AKIATEST"
        assert pool.volumes[0].storage.secret_key == "secret123"

    def test_volume_without_storage(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[[pools.ml.volumes]]\n'
            'bucket = "my-data"\n'
            'mount = "/data"\n'
            '\n'
            '[pools.ml]\nprovider = "a"\nnodes = 1\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool.volumes[0].storage is None


class TestResolvePoolWithDiskGb:
    def test_disk_gb_from_toml(self, tmp_path: Path):
        (tmp_path / "skyward.toml").write_text(
            _PROVIDER_PREAMBLE +
            '[pools.ml]\n'
            'provider = "a"\n'
            'nodes = 1\n'
            'disk_gb = 200\n'
        )
        pool = resolve_pool("ml", project_dir=tmp_path)
        assert pool._specs[0].disk_gb == 200
