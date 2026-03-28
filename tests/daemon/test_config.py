from pathlib import Path

from skyward.config import load_config, resolve_pool_config


class TestDaemonConfig:
    def test_resolve_pool_config_extracts_daemon_flag(self, tmp_path: Path) -> None:
        toml = tmp_path / "skyward.toml"
        toml.write_text("""
[providers.my-vastai]
type = "vastai"

[pools.train]
provider = "my-vastai"
nodes = 4
accelerator = "A100"
daemon = true
ttl = 3600
""")
        result = resolve_pool_config("train", project_dir=tmp_path)
        assert result.daemon is True
        assert result.pool is not None

    def test_resolve_pool_config_defaults_daemon_false(self, tmp_path: Path) -> None:
        toml = tmp_path / "skyward.toml"
        toml.write_text("""
[providers.my-vastai]
type = "vastai"

[pools.train]
provider = "my-vastai"
nodes = 1
""")
        result = resolve_pool_config("train", project_dir=tmp_path)
        assert result.daemon is False
