# Configuration

## TOML configuration files

Skyward loads configuration from two TOML files, merged with project settings taking precedence:

1. **Global defaults:** `~/.skyward/defaults.toml`
2. **Project config:** `skyward.toml` (in the current working directory)

### File format

```toml
[providers.my-aws]
type = "aws"
region = "us-west-2"

[providers.my-vastai]
type = "vastai"
min_reliability = 0.95
geolocation = "US"

[pools.training]
provider = "my-aws"
nodes = 4
accelerator = "A100"

[pools.training.image]
python = "3.13"
pip = ["torch", "transformers"]
apt = ["ffmpeg"]

[[pools.training.volumes]]
bucket = "my-bucket"
mount = "/data"
```

The `[providers]` section defines named provider configurations. Each must have a `type` field matching a supported provider (`aws`, `gcp`, `hyperstack`, `tensordock`, `vastai`, `runpod`, `verda`). All other fields are passed to the provider's config class.

The `[pools]` section defines named pools that reference a provider by name. Pools support `nodes`, `accelerator` (as a string name), `image` (as a sub-table), and `volumes` (as an array of tables).

### Using named pools

```python
with sky.Compute.Named("training") as compute:
    result = train() >> compute
```

## API reference

::: skyward.PoolSpec

::: skyward.Image

::: skyward.DEFAULT_IMAGE

::: skyward.AllocationStrategy

::: skyward.api.spec.PoolState
