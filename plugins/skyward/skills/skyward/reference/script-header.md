# A PEP 723 header declares the compute

The `# /// script` block `uv run` reads, with a `[tool.skyward]` table in it. `requires-python` and `dependencies` become the image, `[tool.skyward]` is the rest of the compute under the names `sky.Compute` gives the same things, and the whole file runs on the node as `__main__`. What every `sky run` shares (the node, the compute's name, the account, output, `--node`) is `reference/script.md`.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
#
# [tool.skyward]
# provider = "runpod"
# accelerator = "A100"
# nodes = 2
# delete_on_exit = false
#
# [tool.skyward.image]
# apt = ["build-essential"]
# env = { OMP_NUM_THREADS = "8" }
# ///
import sys

import numpy as np
import skyward as sky

info = sky.instance_info()
rows = sky.shard(list(range(1000)))
print(f"rank {info.rank} of {info.nodes}: {len(rows)} rows, mean {np.mean(rows):.1f}, args {sys.argv[1:]}")
```

```bash
sky run train.py --node all -- --epochs 3
```

`__name__ == "__main__"`, `__file__` and `sys.argv[0]` are the path as given to `sky run`; `sys.argv[1:]` are the arguments after the file, except `--node` and `--url`, which are `sky run`'s wherever they are; after `--`, everything is the script's.

## The block

Exactly one `# /// script` block. Every line inside starts with `# ` (or is a bare `#`). A file without a `[tool.skyward]` table is not a header script: `sky run` looks for `sky.app` functions instead (`reference/script-app.md`), and refuses a file with neither: `declares no compute`.

PEP 723's own keys:

| key | becomes |
|---|---|
| `requires-python` | the node's interpreter: the first of `3.12`, `3.13`, `3.14` the specifier admits (`>=3.12.4` gives `3.13`). Left out, the image's default |
| `dependencies` | the image's pip packages, as pip spells them |

`[tool.skyward]`:

| key | type | default | meaning |
|---|---|---|---|
| `provider` | string | required | a provider **kind** (`runpod`, `aws`, `container`, ...: the kinds `sky providers list --kinds` prints), with that kind's default settings |
| `accelerator` | string | none | `"A100"`, `"H100"`, ... as `accelerator=` takes it |
| `cpus` | int | none | least vCPUs per machine |
| `memory_gb` | int | none | least memory per machine, in GB |
| `region` | string | none | where to buy |
| `allocation` | string | `"spot_if_available"` | `"spot"`, `"on_demand"`, `"spot_if_available"`, `"cheapest"` |
| `nodes` | int, or table | `1` | a count (at least 1), or `sky.Nodes` as a table: `{ initial = 8, min = 4 }`, `{ min = 2, max = 8 }` |
| `delete_on_exit` | bool | `true` | whether the run deletes the compute when the script ends |

`nodes` as a table needs an `initial` or a `min`; without `initial` it opens at `min`. The bounds must satisfy `0 <= min <= initial <= max` with room for one node.

`[tool.skyward.image]`, the image fields that do not come from PEP 723's keys:

| key | type | meaning |
|---|---|---|
| `base` | string | Docker image the machines start from |
| `apt` | list of strings | apt packages |
| `env` | table of strings | environment variables on the nodes |
| `pip_indexes` | list of `{ url, packages }` | extra package indexes; `packages` scopes which names resolve from `url`, empty makes it an ordinary extra index |
| `includes` | list of strings | local files and directories, counted from the file, sent with every run and importable by their own names (`reference/script.md`, "Local code") |
| `excludes` | list of strings | glob patterns left out of `includes` |
| `skyward` | string | where the node gets skyward from: `"auto"` (default), `"local"` (a wheel built from the checkout the daemon runs from), `"github"`, `"pypi"` |

`python` and `pip` are not accepted here: they are `requires-python` and `dependencies`.

`[[tool.skyward.plugins]]`, one table per plugin: `kind` plus the plugin's own fields, with the SDK's defaults for what is left out.

```toml
[[tool.skyward.plugins]]
kind = "torch"
backend = "gloo"

[[tool.skyward.plugins]]
kind = "huggingface"
```

| kind | fields |
|---|---|
| `torch` | `backend` (`"nccl"`), `cuda` (`"cu128"`), `version` |
| `jax` | `cuda` (`"cu124"`) |
| `keras` | `backend` (`"jax"`) |
| `accelerate` | `config` (table) |
| `huggingface` | `token` |
| `cuml` | `cuda` (`"cu12"`) |
| `sklearn` | `version` |
| `joblib` | `version` |
| `mig` | `profile` (required) |
| `mps` | `active_thread_percentage`, `pinned_memory_limit` |

What the plugins do is the SDK's `plugins/<kind>/` page.

Anything else is refused as an unknown field, including what `sky.Compute` takes but the header does not: `name` (derived, see `reference/script.md`), `ttl`, `executor`, `options`, `ports`, and where the script runs (that is `--node`, which may differ on every run of the same file). Provider settings (a RunPod `cloud_type`, an AWS `region` of the account) cannot be written here either; a file that needs them uses `sky.app`.

The block is read without running the file: a TOML error, an unknown field, a plugin that does not exist or a field it does not have, bounds that disagree, a `requires-python` that admits none of 3.12–3.14, an unknown provider kind, or an include that is not there is refused before anything else happens.
