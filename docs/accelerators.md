# Accelerators

Skyward uses one canonical accelerator catalog for request names and provider
offer normalization. A provider may return names such as `NVIDIA H100 80GB
SXM5` or `H100-80G-PCIe`; the daemon normalizes them to the shared accelerator
name and VRAM fields used by offer queries.

## Request an accelerator

Use a factory under `sky.accelerators`:

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
) as compute:
    ...
```

`count` is the number of accelerators per node:

```python
sky.accelerators.H100(count=4)
sky.accelerators.RTX_4090(count=1)
sky.accelerators.TPU_V5P(count=8)
```

`nodes` is the number of compute nodes. It is independent from accelerator
`count`:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.H100(count=4),
    nodes=2,
) as compute:
    ...
```

Factories return a frozen `Accelerator(name, count)` value. `count` is an
integer. Memory and form-factor variants are represented by the canonical
catalog entries, not by `memory=`, `form_factor=`, or `Custom(...)` arguments.
For CPU-only compute, leave `accelerator=None`.

Factory attributes use Python identifiers for catalog names containing
hyphens: `RTX_4090`, `H100_NVL`, `TPU_V5P_8`, and `RTX_PRO_6000`.

## Catalog metadata

The canonical catalog carries the normalized name, VRAM, manufacturer,
architecture, and CUDA compatibility where those fields are known. Common
entries include:

| Factory | Canonical name | VRAM |
|---|---|---:|
| `A100()` | `a100` | 80 GB |
| `H100()` | `h100` | 80 GB |
| `H200()` | `h200` | 141 GB |
| `B200()` | `b200` | 192 GB |
| `B300()` | `b300` | 288 GB |
| `GH200()` | `gh200` | 96 GB |
| `L4()` | `l4` | 24 GB |
| `L40S()` | `l40s` | 48 GB |
| `T4()` | `t4` | 16 GB |
| `V100()` | `v100` | 32 GB |
| `RTX_4090()` | `rtx-4090` | 24 GB |
| `RTX_5090()` | `rtx-5090` | 32 GB |
| `MI300X()` | `mi300x` | 192 GB |
| `TPU_V5P()` | `tpu-v5p` | 95 GB |
| `GAUDI3()` | `gaudi3` | 128 GB |
| `TRAINIUM2()` | `trainium2` | 64 GB |
| `INFERENTIA2()` | `inferentia2` | 32 GB |

The table is representative. The factory module is generated from the same
catalog used by offer normalization; do not maintain a second list of provider
spellings in application code.

## Inspect live availability

The catalog describes names and hardware metadata. Availability and price come
from provider offers:

```bash
sky offers list --accelerator H100 --min-vram 80 --limit 10
sky offers list --accelerator RTX_4090 --refresh
sky offers summary --accelerator A100
```

Offer rows include the normalized accelerator, VRAM, CPU, memory, region,
provider, and the provider's billing unit. `--refresh` asks the daemon to
refresh stale data before answering. If a provider refresh fails, its stale rows
remain in the cache and the provider records the error.

## Related pages

- [Choosing a provider](choosing-a-provider.md)
- [Compare accelerators](compare.md)
- [Using accelerators](guides/using-accelerators.md)
- [Accelerator API](reference/accelerators.md)
