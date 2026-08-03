# Choosing a provider

Provider selection in v2 is based on the daemon's current offer cache. A
provider kind identifies an adapter; a provider account is a named set of
credentials and non-secret configuration.

## Inspect the current catalog

List the provider kinds supported by the installed build:

```bash
sky providers list --kinds
```

This output includes the credential fields and offer-cache TTL for each kind.
List registered accounts with:

```bash
sky providers list
sky providers check
```

`check` reports the last recorded provider result. It does not authenticate
again. Credentials are used by the daemon and are not returned by these
commands.

## Supported provider kinds

The current adapters expose these account fields and cache intervals:

| Kind | Credential fields | Offer TTL |
|---|---|---:|
| `aws` | `access_key_id`, `secret_access_key` | 30 min |
| `container` | none | 24 h |
| `gcp` | `service_account_json` | 6 h |
| `hyperstack` | `api_key` | 1 h |
| `jarvislabs` | `api_key` | 15 min |
| `lambda` | `api_key` | 5 min |
| `massed_compute` | `api_key` | 10 min |
| `novita` | `api_key` | 15 min |
| `runpod` | `api_key` | 10 min |
| `scaleway` | `secret_key`, `project_id` | 10 min |
| `tensordock` | `api_token` | 5 min |
| `vastai` | `api_key` | 2 min |
| `verda` | `client_id`, `client_secret` | 15 min |
| `vultr` | `api_key` | 6 h |

Use the provider account in Python when you need provider configuration beyond
the CLI's basic `create` flags:

```python
import skyward as sky

aws = sky.AWS(name="production", region="us-east-1")
runpod = sky.RunPod(name="experiments", data_center_ids=("EU-RO-1",))

with sky.Compute(provider=aws, accelerator=sky.accelerators.H100()) as compute:
    ...
```

Factories read credentials from their arguments and the provider-specific
environment or credential files. See [Providers](providers.md) for the
configuration details of each adapter.

The CLI creates an account from the current process when it runs:

```bash
sky compute create --provider aws --accelerator A100 --name research
sky compute create --provider runpod --accelerator RTX_4090 --name experiments
```

The CLI `--name` flag names the compute, not the provider account. The CLI
accepts the provider kind, not an account object. Use the SDK when selecting a
named account or provider-specific options.

## Query offers

The daemon refreshes a provider when its cached offers are stale. A failed
refresh leaves the existing rows in place and records the provider error.

```bash
sky offers list --accelerator H100 --min-vram 80 --limit 10
sky offers list --provider production --max-price 3
sky offers list --refresh --output json
sky offers fetch --provider experiments
sky offers summary --accelerator A100
```

`list` accepts provider id/name, accelerator, minimum accelerator count,
minimum VRAM, maximum price, row limit, and `--refresh`. The price is displayed
with the billing unit returned by the offer. `summary` aggregates the cached
rows by normalized accelerator and provider.

Use the live catalog rather than hard-coded price tables. Availability and
prices vary by account, region, and cache age.

## One provider

The usual form supplies a provider and a hardware request directly:

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(region="us-east-1"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    selection="cheapest",
) as compute:
    ...
```

`Compute` also accepts `cpus`, `memory_gb`, `region`, `allocation`, `image`,
`executor`, `options`, `ports`, `volumes`, and `ttl`. The provider-specific
hardware translation is performed from the normalized offer catalog.

## Multiple providers

Pass one `Spec` per provider preference:

```python
with sky.Compute(
    sky.Spec(provider=sky.Verda(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.AWS(), accelerator=sky.accelerators.H100()),
    nodes=1,
    selection="cheapest",
) as compute:
    ...
```

`Spec` contains the provider, accelerator, CPU and memory constraints, region,
disk size, architecture, and an optional maximum hourly cost. Node count,
allocation, image, executor, volumes, and ports belong to `Compute`, not to
`Spec`.

## Choosing by workload

Use the provider capabilities returned by `sky providers list --kinds` and the
current offers to decide. The repository currently includes cloud adapters for
AWS, GCP, Hyperstack, JarvisLabs, Lambda, Massed Compute, Novita, RunPod,
Scaleway, TensorDock, Vast.ai, Verda, Vultr, and the local Container provider.

Cluster networking, monitor/live behavior, and provider changes that are still
uncommitted are intentionally not specified on this page.

## Related pages

- [Providers](providers.md)
- [Accelerators](accelerators.md)
- [Compare accelerators](compare.md)
- [CLI](cli.md)
