# Cloud providers

A provider has two identities in v2:

- a **kind**, such as `aws` or `runpod`, which selects an adapter;
- an **account**, which is a named set of credentials and non-secret configuration stored by the daemon.

The public factories return account descriptors. They read credentials in the client process; an explicitly passed credential takes precedence over its environment source.

```python
import skyward as sky

aws = sky.AWS(name="production", region="us-east-1")
vast = sky.VastAI(name="marketplace", geolocation="US")

with sky.Compute(
    sky.Spec(aws, accelerator="A100"),
    sky.Spec(vast, accelerator="A100", max_hourly_cost=2.0),
    selection="cheapest",
) as compute:
    train(data) >> compute
```

`name` is the account alias. It defaults to the provider kind, so use explicit names when more than one account of a kind is needed. Entering a `Compute` registers missing accounts with the daemon. Provider read operations return the account id, name, kind, configuration, offer-cache metadata, and the last error; they never return credentials.

## Supported provider kinds

The credentials column is the field required by the provider adapter. Environment variables are listed in the next section.

| Factory | Kind | Required credential fields | Catalog TTL |
|---|---|---|---:|
| `sky.AWS` | `aws` | `access_key_id`, `secret_access_key` | 30 min |
| `sky.GCP` | `gcp` | `service_account_json` | 6 h |
| `sky.Hyperstack` | `hyperstack` | `api_key` | 1 h |
| `sky.JarvisLabs` | `jarvislabs` | `api_key` | 15 min |
| `sky.Lambda` | `lambda` | `api_key` | 5 min |
| `sky.MassedCompute` | `massed_compute` | `api_key` | 10 min |
| `sky.Novita` | `novita` | `api_key` | 15 min |
| `sky.RunPod` | `runpod` | `api_key` | 10 min |
| `sky.Salad` | `salad` | `api_key` | 10 min |
| `sky.Scaleway` | `scaleway` | `secret_key`, `project_id` | 10 min |
| `sky.TensorDock` | `tensordock` | `api_token` | 5 min |
| `sky.VastAI` | `vastai` | `api_key` | 2 min |
| `sky.Verda` | `verda` | `client_id`, `client_secret` | 15 min |
| `sky.Vultr` | `vultr` | `api_key` | 6 h |
| `sky.Container` | `container` | none | 1 day |

Each factory's complete signature is available in the [provider reference pages](reference/providers/aws.md).

## Credential sources

| Provider | Environment or local source |
|---|---|
| AWS | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`; otherwise the selected profile in `~/.aws/credentials` |
| GCP | `GOOGLE_APPLICATION_CREDENTIALS` points to a service-account JSON file; `GOOGLE_CLOUD_PROJECT` supplies the project |
| Hyperstack | `HYPERSTACK_API_KEY` |
| JarvisLabs | `JL_API_KEY` |
| Lambda | `LAMBDA_API_KEY` |
| Massed Compute | `MASSED_API_KEY` |
| Novita | `NOVITA_API_KEY` |
| RunPod | `RUNPOD_API_KEY` |
| Salad | `SALAD_API_KEY`; `SALAD_ORGANIZATION` and `SALAD_PROJECT` are required configuration |
| Scaleway | `SCW_SECRET_KEY`, `SCW_DEFAULT_PROJECT_ID` |
| TensorDock | `TENSORDOCK_API_KEY`, `TENSORDOCK_API_TOKEN` |
| VastAI | `VAST_API_KEY` |
| Verda | `VERDA_CLIENT_ID`, `VERDA_CLIENT_SECRET` |
| Vultr | `VULTR_API_KEY` |
| Container | no credentials |

Pass credentials directly to the factory when the process should not read them from the environment. The daemon adapter receives resolved credentials through the provider account record; it does not read environment variables on its own.

## Cached offers

The daemon keeps one hardware catalog per registered account. Each adapter declares its own freshness interval because marketplace capacity changes at a different rate from fixed instance types.

The catalog is queried through `GET /v1/offers` or the CLI:

```bash
sky offers list --accelerator A100 --min-vram 40 --max-price 3
sky offers fetch
sky offers summary --accelerator H100
```

Queries filter the cached rows by provider, accelerator, accelerator count, VRAM, and price. An expired account is refreshed before its rows are returned. `refresh=true` or the CLI `--refresh` flag forces a refetch.

A failed refresh does not erase the previous rows. The stale rows remain available and the provider account records the last error. The compute market uses the same cache when matching `Spec` alternatives; it does not call every provider separately for each scheduling decision.

## Local containers

`Container` uses a local container runtime and needs no cloud account:

```python
with sky.Compute(
    provider=sky.Container(binary="docker"),
    image=sky.Image(pip=["numpy"]),
) as compute:
    train(data) >> compute
```

Use it for local development and CI. Its provider factory accepts the container image, SSH user, runtime binary, container-name prefix, network, and account name.

## Choosing a provider

Choose based on the catalog returned for the account rather than a fixed hardware list. Accelerator names and VRAM are normalized into the shared vocabulary, while provider-specific filters remain in each factory's configuration.

- Use AWS or GCP for account-managed infrastructure and broad regional catalogs.
- Use marketplace providers when price and rapidly changing capacity matter.
- Use RunPod, VastAI, TensorDock, or similar providers when their networking and capacity model fits the workload.
- Use Salad for standalone GPU containers. It does not provide the private network required by clustered computes.
- Use Container before provisioning a cloud account.

See [Compute and task dispatch](reference/pool.md) for `Spec` alternatives and [Accelerators](accelerators.md) for the shared accelerator vocabulary.

## Provider references

- [AWS](reference/providers/aws.md)
- [GCP](reference/providers/gcp.md)
- [Hyperstack](reference/providers/hyperstack.md)
- [JarvisLabs](reference/providers/jarvislabs.md)
- [Lambda](reference/providers/lambda.md)
- [Massed Compute](reference/providers/massed-compute.md)
- [Novita](reference/providers/novita.md)
- [RunPod](reference/providers/runpod.md)
- [Salad](reference/providers/salad.md)
- [Scaleway](reference/providers/scaleway.md)
- [TensorDock](reference/providers/tensordock.md)
- [VastAI](reference/providers/vastai.md)
- [Verda](reference/providers/verda.md)
- [Vultr](reference/providers/vultr.md)
- [Container](reference/providers/container.md)
