# Volumes

A `Volume` maps object storage or provider-native storage to a directory on each compute node. The compute and the stored data have independent lifecycles.

## `Volume`

Construct volumes with `bucket=`, never with a display name:

```python
import skyward as sky

data = sky.Volume(
    bucket="my-datasets",
    mount="/data",
    prefix="imagenet/train/",
    read_only=True,
)

checkpoints = sky.Volume(
    bucket="my-experiments",
    mount="/checkpoints",
    read_only=False,
)
```

Fields:

- `bucket`: bucket name, or a provider-native volume id/name where supported;
- `mount`: absolute path on every node;
- `prefix`: optional subdirectory within the bucket;
- `read_only`: defaults to `True`;
- `storage`: optional explicit `sky.Storage` endpoint and credentials.

The constructor rejects relative paths and protected system paths such as `/`, `/root`, `/tmp`, and `/opt`.

## Attach volumes to a compute

Pass volumes to `Compute`:

```python
import skyward as sky


@sky.function
def train(data_dir: str, checkpoint_dir: str) -> None:
    dataset = load(data_dir)
    model = fit(dataset)
    save(model, f"{checkpoint_dir}/model.pt")


with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    volumes=[
        sky.Volume(bucket="my-datasets", mount="/data"),
        sky.Volume(bucket="my-experiments", mount="/checkpoints", read_only=False),
    ],
) as compute:
    train("/data", "/checkpoints") >> compute
```

The function sees ordinary filesystem paths. The provider adapter decides how the mount is prepared before the worker starts.

If `storage` is omitted, the daemon asks the compute's provider to resolve the storage endpoint. Use `storage=` when the bucket belongs to a different storage account or endpoint.

## S3-compatible storage

`Storage` is a context manager for local CRUD operations and can also be passed to a `Volume`:

```python
import skyward as sky

r2 = sky.storage.R2(
    account_id="account-id",
    access_key="access-key",
    secret_key="secret-key",
)

with sky.Compute(
    provider=sky.AWS(),
    volumes=[sky.Volume(bucket="training-data", mount="/data", storage=r2)],
) as compute:
    ...
```

Available presets and their signatures are:

```python
sky.storage.S3(
    region="us-east-1",
    access_key=None,
    secret_key=None,
)
sky.storage.GCS(access_key="...", secret_key="...")
sky.storage.R2(account_id="...", access_key="...", secret_key="...")
sky.storage.Wasabi(region="...", access_key="...", secret_key="...")
sky.storage.Backblaze(region="...", key_id="...", app_key="...")
sky.storage.Hyperstack(
    access_key="...",
    secret_key="...",
    endpoint="https://ca1.obj.nexgencloud.io",
)
```

Or construct an endpoint directly:

```python
storage = sky.Storage(
    endpoint="https://s3.example.com",
    access_key="...",
    secret_key="...",
    path_style=False,
)
```

Open a `Storage` before using its local CRUD methods:

```python
with storage:
    storage.upload("my-bucket", "./data.csv", key="data.csv")
    storage.download("my-bucket", "data.csv", "./copy.csv")
    keys = storage.ls("my-bucket", prefix="runs/")
    exists = storage.exists("my-bucket", "data.csv")
    storage.rm("my-bucket", "old/data.csv")
```

`upload`, `download`, `ls`, `exists`, and `rm` are synchronous methods on the context-managed object.

Credentials may be strings, synchronous callables, or asynchronous callables.

## Provider strategies

These providers have volume mount adapters:

| Provider | Current strategy |
|---|---|
| AWS | S3-compatible mount using the machine identity |
| GCP | S3-compatible mount with a per-compute HMAC key |
| Hyperstack | S3-compatible mount with a per-compute object-storage key |
| RunPod | One provider network-volume attachment, projected through `prefix` |

Providers without a mount adapter report a capability mismatch when a compute requests volumes. The provider adapter owns the bootstrap details; the public API remains `Volume(bucket=..., mount=...)`.

RunPod accepts one network volume per pod. Multiple `Volume` objects can share that `bucket` and use different prefixes:

```python
with sky.Compute(
    provider=sky.RunPod(),
    volumes=[
        sky.Volume(bucket="checkpoints", mount="/train", prefix="train"),
        sky.Volume(bucket="checkpoints", mount="/eval", prefix="eval"),
    ],
) as compute:
    ...
```

## Shared buckets

Multiple volumes can reference the same bucket. The bootstrap deduplicates the underlying FUSE mount and creates one link per `mount`/`prefix` pair. If any volume for that bucket is writable, the shared mount is writable.

```python
volumes = [
    sky.Volume(bucket="datasets", mount="/train", prefix="train"),
    sky.Volume(bucket="datasets", mount="/validation", prefix="validation"),
]
```

There is no volume configuration file. Declare volumes in `Compute`.

## Next steps

- **[S3 volumes](guides/s3-volumes.md)** — a walkthrough with runnable code
- **[Providers](providers.md)** — which providers mount buckets and which attach volumes
- **[CLI](cli.md)** — inspecting a compute's mounts from the terminal
