# S3 volumes

This guide walks through mounting S3 buckets as local filesystems on remote workers. You'll mount a read-only volume for input data and a writable volume for saving results — both accessible as ordinary directories inside your `@sky.compute` functions.

## Declaring volumes

A `Volume` maps an S3 bucket (or a prefix within it) to a local path on every worker:

```python
--8<-- "examples/guides/17_s3_volumes.py:57:70"
```

The first volume mounts the `imagenet/` prefix of `my-datasets` at `/data` — read-only, so your training code can read from it but can't accidentally modify the dataset. The second mounts `run-042/` of `my-experiments` at `/checkpoints` — writable, so workers can save artifacts that persist to S3 after the pool is torn down.

## Reading from a volume

Inside a `@sky.compute` function, the mount path is a regular directory. Standard filesystem operations — `Path.iterdir()`, `open()`, `glob()` — work as expected:

```python
--8<-- "examples/guides/17_s3_volumes.py:8:14"
```

Because the volume is backed by s3fs-fuse, reads are transparently fetched from S3. There's no download step, no SDK calls, no boto3 — just file paths.

## Writing to a volume

Writable volumes (`read_only=False`) let workers save files that persist to S3:

```python
--8<-- "examples/guides/17_s3_volumes.py:17:31"
```

Each node writes its own checkpoint file. Because the volume is backed by S3, these files survive instance termination — they're in the bucket, not on ephemeral local disk. Note that s3fs syncs writes asynchronously, so freshly written files may take a moment to appear on other nodes or in the S3 console.

## Broadcasting across nodes

Volumes are mounted on every node, so broadcast operations naturally see the same storage:

```python
--8<-- "examples/guides/17_s3_volumes.py:34:50"
```

When dispatched with `@` (broadcast), each node independently counts files in `/data`. They all see the same S3 data, so the counts should match — but each reports from its own perspective via `instance_info()`.

## Putting it together

```python
--8<-- "examples/guides/17_s3_volumes.py:53:84"
```

The pool provisions two nodes, mounts both volumes on each, and then the three operations — count, checkpoint, verify — run against familiar filesystem paths. When the `with` block exits, the instances are destroyed, but the checkpoints remain in S3.

## IAM authentication (AWS)

On AWS, the cleanest approach is an instance profile with S3 permissions. Pass `instance_profile_arn="auto"` and Skyward uses IAM role authentication — no credentials are written to disk, no access keys in environment variables:

```python
sky.AWS(instance_profile_arn="auto")
```

The instance profile must have `s3:GetObject` and `s3:ListBucket` permissions for read-only volumes, plus `s3:PutObject` and `s3:DeleteObject` for writable ones.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/17_s3_volumes.py
```

---

**What you learned:**

- **`sky.Volume`** maps an S3 bucket to a local path on every worker — read or read-write.
- **`prefix`** scopes a volume to a subfolder within the bucket, keeping mount points clean.
- **s3fs-fuse** handles the mounting transparently — no SDK, no download step, just file paths.
- **Writable volumes** persist data to S3, surviving instance termination and pool teardown.
- **IAM roles** (`instance_profile_arn="auto"`) are the cleanest auth strategy on AWS — no credentials on disk.
