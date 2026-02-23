"""S3 Volumes Example.

Demonstrates how to mount S3-compatible buckets as local filesystems using Volume.
This enables:
- Reading datasets directly from cloud storage
- Saving checkpoints/artifacts to persistent storage
- Sharing data between nodes via mounted volumes

Uses s3fs-fuse under the hood for FUSE-based mounting.
Works with AWS (native S3), GCP (S3-compatible via HMAC), and RunPod (S3 API).
"""

import time
from pathlib import Path

import skyward as sky


@sky.compute
def list_files(data_dir: str, pattern: str = "*") -> list[str]:
    """List files in the mounted S3 volume."""
    path = Path(data_dir)

    if not path.exists():
        return [f"Directory {data_dir} does not exist"]

    files = list(path.glob(pattern))
    return [f.name for f in sorted(files)[:20]]


@sky.compute
def save_checkpoint(checkpoint_dir: str, epoch: int, metrics: dict) -> dict:
    """Save a training checkpoint to S3 and verify it was written."""
    import json

    info = sky.instance_info()

    checkpoint = {
        "epoch": epoch,
        "node": info.node,
        "metrics": metrics,
    }

    path = Path(checkpoint_dir) / f"checkpoint_node{info.node}_epoch{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return {
        "node": info.node,
        "path": str(path),
        "size": path.stat().st_size,
    }


@sky.compute
def process_dataset(data_dir: str) -> dict:
    """Process files from S3 and return statistics."""
    path = Path(data_dir)
    info = sky.instance_info()

    if not path.exists():
        return {"node": info.node, "error": "Data directory not found"}

    files = list(path.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    return {
        "node": info.node,
        "file_count": file_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "sample_files": [f.name for f in files[:5] if f.is_file()],
    }


if __name__ == "__main__":
    # =================================================================
    # Configure S3 Volumes
    # =================================================================
    # Replace with your actual bucket names
    DATA_BUCKET = "skyward-example"
    DATA_PREFIX = "datasets/imagenet/"

    CHECKPOINT_BUCKET = "skyward-example"
    CHECKPOINT_PREFIX = "checkpoints/experiment-001/"

    with sky.ComputePool(
        provider=sky.AWS(instance_profile_arn="auto"),
        nodes=2,
        volumes=[
            sky.Volume(
                bucket=DATA_BUCKET,
                mount="/data",
                prefix=DATA_PREFIX,
                read_only=True,
            ),
            sky.Volume(
                bucket=CHECKPOINT_BUCKET,
                mount="/checkpoints",
                prefix=CHECKPOINT_PREFIX,
                read_only=False,
            ),
        ],
    ) as pool:
        # =================================================================
        # Process dataset across nodes
        # =================================================================
        print("Processing dataset from /data:")
        stats = process_dataset("/data") @ pool
        for s in stats:
            if "error" in s:
                print(f"  Node {s['node']}: {s['error']}")
            else:
                print(f"  Node {s['node']}: {s['file_count']} files, {s['total_size_mb']}MB")

        # =================================================================
        # Save checkpoints from each node
        # =================================================================
        print("\nSaving checkpoints:")
        metrics = {"loss": 0.5, "accuracy": 0.85}

        results = save_checkpoint("/checkpoints", epoch=1, metrics=metrics) @ pool
        for r in results:
            print(f"  Node {r['node']}: {r['path']} ({r['size']} bytes)")

        # s3fs syncs writes to S3 asynchronously — wait for propagation
        print("\nWaiting for S3 sync...")
        time.sleep(10)

        # Each node lists its own view of /checkpoints
        print("\nCheckpoints visible per node:")
        all_views = list_files("/checkpoints", "*.json") @ pool
        for i, files in enumerate(all_views):
            print(f"  Node {i}: {files}")
