"""Hyperstack Object Storage Volumes.

Mount Hyperstack S3-compatible buckets as local filesystems on GPU instances.
Hyperstack Object Storage uses the endpoint ca1.obj.nexgencloud.io (CANADA-1 only).

Access keys are created/deleted automatically — no manual setup needed.
When no region is configured, Skyward automatically selects CANADA-1.

    ┌─────────────────────────────────────────────┐
    │  Hyperstack VM (CANADA-1)                   │
    │                                             │
    │  /data        → s3://my-dataset/train/  RO  │
    │  /checkpoints → s3://my-outputs/ckpt/   RW  │
    │                                             │
    │  Mounted via s3fs-fuse                      │
    └─────────────────────────────────────────────┘
"""

import time
from pathlib import Path

import skyward as sky


@sky.compute
def explore_data(data_dir: str) -> dict:
    """Explore the mounted dataset volume."""
    path = Path(data_dir)
    info = sky.instance_info()

    if not path.exists():
        return {"node": info.node, "error": f"{data_dir} not found"}

    files = [f for f in path.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    return {
        "node": info.node,
        "files": len(files),
        "total_mb": round(total_size / (1024 * 1024), 2),
        "sample": [f.name for f in files[:5]],
    }


@sky.compute
def save_checkpoint(checkpoint_dir: str, epoch: int, loss: float) -> dict:
    """Save a training checkpoint to the writable volume."""
    import json

    info = sky.instance_info()
    path = Path(checkpoint_dir) / f"node{info.node}_epoch{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {"epoch": epoch, "loss": loss, "node": info.node}
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return {"node": info.node, "path": str(path), "size": path.stat().st_size}


@sky.compute
def list_checkpoints(checkpoint_dir: str) -> list[str]:
    """List checkpoint files visible from this node."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    return sorted(f.name for f in path.glob("*.json"))


if __name__ == "__main__":
    # Replace with your Hyperstack Object Storage bucket names
    DATA_BUCKET = "my-dataset-bucket"
    CHECKPOINT_BUCKET = "my-checkpoint-bucket"

    with sky.ComputePool(
        provider=sky.Hyperstack(),
        accelerator=sky.accelerators.L4(),
        nodes=2,
        volumes=[
            sky.Volume(
                bucket=DATA_BUCKET,
                mount="/data",
                prefix="train/",
                read_only=True,
            ),
            sky.Volume(
                bucket=CHECKPOINT_BUCKET,
                mount="/checkpoints",
                prefix="experiment-001/",
                read_only=False,
            ),
        ],
    ) as pool:
        # Each node explores its view of the dataset
        print("Dataset stats per node:")
        for stats in explore_data("/data") @ pool:
            if "error" in stats:
                print(f"  Node {stats['node']}: {stats['error']}")
            else:
                print(f"  Node {stats['node']}: {stats['files']} files, {stats['total_mb']}MB")

        # Each node saves a checkpoint
        print("\nSaving checkpoints:")
        for result in save_checkpoint("/checkpoints", epoch=1, loss=0.42) @ pool:
            print(f"  Node {result['node']}: {result['path']} ({result['size']} bytes)")

        # Wait for s3fs async sync
        print("\nWaiting for S3 sync...")
        time.sleep(10)

        # Verify all checkpoints are visible
        print("\nCheckpoints visible per node:")
        for i, files in enumerate(list_checkpoints("/checkpoints") @ pool):
            print(f"  Node {i}: {files}")
