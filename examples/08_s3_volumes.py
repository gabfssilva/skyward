"""S3 Volumes Example.

Demonstrates how to mount S3 buckets as local filesystems using S3Volume.
This enables:
- Reading datasets directly from S3
- Saving checkpoints/artifacts to S3
- Sharing data between nodes via S3

Uses AWS Mountpoint for Amazon S3 under the hood.
"""

from pathlib import Path

import skyward as sky


@sky.compute
def list_files(data_dir: str, pattern: str = "*") -> list[str]:
    """List files in the mounted S3 volume."""
    path = Path(data_dir)

    if not path.exists():
        return [f"Directory {data_dir} does not exist"]

    files = list(path.glob(pattern))
    return [f.name for f in sorted(files)[:20]]  # Limit to first 20


@sky.compute
def read_file_sample(data_dir: str, filename: str, lines: int = 5) -> dict:
    """Read first N lines of a file from S3."""
    path = Path(data_dir) / filename

    if not path.exists():
        return {"error": f"File {filename} not found"}

    with open(path) as f:
        content = [f.readline().strip() for _ in range(lines)]

    return {
        "filename": filename,
        "size_bytes": path.stat().st_size,
        "first_lines": content,
    }


@sky.compute
def save_checkpoint(checkpoint_dir: str, epoch: int, metrics: dict) -> str:
    """Save a training checkpoint to S3."""
    import json

    pool = sky.instance_info()

    # Create checkpoint data
    checkpoint = {
        "epoch": epoch,
        "node": pool.node,
        "metrics": metrics,
    }

    # Save to the mounted S3 volume
    path = Path(checkpoint_dir) / f"checkpoint_node{pool.node}_epoch{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    return str(path)


@sky.compute
def process_dataset(data_dir: str) -> dict:
    """Process files from S3 and return statistics."""
    import os

    path = Path(data_dir)
    pool = sky.instance_info()

    if not path.exists():
        return {"node": pool.node, "error": "Data directory not found"}

    # Count files and total size
    files = list(path.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    return {
        "node": pool.node,
        "file_count": file_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "sample_files": [f.name for f in files[:5] if f.is_file()],
    }


if __name__ == "__main__":
    # =================================================================
    # Configure S3 Volumes
    # =================================================================
    # Replace with your actual bucket names
    DATA_BUCKET = "my-ml-bucket"
    DATA_PREFIX = "datasets/imagenet/"

    CHECKPOINT_BUCKET = "my-ml-bucket"
    CHECKPOINT_PREFIX = "checkpoints/experiment-001/"

    with sky.ComputePool(
        provider=sky.AWS(),
        nodes=2,
        accelerator=sky.NVIDIA.A100,
        allocation="spot-if-available",
        volumes=[
            # Read-only volume for input data
            sky.S3Volume(
                mount_path="/data",
                bucket=DATA_BUCKET,
                prefix=DATA_PREFIX,
                read_only=True,
            ),
            # Read-write volume for checkpoints
            sky.S3Volume(
                mount_path="/checkpoints",
                bucket=CHECKPOINT_BUCKET,
                prefix=CHECKPOINT_PREFIX,
                read_only=False,
            ),
        ],
    ) as pool:
        # =================================================================
        # List files from S3
        # =================================================================
        print("Files in /data:")
        files = list_files("/data", "*.jpg") >> pool
        for f in files[:10]:
            print(f"  {f}")

        # =================================================================
        # Process dataset across nodes
        # =================================================================
        print("\nProcessing dataset:")
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

        # Simulate training metrics
        metrics = {"loss": 0.5, "accuracy": 0.85}

        checkpoint_paths = save_checkpoint("/checkpoints", epoch=1, metrics=metrics) @ pool
        for path in checkpoint_paths:
            print(f"  Saved: {path}")

        # Verify checkpoints were saved
        print("\nCheckpoints in /checkpoints:")
        checkpoints = list_files("/checkpoints", "*.json") >> pool
        for f in checkpoints:
            print(f"  {f}")
