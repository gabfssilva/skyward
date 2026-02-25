"""S3 Volumes — mount cloud storage as local filesystems on remote workers."""

from pathlib import Path

import skyward as sky


@sky.compute
def list_files(directory: str) -> list[str]:
    """List files in a mounted volume."""
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(f.name for f in path.iterdir() if f.is_file())[:20]


@sky.compute
def save_checkpoint(checkpoint_dir: str, epoch: int, loss: float) -> str:
    """Save a training checkpoint to the writable volume."""
    import json

    info = sky.instance_info()

    checkpoint = {"epoch": epoch, "node": info.node, "loss": loss}
    path = Path(checkpoint_dir) / f"node{info.node}_epoch{epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(checkpoint, f)

    return f"Node {info.node}: saved {path.name} ({path.stat().st_size} bytes)"


@sky.compute
def count_files(directory: str) -> dict:
    """Count files and total size in a mounted volume."""
    info = sky.instance_info()
    path = Path(directory)

    if not path.exists():
        return {"node": info.node, "files": 0, "size_mb": 0}

    files = [f for f in path.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    return {
        "node": info.node,
        "files": len(files),
        "size_mb": round(total_size / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(instance_profile_arn="auto"),
        nodes=2,
        volumes=[
            sky.Volume(
                bucket="my-datasets",
                mount="/data",
                prefix="imagenet/",
                read_only=True,
            ),
            sky.Volume(
                bucket="my-experiments",
                mount="/checkpoints",
                prefix="run-042/",
                read_only=False,
            ),
        ],
    ) as pool:
        # Each node sees the same /data — read-only from S3
        stats = count_files("/data") @ pool
        for s in stats:
            print(f"  Node {s['node']}: {s['files']} files, {s['size_mb']} MB")

        # Each node writes to /checkpoints — writable to S3
        results = save_checkpoint("/checkpoints", epoch=1, loss=0.42) @ pool
        for r in results:
            print(f"  {r}")

        # Verify checkpoints are visible
        files = list_files("/checkpoints") >> pool
        print(f"  Checkpoints: {files}")
