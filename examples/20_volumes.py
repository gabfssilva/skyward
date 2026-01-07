"""
S3 Volume Mounting.

Mount S3 buckets as local filesystems using AWS Mountpoint.
Great for large datasets that don't fit in instance storage.
"""

import skyward as sky


@sky.compute
def list_data_files() -> list[str]:
    """List files in the mounted S3 bucket."""
    import os

    files = []
    for root, _, filenames in os.walk("/data"):
        for f in filenames:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            files.append(f"{path} ({size} bytes)")
    return files[:10]  # First 10 files


@sky.compute
def process_data_file(path: str) -> dict:
    """Process a file from the mounted volume."""
    with open(path, "rb") as f:
        data = f.read()
    return {
        "path": path,
        "size": len(data),
        "checksum": hash(data) & 0xFFFFFFFF,
    }


@sky.compute
def save_checkpoint(data: dict) -> str:
    """Save checkpoint to writable volume."""
    import json
    import os

    os.makedirs("/checkpoints/run1", exist_ok=True)
    path = "/checkpoints/run1/checkpoint.json"

    with open(path, "w") as f:
        json.dump(data, f)

    return f"Saved to {path}"


def main():
    # Read-only volume for input data
    data_volume = sky.S3Volume(
        mount_path="/data",
        bucket="my-ml-datasets",
        prefix="training/",
        read_only=True,
    )

    # Read-write volume for checkpoints
    checkpoint_volume = sky.S3Volume(
        mount_path="/checkpoints",
        bucket="my-ml-outputs",
        prefix="checkpoints/",
        read_only=False,
    )

    # You can also use dict syntax
    volumes_dict = {
        "/data": "s3://my-ml-datasets/training/",
        "/checkpoints": "s3://my-ml-outputs/checkpoints/",
    }

    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator="T4",
        volume=[data_volume, checkpoint_volume],
        # Or: volume=volumes_dict,
        image=sky.Image(pip=["numpy"]),
        allocation="spot-if-available",
    ) as pool:
        # List files in mounted volume
        files = list_data_files() >> pool
        print("Data files found:")
        for f in files:
            print(f"  {f}")

        # Process a file (if any exist)
        if files:
            # Extract path from listing
            path = files[0].split(" ")[0]
            result = process_data_file(path) >> pool
            print(f"\nProcessed: {result}")

        # Save checkpoint
        checkpoint_data = {"epoch": 10, "loss": 0.01, "accuracy": 0.99}
        save_path = save_checkpoint(checkpoint_data) >> pool
        print(f"\n{save_path}")


if __name__ == "__main__":
    main()
