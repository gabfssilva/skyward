"""Volume abstraction for mounting remote storage as local filesystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Volume(Protocol):
    """Protocol for mountable volumes.

    Volumes provide a way to mount remote storage as local filesystem.
    Each implementation handles its own mount tool installation and mounting.

    Example:
        class MyVolume:
            @property
            def mount_path(self) -> str:
                return "/data"

            def install_commands(self) -> list[str]:
                return ["apt-get install -y my-mount-tool"]

            def mount_commands(self) -> list[str]:
                return ["my-mount-tool /data"]
    """

    @property
    def mount_path(self) -> str:
        """Local path where volume will be mounted (e.g., '/data')."""
        ...

    def install_commands(self) -> list[str]:
        """Shell commands to install mount tools.

        Returns:
            List of shell commands to execute.
        """
        ...

    def mount_commands(self) -> list[str]:
        """Shell commands to mount the volume.

        Returns:
            List of shell commands to execute.
        """
        ...


@dataclass(frozen=True, slots=True)
class S3Volume:
    """Volume backed by S3 via AWS Mountpoint.

    Uses Mountpoint for Amazon S3 - AWS's official high-throughput
    file client optimized for ML workloads.

    Args:
        mount_path: Local path where volume will be mounted (e.g., '/data').
        bucket: S3 bucket name (e.g., 'my-bucket').
        prefix: Optional prefix/path within bucket (e.g., 'datasets/').
        read_only: If True, mount in read-only mode.

    Example:
        S3Volume("/data", "my-bucket", "datasets/")
        S3Volume("/checkpoints", "my-bucket", "checkpoints/", read_only=False)
    """

    mount_path: str
    bucket: str
    prefix: str = ""
    read_only: bool = False

    def install_commands(self) -> list[str]:
        """Install Mountpoint for Amazon S3."""
        # Use DPkg::Lock::Timeout to wait for apt lock (up to 10 min)
        # This is the official apt solution for lock contention with unattended-upgrades
        # See: https://blog.sinjakli.co.uk/2021/10/25/waiting-for-apt-locks-without-the-hacky-bash-scripts/
        apt_opts = "-o DPkg::Lock::Timeout=600"
        return [
            # Try Amazon Linux repo first, then download for Ubuntu/Debian
            f"dnf install -y mount-s3 2>/dev/null || {{ "
            f"apt-get {apt_opts} update -qq && "
            f"apt-get {apt_opts} install -y -qq libfuse2 && "
            f"wget -q https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb && "
            f"dpkg -i mount-s3.deb; "
            f"}}"
        ]

    def mount_commands(self) -> list[str]:
        """Mount S3 bucket/prefix to local path using fstab (recommended by AWS)."""
        # Build S3 URI with optional prefix
        if self.prefix:
            # mount-s3 requires prefix to end with /
            prefix = self.prefix if self.prefix.endswith("/") else f"{self.prefix}/"
            s3_uri = f"s3://{self.bucket}/{prefix}"
        else:
            s3_uri = f"s3://{self.bucket}/"

        # Mount options (see CONFIGURATION.md in mountpoint-s3 repo)
        # Required: _netdev, nosuid, nodev
        # Recommended: nofail (allow boot if mount fails)
        # Permissions: allow-other (FUSE access), uid/gid (file ownership)
        # Write support: allow-delete, allow-overwrite
        # rw/ro: read-write or read-only
        rw_mode = "ro" if self.read_only else "rw,allow-delete,allow-overwrite"
        # Use placeholder UID/GID - will be replaced by _generate_volume_script
        mount_opts = f"_netdev,nosuid,nodev,nofail,allow-other,{rw_mode},uid=UID_PLACEHOLDER,gid=GID_PLACEHOLDER"

        return [
            f"mkdir -p {self.mount_path}",
            f'echo "{s3_uri} {self.mount_path} mount-s3 {mount_opts}" >> /etc/fstab',
            "systemctl daemon-reload",
            "mount -a",
        ]


def parse_volume_uri(mount_path: str, uri: str) -> Volume:
    """Parse a volume URI string into a Volume object.

    Supported URI formats:
        - s3://bucket/prefix -> S3Volume

    Args:
        mount_path: Local path to mount the volume.
        uri: Remote storage URI.

    Returns:
        Volume instance.

    Raises:
        ValueError: If URI format is not supported.

    Example:
        >>> parse_volume_uri("/data", "s3://my-bucket/datasets")
        S3Volume(mount_path='/data', bucket='my-bucket', prefix='datasets', read_only=False)
    """
    if uri.startswith("s3://"):
        path = uri[5:]  # Remove "s3://"
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3Volume(mount_path=mount_path, bucket=bucket, prefix=prefix)

    raise ValueError(
        f"Unsupported volume URI scheme: {uri}. "
        f"Supported schemes: s3://"
    )
