"""Helpers that translate `Volume` requests into `MountPlan` values.

Providers call one of these helpers inside ``Mountable.mount_plan``:

- :func:`fuse_mount_plan` — geesefs + S3 endpoint (AWS/GCP/Hyperstack).
- :func:`native_mount_plan` — host-attached volume + symlinks (RunPod NV).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from skyward.api.model import MountPlan
from skyward.providers.bootstrap.ops import mount_volumes, symlink_volumes

if TYPE_CHECKING:
    from skyward.api.spec import Volume
    from skyward.storage import Storage


def fuse_mount_plan(volumes: tuple[Volume, ...], storage: Storage) -> MountPlan:
    """Build a FUSE-based `MountPlan` — one `Storage` endpoint for all volumes.

    Parameters
    ----------
    volumes
        Volumes to mount via geesefs.
    storage
        S3-compatible endpoint + credentials used for every volume.

    Returns
    -------
    MountPlan
        Plan with empty ``deploy_hints`` and a geesefs install-and-mount
        bootstrap op.
    """
    pairs = tuple((v, storage) for v in volumes)
    return MountPlan(bootstrap=mount_volumes(pairs))


def native_mount_plan(
    volumes: tuple[Volume, ...],
    *,
    base: str,
    **hints: Any,
) -> MountPlan:
    """Build a native-attachment `MountPlan` — deploy hints + symlink bootstrap.

    Parameters
    ----------
    volumes
        Volumes to project into the pre-mounted base directory.
    base
        Absolute path where the host runtime mounts the native volume
        (e.g. ``"/workspace"`` for RunPod).
    **hints
        Opaque hints passed back to ``Provider.provision`` (e.g.
        ``networkVolumeId="vol-abc"``).

    Returns
    -------
    MountPlan
        Plan with the supplied hints in an immutable mapping and a
        symlink-only bootstrap op.
    """
    return MountPlan(
        deploy_hints=MappingProxyType(dict(hints)),
        bootstrap=symlink_volumes(volumes, base=base),
    )
