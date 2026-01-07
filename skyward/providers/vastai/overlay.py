"""Overlay network management for Vast.ai.

Vast.ai overlay networks create virtual LANs on top of physical clusters,
enabling direct communication between instances on all ports. This is
required for NCCL and distributed training frameworks.

The VastAI SDK doesn't expose overlay network methods, so we use the CLI
through subprocess:
- vastai create overlay CLUSTER_ID OVERLAY_NAME
- vastai join overlay OVERLAY_NAME INSTANCE_ID
- vastai show overlays
- vastai delete overlay OVERLAY_NAME
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

if TYPE_CHECKING:
    from skyward.types import Instance


class OverlayError(Exception):
    """Error during overlay network operations."""


class _OverlayNotFoundError(Exception):
    """Internal: overlay not yet visible in API."""


@dataclass(frozen=True, slots=True)
class OverlayNetwork:
    """Represents a VastAI overlay network."""

    id: int
    name: str
    cluster_id: int
    instance_ids: frozenset[int]


def _check_vastai_cli() -> None:
    """Verify VastAI CLI is available."""
    if shutil.which("vastai") is None:
        raise OverlayError(
            "VastAI CLI not found. Install with: pip install vastai\n"
            "The CLI is required for overlay network functionality."
        )


def _run_vastai_cli(
    args: list[str],
    api_key: str | None = None,
    timeout: int = 60,
) -> dict[str, Any] | list[Any]:
    """Execute vastai CLI command and return JSON result.

    Args:
        args: CLI arguments (e.g., ["create", "overlay", "123", "my-overlay"]).
        api_key: VastAI API key (uses CLI default if None).
        timeout: Command timeout in seconds.

    Returns:
        Parsed JSON response from CLI.

    Raises:
        OverlayError: If CLI fails or returns non-JSON output.
    """
    _check_vastai_cli()

    cmd = ["vastai", "--raw"]
    if api_key:
        cmd.extend(["--api-key", api_key])
    cmd.extend(args)

    logger.debug(f"Running VastAI CLI: vastai --raw {' '.join(args[:3])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise OverlayError(f"VastAI CLI timed out after {timeout}s") from e

    if result.returncode != 0:
        raise OverlayError(f"VastAI CLI failed: {result.stderr.strip() or result.stdout.strip()}")

    stdout = result.stdout.strip()
    if not stdout:
        # Some commands return empty on success
        return {"success": True}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # Some commands return plain text on success
        return {"success": True, "message": stdout}


def _fetch_overlay_id_with_retry(
    overlay_name: str,
    api_key: str | None,
    timeout: int,
) -> int | None:
    """Fetch overlay ID by name with retries for API propagation delay."""

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), reraise=True)
    def _fetch() -> int:
        overlays = show_overlays(api_key=api_key, timeout=timeout)
        for overlay in overlays:
            if overlay.name == overlay_name:
                logger.debug(f"Found overlay '{overlay_name}' with ID {overlay.id}")
                return overlay.id
        raise _OverlayNotFoundError(f"Overlay '{overlay_name}' not visible yet")

    try:
        return _fetch()
    except _OverlayNotFoundError:
        return None


def create_overlay(
    cluster_id: int,
    overlay_name: str,
    api_key: str | None = None,
    timeout: int = 60,
) -> int:
    """Create an overlay network on a physical cluster.

    Args:
        cluster_id: Physical cluster ID from offer's cluster_id field.
        overlay_name: Name for the overlay network (must be unique).
        api_key: VastAI API key (uses CLI default if None).
        timeout: Command timeout in seconds.

    Returns:
        Overlay network ID.

    Raises:
        OverlayError: If creation fails.
    """
    logger.info(f"Creating overlay network '{overlay_name}' on cluster {cluster_id}")

    result = _run_vastai_cli(
        ["create", "overlay", str(cluster_id), overlay_name],
        api_key=api_key,
        timeout=timeout,
    )

    if isinstance(result, dict):
        if overlay_id := result.get("id") or result.get("overlay_id"):
            logger.debug(f"Created overlay network: {overlay_name} (ID: {overlay_id})")
            return int(overlay_id)
        if result.get("success"):
            # Success but no ID - fetch with retries (API propagation delay)
            overlay_id = _fetch_overlay_id_with_retry(overlay_name, api_key, timeout)
            if overlay_id is not None:
                return overlay_id

            # Created but can't retrieve ID - return 0 as placeholder
            logger.warning(
                f"Overlay '{overlay_name}' created but ID not retrievable. Proceeding with ID=0."
            )
            return 0

    raise OverlayError(f"Failed to create overlay network: {result}")


def join_overlay(
    overlay_name: str,
    instance_id: int,
    api_key: str | None = None,
    timeout: int = 60,
) -> None:
    """Join an instance to an overlay network.

    Args:
        overlay_name: Overlay network name.
        instance_id: VastAI instance ID.
        api_key: VastAI API key.
        timeout: Command timeout in seconds.

    Raises:
        OverlayError: If join fails.
    """
    logger.debug(f"Joining instance {instance_id} to overlay '{overlay_name}'")

    _run_vastai_cli(
        ["join", "overlay", overlay_name, str(instance_id)],
        api_key=api_key,
        timeout=timeout,
    )

    logger.debug(f"Instance {instance_id} joined overlay '{overlay_name}'")


def show_overlays(
    api_key: str | None = None,
    timeout: int = 30,
) -> list[OverlayNetwork]:
    """List all overlay networks for this account.

    Args:
        api_key: VastAI API key.
        timeout: Command timeout in seconds.

    Returns:
        List of OverlayNetwork objects.
    """
    result = _run_vastai_cli(
        ["show", "overlays"],
        api_key=api_key,
        timeout=timeout,
    )

    overlays: list[OverlayNetwork] = []

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                overlays.append(
                    OverlayNetwork(
                        id=item.get("overlay_id") or item.get("id", 0),
                        name=item.get("name", ""),
                        cluster_id=item.get("cluster_id", 0),
                        instance_ids=frozenset(item.get("instances", [])),
                    )
                )

    return overlays


def delete_overlay(
    overlay_name: str,
    api_key: str | None = None,
    timeout: int = 60,
) -> None:
    """Delete an overlay network.

    Args:
        overlay_name: Overlay network name.
        api_key: VastAI API key.
        timeout: Command timeout in seconds.

    Note:
        This function logs warnings but doesn't raise on failure,
        as the overlay may have already been deleted.
    """
    logger.info(f"Deleting overlay network: {overlay_name}")

    try:
        _run_vastai_cli(
            ["delete", "overlay", overlay_name],
            api_key=api_key,
            timeout=timeout,
        )
        logger.debug(f"Deleted overlay network: {overlay_name}")
    except OverlayError as e:
        logger.warning(f"Failed to delete overlay '{overlay_name}': {e}")


def get_instance_overlay_info(
    instance: Instance,
    timeout: int = 10,
) -> tuple[str, str] | None:
    """Get the overlay network IP and interface from an instance.

    Discovers the overlay network dynamically by finding which interface
    has a route to 10.x.x.x (VastAI overlay network range).

    Args:
        instance: Instance to query.
        timeout: SSH command timeout in seconds.

    Returns:
        Tuple of (ip_address, interface_name) or None if not found.
    """
    try:
        # Find interface with 10.x.x.x route and get overlay IP in one command
        # /proc/net/route uses little-endian hex, so 10.x.x.x has "0A" at position 7-8
        cmd = """
IFACE=$(awk 'NR>1 && substr($2,7,2)=="0A" {print $1; exit}' /proc/net/route)
IP=$(hostname -I | tr ' ' '\\n' | grep '^10\\.')
echo "$IFACE $IP"
"""
        output = instance.run_command(cmd.strip(), timeout=timeout)
        parts = output.strip().split()

        if len(parts) >= 2:
            iface, ip = parts[0], parts[1]
            logger.debug(f"Got overlay: {ip} on {iface}")
            return ip, iface

    except Exception as e:
        logger.debug(f"Failed to get overlay info: {e}")

    return None


def get_instance_overlay_ip(
    instance: Instance,
    interface: str = "eth0",
    timeout: int = 10,
) -> str | None:
    """Get the overlay network IP from an instance.

    Args:
        instance: Instance to query.
        interface: Deprecated, interface is discovered dynamically.
        timeout: SSH command timeout in seconds.

    Returns:
        IP address string or None if not found.
    """
    result = get_instance_overlay_info(instance, timeout)
    return result[0] if result else None
