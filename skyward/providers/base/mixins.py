"""Reusable mixins and utilities for providers.

Provides factory functions for common provider patterns like instance
polling, reducing code duplication across providers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)


class InstancePendingError(Exception):
    """Instance not yet in target state - retry."""


def create_instance_poller[T](
    fetch_status: Callable[[T], tuple[str, dict[str, Any]]],
    target_status: str,
    timeout: float = 300,
) -> Callable[[T], dict[str, Any]]:
    """Create a polling function for instance state transitions.

    This factory creates a retry-enabled function that polls an instance
    until it reaches the target status, returning updated instance info.

    Type Parameters:
        T: The instance type being polled (e.g., _Droplet, _VerdaInstance).

    Args:
        fetch_status: Function that fetches current status and info.
                     Returns (status, info_dict) where info_dict contains
                     updated fields like ip, private_ip, etc.
        target_status: Status string to wait for (e.g., "active", "running").
        timeout: Maximum wait time in seconds.

    Returns:
        A function that polls until target_status is reached.

    Example:
        def fetch_droplet_status(droplet: _Droplet) -> tuple[str, dict]:
            resp = client.droplets.get(droplet.id)
            data = resp["droplet"]
            return data["status"], {
                "ip": data["networks"]["v4"][0]["ip_address"],
                "private_ip": data["networks"]["v4"][1]["ip_address"],
            }

        poll = create_instance_poller(
            fetch_status=fetch_droplet_status,
            target_status="active",
            timeout=300,
        )

        info = poll(droplet)  # Blocks until active
        droplet.ip = info["ip"]
    """

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(InstancePendingError),
        reraise=True,
    )
    def poll(instance: T) -> dict[str, Any]:
        status, info = fetch_status(instance)
        if status != target_status:
            raise InstancePendingError(f"Instance status: {status}, waiting for: {target_status}")
        return info

    return poll


def poll_instances[T](
    instances: Sequence[T],
    fetch_status: Callable[[T], tuple[str, dict[str, Any]]],
    target_status: str,
    update_instance: Callable[[T, dict[str, Any]], None],
    timeout: float = 300,
) -> None:
    """Poll multiple instances concurrently until all reach target status.

    This is a higher-level helper that:
    1. Creates a poller using create_instance_poller
    2. Polls all instances concurrently using for_each_async
    3. Updates each instance with the returned info

    Args:
        instances: Sequence of instances to poll.
        fetch_status: Function to fetch current status and info.
        target_status: Status string to wait for.
        update_instance: Function to update instance with fetched info.
        timeout: Maximum wait time per instance.

    Example:
        def fetch_status(droplet: _Droplet) -> tuple[str, dict]:
            resp = client.droplets.get(droplet.id)
            data = resp["droplet"]
            return data["status"], {"ip": ..., "private_ip": ...}

        def update_droplet(droplet: _Droplet, info: dict) -> None:
            object.__setattr__(droplet, "ip", info["ip"])
            object.__setattr__(droplet, "private_ip", info["private_ip"])

        poll_instances(
            instances=droplets,
            fetch_status=fetch_status,
            target_status="active",
            update_instance=update_droplet,
        )
    """
    from skyward.utils.conc import for_each_async

    poll = create_instance_poller(
        fetch_status=fetch_status,
        target_status=target_status,
        timeout=timeout,
    )

    def poll_and_update(instance: T) -> None:
        try:
            info = poll(instance)
            update_instance(instance, info)
        except RetryError as e:
            raise RuntimeError(f"Timeout waiting for instance to reach {target_status}") from e

    for_each_async(poll_and_update, list(instances))
