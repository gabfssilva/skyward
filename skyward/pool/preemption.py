"""Preemption detection and handling for spot/bid instances.

This module provides:
- preemption_check: Factory for creating check functions for the Monitor
- PreemptionHandler: Event consumer that handles preemption based on policy

The preemption system follows a detect-emit-react pattern:
1. Monitor calls preemption_check() periodically to detect preempted instances
2. preemption_check() returns InstancePreempted events
3. PreemptionHandler receives events and applies the configured policy
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, cast

from loguru import logger

from skyward.core.callback import emit
from skyward.core.events import (
    InstancePreempted,
    InstanceReplaced,
    ProviderName,
    ProvisionedInstance,
    SkywardEvent,
)
from skyward.core.exceptions import PreemptionError

if TYPE_CHECKING:
    from skyward.spec.preemption import Preemption
    from skyward.types import ComputeSpec, Instance, Provider


def _to_provisioned(inst: Instance, provider: Provider) -> ProvisionedInstance:
    """Convert Instance to ProvisionedInstance for events."""
    # Map provider name to enum
    provider_map = {
        "vastai": ProviderName.VastAI,
        "aws": ProviderName.AWS,
        "digitalocean": ProviderName.DigitalOcean,
        "verda": ProviderName.Verda,
    }
    provider_name = provider_map.get(provider.name, ProviderName.VastAI)

    return ProvisionedInstance(
        instance_id=inst.id,
        node=inst.node,
        provider=provider_name,
        spot=inst.spot,
        ip=inst.public_ip or inst.private_ip,
    )


def preemption_check(
    provider: Provider,
    get_instances: Callable[[], tuple[Instance, ...]],
) -> Callable[[], list[SkywardEvent]]:
    """Create a check function for preemption monitoring.

    The returned function is pure - it only detects preempted instances
    and returns events. It does not modify state or trigger side effects.

    Args:
        provider: Cloud provider with get_instance_status() and classify_preemption().
        get_instances: Function that returns current instances tuple.

    Returns:
        A check function suitable for use with Monitor.
    """

    def check() -> list[SkywardEvent]:
        events: list[SkywardEvent] = []

        for inst in get_instances():
            # Only monitor spot/bid instances
            if not inst.spot:
                continue

            status = provider.get_instance_status(inst.id)
            if not status:
                continue

            reason = provider.classify_preemption(status.status)
            if reason:
                events.append(
                    InstancePreempted(
                        instance=_to_provisioned(inst, provider),
                        reason=reason,
                    )
                )

        return events

    return check


class PreemptionHandler:
    """Event consumer that handles preemption based on configured policy.

    Registered as a callback to react to InstancePreempted events.
    Supports three policies:
    - replace: Automatically provision a replacement instance
    - fail: Raise PreemptionError immediately
    - ignore: Log and continue without the instance

    Example:
        handler = PreemptionHandler(
            config=Preemption(policy="replace"),
            provider=provider,
            get_instances=lambda: pool.instances,
            set_instances=pool._set_instances,
            compute_spec=compute,
        )
        register_consumer(handler)
    """

    def __init__(
        self,
        config: Preemption,
        provider: Provider,
        get_instances: Callable[[], tuple[Instance, ...]],
        set_instances: Callable[[tuple[Instance, ...]], None],
        compute_spec: ComputeSpec,
    ) -> None:
        self.config = config
        self.provider = provider
        self.get_instances = get_instances
        self.set_instances = set_instances
        self.compute_spec = compute_spec
        self._retry_counts: dict[str, int] = {}

    def __call__(self, event: SkywardEvent) -> None:
        """Handle incoming events."""
        match event:
            case InstancePreempted(instance=prov_inst, reason=reason):
                self._handle(prov_inst, reason)

    def _handle(self, prov_inst: ProvisionedInstance, reason: str) -> None:
        """Handle a preemption event based on policy."""
        instance_id = prov_inst.instance_id
        logger.warning(f"Instance {instance_id} preempted: {reason}")

        match self.config.policy:
            case "replace":
                self._replace(prov_inst, reason)
            case "fail":
                raise PreemptionError(instance_id, reason)
            case "ignore":
                logger.info(f"Ignoring preemption of {instance_id} (policy=ignore)")

    def _replace(self, old_prov: ProvisionedInstance, reason: str) -> None:
        """Attempt to replace a preempted instance."""
        instance_id = old_prov.instance_id
        retries = self._retry_counts.get(instance_id, 0)

        if retries >= self.config.max_retries:
            raise PreemptionError(instance_id, reason, attempts=retries)

        for attempt in range(retries, self.config.max_retries):
            self._retry_counts[instance_id] = attempt + 1

            try:
                logger.info(f"Replacing {instance_id} (attempt {attempt + 1}/{self.config.max_retries})")

                # Provision single replacement instance
                # Cast to Any since ComputeSpec is a Protocol but actual impl is a dataclass
                spec = replace(cast(Any, self.compute_spec), nodes=1)
                new_instances = self.provider.provision(spec)
                new_inst = new_instances[0]

                # Find old instance in current list
                old_inst = next(
                    (i for i in self.get_instances() if i.id == instance_id),
                    None,
                )
                if not old_inst:
                    logger.warning(f"Instance {instance_id} not found in pool, skipping replace")
                    return

                # Swap old instance with new one
                updated = tuple(
                    new_inst if i.id == instance_id else i
                    for i in self.get_instances()
                )
                self.set_instances(updated)

                # Destroy old instance
                old_inst.destroy()

                # Emit success event
                emit(
                    InstanceReplaced(
                        old_instance=old_prov,
                        new_instance=_to_provisioned(new_inst, self.provider),
                        retry_attempt=attempt,
                    )
                )

                # Clear retry count on success
                del self._retry_counts[instance_id]
                logger.info(f"Successfully replaced {instance_id} with {new_inst.id}")
                return

            except PreemptionError:
                raise
            except Exception as e:
                logger.warning(f"Replace attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay)

        raise PreemptionError(instance_id, reason, attempts=self.config.max_retries)


__all__ = ["preemption_check", "PreemptionHandler"]
