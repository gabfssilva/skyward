"""Cost tracking callback for Skyward events."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from skyward.events import (
    BootstrapCompleted,
    CostFinal,
    CostUpdate,
    InstanceProvisioned,
    InstanceStopping,
    Metrics,
    PoolStopping,
    SkywardEvent,
)
from skyward.pricing import get_instance_pricing

if TYPE_CHECKING:
    from skyward.callback import Callback


@dataclass
class _InstanceCost:
    """Tracks cost for a single instance."""

    instance_id: str
    instance_type: str
    is_spot: bool
    hourly_rate: float
    start_time: float | None = None
    end_time: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.monotonic()
        return end - self.start_time

    @property
    def cost(self) -> float:
        return (self.elapsed_seconds / 3600) * self.hourly_rate


@dataclass
class _CostState:
    """Internal state for cost tracking."""

    region: str
    provider: Literal["aws", "azure", "gcp"]
    instances: dict[str, _InstanceCost] = field(default_factory=dict)

    def register(self, instance_id: str, is_spot: bool, instance_type: str) -> None:
        """Register an instance and fetch its pricing."""
        pricing = get_instance_pricing(instance_type, self.provider, self.region)

        if pricing is None:
            hourly_rate = 0.0
        elif is_spot and pricing.spot_avg is not None:
            hourly_rate = pricing.spot_avg
        else:
            hourly_rate = pricing.ondemand or 0.0

        self.instances[instance_id] = _InstanceCost(
            instance_id=instance_id,
            instance_type=instance_type,
            is_spot=is_spot,
            hourly_rate=hourly_rate,
        )

    def start_billing(self, instance_id: str) -> None:
        """Start billing for an instance."""
        if instance_id in self.instances:
            self.instances[instance_id].start_time = time.monotonic()

    def stop_billing(self, instance_id: str) -> None:
        """Stop billing for an instance."""
        if instance_id in self.instances:
            self.instances[instance_id].end_time = time.monotonic()

    def calculate_update(self) -> CostUpdate | None:
        """Calculate current cost update."""
        if not self.instances:
            return None

        total_elapsed = 0.0
        total_cost = 0.0
        total_hourly = 0.0
        spot_count = 0
        ondemand_count = 0

        for inst in self.instances.values():
            if inst.start_time is not None:
                total_elapsed = max(total_elapsed, inst.elapsed_seconds)
                total_cost += inst.cost
                total_hourly += inst.hourly_rate

                if inst.is_spot:
                    spot_count += 1
                else:
                    ondemand_count += 1

        if total_elapsed == 0:
            return None

        return CostUpdate(
            elapsed_seconds=total_elapsed,
            accumulated_cost=total_cost,
            hourly_rate=total_hourly,
            spot_count=spot_count,
            ondemand_count=ondemand_count,
        )

    def finalize(self) -> CostFinal | None:
        """Finalize and return cost summary."""
        if not self.instances:
            return None

        # Stop all billing
        now = time.monotonic()
        for inst in self.instances.values():
            if inst.end_time is None and inst.start_time is not None:
                inst.end_time = now

        # Calculate totals
        total_elapsed = 0.0
        total_cost = 0.0
        total_hourly = 0.0
        spot_count = 0
        ondemand_count = 0
        actual_cost = 0.0
        ondemand_cost = 0.0

        for inst in self.instances.values():
            if inst.start_time is not None:
                total_elapsed = max(total_elapsed, inst.elapsed_seconds)
                total_cost += inst.cost
                total_hourly += inst.hourly_rate
                actual_cost += inst.cost

                # Calculate what on-demand would have cost
                pricing = get_instance_pricing(
                    inst.instance_type, self.provider, self.region
                )
                od_rate = pricing.ondemand if pricing else inst.hourly_rate
                ondemand_cost += (inst.elapsed_seconds / 3600) * (od_rate or 0)

                if inst.is_spot:
                    spot_count += 1
                else:
                    ondemand_count += 1

        savings = ondemand_cost - actual_cost

        return CostFinal(
            total_cost=total_cost,
            total_seconds=total_elapsed,
            hourly_rate=total_hourly,
            spot_count=spot_count,
            ondemand_count=ondemand_count,
            savings_vs_ondemand=savings,
        )


def cost_tracker(
    region: str = "us-east-1",
    provider: Literal["aws", "azure", "gcp"] = "aws",
) -> Callback:
    """Create a stateful callback that tracks fleet costs.

    Returns derived events (CostUpdate on Metrics, CostFinal on PoolStopping)
    which are dispatched after this callback returns, ensuring thread safety.

    Args:
        region: Cloud region for pricing lookup.
        provider: Cloud provider name.

    Returns:
        A callback that tracks costs and emits cost events.

    Example:
        callback = compose(cost_tracker(region="us-west-2"), log)
        with use_callback(callback):
            emit(Metrics(...))  # Triggers CostUpdate emission
    """
    state = _CostState(region=region, provider=provider)
    lock = threading.Lock()

    def track(event: SkywardEvent) -> CostUpdate | CostFinal | None:
        # All state access is protected by lock
        # Lock is released BEFORE dispatcher processes returned events
        with lock:
            match event:
                case InstanceProvisioned(
                    instance_id=iid, spot=spot, instance_type=itype
                ) if itype:
                    state.register(iid, spot, itype)
                    return None

                case BootstrapCompleted(instance_id=iid):
                    state.start_billing(iid)
                    return None

                case InstanceStopping(instance_id=iid):
                    state.stop_billing(iid)
                    return None

                case Metrics():
                    return state.calculate_update()

                case PoolStopping():
                    return state.finalize()

                case _:
                    return None
        # Lock released here - safe for dispatcher to process derived events

    return track
