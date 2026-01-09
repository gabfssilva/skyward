"""Cost tracking callback for Skyward events."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Literal

from skyward.core.events import (
    BootstrapCompleted,
    CostFinal,
    CostUpdate,
    InstanceProvisioned,
    InstanceStopping,
    MetricValue,
    PoolStopping,
    SkywardEvent,
)
from skyward.utils.pricing import get_instance_pricing

if TYPE_CHECKING:
    from skyward.core.callback import Callback


@dataclass
class _InstanceCost:
    """Tracks cost for a single instance."""

    instance_id: str
    instance_type: str
    is_spot: bool
    hourly_rate: float
    on_demand_rate: float  # Stored for savings calculation
    billing_increment_minutes: int | None = None  # None = per-second billing
    start_time: float | None = None
    end_time: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.monotonic()
        return end - self.start_time

    @property
    def billable_hours(self) -> float:
        """Calculate billable hours, respecting billing increment."""
        elapsed_minutes = self.elapsed_seconds / 60
        if self.billing_increment_minutes is not None:
            # Round up to next billing increment
            increment = self.billing_increment_minutes
            billable_minutes = ceil(elapsed_minutes / increment) * increment
        else:
            # Per-second billing
            billable_minutes = elapsed_minutes
        return billable_minutes / 60

    @property
    def cost(self) -> float:
        return self.billable_hours * self.hourly_rate


@dataclass
class _CostState:
    """Internal state for cost tracking."""

    region: str
    provider: Literal["aws", "azure", "gcp"]
    instances: dict[str, _InstanceCost] = field(default_factory=dict)

    def register(
        self,
        instance_id: str,
        is_spot: bool,
        instance_type: str,
        price_on_demand: float | None = None,
        price_spot: float | None = None,
        billing_increment_minutes: int | None = None,
    ) -> None:
        """Register an instance and determine its pricing.

        Prefers pricing from InstanceSpec if available, falls back to Vantage API.
        """
        hourly_rate: float = 0.0
        on_demand_rate: float = 0.0

        # Prefer pricing from InstanceSpec if available
        if price_on_demand is not None:
            on_demand_rate = price_on_demand
        if price_spot is not None and is_spot:
            hourly_rate = price_spot
        elif price_on_demand is not None:
            hourly_rate = price_on_demand

        # Fallback to Vantage API lookup if not provided
        if hourly_rate == 0.0 or on_demand_rate == 0.0:
            pricing = get_instance_pricing(instance_type, self.provider, self.region)
            if pricing is not None:
                if hourly_rate == 0.0:
                    if is_spot and pricing.spot_avg is not None:
                        hourly_rate = pricing.spot_avg
                    else:
                        hourly_rate = pricing.ondemand or 0.0
                if on_demand_rate == 0.0:
                    on_demand_rate = pricing.ondemand or 0.0

        self.instances[instance_id] = _InstanceCost(
            instance_id=instance_id,
            instance_type=instance_type,
            is_spot=is_spot,
            hourly_rate=hourly_rate,
            on_demand_rate=on_demand_rate,
            billing_increment_minutes=billing_increment_minutes,
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

                # Calculate what on-demand would have cost (using stored rate and billable hours)
                ondemand_cost += inst.billable_hours * inst.on_demand_rate

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
                case InstanceProvisioned(instance=inst) if inst.spec:
                    state.register(
                        inst.instance_id,
                        inst.spot,
                        inst.spec.name,
                        inst.spec.price_on_demand,
                        inst.spec.price_spot,
                        inst.spec.billing_increment_minutes,
                    )
                    return None

                case BootstrapCompleted(instance=inst):
                    state.start_billing(inst.instance_id)
                    return None

                case InstanceStopping(instance=inst):
                    state.stop_billing(inst.instance_id)
                    return None

                case MetricValue():
                    return state.calculate_update()

                case PoolStopping():
                    return state.finalize()

                case _:
                    return None
        # Lock released here - safe for dispatcher to process derived events

    return track
