"""Cost tracking for skyward fleet operations."""

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
)
from skyward.pricing import get_instance_pricing

if TYPE_CHECKING:
    from skyward.pool import ComputePool


@dataclass
class _InstanceCost:
    """Tracks cost for a single instance."""

    instance_id: str
    instance_type: str
    is_spot: bool
    hourly_rate: float
    start_time: float | None = None  # None until billing starts
    end_time: float | None = None  # None until billing stops

    @property
    def billing_active(self) -> bool:
        return self.start_time is not None and self.end_time is None

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
class CostConsumer:
    """Tracks fleet costs via EventBus handlers.

    Uses @pool.on to receive events and @pool.respond_with to emit
    CostUpdate events when Metrics are received.

    Events consumed:
        - InstanceProvisioned: Register instance and fetch pricing
        - BootstrapCompleted: Start billing for instance
        - Metrics: Emit CostUpdate via @respond_with
        - InstanceStopping: Stop billing for instance
        - PoolStopping: Emit CostFinal via @respond_with

    Events emitted (via @respond_with):
        - CostUpdate: On Metrics events (if instances registered)
        - CostFinal: On PoolStopping event
    """

    pool: ComputePool
    region: str = "us-east-1"
    provider: Literal["aws", "azure", "gcp"] = "aws"

    _instances: dict[str, _InstanceCost] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Register event handlers after dataclass initialization."""
        pool = self.pool

        @pool.on(InstanceProvisioned)
        def on_provisioned(event: InstanceProvisioned) -> None:
            if event.instance_type:
                with self._lock:
                    self._register_instance(
                        event.instance_id, event.spot, event.instance_type
                    )

        @pool.on(BootstrapCompleted)
        def on_bootstrap_completed(event: BootstrapCompleted) -> None:
            with self._lock:
                self._start_billing(event.instance_id)

        @pool.on(InstanceStopping)
        def on_stopping(event: InstanceStopping) -> None:
            with self._lock:
                self._stop_billing(event.instance_id)

        @pool.on(Metrics)
        @pool.respond_with
        def on_metrics(event: Metrics) -> CostUpdate | None:
            with self._lock:
                if self._instances:
                    elapsed, cost, hourly, spot, ondemand = self._calculate_totals()
                    return CostUpdate(
                        elapsed_seconds=elapsed,
                        accumulated_cost=cost,
                        hourly_rate=hourly,
                        spot_count=spot,
                        ondemand_count=ondemand,
                    )
            return None

        @pool.on(PoolStopping)
        @pool.respond_with
        def on_stop(event: PoolStopping) -> CostFinal | None:
            return self._finalize()

    def _finalize(self) -> CostFinal | None:
        """Finalize and return CostFinal event."""
        with self._lock:
            if not self._instances:
                return None

            # Stop all billing first
            now = time.monotonic()
            for inst in self._instances.values():
                if inst.end_time is None and inst.start_time is not None:
                    inst.end_time = now

            elapsed, cost, hourly, spot, ondemand = self._calculate_totals()
            savings = self._calculate_savings()

        return CostFinal(
            total_cost=cost,
            total_seconds=elapsed,
            hourly_rate=hourly,
            spot_count=spot,
            ondemand_count=ondemand,
            savings_vs_ondemand=savings,
        )

    def _register_instance(
        self, instance_id: str, is_spot: bool, instance_type: str
    ) -> None:
        """Register an instance and fetch its pricing."""
        pricing = get_instance_pricing(instance_type, self.provider, self.region)

        if pricing is None:
            # Fallback: can't get pricing, use 0
            hourly_rate = 0.0
        elif is_spot and pricing.spot_avg is not None:
            hourly_rate = pricing.spot_avg
        else:
            hourly_rate = pricing.ondemand or 0.0

        self._instances[instance_id] = _InstanceCost(
            instance_id=instance_id,
            instance_type=instance_type,
            is_spot=is_spot,
            hourly_rate=hourly_rate,
        )

    def _start_billing(self, instance_id: str) -> None:
        """Start billing for an instance (called when bootstrap completes)."""
        if instance_id in self._instances:
            self._instances[instance_id].start_time = time.monotonic()

    def _stop_billing(self, instance_id: str) -> None:
        """Stop billing for an instance."""
        if instance_id in self._instances:
            self._instances[instance_id].end_time = time.monotonic()

    def _calculate_totals(
        self,
    ) -> tuple[float, float, float, int, int]:
        """Calculate current totals.

        Returns:
            (elapsed_seconds, accumulated_cost, hourly_rate, spot_count, ondemand_count)
        """
        total_elapsed = 0.0
        total_cost = 0.0
        total_hourly = 0.0
        spot_count = 0
        ondemand_count = 0

        for inst in self._instances.values():
            if inst.start_time is not None:
                total_elapsed = max(total_elapsed, inst.elapsed_seconds)
                total_cost += inst.cost
                total_hourly += inst.hourly_rate

                if inst.is_spot:
                    spot_count += 1
                else:
                    ondemand_count += 1

        return total_elapsed, total_cost, total_hourly, spot_count, ondemand_count

    def _calculate_savings(self) -> float:
        """Calculate savings vs all on-demand."""
        actual_cost = 0.0
        ondemand_cost = 0.0

        for inst in self._instances.values():
            if inst.start_time is not None:
                actual_cost += inst.cost
                # Get on-demand rate for comparison
                pricing = get_instance_pricing(
                    inst.instance_type, self.provider, self.region
                )
                od_rate = pricing.ondemand if pricing else inst.hourly_rate
                ondemand_cost += (inst.elapsed_seconds / 3600) * (od_rate or 0)

        return ondemand_cost - actual_cost
