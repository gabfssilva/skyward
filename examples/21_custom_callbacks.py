"""
Custom Callbacks and Event Monitoring.

Skyward emits events throughout the pool lifecycle. You can subscribe
to these events with custom callbacks for logging, monitoring, cost
tracking, and more.
"""

import skyward as sky
from skyward.callback import compose, emit, use_callback
from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    CostFinal,
    CostUpdate,
    FunctionCall,
    FunctionResult,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    Metrics,
    PoolReady,
    ProvisioningCompleted,
    ProvisioningStarted,
    SkywardEvent,
)


def logging_callback(event: SkywardEvent) -> None:
    """Log all events with pattern matching."""
    match event:
        case ProvisioningStarted():
            print("[LOG] Provisioning started...")

        case InstanceLaunching(count=n):
            print(f"[LOG] Launching {n} instance(s)")

        case InstanceProvisioned(instance=inst):
            print(f"[LOG] Instance {inst.instance_id} provisioned at {inst.ip}")

        case BootstrapStarting(instance=inst):
            print(f"[LOG] Bootstrap starting on {inst.instance_id}")

        case BootstrapProgress(percentage=pct, message=msg):
            print(f"[LOG] Bootstrap: {pct}% - {msg}")

        case BootstrapCompleted(instance=inst):
            print(f"[LOG] Bootstrap completed on {inst.instance_id}")

        case PoolReady():
            print("[LOG] Pool ready!")

        case FunctionCall(function_name=name):
            print(f"[LOG] Calling: {name}")

        case FunctionResult(function_name=name, duration_ms=ms):
            print(f"[LOG] {name} completed in {ms:.1f}ms")

        case InstanceStopping(instance=inst):
            print(f"[LOG] Stopping {inst.instance_id}")


def cost_tracker(event: SkywardEvent) -> None:
    """Track and display cost information."""
    match event:
        case CostUpdate(total_cost=cost, hourly_rate=rate):
            print(f"[COST] Running: ${cost:.3f} (${rate:.2f}/hr)")

        case CostFinal(total_cost=cost, duration_seconds=secs):
            mins = secs / 60
            print(f"[COST] Final: ${cost:.3f} ({mins:.1f} minutes)")


def metrics_monitor(event: SkywardEvent) -> None:
    """Monitor GPU utilization and memory."""
    match event:
        case Metrics(gpu_utilization=gpu, gpu_memory_used=mem) if gpu is not None:
            if gpu > 80:
                print(f"[METRICS] GPU high: {gpu}% (mem: {mem}GB)")
        case _:
            pass  # Ignore other events


@sky.compute
def train_step(batch_id: int) -> dict:
    """Simulated training step."""
    import time

    time.sleep(0.1)
    return {"batch": batch_id, "loss": 1.0 / (batch_id + 1)}


def main():
    # Compose multiple callbacks into one
    combined = compose(
        logging_callback,
        cost_tracker,
        metrics_monitor,
    )

    # Use callback as context manager
    with use_callback(combined):
        with sky.ComputePool(
            provider=sky.AWS(),
            accelerator="T4",
            image=sky.Image(pip=["numpy"]),
            allocation="spot-if-available",
        ) as pool:
            # Run some training steps
            for i in range(5):
                result = train_step(i) >> pool
                print(f"Batch {i}: loss={result['loss']:.4f}")


def main_with_pool_callback():
    """Alternative: pass callback directly to pool."""

    def simple_callback(event: SkywardEvent) -> None:
        match event:
            case PoolReady():
                print("Pool is ready!")
            case CostFinal(total_cost=cost):
                print(f"Total cost: ${cost:.3f}")
            case _:
                pass

    # Pass callback via on_event parameter
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator="T4",
        allocation="spot-if-available",
        on_event=simple_callback,
    ) as pool:
        result = train_step(0) >> pool
        print(f"Result: {result}")


if __name__ == "__main__":
    print("=== Using use_callback context manager ===\n")
    main()

    print("\n=== Using on_event parameter ===\n")
    main_with_pool_callback()
