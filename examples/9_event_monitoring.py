"""Event Monitoring Example.

Demonstrates Skyward's event system for monitoring:
- Infrastructure provisioning
- Bootstrap progress
- Runtime metrics (CPU, memory, GPU)
- Errors

Events enable real-time visibility into the execution lifecycle.
"""

from skyward import (
    AWS,
    NVIDIA,
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    ComputePool,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    Metrics,
    PoolStarted,
    PoolStopping,
    compute,
)


@compute
def heavy_computation(iterations: int) -> dict:
    """CPU-intensive computation for metrics demonstration."""
    import time

    start = time.time()
    result = 0

    for i in range(iterations):
        result += i**2 % 1000000007

        # Add some variation to make metrics interesting
        if i % (iterations // 10) == 0:
            time.sleep(0.1)

    return {
        "iterations": iterations,
        "result": result,
        "elapsed_seconds": round(time.time() - start, 2),
    }


@compute
def gpu_computation(size: int) -> dict:
    """GPU computation to generate GPU metrics."""
    import torch

    if not torch.cuda.is_available():
        return {"error": "No GPU available"}

    device = "cuda"

    # Allocate memory and do work
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for _ in range(10):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    return {
        "matrix_size": size,
        "device": torch.cuda.get_device_name(0),
    }


# =============================================================================
# Event Handlers
# =============================================================================


def on_infra_creating(event: InfraCreating):
    """Called when infrastructure creation starts."""
    print(f"[INFRA] Creating infrastructure...")


def on_infra_created(event: InfraCreated):
    """Called when infrastructure is ready."""
    print(f"[INFRA] Infrastructure created")


def on_instance_launching(event: InstanceLaunching):
    """Called when an instance is being launched."""
    print(f"[INSTANCE] Launching instance...")


def on_instance_provisioned(event: InstanceProvisioned):
    """Called when an instance is ready."""
    print(f"[INSTANCE] Instance provisioned: {event.instance_id}")


def on_bootstrap_starting(event: BootstrapStarting):
    """Called when bootstrap phase begins."""
    print(f"[BOOTSTRAP] Starting bootstrap...")


def on_bootstrap_progress(event: BootstrapProgress):
    """Called during bootstrap with progress updates."""
    print(f"[BOOTSTRAP] {event.message}")


def on_bootstrap_completed(event: BootstrapCompleted):
    """Called when bootstrap is complete."""
    print(f"[BOOTSTRAP] Completed!")


def on_pool_started(event: PoolStarted):
    """Called when the pool is ready for execution."""
    print(f"[POOL] Pool started with {event.nodes} nodes")


def on_metrics(event: Metrics):
    """Called periodically with resource metrics."""
    gpu_info = ""
    if event.gpu_utilization is not None:
        gpu_info = f", GPU: {event.gpu_utilization}%"

    print(
        f"[METRICS] CPU: {event.cpu_percent}%, "
        f"Memory: {event.memory_percent}%{gpu_info}"
    )


def on_instance_stopping(event: InstanceStopping):
    """Called when an instance is being terminated."""
    print(f"[INSTANCE] Stopping instance: {event.instance_id}")


def on_pool_stopping(event: PoolStopping):
    """Called when the pool is shutting down."""
    print(f"[POOL] Stopping pool...")


def on_error(event: Error):
    """Called when an error occurs."""
    print(f"[ERROR] {event.message}")


if __name__ == "__main__":
    # Create pool
    pool = ComputePool(
        provider=AWS(),
        accelerator=NVIDIA.A100,
        pip=["torch"],
        spot="always",
    )

    # =================================================================
    # Register Event Handlers
    # =================================================================
    # Provision phase
    pool.on(InfraCreating, on_infra_creating)
    pool.on(InfraCreated, on_infra_created)
    pool.on(InstanceLaunching, on_instance_launching)
    pool.on(InstanceProvisioned, on_instance_provisioned)

    # Bootstrap phase
    pool.on(BootstrapStarting, on_bootstrap_starting)
    pool.on(BootstrapProgress, on_bootstrap_progress)
    pool.on(BootstrapCompleted, on_bootstrap_completed)

    # Execution phase
    pool.on(PoolStarted, on_pool_started)
    pool.on(Metrics, on_metrics)

    # Shutdown phase
    pool.on(InstanceStopping, on_instance_stopping)
    pool.on(PoolStopping, on_pool_stopping)

    # Error handling
    pool.on(Error, on_error)

    # =================================================================
    # Run computations with event monitoring
    # =================================================================
    print("\n" + "=" * 60)
    print("Starting pool with event monitoring...")
    print("=" * 60 + "\n")

    with pool:
        print("\n--- Running CPU computation ---")
        cpu_result = heavy_computation(5_000_000) >> pool
        print(f"\nResult: {cpu_result['elapsed_seconds']}s for {cpu_result['iterations']:,} iterations")

        print("\n--- Running GPU computation ---")
        gpu_result = gpu_computation(4096) >> pool
        if "error" not in gpu_result:
            print(f"\nResult: {gpu_result['matrix_size']}x{gpu_result['matrix_size']} on {gpu_result['device']}")

    print("\n" + "=" * 60)
    print("Pool shutdown complete")
    print("=" * 60)
