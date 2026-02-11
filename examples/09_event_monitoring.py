"""Event Monitoring Example.

Demonstrates Skyward's event system for monitoring:
- Infrastructure provisioning
- Bootstrap progress
- Runtime metrics (CPU, memory, GPU)
- Errors

Events enable real-time visibility into the execution lifecycle.
"""

import skyward as sky


@sky.compute
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


@sky.compute
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
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    return {
        "matrix_size": size,
        "device": torch.cuda.get_device_name(0),
    }


# =============================================================================
# Event Handlers
# =============================================================================


def on_infra_creating(event: sky.InfraCreating):
    """Called when infrastructure creation starts."""
    print("[INFRA] Creating infrastructure...")


def on_infra_created(event: sky.InfraCreated):
    """Called when infrastructure is ready."""
    print("[INFRA] Infrastructure created")


def on_instance_launching(event: sky.InstanceLaunching):
    """Called when an instance is being launched."""
    print("[INSTANCE] Launching instance...")


def on_instance_provisioned(event: sky.InstanceProvisioned):
    """Called when an instance is ready."""
    print(f"[INSTANCE] Instance provisioned: {event.instance_id}")


def on_bootstrap_starting(event: sky.BootstrapStarting):
    """Called when bootstrap phase begins."""
    print("[BOOTSTRAP] Starting bootstrap...")


def on_bootstrap_progress(event: sky.BootstrapProgress):
    """Called during bootstrap with progress updates."""
    print(f"[BOOTSTRAP] {event.message}")


def on_bootstrap_completed(event: sky.BootstrapCompleted):
    """Called when bootstrap is complete."""
    print("[BOOTSTRAP] Completed!")


def on_pool_started(event: sky.PoolStarted):
    """Called when the pool is ready for execution."""
    print(f"[POOL] Pool started with {event.nodes} nodes")


def on_metrics(event: sky.Metrics):
    """Called periodically with resource metrics."""
    gpu_info = ""
    if event.gpu_utilization is not None:
        gpu_info = f", GPU: {event.gpu_utilization}%"

    print(
        f"[METRICS] CPU: {event.cpu_percent}%, "
        f"Memory: {event.memory_percent}%{gpu_info}"
    )


def on_instance_stopping(event: sky.InstanceStopping):
    """Called when an instance is being terminated."""
    print(f"[INSTANCE] Stopping instance: {event.instance_id}")


def on_pool_stopping(event: sky.PoolStopping):
    """Called when the pool is shutting down."""
    print("[POOL] Stopping pool...")


def on_error(event: sky.Error):
    """Called when an error occurs."""
    print(f"[ERROR] {event.message}")


if __name__ == "__main__":
    # Create pool
    pool = sky.ComputePool(
        provider=sky.AWS(),
        accelerator=sky.NVIDIA.A100,
        pip=["torch"],
        allocation="spot-if-available",
    )

    # =================================================================
    # Register Event Handlers
    # =================================================================
    # Provision phase
    pool.on(sky.InfraCreating, on_infra_creating)
    pool.on(sky.InfraCreated, on_infra_created)
    pool.on(sky.InstanceLaunching, on_instance_launching)
    pool.on(sky.InstanceProvisioned, on_instance_provisioned)

    # Bootstrap phase
    pool.on(sky.BootstrapStarting, on_bootstrap_starting)
    pool.on(sky.BootstrapProgress, on_bootstrap_progress)
    pool.on(sky.BootstrapCompleted, on_bootstrap_completed)

    # Execution phase
    pool.on(sky.PoolStarted, on_pool_started)
    pool.on(sky.Metrics, on_metrics)

    # Shutdown phase
    pool.on(sky.InstanceStopping, on_instance_stopping)
    pool.on(sky.PoolStopping, on_pool_stopping)

    # Error handling
    pool.on(sky.Error, on_error)

    # =================================================================
    # Run computations with event monitoring
    # =================================================================
    print("\n" + "=" * 60)
    print("Starting pool with event monitoring...")
    print("=" * 60 + "\n")

    with pool:
        print("\n--- Running CPU computation ---")
        cpu_result = heavy_computation(5_000_000) >> pool
        print(
            f"\nResult: {cpu_result['elapsed_seconds']}s "
            f"for {cpu_result['iterations']:,} iterations"
        )

        print("\n--- Running GPU computation ---")
        gpu_result = gpu_computation(4096) >> pool
        if "error" not in gpu_result:
            print(
                f"\nResult: {gpu_result['matrix_size']}x"
                f"{gpu_result['matrix_size']} on {gpu_result['device']}"
            )

    print("\n" + "=" * 60)
    print("Pool shutdown complete")
    print("=" * 60)
