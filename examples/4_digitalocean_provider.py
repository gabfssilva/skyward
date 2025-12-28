"""DigitalOcean Provider Example.

Demonstrates how to use DigitalOcean as an alternative cloud provider.
DigitalOcean Droplets are simpler and often more cost-effective for
CPU-based workloads.

Note: DigitalOcean does not support GPUs, use AWS for GPU workloads.
"""

from skyward import ComputePool, DigitalOcean, compute


@compute
def system_info() -> dict:
    """Get system information from the Droplet."""
    import os
    import platform

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


@compute
def memory_info() -> dict:
    """Get memory information."""
    import psutil

    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "percent_used": mem.percent,
    }


@compute
def cpu_intensive_task(iterations: int) -> dict:
    """Run a CPU-intensive computation."""
    import time

    start = time.time()
    result = 0
    for i in range(iterations):
        result += i**2 % 1000000007
    elapsed = time.time() - start

    return {
        "iterations": iterations,
        "result": result,
        "elapsed_seconds": round(elapsed, 3),
    }


if __name__ == "__main__":
    # =================================================================
    # DigitalOcean Droplet with 4 CPUs and 8GB RAM
    # =================================================================
    with ComputePool(
        provider=DigitalOcean(region="nyc1"),
        cpu=4,
        memory=8,  # GB
        spot="always",  # Use spot pricing when available
    ) as pool:
        # Get system info
        sys_info = system_info() >> pool
        print("System Info:")
        for key, value in sys_info.items():
            print(f"  {key}: {value}")

        # Get memory info
        mem_info = memory_info() >> pool
        print("\nMemory Info:")
        for key, value in mem_info.items():
            print(f"  {key}: {value}")

        # Run CPU-intensive task
        result = cpu_intensive_task(10_000_000) >> pool
        print(f"\nCPU Task: {result['iterations']:,} iterations in {result['elapsed_seconds']}s")

    # =================================================================
    # Available DigitalOcean regions
    # =================================================================
    # nyc1, nyc3  - New York
    # sfo2, sfo3  - San Francisco
    # ams3        - Amsterdam
    # sgp1        - Singapore
    # lon1        - London
    # fra1        - Frankfurt
    # tor1        - Toronto
    # blr1        - Bangalore
