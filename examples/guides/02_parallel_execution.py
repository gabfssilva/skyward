"""Parallel Execution — run multiple computations concurrently."""

import time

import skyward as sky


@sky.compute
def process_chunk(data: list[int]) -> int:
    """Sum all numbers in a chunk."""
    return sum(data)


@sky.compute
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@sky.compute
def factorial(n: int) -> int:
    """Calculate factorial."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    with sky.ComputePool(provider=sky.AWS()) as pool:
        chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # gather() runs all calls in parallel
        results = sky.gather(*[process_chunk(c) for c in chunks]) >> pool
        print(f"Chunk sums: {results}")  # (6, 15, 24)

        # & operator for type-safe parallel execution
        a, b = (multiply(2, 3) & multiply(4, 5)) >> pool
        print(f"Products: {a}, {b}")  # 6, 20

        # Mix different computations
        s, p, f = (process_chunk([1, 2, 3]) & multiply(10, 20) & factorial(5)) >> pool
        print(f"Mixed: sum={s}, product={p}, factorial={f}")  # 6, 200, 120

        # gather(stream=True) yields results as they complete
        tasks = [process_chunk([i] * 1000) for i in range(5)]
        start = time.monotonic()
        for result in sky.gather(*tasks, stream=True) >> pool:
            elapsed = time.monotonic() - start
            print(f"  [{elapsed:.1f}s] Got: {result}")
