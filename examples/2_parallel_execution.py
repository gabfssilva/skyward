"""Parallel Execution Example.

Demonstrates how to execute multiple computations in parallel using:
- gather(): Group multiple computations for parallel execution
- & operator: Chain computations for parallel execution with type safety

All computations are executed on a single remote instance,
but processed concurrently for better throughput.
"""

from skyward import AWS, ComputePool, compute, gather


@compute
def process_chunk(data: list[int]) -> int:
    """Sum all numbers in a chunk."""
    return sum(data)


@compute
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@compute
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    with ComputePool(provider=AWS(), spot="always") as pool:
        # =================================================================
        # Using gather() for parallel execution
        # =================================================================
        # All process_chunk calls execute in parallel on the remote instance
        results = gather(*[process_chunk(c) for c in chunks]) >> pool
        print(f"Chunk sums: {results}")  # (6, 15, 24, 33)

        # =================================================================
        # Using & operator for type-safe parallel execution
        # =================================================================
        # The & operator chains computations and preserves types
        a, b = (multiply(2, 3) & multiply(4, 5)) >> pool
        print(f"Products: {a}, {b}")  # 6, 20

        # Chain up to 8 computations with full type inference
        f3, f4, f5 = (factorial(3) & factorial(4) & factorial(5)) >> pool
        print(f"Factorials: 3!={f3}, 4!={f4}, 5!={f5}")  # 6, 24, 120

        # =================================================================
        # Mixing different computations
        # =================================================================
        sum_result, product_result, fact_result = (
            process_chunk([1, 2, 3, 4, 5])
            & multiply(10, 20)
            & factorial(6)
        ) >> pool

        print(f"Mixed: sum={sum_result}, product={product_result}, factorial={fact_result}")
