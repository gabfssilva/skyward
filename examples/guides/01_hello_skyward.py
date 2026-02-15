"""Hello, Skyward! — run your first function on the cloud."""

import skyward as sky


@sky.compute
def add(a: int, b: int) -> int:
    """Add two numbers on a remote instance."""
    return a + b


@sky.compute
def process(data: list[int]) -> dict:
    """Process data remotely and return statistics."""
    return {
        "count": len(data),
        "sum": sum(data),
        "mean": sum(data) / len(data),
    }


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.AWS(),
    ) as pool:
        result = add(2, 3) >> pool
        print(f"2 + 3 = {result}")

        stats = process([1, 2, 3, 4, 5]) >> pool
        print(f"Stats: {stats}")
