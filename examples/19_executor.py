"""
Executor - Drop-in replacement for concurrent.futures.

sky.Executor provides the familiar concurrent.futures.Executor interface
while running tasks on provisioned cloud instances. Compatible with
submit(), map(), and as_completed().
"""

import concurrent.futures
import time

import skyward as sky


def process_item(item: int) -> dict:
    """Process a single item (runs on cloud)."""
    time.sleep(0.5)  # Simulate work
    return {"item": item, "result": item ** 2}


def main():
    items = list(range(20))

    # Executor provisions cloud instances on __enter__
    with sky.Executor(
        provider=sky.AWS(),
        nodes=4,
        worker=sky.Worker(concurrency=4),  # 4 nodes x 4 concurrent = 16 parallel slots
        cpu=2,
        memory="4GB",
        allocation="spot-if-available",
    ) as executor:
        print(f"Executor ready with {executor.total_slots} slots")

        # Example 1: Submit individual tasks
        print("\n--- submit() example ---")
        future = executor.submit(process_item, 42)
        result = future.result()
        print(f"Single result: {result}")

        # Example 2: Map over items (preserves order)
        print("\n--- map() example ---")
        results = list(executor.map(process_item, items[:5]))
        print(f"Map results: {results}")

        # Example 3: as_completed (process results as they finish)
        print("\n--- as_completed() example ---")
        futures = [executor.submit(process_item, x) for x in items]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            if completed <= 5:  # Print first 5
                print(f"Completed: {result}")
            elif completed == 6:
                print("...")

        print(f"Total completed: {completed}")

        # Example 4: Exception handling
        print("\n--- error handling example ---")

        def might_fail(x: int) -> int:
            if x == 7:
                raise ValueError("Unlucky number!")
            return x * 2

        futures = [executor.submit(might_fail, x) for x in range(10)]
        for i, future in enumerate(futures):
            try:
                print(f"Item {i}: {future.result()}")
            except ValueError as e:
                print(f"Item {i}: Error - {e}")


if __name__ == "__main__":
    main()
