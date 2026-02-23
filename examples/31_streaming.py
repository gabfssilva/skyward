"""Streaming with @sky.compute.

Demonstrates three streaming patterns:

1. Output streaming — generator function yields results back incrementally
2. Input streaming — Iterator[T]-annotated param streams data to the worker
3. Bidirectional — combines both: stream data in, yield results back

Uses the Container provider so no cloud credentials are needed.

Requirements:
    - A container runtime running locally (Docker, podman, etc.)
"""

from collections.abc import Iterator

import skyward as sky


@sky.compute
def fibonacci(n: int):
    """Output streaming: yields Fibonacci numbers one at a time."""
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b

        if i > 1000:
            break


@sky.compute
def running_mean(data: Iterator[float]) -> list[float]:
    """Input streaming: consumes an iterator of floats, returns running means."""
    total = 0.0
    means = []
    for i, x in enumerate(data, 1):
        total += x
        means.append(total / i)
    return means


@sky.compute
def moving_average(data: Iterator[float], window: int = 3):
    """Bidirectional: streams data in, yields moving averages out."""
    from collections import deque

    buf: deque[float] = deque(maxlen=window)
    for x in data:
        buf.append(x)
        yield sum(buf) / len(buf)


if __name__ == "__main__":
    with sky.ComputePool(
        provider=sky.Container(),
        nodes=1,
        vcpus=1,
        memory_gb=1,
        logging=True
    ) as pool:
        #warm up
        # for val in fibonacci(5) >> pool:
        #     pass
        #
        # print("warm up done")
        # print("Fibonacci (first 10):")

        for t in range(100):
            for val in fibonacci(100) >> pool:
                print(f"#{t}: {val}")

        #
        # print("done! ;)")
        #
        # # --- Input streaming ---
        # print("\nRunning mean of 1..5:")
        # data = iter([1.0, 2.0, 3.0, 4.0, 5.0])
        # means = running_mean(data) >> pool
        # for i, m in enumerate(means, 1):
        #     print(f"  after {i} values: {m:.2f}")
        #
        # # --- Bidirectional ---
        # print("\nMoving average (window=3) of 1..6:")
        # data = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # for avg in moving_average(data, window=3) >> pool:
        #     print(f"  {avg:.2f}")
