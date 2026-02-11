from time import perf_counter, sleep

from joblib import Parallel, delayed

import skyward as sky


def slow_task(x):
    print(f"Task {x} starting")
    sleep(5)
    print(f"Task {x} done")
    return x * 2


if __name__ == '__main__':
    with sky.integrations.JoblibPool(
        provider=sky.AWS(),
        nodes=10,
        image=sky.Image(pip=["joblib"], skyward_source='local'),
        concurrency=10,
    ) as pool:
        # just to warm up
        for _ in range(10):
            sky.compute(lambda: f"pong from {sky.instance_info().node}")() @ pool

        t_ping = perf_counter()
        pongs = sky.compute(lambda: f"pong from {sky.instance_info().node}")() @ pool
        ping_elapsed = perf_counter() - t_ping
        print(f"\nBroadcast ping: {pongs} in {ping_elapsed * 1000:.0f}ms")

        t0 = perf_counter()
        results = Parallel(n_jobs=-1)(
            delayed(slow_task)(i) for i in range(2000)
        )
        elapsed = perf_counter() - t0

        print(f"\n{'=' * 50}")
        print("Tasks: 200 | Nodes: 5 | Concurrency: 10")
        print("Effective workers: 50")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {200 / elapsed:.2f} tasks/s")
        print(f"Ideal time (200 tasks / 50 workers * 5s): {200 / 50 * 5:.0f}s")
        print(f"Efficiency: {(200 / 50 * 5) / elapsed * 100:.1f}%")
        print(f"{'=' * 50}")
        print(results)
