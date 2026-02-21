from time import perf_counter, sleep

from joblib import Parallel, delayed

import skyward as sky


def slow_task(x):
    sleep(5)
    return x * 2


if __name__ == '__main__':
    nodes = 10
    concurrency = 10
    tasks = 2000

    effective_workers = nodes * concurrency

    with sky.integrations.JoblibPool(
        provider=sky.AWS(),
        nodes=nodes,
        image=sky.Image(pip=["joblib"], skyward_source='local'),
        worker=sky.Worker(concurrency=concurrency),
        max_inflight=tasks
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
            delayed(slow_task)(i) for i in range(tasks)
        )

        elapsed = perf_counter() - t0

        print(f"\n{'=' * 50}")
        print(f"Tasks: {tasks} | Nodes: {nodes} | Concurrency: {concurrency}")
        print(f"Effective workers: {effective_workers}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {tasks / elapsed:.2f} tasks/s")
        ideal_time = tasks / effective_workers * 5
        print(f"Ideal time ({tasks} tasks / {effective_workers} workers * 5s): {ideal_time:.0f}s")
        print(f"Efficiency: {(tasks / effective_workers * 5) / elapsed * 100:.1f}%")
        print(f"{'=' * 50}")
        print(results)
