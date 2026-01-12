"""SkywardExecutor - ThreadPoolExecutor drop-in for cloud compute."""

import time
from time import sleep

import skyward as sky


def task(x: int) -> int:
    sleep(5)
    return x * 2

def do_work(n: int):
    futures = [executor.submit(task, i) for i in range(n)]
    return [f.result() for f in futures]

if __name__ == "__main__":
    with sky.Executor(
        provider=sky.AWS(),
        nodes=5,
        concurrency=10,
        image=sky.Image(skyward_source="local"),
    ) as executor:
        do_work(5) # warm up
        t0 = time.perf_counter()
        do_work(100)
        print(f"Submit time: {time.perf_counter() - t0:.1f}s")
