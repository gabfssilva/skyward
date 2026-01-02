from time import sleep

from joblib import Parallel, delayed

from skyward import AWS, Image
from skyward.integrations import JoblibPool


def slow_task(x):
    print(f"Task {x} starting")
    sleep(5)
    print(f"Task {x} done")
    return x * 2


if __name__ == '__main__':
    with JoblibPool(
        provider=AWS(),
        nodes=5,
        concurrency=5,
        image=Image(pip=["joblib"])
    ):
        results = Parallel(n_jobs=50)(
            delayed(slow_task)(i) for i in range(100)
        )
        print(results)
