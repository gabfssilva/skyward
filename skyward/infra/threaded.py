import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor


class ThreadPoolRunner[T]:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=self.workers)

    async def run[R](self, fn: Callable[[T], R], obj: T) -> R:
        return await self.as_async(fn)(obj)

    def as_async[**P, R](self, fn: Callable[P, R]) -> Callable[P, Awaitable[R]]:
        async def run(*args: P.args, **kwargs: P.kwargs) -> R:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

        return run
