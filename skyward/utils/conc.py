"""Concurrent utilities - functional primitives for parallel execution."""

import contextvars
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import overload


def map_async[I, O](
    fn: Callable[[I], O],
    items: Iterable[I],
    concurrency: int | None = None,
) -> Iterator[O]:
    """Apply function to items concurrently, preserving order.

    Automatically propagates contextvars to worker threads.

    Args:
        fn: Function to apply to each item.
        items: Items to process.
        concurrency: Max concurrent workers. None = len(items).

    Yields:
        Results in same order as input items.

    Example:
        >>> list(map_async(fetch_url, urls, concurrency=10))
        [response1, response2, ...]
    """
    items_list = list(items)
    if not items_list:
        return

    # Create a fresh context copy for EACH task (ctx.run cannot be concurrent on same object)
    workers = concurrency if concurrency is not None else len(items_list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, fn, item)
            for item in items_list
        ]
        for future in futures:
            yield future.result()


def for_each_async[I](
    fn: Callable[[I], object],
    items: Iterable[I],
    concurrency: int | None = None,
) -> None:
    """Apply function to items concurrently, discarding results.

    Args:
        fn: Function to apply (side-effectful).
        items: Items to process.
        concurrency: Max concurrent workers. None = len(items).

    Raises:
        Exception: First exception encountered (fails fast).

    Example:
        >>> for_each_async(upload_file, files, concurrency=5)
    """
    # Consume the iterator to execute all items
    for _ in map_async(fn, items, concurrency):
        pass


@overload
def map_async_indexed[I, O](
    fn: Callable[[int, I], O],
    items: Iterable[I],
    concurrency: int | None = None,
) -> Iterator[O]: ...


@overload
def map_async_indexed[I, O](
    fn: Callable[[int, I], O],
    items: Iterable[I],
    concurrency: int | None,
    *,
    with_index: bool,
) -> Iterator[tuple[int, O]]: ...


def map_async_indexed[I, O](
    fn: Callable[[int, I], O],
    items: Iterable[I],
    concurrency: int | None = None,
    *,
    with_index: bool = False,
) -> Iterator[O] | Iterator[tuple[int, O]]:
    """Apply function with index to items concurrently, preserving order.

    Automatically propagates contextvars to worker threads.

    Args:
        fn: Function (index, item) -> result.
        items: Items to process.
        concurrency: Max concurrent workers. None = len(items).
        with_index: If True, yields (index, result) tuples.

    Yields:
        Results (or (index, result) tuples) in same order as input.

    Example:
        >>> list(map_async_indexed(lambda i, x: f"{i}: {x}", ["a", "b"]))
        ["0: a", "1: b"]
    """
    items_list = list(items)
    if not items_list:
        return

    # Create a fresh context copy for EACH task (ctx.run cannot be concurrent on same object)
    workers = concurrency if concurrency is not None else len(items_list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, fn, i, item)
            for i, item in enumerate(items_list)
        ]
        for i, future in enumerate(futures):
            result = future.result()
            if with_index:
                yield (i, result)
            else:
                yield result
