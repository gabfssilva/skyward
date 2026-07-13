"""Application stub — type-checking interface for the dashboard runner.

``Application`` runs a workload under a live full-screen Skyward dashboard.
It takes the same arguments as :func:`Compute`, but because the dashboard
owns the terminal and the main thread, the workload is passed as a callable
to :meth:`run` rather than written inside a ``with`` body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack, overload

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate

    from skyward.api.pool import Pool
    from skyward.api.spec import Options, Spec, SpecKwargs

__all__ = ["Application", "app"]


class Application:
    """Run a workload under a live full-screen Skyward dashboard.

    Parameters
    ----------
    *specs
        One or more ``Spec`` objects (multi-provider fallback).  Mutually
        exclusive with ``name`` and flat keyword arguments.
    name
        Named pool from ``skyward.toml``.  Mutually exclusive with specs
        and kwargs.
    options
        Operational tuning, identical to :func:`Compute`.  ``options.console``
        is ignored — the dashboard replaces the console.
    dark
        Start the dashboard in the dark theme.
    **kwargs
        Flat ``Spec`` fields assembled into a single ``Spec`` when no
        positional specs are given.

    Examples
    --------
    >>> def workload(pool):
    ...     return train(data) @ pool
    >>> results = sky.Application(provider=sky.AWS(), nodes=4).run(workload)
    """

    @overload
    def __init__(self, *specs: Spec, options: Options = ..., dark: bool = ...) -> None: ...

    @overload
    def __init__(self, *, name: str, options: Options = ..., dark: bool = ...) -> None: ...

    @overload
    def __init__(
        self, *, options: Options = ..., dark: bool = ..., **kwargs: Unpack[SpecKwargs],
    ) -> None: ...

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def run[T](self, fn: Callable[[Pool], T]) -> T:
        """Provision the pool, run ``fn(pool)`` under the dashboard, return its result.

        In a terminal the dashboard takes over the screen on the main thread
        while ``fn`` runs on a worker thread; pressing ``q`` closes the
        dashboard but lets the in-flight workload finish.  Without a TTY the
        workload runs directly under the ``log`` console.

        Parameters
        ----------
        fn
            Callable invoked with the provisioned pool.  Its return value is
            returned by ``run``; an exception it raises propagates out.

        Returns
        -------
        T
            Whatever ``fn`` returns.
        """
        ...


type _AppDecorator[**P, T] = Callable[[Callable[Concatenate[Pool, P], T]], Callable[P, T]]


@overload
def app[**P, T](*specs: Spec, options: Options = ..., dark: bool = ...) -> _AppDecorator[P, T]: ...


@overload
def app[**P, T](*, name: str, options: Options = ..., dark: bool = ...) -> _AppDecorator[P, T]: ...


@overload
def app[**P, T](
    *, options: Options = ..., dark: bool = ..., **kwargs: Unpack[SpecKwargs],
) -> _AppDecorator[P, T]: ...


def app(*args: object, **kwargs: object) -> object:
    """Decorate an entry point to run under a live Skyward dashboard.

    Sugar over :class:`Application`: the decorated ``fn(pool, *args) -> T``
    becomes ``f(*args) -> T``; calling it provisions the pool, runs the body
    under the dashboard, and returns the result.

    Examples
    --------
    >>> @sky.app(provider=sky.AWS(), accelerator="A100", nodes=4)
    ... def main(pool):
    ...     return train(data) @ pool
    >>> results = main()
    """
    ...
