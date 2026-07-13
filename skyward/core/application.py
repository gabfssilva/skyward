"""Application — run a workload under a live full-screen dashboard.

``Application`` mirrors :func:`Compute`'s constructor but inverts the
execution model.  Textual must own the main thread (it installs OS signal
handlers — ``signal.signal`` only works there), so the dashboard cannot run
on the session's background loop the way the Rich console does.  Instead
:meth:`Application.run` keeps the dashboard on the main thread and runs the
workload on a worker thread, returning its result.

Outside a TTY (CI, pipes, redirected output) there is nothing to draw, so
``run`` falls back to executing the workload directly under the ``log``
console — same result, no dashboard.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Unpack

from skyward.core.spec import Options, Spec, SpecKwargs

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate

    from skyward.api.pool import Pool

__all__ = ["Application", "app"]

_DEFAULT_OPTIONS = Options()


def _wants_dashboard() -> bool:
    """Return ``True`` when the dashboard should be drawn.

    Honors ``SKYWARD_CONSOLE_FORCE_TTY`` (``1``/``true``/``yes``) so the
    dashboard can be forced on when stdout is not a terminal — used by the
    tests and for debugging the renderer.
    """
    override = os.environ.get("SKYWARD_CONSOLE_FORCE_TTY", "").strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    return bool(sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


class Application:
    """Run a workload under a live full-screen Skyward dashboard.

    The constructor takes the same arguments as :func:`Compute`.  Because
    the dashboard owns the terminal and the main thread, the workload is
    passed as a callable to :meth:`run` rather than written inside a ``with``
    body.

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
        Flat ``Spec`` fields (``provider``, ``accelerator``, ``nodes``, …)
        assembled into a single ``Spec`` when no positional specs are given.

    Examples
    --------
    >>> def workload(pool):
    ...     return train(data) @ pool
    >>> results = sky.Application(provider=sky.AWS(), nodes=4).run(workload)
    """

    def __init__(
        self,
        *specs: Spec,
        name: str | None = None,
        options: Options = _DEFAULT_OPTIONS,
        dark: bool = False,
        **kwargs: Unpack[SpecKwargs],
    ) -> None:
        self._specs = specs
        self._name = name
        self._options = options
        self._dark = dark
        self._kwargs = kwargs

    def run[T](self, fn: Callable[[Pool], T]) -> T:
        """Provision the pool, run ``fn(pool)``, and return its result.

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
        specs, options = self._resolve()
        if not _wants_dashboard():
            return self._run_headless(fn, specs, options)
        return self._run_with_dashboard(fn, specs, options)

    def _resolve(self) -> tuple[tuple[Spec, ...], Options]:
        """Resolve the three calling conventions to ``(specs, options)``."""
        if self._name is not None:
            if self._specs or self._kwargs:
                raise ValueError("Cannot mix 'name' with specs or keyword arguments")
            from skyward.config import resolve_pool_specs

            specs, options = resolve_pool_specs(self._name)
            return tuple(specs), options
        if self._specs and self._kwargs:
            raise ValueError("Cannot mix positional Spec objects with flat keyword arguments")
        if not self._specs and not self._kwargs:
            raise ValueError("Either Spec objects or keyword arguments (provider, ...) must be provided")
        specs = self._specs if self._specs else (Spec(**self._kwargs),)
        return specs, self._options

    def _run_headless[T](
        self, fn: Callable[[Pool], T], specs: tuple[Spec, ...], options: Options,
    ) -> T:
        from skyward.core.context import _active_pool
        from skyward.core.session import Session

        with Session(
            console="log",
            logging=options.logging,
            shutdown_timeout=options.shutdown_timeout,
        ) as session:
            pool = session.compute(*specs, name=self._name, options=options)
            token = _active_pool.set(pool)
            try:
                return fn(pool)
            finally:
                _active_pool.reset(token)

    def _run_with_dashboard[T](
        self, fn: Callable[[Pool], T], specs: tuple[Spec, ...], options: Options,
    ) -> T:
        from skyward.core.context import _active_pool
        from skyward.core.session import Session
        from skyward.tui.app import SkywardTUI
        from skyward.tui.sources import ProjectionSource

        box: dict[str, object] = {}
        ready = threading.Event()
        session = Session(
            console="silent",
            logging=options.logging,
            shutdown_timeout=options.shutdown_timeout,
        )
        session.__enter__()
        source = ProjectionSource(session.projection)
        tui = SkywardTUI(source, start_dark=self._dark, on_ready=ready.set)

        def worker() -> None:
            ready.wait(timeout=30)
            token = None
            try:
                pool = session.compute(*specs, name=self._name, options=options)
                token = _active_pool.set(pool)
                box["result"] = fn(pool)
            except Exception as exc:  # surfaced to the caller after teardown
                box["error"] = exc
            finally:
                if token is not None:
                    _active_pool.reset(token)
                with suppress(Exception):
                    tui.call_from_thread(tui.exit)

        thread = threading.Thread(target=worker, name="skyward-app", daemon=True)
        thread.start()
        try:
            tui.run()
        finally:
            thread.join()
            source.close()
            session.__exit__(None, None, None)

        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box.get("result")  # type: ignore[return-value]


def app[**P, T](
    *specs: Spec,
    name: str | None = None,
    options: Options = _DEFAULT_OPTIONS,
    dark: bool = False,
    **kwargs: Unpack[SpecKwargs],
) -> Callable[[Callable[Concatenate[Pool, P], T]], Callable[P, T]]:
    """Decorate an entry point to run under a live Skyward dashboard.

    Sugar over :class:`Application`: the decorated function receives the
    provisioned pool as its first argument, and calling it provisions the
    pool, runs the body under the dashboard (on the worker thread while the
    dashboard owns the main thread), and returns the result.  Any extra
    arguments passed at call time are forwarded after the pool.

    Parameters
    ----------
    *specs, name, options, dark, **kwargs
        Identical to :class:`Application` / :func:`Compute`.

    Returns
    -------
    Callable
        A decorator turning ``fn(pool, *args) -> T`` into ``f(*args) -> T``.

    Examples
    --------
    >>> @sky.app(provider=sky.AWS(), accelerator="A100", nodes=4)
    ... def main(pool):
    ...     return train(data) @ pool
    >>> results = main()
    """

    def decorator(fn: Callable[Concatenate[Pool, P], T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **call_kwargs: P.kwargs) -> T:
            application = Application(*specs, name=name, options=options, dark=dark, **kwargs)
            return application.run(lambda pool: fn(pool, *args, **call_kwargs))

        return wrapper

    return decorator
