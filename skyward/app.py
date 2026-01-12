"""Application infrastructure: @component, @on, @monitor, create_app.

Core decorators for the event-driven architecture:
- @component: Auto-generates __init__, applies DI, wires handlers
- @on: Marks methods as event handlers
- @monitor: Transforms async functions into background loops
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import field
from typing import TYPE_CHECKING, Any, AsyncIterator, dataclass_transform, get_type_hints

from injector import Injector, Module, inject, singleton

from .bus import AsyncEventBus

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================

_HANDLERS_ATTR = "__event_handlers__"
_MONITOR_ATTR = "__monitor__"
_COMPONENT_MARKER = "__is_component__"

_COMPONENT_REGISTRY: list[type] = []
_MONITOR_REGISTRY: list[Callable[..., Coroutine[Any, Any, Any]]] = []


# =============================================================================
# @on decorator
# =============================================================================


def on[E](
    event_type: type[E],
    *,
    match: Callable[[Any, E], bool] | None = None,
    audit: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Mark method as event handler for given event type.

    The decorated method will be automatically connected to the event bus
    when the component is instantiated.

    Args:
        event_type: The event class to handle.
        match: Optional filter function (self, event) -> bool.
            If provided, handler only runs when match returns True.
            This eliminates boilerplate filtering in handlers.
        audit: If True (default), log entry/exit with timing.
            Set to False for noisy/high-frequency events.

    Usage:
        @component
        class MyComponent:
            bus: AsyncEventBus

            # Without filter (manual filtering)
            @on(InstancePreempted)
            async def handle_preemption(self, sender, event):
                if event.instance.node != self.id:
                    return
                # Handle event

            # With filter (declarative)
            @on(InstanceProvisioned, match=lambda self, e: e.instance.node == self.id)
            async def handle_provisioned(self, sender, event):
                # No filtering needed - only called when match is True
                pass

            # Disable audit for noisy handlers
            @on(Metric, audit=False)
            async def handle_metric(self, sender, event):
                pass
    """
    import functools
    import time

    from loguru import logger

    def _truncate(s: str, max_len: int = 200) -> str:
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        method_name = method.__name__

        @functools.wraps(method)
        async def handler(self: Any, sender: Any, event: E) -> Any:
            # Apply match filter if provided
            if match is not None and not match(self, event):
                return None

            # Audit logging
            if audit:
                component = type(self).__name__
                event_repr = _truncate(repr(event))
                op = f"{component}.{method_name}"

                logger.opt(depth=1).debug(f"→ {op}({event_repr})")
                start = time.perf_counter()

                try:
                    result = await method(self, sender, event)
                except Exception:
                    elapsed = time.perf_counter() - start
                    logger.opt(depth=1).exception(f"✗ {op} [{elapsed:.2f}s]")
                    raise

                elapsed = time.perf_counter() - start
                logger.opt(depth=1).debug(f"← {op} [{elapsed:.2f}s]")
                return result
            else:
                return await method(self, sender, event)

        if not hasattr(handler, _HANDLERS_ATTR):
            setattr(handler, _HANDLERS_ATTR, [])
        getattr(handler, _HANDLERS_ATTR).append(event_type)
        return handler

    return decorator


# =============================================================================
# @monitor decorator
# =============================================================================


def monitor(
    interval: float = 5.0,
    name: str | None = None,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, Any]]],
    Callable[..., Coroutine[Any, Any, Any]],
]:
    """
    Transform async function into background monitoring loop.

    The decorated function becomes the body of a loop that runs every `interval`
    seconds. Use with DI - dependencies are injected when the monitor starts.

    Usage:
        class MonitorModule(Module):

            @monitor(interval=5.0)
            async def check_preemption(
                self,
                registry: InstanceRegistry,
                bus: AsyncEventBus,
            ):
                # This runs every 5 seconds
                for instance in registry.instances:
                    if await is_preempted(instance):
                        bus.emit(InstancePreempted(instance=instance))
    """

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        fn.__monitor__ = {  # type: ignore[attr-defined]
            "interval": interval,
            "name": name or fn.__name__,
        }
        _MONITOR_REGISTRY.append(fn)
        return fn

    return decorator


# =============================================================================
# @component decorator
# =============================================================================


def _has_event_handlers(cls: type) -> bool:
    """Check if class has any @on decorated methods."""
    for name in dir(cls):
        # Skip dunder methods but allow single-underscore prefixed handlers
        if name.startswith("__"):
            continue
        method = getattr(cls, name, None)
        if method and hasattr(method, _HANDLERS_ATTR):
            return True
    return False


def _wire_handlers(instance: object, event_bus: AsyncEventBus) -> None:
    """Connect all @on handlers of instance to event bus."""
    from loguru import logger

    wired_count = 0
    for name in dir(instance):
        # Skip dunder methods but allow single-underscore prefixed handlers
        if name.startswith("__"):
            continue
        method = getattr(instance, name)
        if event_types := getattr(method, _HANDLERS_ATTR, None):
            for event_type in event_types:
                event_bus.connect(event_type, method)
                logger.trace(f"Wired {type(instance).__name__}.{name} -> {event_type.__name__}")
                wired_count += 1
    if wired_count > 0:
        logger.trace(f"Wired {wired_count} handlers for {type(instance).__name__}")


def _generate_init(cls: type, hints: dict[str, type]) -> Callable[..., None]:
    """Generate __init__ from type hints.

    Attributes with default values on the class are treated as optional
    and excluded from injection. Attributes without defaults are required
    and will be injected by the DI container.

    Special handling for dataclass Field objects with default_factory.
    """
    from dataclasses import Field, MISSING

    # Find attributes that need injection (no default value, not private, not callable)
    required_deps = []
    # Collect fields with default_factory for initialization
    factory_fields: list[tuple[str, Any]] = []

    for name in hints:
        attr = getattr(cls, name, None)

        # Handle Field objects (from dataclass field())
        if isinstance(attr, Field):
            if attr.default_factory is not MISSING:
                # Has a factory, store for initialization
                factory_fields.append((name, attr.default_factory))
            continue

        if name.startswith("_"):
            continue
        if callable(attr) or isinstance(attr, property):
            continue
        # Skip if class has a default value for this attribute
        if attr is not None:
            continue
        required_deps.append(name)

    # Build factory initializations
    factory_refs: dict[str, Any] = {}
    factory_assignments = []
    for name, factory in factory_fields:
        ref_name = f"_factory_{name}"
        factory_refs[ref_name] = factory
        factory_assignments.append(f"self.{name} = {ref_name}()")

    if not required_deps and not factory_assignments:

        def empty_init(self: Any) -> None:
            pass

        return empty_init

    # Build __init__ dynamically - only required dependencies
    # The @inject decorator handles the type resolution
    params = ", ".join(required_deps) if required_deps else ""
    dep_assignments = "; ".join(f"self.{name} = {name}" for name in required_deps)
    factory_init = "; ".join(factory_assignments)

    # Combine all assignments
    all_assignments = "; ".join(filter(None, [dep_assignments, factory_init]))

    # Check if class has __post_init__ to call after assignments
    has_post_init = hasattr(cls, "__post_init__")
    post_init_call = "; self.__post_init__()" if has_post_init else ""

    code = f"def __init__(self, {params}) -> None: {all_assignments}{post_init_call}"

    local_ns: dict[str, Any] = factory_refs.copy()
    exec(code, local_ns, local_ns)  # noqa: S102

    # Attach original annotations for injector to use
    init_func = local_ns["__init__"]
    init_func.__annotations__ = {name: hints[name] for name in required_deps}
    init_func.__annotations__["return"] = None

    return init_func


def _wrap_init_with_auto_wire(
    cls: type,
    original_init: Callable[..., None],
) -> Callable[..., None]:
    """Wrap __init__ to auto-wire handlers after initialization."""
    import functools
    from loguru import logger

    @functools.wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        logger.trace(f"wrapped_init called for {cls.__name__} with args={args}, kwargs={list(kwargs.keys())}")
        original_init(self, *args, **kwargs)
        # Auto-wire if instance has bus attribute
        if hasattr(self, "bus") and self.bus is not None:
            logger.trace(f"Auto-wiring {cls.__name__} to bus {id(self.bus)}")
            _wire_handlers(self, self.bus)
        else:
            logger.warning(f"{cls.__name__} has no bus attribute after init!")

    # Preserve annotations for @inject
    wrapped_init.__annotations__ = original_init.__annotations__
    return wrapped_init


@dataclass_transform(field_specifiers=(field,))
def component[T](cls: type[T]) -> type[T]:
    """
    Transform class into a component.

    1. Generate __init__ from type hints (like @dataclass)
    2. Apply @inject for DI
    3. Register for auto-wiring of @on handlers
    4. Handle field(default_factory=...) for lazy initialization

    Usage:
        @component
        class Node:
            # Required dependencies (injected)
            bus: AsyncEventBus
            provider: ProviderName

            # Optional with defaults
            _count: int = 0

            # Fields with factories
            _cache: dict[str, Any] = field(default_factory=dict)

            @on(InstancePreempted)
            async def handle(self, sender, event): ...
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    # Check if class needs auto-wiring (has handlers and bus)
    has_handlers = _has_event_handlers(cls)
    has_bus = "bus" in hints

    # Generate __init__ if not present
    if hints and (
        "__init__" not in cls.__dict__ or cls.__init__ is object.__init__  # type: ignore[comparison-overlap]
    ):
        generated_init = _generate_init(cls, hints)
        generated_init = inject(generated_init)
        # Wrap with auto-wire if needed
        if has_handlers and has_bus:
            generated_init = _wrap_init_with_auto_wire(cls, generated_init)
        cls.__init__ = generated_init  # type: ignore[method-assign]
    elif hasattr(cls, "__init__") and not hasattr(cls.__init__, "__bindings__"):
        init_func = inject(cls.__init__)
        # Wrap with auto-wire if needed
        if has_handlers and has_bus:
            init_func = _wrap_init_with_auto_wire(cls, init_func)
        cls.__init__ = init_func  # type: ignore[method-assign]

    # Mark as component and register
    setattr(cls, _COMPONENT_MARKER, True)
    _COMPONENT_REGISTRY.append(cls)

    return cls


# =============================================================================
# MonitorManager
# =============================================================================


class MonitorManager:
    """Manages background monitor tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._shutdown = False

    def start(
        self,
        name: str,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        interval: float,
        injector: Injector,
    ) -> None:
        """Start a monitor loop."""
        if name in self._tasks:
            raise ValueError(f"Monitor {name} already running")

        async def loop() -> None:
            while not self._shutdown:
                try:
                    # Inject dependencies and call
                    await injector.call_with_injection(fn)
                except Exception as e:
                    print(f"[ERROR] Monitor {name} failed: {e}")
                await asyncio.sleep(interval)

        task = asyncio.create_task(loop())
        self._tasks[name] = task

    async def stop(self, name: str) -> None:
        """Stop a specific monitor."""
        task = self._tasks.pop(name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop_all(self) -> None:
        """Stop all monitors."""
        self._shutdown = True
        for task in self._tasks.values():
            task.cancel()

        for _, task in list(self._tasks.items()):
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    @property
    def running(self) -> list[str]:
        """List of running monitor names."""
        return list(self._tasks.keys())


# =============================================================================
# Bootstrap
# =============================================================================


class _ComponentModule(Module):
    """Binds all registered components as singletons."""

    def configure(self, binder: Any) -> None:
        for comp_cls in _COMPONENT_REGISTRY:
            binder.bind(comp_cls, to=comp_cls, scope=singleton)


def create_app(*modules: Module) -> tuple[Injector, MonitorManager]:
    """
    Create app with DI and wire components.

    Returns:
        Tuple of (injector, monitor_manager)
    """
    # Create injector with all modules
    injector = Injector([*modules, _ComponentModule()])

    # Get event bus
    event_bus = injector.get(AsyncEventBus)

    # Wire handlers for all components
    for comp_cls in _COMPONENT_REGISTRY:
        if _has_event_handlers(comp_cls):
            instance = injector.get(comp_cls)
            _wire_handlers(instance, event_bus)

    # Create monitor manager
    monitor_manager = MonitorManager()

    # Start monitors
    for fn in _MONITOR_REGISTRY:
        config = getattr(fn, _MONITOR_ATTR, None)
        if config:
            monitor_manager.start(
                name=config["name"],
                fn=fn,
                interval=config["interval"],
                injector=injector,
            )

    return injector, monitor_manager


@asynccontextmanager
async def app_context(*modules: Module) -> AsyncIterator[Injector]:
    """
    Full lifecycle context manager.

    Creates app, starts monitors, yields injector, then shuts down.

    Usage:
        async with app_context(AppModule(), MonitorModule()) as app:
            pool = app.get(ComputePool)
            await pool.start()
            ...
    """
    injector, monitor_manager = create_app(*modules)
    event_bus = injector.get(AsyncEventBus)

    try:
        yield injector
    finally:
        # Stop monitors
        await monitor_manager.stop_all()

        # Shutdown event bus
        await event_bus.shutdown()


# =============================================================================
# Utilities
# =============================================================================


def clear_registries() -> None:
    """Clear component and monitor registries. Useful for testing."""
    _COMPONENT_REGISTRY.clear()
    _MONITOR_REGISTRY.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Decorators
    "component",
    "on",
    "monitor",
    # Bootstrap
    "create_app",
    "app_context",
    # Manager
    "MonitorManager",
    # Testing
    "clear_registries",
]
