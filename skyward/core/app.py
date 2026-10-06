"""A function the command line runs on the compute declared above it.

``sky.app`` is what ``sky.Compute`` is told about the machines — provider, shape,
size, image, plugins — and the options its session runs with, as a value, without
the pool. Applied to a function, it makes that function a command of ``sky run``::

    @sky.app(provider=sky.AWS(), accelerator=sky.accelerators.RTX_3090(), nodes=2)
    def train(epochs: int = 10) -> float: ...

    $ sky run train.py --epochs 5

The function's parameters are the command's, parsed and checked before a machine
is bought. Several functions under one ``sky.app`` are commands of one compute.
Called from Python, the function is only itself, and runs where it is called.

Nothing here may need the client. ``sky run`` sends the file's text to a node and
the node runs all of it, decorator included, on an install of skyward that has no
httpx.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from skyward.core.accelerators import Accelerator
from skyward.core.spec import NodeSpec, Options
from skyward.shared.providers import Provider
from skyward.shared.schemas import Allocation, Image
from skyward.worker.plugins import Plugin


@dataclass(frozen=True, slots=True, kw_only=True)
class App:
    """The compute a function runs on, under the names ``sky.Compute`` gives the same things."""

    provider: Provider
    accelerator: str | Accelerator | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None
    nodes: NodeSpec = 1
    allocation: Allocation = "spot_if_available"
    image: Image = Image()
    plugins: Sequence[Plugin] = ()
    delete_on_exit: bool = True
    options: Options = Options()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugins", tuple(self.plugins))

    def __call__[**P, T](self, fn: Callable[P, T]) -> Entry[P, T]:
        return Entry(fn, self)


@dataclass(frozen=True, slots=True)
class Entry[**P, T]:
    """A function, and the compute ``sky run`` runs it on."""

    fn: Callable[P, T]
    app: App

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.fn(*args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[P, T]:
        """What :func:`inspect.signature` follows, so an entry reads as the function's parameters and not ``*args, **kwargs``."""
        return self.fn


app = App

__all__ = ["App", "Entry", "app"]
