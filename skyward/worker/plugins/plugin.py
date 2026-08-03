"""What a plugin is, and the places it gets to speak.

A plugin is a value, not a callback. It travels in the spec, is written to the
database with it, and is rebuilt from it on the node — so it cannot be a closure,
a lambda or an object holding a live handle. It is a struct of parameters, and its
behaviour is what the class does with them.

Which is what lets the same object be constructed by the user, validated by the
daemon and executed by the worker: three processes, on two machines, agreeing about
a plugin because they agree about its name and its fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from msgspec import Struct, to_builtins

from skyward.shared.schemas import Image, PluginRef
from skyward.worker.api import Info

if TYPE_CHECKING:
    from skyward.core.compute import Compute


class Plugin(Struct, frozen=True):
    """A plugin, with every hook optional and none of them doing anything by default.

    Attributes
    ----------
    kind : str
        Its name on the wire, and how it is found again on the node.
    collective : bool
        Whether the plugin makes the nodes depend on each other. A collective
        freezes the world when the last rank joins it, so a compute running one
        cannot be resized: taking a rank away does not shrink the job, it hangs it
        at the next all-reduce on a peer that is never going to answer. The
        reconciler reads this and refuses to scale such a compute at all.
    """

    kind: ClassVar[str]
    collective: ClassVar[bool] = False

    def image(self, image: Image) -> Image:
        """What the machine needs installed before the plugin can run at all.

        The daemon calls this once, when the compute is provisioned. It is a
        transform rather than a list of packages because plugins compose: each is
        handed what the ones before it asked for.
        """
        return image

    def bootstrap(self, image: Image, concurrency: int) -> Sequence[str]:
        """Extra shell phases, appended after the image's own bootstrap.

        Runs on the daemon at script-generation time, like :meth:`image` — not on
        the node — so it may only return the phases the script will run, never do
        anything itself. ``concurrency`` is the worker's width, the one datum a
        phase needs that the image does not carry: a plugin that partitions the
        machine has to know how many ways.
        """
        return ()

    @contextmanager
    def setup(self, info: Info) -> Iterator[None]:
        """The worker's lifetime, on the node.

        Entered once, before the worker takes a task, and left when it stops. This
        is where a process group is formed and where an environment variable that
        a library reads at import time is set — both being things that must exist
        before the first task, not around each one.
        """
        yield

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        """One task, on the node.

        Plugins wrap in the order they were listed: the first is outermost, and
        therefore the one that sees the others' work.
        """
        return call()

    @contextmanager
    def client(self, compute: Compute) -> Iterator[None]:
        """The plugin's say on the client, for as long as the pool is up.

        Unlike the others, this hook does not travel: it runs on the instance the
        user constructed, in the process that opened the ``with`` block, and never
        on a node. It is entered once the compute is ready and left before it is
        torn down — the place a plugin reaches back into the live pool, as joblib
        does to point its parallel backend at it.
        """
        yield

    def ref(self) -> PluginRef:
        return PluginRef(kind=self.kind, params=to_builtins(self))
