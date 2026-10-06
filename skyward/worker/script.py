"""A file, run on the node: all of it as ``__main__``, or one of its functions.

``sky compute run`` and ``sky run`` hand a node a file rather than a function.
What crosses the wire is :func:`run` or :func:`call` bound to the file's text, and
they are functions of this module on purpose: a function of an installed module
travels by reference, and the node has its own copy of this one. A closure would
travel by value, as bytecode, and bytecode is only good on the interpreter that
compiled it — a script sent from a 3.13 terminal to a node its header put on 3.12
would kill the worker unpickling it.

For the same reason the arguments of :func:`call` arrive as msgpack and its answer
leaves as JSON: values the node rebuilds against the function's own annotations,
not objects pickled on one interpreter for another.

What the file's image includes arrives beside the text, as the tar.gz ``sky run``
packed, and is unpacked for that run alone. It travels with every run rather than
once with the machine, so a compute attached to runs the code as it is now, not as
it was when the machine was set up.
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import tarfile
import tempfile
import traceback
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import GenericAlias, ModuleType

import msgspec


@dataclass(frozen=True, slots=True)
class Returned:
    """What the function returned, as JSON."""

    value: bytes


@dataclass(frozen=True, slots=True)
class Exited:
    """The function did not return, and left with this status instead."""

    status: int


type Outcome = Returned | Exited


def run(source: str, argv: tuple[str, ...], includes: bytes | None = None) -> int:
    """Execute ``source`` as ``__main__``, with ``argv`` as ``sys.argv``, and answer with its exit status.

    ``sys.exit`` is the script saying how it ended, so it is a status and not a
    failure; an exception is printed the way the interpreter would print it and
    reads as ``1``. What is not an ``Exception`` is left to go through: the worker
    stops an attempt by raising one of those inside it, and a script that caught
    it would report a stopped run as one that failed on its own.

    ``includes`` is importable while the file runs, and while its traceback is
    printed, which reads the lines of the files it passes through.
    """
    with _unpacked(includes):
        held, sys.argv = sys.argv, list(argv)
        try:
            exec(compile(source, argv[0], "exec"), {"__name__": "__main__", "__file__": argv[0]})
        except SystemExit as stop:
            return _status(stop)
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            sys.argv = held
        return 0


def call(source: str, path: str, entry: str, arguments: bytes, includes: bytes | None = None) -> Outcome:
    """Load ``source`` as a module, call its function ``entry`` with ``arguments``, and answer with what it returned.

    The arguments are the ones the command line was parsed into, encoded as
    msgpack by name, and each is converted back to what its parameter is annotated
    with. The answer is JSON; a value JSON has no form for is written as its
    ``str``, since what reads it is a person or a pipe, not a Python. Exiting and
    raising end the call as they end :func:`run`, and ``includes`` is there as it
    is there for :func:`run`.
    """
    with _unpacked(includes):
        try:
            with loaded(source, path) as module:
                function = vars(module)[entry]
                signature = inspect.signature(function, eval_str=True)
                given = msgspec.msgpack.decode(arguments, type=dict[str, object])
                bound = inspect.BoundArguments(signature, OrderedDict((name, _argument(value, signature.parameters[name])) for name, value in given.items()))
                return Returned(msgspec.json.encode(function(*bound.args, **bound.kwargs), enc_hook=str))
        except SystemExit as stop:
            return Exited(_status(stop))
        except Exception:
            traceback.print_exc()
            return Exited(1)


@contextmanager
def loaded(source: str, path: str) -> Iterator[ModuleType]:
    """``source`` as the module named after its file, for as long as the block lasts.

    Named after the file and not ``__main__``, so the ``if __name__ == "__main__"``
    block of a file written to be run by hand stays out of it. The module is in
    ``sys.modules`` while the block lasts because a dataclass written under
    ``from __future__ import annotations`` resolves its fields through it, both
    when the file defines it and when an argument is converted into it; whatever
    held the name before gets it back.
    """
    module = ModuleType(Path(path).stem)
    module.__file__ = path
    held = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, path, "exec"), vars(module))
        yield module
    finally:
        if held is None:
            del sys.modules[module.__name__]
        else:
            sys.modules[module.__name__] = held


@contextmanager
def _unpacked(includes: bytes | None) -> Iterator[None]:
    """``includes`` unpacked into a directory of its own, first on ``sys.path`` for as long as the block lasts.

    First, so the code shipped with the run wins over an installed package of the
    same name. What was imported from it leaves ``sys.modules`` with it: the next
    run's copy is another directory, and a module cached from this one would be
    found before it.
    """
    if includes is None:
        yield
        return
    with tempfile.TemporaryDirectory(prefix="skyward-includes-") as directory:
        with tarfile.open(fileobj=io.BytesIO(includes), mode="r:gz") as archive:
            archive.extractall(directory, filter="data")
        sys.path.insert(0, directory)
        try:
            yield
        finally:
            sys.path.remove(directory)
            sys.path_importer_cache.pop(directory, None)
            for name in [name for name, module in list(sys.modules.items()) if _within(module, directory)]:
                del sys.modules[name]


def _within(module: object, directory: str) -> bool:
    """Whether ``module`` was imported from ``directory``: its file, or for a namespace package, one of its paths.

    Both are read from the module's own namespace: a module may make up any attribute it lacks
    (``torch.classes`` does), and ``getattr`` would hand back whatever it made up.
    """
    namespace = getattr(module, "__dict__", {})
    places = (namespace.get("__file__"), *(namespace.get("__path__") or ()))
    return any(isinstance(place, str) and place.startswith(directory + os.sep) for place in places)


def _status(stop: SystemExit) -> int:
    match stop.code:
        case None:
            return 0
        case int(code):
            return code
        case message:
            print(message, file=sys.stderr)
            return 1


def _argument(value: object, parameter: inspect.Parameter) -> object:
    """``value`` as ``parameter`` takes it: its annotation, or a tuple or a mapping of it for ``*args`` and ``**kwargs``."""
    annotation = object if parameter.annotation is inspect.Parameter.empty else parameter.annotation
    match parameter.kind:
        case inspect.Parameter.VAR_POSITIONAL:
            wanted = GenericAlias(tuple, (annotation, ...))
        case inspect.Parameter.VAR_KEYWORD:
            wanted = GenericAlias(dict, (str, annotation))
        case _:
            wanted = annotation
    return msgspec.convert(value, wanted, dec_hook=_path)


def _path(kind: type, value: object) -> object:
    """A path, which msgpack carries as its text."""
    if issubclass(kind, PurePath) and isinstance(value, str):
        return kind(value)
    raise NotImplementedError(kind)


__all__ = ["Exited", "Outcome", "Returned", "call", "loaded", "run"]
