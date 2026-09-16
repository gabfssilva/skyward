"""A function written out as text, for a caller with no interpreter to pickle one with.

The SDK hands the daemon a live callable it has already pickled. Nothing else can:
a browser, or anything else that speaks HTTP and nothing else, has only the text
somebody typed, and text is not what a worker takes.

What is built here is the bridge — the module's source and the name to call in it,
bound to a function of this module, and pickled and dispatched like any other
function. The text is not executed where it is captured. It is compiled and run on
the machine, once per call, which is the only place the imports it names exist and
the only place anything was ever going to run.

It is a partial and not a closure because of what each pickles as. A closure
travels by value, as bytecode, and bytecode is only good on the interpreter that
compiled it: the daemon's 3.13 closure crashes a worker on 3.12. A function of
this module travels by reference, and the node has this module — its own copy,
for its own Python — so the only code crossing is the text.

The name is checked against the source here rather than trusted, because the two
ways this can be wrong are both wrong on every machine equally. A dispatch is a
slow and expensive place to learn that a module does not parse.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from functools import partial

from skyward.shared.errors import SourceRejectedError


def authored(source: str, name: str) -> Callable[..., object]:
    """The module's ``name``, as something a node can be handed.

    Parameters
    ----------
    source
        A Python module, as text.
    name
        The function in it to call.

    Raises
    ------
    SourceRejectedError
        The text does not parse, or defines no function under that name.
    """
    try:
        module = ast.parse(source)
    except SyntaxError as broken:
        raise SourceRejectedError(f"line {broken.lineno}: {broken.msg}", line=broken.lineno, offset=broken.offset) from broken

    if name not in (defined := _defines(module)):
        raise SourceRejectedError(f"the source defines no function called {name!r}", defines=sorted(defined))

    return partial(_run, source, name)


def _run(source: str, name: str, /, *args: object, **kwargs: object) -> object:
    origin = f"<{name}>"
    namespace: dict[str, object] = {"__name__": name, "__file__": origin}
    exec(compile(source, origin, "exec"), namespace)
    target = namespace[name]
    if not callable(target):
        raise TypeError(f"{name} is {type(target).__name__} by the time the module finishes, not a function")
    return target(*args, **kwargs)


def _defines(module: ast.Module) -> frozenset[str]:
    """The functions the module defines at its top level, which are the callable ones.

    A nested one is a detail of its parent and a name bound to a call is a value
    whose type is only known once the module has run — neither is something to
    promise a caller before anything has.
    """
    return frozenset(name for node in module.body if (name := _defined(node)))


def _defined(node: ast.stmt) -> str | None:
    match node:
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
            return name
        case _:
            return None
