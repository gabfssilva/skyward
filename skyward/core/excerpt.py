"""The part of a module a function needs, as text, so the daemon has code to show for it.

A function reaches the daemon as a pickle, and a pickle is compiled code with no
text in it. The text exists only here, in the process that defined the function,
and only while the file it came from is on disk — so it is read at the upload, or
never.

``inspect.getsource`` alone gives the function and nothing around it: a body that
calls ``col.Counter`` and a helper defined three screens up reads as a fragment. So
the names the function uses are followed through its globals — the live ones,
which is what writes an alias out as ``import collections as col`` instead of
guessing at it — and what they are bound to is written above it: the imports, the
constants, and the functions and classes of its own module, each followed in turn.

It is a transcript for a person to read, not a module anything runs. A function
whose text was never kept — fed through stdin, built by an ``exec``, typed at a
REPL older than 3.13, or out of a file gone by the time it is sent — has none.
"""

from __future__ import annotations

import ast
import functools
import inspect
import sys
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass, field

LONGEST = 120
"""The widest ``repr`` written out as a constant; past it, a value is named and not shown."""


@dataclass(slots=True)
class _Found:
    home: str
    scope: dict[str, object]
    imports: set[str] = field(default_factory=set)
    constants: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, tuple[int, str]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)


def defined(fn: Callable[..., object]) -> Callable[..., object]:
    """The function a bound method or a partial was made of, which is what has a name and a text.

    Parameters
    ----------
    fn
        What was handed to ``@sky.function``.
    """
    match fn:
        case types.MethodType(__func__=inner) | functools.partial(func=inner):
            return defined(inner)
        case _:
            return fn


def excerpt(fn: Callable[..., object]) -> str | None:
    """The imports, constants, local definitions and the function itself, in the order of its file.

    Parameters
    ----------
    fn
        The function as it was defined: what :func:`defined` answers, not a wrapper around it.

    Returns
    -------
    str or None
        ``None`` when there is no source to read.
    """
    if not isinstance(fn, types.FunctionType) or (definition := _definition(fn)) is None:
        return None

    found = _Found(home=fn.__module__, scope=fn.__globals__)
    found.definitions[fn.__name__] = definition
    found.seen.add(fn.__name__)
    _follow(definition[1], found)

    straight = sorted(line for line in found.imports if line.startswith("import "))
    taken = sorted(line for line in found.imports if line.startswith("from "))
    blocks = (
        "\n".join([*straight, *taken]),
        "\n".join(found.constants.values()),
        *(text for _, text in sorted(found.definitions.values())),
    )
    return "\n\n\n".join(block.strip("\n") for block in blocks if block.strip()) + "\n"


def _definition(obj: types.FunctionType | type) -> tuple[int, str] | None:
    """Where a function or class is defined in its file, and its text there."""
    try:
        lines, first = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    return first, textwrap.dedent("".join(lines))


def _follow(text: str, found: _Found) -> None:
    for name in _names(text):
        if name in found.seen or name not in found.scope:
            continue
        found.seen.add(name)
        _place(name, found.scope[name], found)


def _names(text: str) -> list[str]:
    """Every name a definition reads, in order: its body, its annotations, its decorators, its bases.

    A parameter or a local that happens to share a global's name is counted too,
    which costs at most an import line nobody needed; parsing scopes to avoid it
    would cost far more than that.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return list(dict.fromkeys(node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)))


def _place(name: str, value: object, found: _Found) -> None:
    match value:
        case types.ModuleType(__name__=module) if module == name:
            found.imports.add(f"import {module}")
        case types.ModuleType(__name__=module) if module.endswith(f".{name}"):
            found.imports.add(f"from {module.rpartition('.')[0]} import {name}")
        case types.ModuleType(__name__=module):
            found.imports.add(f"import {module} as {name}")
        case types.FunctionType() | type() if value.__module__ == found.home and (definition := _definition(value)) is not None:
            found.definitions[name] = definition
            _follow(definition[1], found)
        case _ if (line := _imported(name, value, found.home)) is not None:
            found.imports.add(line)
        case _:
            shown = repr(value)
            found.constants[name] = f"{name} = {shown}" if len(shown) <= LONGEST else f"{name} = ...  # {type(value).__name__}"


def _imported(name: str, value: object, home: str) -> str | None:
    """The line that brings ``value`` in under ``name``, when its own module can be shown to hold it."""
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or module == home or (source := sys.modules.get(module)) is None:
        return None
    if getattr(source, name, None) is value:
        return f"from {module} import {name}"
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str) and "." not in qualname and getattr(source, qualname, None) is value:
        return f"from {module} import {qualname} as {name}"
    return None
