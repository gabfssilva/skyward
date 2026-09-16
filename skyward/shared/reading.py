"""What a pickled function says about itself, read without running any of it.

A function reaches the daemon as cloudpickle bytes, and what it is — its name,
the file it was written in, the code it is made of — is in there. The obvious way
to get at it is to unpickle it, and that is the one way that is out of the
question: unpickling *is* executing, and the daemon is not the machine the user is
paying to run their code on.

So the stream is walked instead of loaded. :mod:`pickletools` disassembles a
pickle without evaluating it, and this simulates the stack it describes — pushes,
pops, the memo, tuples — far enough to see the code objects cloudpickle ships and
the function they were rebuilt into. Nothing here calls anything the pickle names.

What it is read for is identity. A blob's hash changes when the function moves
three lines down the file, or when a run captures a different id, neither of which
is a change to what the function *does* — so the shape is hashed over the code and
the names alone, and it is edits that move it.
"""

from __future__ import annotations

import asyncio
import hashlib
import pickletools
from collections.abc import Sequence
from dataclasses import dataclass

import lz4.frame

from skyward.shared.codec import THRESHOLD

_CODE = 6
"""Where each field sits in the arguments ``types.CodeType`` is rebuilt from."""
_NAMES = 8
_VARNAMES = 9
_FILENAME = 10
_NAME = 11
_QUALNAME = 12


@dataclass(frozen=True, slots=True)
class Reading:
    """What one pickled function turned out to be.

    Every field is absent rather than guessed when the stream does not say: a
    payload this cannot make sense of is read as an empty reading, never as a
    wrong one.
    """

    qualname: str | None = None
    """The dotted name the function was defined under, ``Trainer.train`` and all."""
    origin: str | None = None
    """The file it was written in, as the code object remembers it — the client's path."""
    shape: str | None = None
    """A digest of the code and the names, and of nothing that says where it lives
    or what it captured. Two uploads that share it are the same function run twice."""


async def read(blob: bytes) -> Reading:
    """Read a ``cloudpickle+lz4`` payload, on a thread when it is big enough to matter."""
    if len(blob) < THRESHOLD:
        return _read(blob)
    return await asyncio.to_thread(_read, blob)


def _read(blob: bytes) -> Reading:
    try:
        walked = _walk(lz4.frame.decompress(blob))
    except Exception:
        return Reading()

    if not walked.codes:
        return Reading()

    entry = walked.functions[0] if walked.functions else walked.codes[-1]
    return Reading(
        qualname=_string(entry, _QUALNAME) or _string(entry, _NAME),
        origin=_string(entry, _FILENAME),
        shape=_shape(walked.codes),
    )


def _shape(codes: Sequence[tuple[object, ...]]) -> str:
    """One digest over every code object the payload carries, in a fixed order.

    The file, the line and the constants are left out on purpose. A function that
    moved down the file, or that captured a different job id on a later run, is
    the same function; a function whose bytecode changed is not.
    """
    digest = hashlib.sha256()
    for args in sorted(codes, key=lambda args: (_string(args, _QUALNAME) or "", _strings(args, _NAMES))):
        digest.update(repr((_field(args, _CODE), _strings(args, _NAMES), _strings(args, _VARNAMES), _string(args, _QUALNAME))).encode())
    return digest.hexdigest()


def _field(args: tuple[object, ...], at: int) -> object:
    return args[at] if at < len(args) else None


def _string(args: tuple[object, ...], at: int) -> str | None:
    return value if isinstance(value := _field(args, at), str) else None


def _strings(args: tuple[object, ...], at: int) -> tuple[str, ...]:
    value = _field(args, at)
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, tuple) else ()


class _Mark:
    """The stack mark, and the stand-in for every value this does not need to know."""

    __slots__ = ()


MARK = _Mark()
OPAQUE = _Mark()
BUILTIN_TYPE = _Mark()
"""``cloudpickle._builtin_type``, the indirection every code object is rebuilt through."""
CODE_TYPE = _Mark()
"""``types.CodeType`` itself, which is what that indirection returns for ``"CodeType"``."""
MAKE_FUNCTION = _Mark()
"""``cloudpickle._make_function``, whose first argument is the code of a function being rebuilt."""

_FOLLOWED = {"_builtin_type": BUILTIN_TYPE, "CodeType": CODE_TYPE, "_make_function": MAKE_FUNCTION}


class _Code(tuple[object, ...]):
    """The arguments one code object is rebuilt from, kept on the stack so a function can claim it."""


@dataclass(slots=True)
class _Walked:
    codes: list[_Code]
    functions: list[_Code]
    """The code of each function rebuilt, in the order they are: the first is the one that was sent."""


_LITERALS = frozenset(
    {
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "UNICODE",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "SHORT_BINBYTES",
        "BINBYTES",
        "BINBYTES8",
        "BININT",
        "BININT1",
        "BININT2",
        "INT",
        "LONG",
        "LONG1",
        "LONG4",
    }
)
_REMEMBER = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_RECALL = frozenset({"GET", "BINGET", "LONG_BINGET"})
_PAIRS = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}


def _walk(raw: bytes) -> _Walked:
    """The stack pickle describes, run far enough to see code objects and the functions made of them.

    Values this has no use for are a single opaque marker, so the simulation costs
    one pass and holds nothing. What it does keep is strings, bytes and tuples of
    them — which is all a global's name or a code object's fields are made of.
    """
    stack: list[object] = []
    memo: dict[object, object] = {}
    walked = _Walked(codes=[], functions=[])

    for operation, argument, _ in pickletools.genops(raw):
        match operation.name:
            case "MARK":
                stack.append(MARK)
            case name if name in _REMEMBER:
                memo[argument] = stack[-1] if stack else OPAQUE
            case "MEMOIZE":
                memo[len(memo)] = stack[-1] if stack else OPAQUE
            case name if name in _RECALL:
                stack.append(memo.get(argument, OPAQUE))
            case "EMPTY_TUPLE":
                stack.append(())
            case name if name in _PAIRS:
                stack.append(_take(stack, _PAIRS[name]))
            case "TUPLE":
                stack.append(_to_mark(stack))
            case "STACK_GLOBAL":
                stack.append(_global(_take(stack, 2)))
            case "GLOBAL" if isinstance(argument, str) and " " in argument:
                stack.append(_global(tuple(argument.split(" ", 1))))
            case "REDUCE":
                stack.append(_reduce(_take(stack, 2), walked))
            case name:
                _pop(stack, operation)
                stack.extend(argument if name in _LITERALS else OPAQUE for _ in operation.stack_after)

    return walked


def _take(stack: list[object], count: int) -> tuple[object, ...]:
    taken = tuple(stack[-count:]) if len(stack) >= count else ()
    del stack[len(stack) - min(count, len(stack)) :]
    return taken


def _to_mark(stack: list[object]) -> tuple[object, ...]:
    at = len(stack) - 1
    while at >= 0 and stack[at] is not MARK:
        at -= 1
    taken = tuple(stack[at + 1 :])
    del stack[max(at, 0) :]
    return taken


def _pop(stack: list[object], operation: pickletools.OpcodeInfo) -> None:
    if pickletools.markobject in operation.stack_before:
        while stack and stack.pop() is not MARK:
            pass
        return
    for _ in operation.stack_before:
        if stack:
            stack.pop()


def _global(taken: tuple[object, ...]) -> object:
    match taken:
        case (str(module), str(attribute)):
            return _FOLLOWED.get(attribute, OPAQUE) if module.startswith(("cloudpickle", "types")) else OPAQUE
        case _:
            return OPAQUE


def _reduce(taken: tuple[object, ...], walked: _Walked) -> object:
    """What a call leaves behind, for the three calls this follows.

    Cloudpickle does not name ``types.CodeType`` directly — it calls its own
    ``_builtin_type("CodeType")`` — so that call is followed to the type, and
    calling *that* is a code object, kept whole on the stack. A function is rebuilt
    by ``_make_function`` with its code first, which is how the function that was
    sent is told apart from the code it merely carries: its nested functions are
    rebuilt before it, as constants of its own code.
    """
    match taken:
        case (callee, ("CodeType",)) if callee is BUILTIN_TYPE:
            return CODE_TYPE
        case (callee, tuple(arguments)) if callee is CODE_TYPE:
            code = _Code(arguments)
            walked.codes.append(code)
            return code
        case (callee, (_Code() as code, *_)) if callee is MAKE_FUNCTION:
            walked.functions.append(code)
        case _:
            pass
    return OPAQUE
