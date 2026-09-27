"""A whole file, run on the node as ``__main__``.

``sky compute run`` and ``sky run`` hand a node a script rather than a function.
What crosses the wire is :func:`run` bound to the script's text and argv, and it
is a function of this module on purpose: a function of an installed module
travels by reference, and the node has its own copy of this one. A closure would
travel by value, as bytecode, and bytecode is only good on the interpreter that
compiled it — a script sent from a 3.13 terminal to a node its header put on 3.12
would kill the worker unpickling it.
"""

from __future__ import annotations

import sys
import traceback


def run(source: str, argv: tuple[str, ...]) -> int:
    """Execute ``source`` as ``__main__``, with ``argv`` as ``sys.argv``, and answer with its exit status.

    ``sys.exit`` is the script saying how it ended, so it is a status and not a
    failure; an exception is printed the way the interpreter would print it and
    reads as ``1``. What is not an ``Exception`` is left to go through: the worker
    stops an attempt by raising one of those inside it, and a script that caught
    it would report a stopped run as one that failed on its own.
    """
    held, sys.argv = sys.argv, list(argv)
    try:
        exec(compile(source, argv[0], "exec"), {"__name__": "__main__", "__file__": argv[0]})
    except SystemExit as stop:
        match stop.code:
            case None:
                return 0
            case int(code):
                return code
            case message:
                print(message, file=sys.stderr)
                return 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        sys.argv = held
    return 0


__all__ = ["run"]
