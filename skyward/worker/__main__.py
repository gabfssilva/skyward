"""``python -m skyward.worker``: the worker, with its module under its own name.

An actor type is named after the module its body lives in. Run as the main module,
``skyward.worker.worker`` would name its actors ``__main__:execution`` on the node,
while the daemon asks for ``skyward.worker.worker:execution``, and the node would not
know what it was asked for.
"""

from skyward.worker.worker import cli

cli()
