"""Execute @sky.main functions from the sidecar."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def run_main(file: str, fn_name: str, args: dict[str, Any], pool: str) -> Any:
    """Import a module, find the ``@sky.main`` function, and call it within a pool context.

    Connects to the daemon pool by name via ``DaemonPool``,
    setting the active pool context so ``>> sky`` / ``@ sky`` works.
    """
    path = Path(file).resolve()
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    func = getattr(module, fn_name, None)
    if func is None:
        raise AttributeError(f"Function '{fn_name}' not found in {file}")
    if not getattr(func, "__sky_main__", False):
        raise ValueError(f"Function '{fn_name}' is not decorated with @sky.main")

    from skyward.daemon.pool import DaemonPool

    daemon_pool = DaemonPool(name=pool)
    with daemon_pool:
        return func(**args)
