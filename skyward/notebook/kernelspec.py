"""Skyward remote-kernel kernelspec builder + installer.

Builds the ``kernel.json`` binding a Jupyter kernel to a Skyward session
via the custom ``skyward`` provisioner and installs/removes it through
``jupyter_client``'s :class:`~jupyter_client.kernelspec.KernelSpecManager`
(user-level).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

KERNEL_NAME_PREFIX = "skyward-"

__all__ = ["KERNEL_NAME_PREFIX", "install_kernelspec", "kernel_json", "remove_kernelspec"]


def kernel_json(session: str) -> dict[str, Any]:
    """Build the ``kernel.json`` for a session.

    Parameters
    ----------
    session
        The Skyward session/pool name.

    Returns
    -------
    dict[str, Any]
        The kernelspec content. ``argv`` is inert — a custom provisioner
        never substitutes ``{connection_file}`` — and the real launch command
        is assembled by the provisioner at ``pre_launch``.
    """
    return {
        "argv": ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"Skyward ({session})",
        "language": "python",
        "interrupt_mode": "message",
        "metadata": {
            "kernel_provisioner": {
                "provisioner_name": "skyward",
                "config": {"session": session},
            },
        },
    }


def install_kernelspec(session: str) -> str:
    """Install the Skyward kernelspec for a session (user-level).

    Parameters
    ----------
    session
        The Skyward session/pool name.

    Returns
    -------
    str
        The installed kernel name (``skyward-<session>``).
    """
    from jupyter_client.kernelspec import KernelSpecManager

    name = f"{KERNEL_NAME_PREFIX}{session}"
    with tempfile.TemporaryDirectory() as tmp:
        spec_dir = Path(tmp) / name
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_json(session), indent=2))
        KernelSpecManager().install_kernel_spec(str(spec_dir), kernel_name=name, user=True)
    return name


def remove_kernelspec(session: str) -> None:
    """Remove the Skyward kernelspec for a session.

    Parameters
    ----------
    session
        The Skyward session/pool name.
    """
    from jupyter_client.kernelspec import KernelSpecManager

    KernelSpecManager().remove_kernel_spec(f"{KERNEL_NAME_PREFIX}{session}")
