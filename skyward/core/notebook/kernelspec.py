"""The kernelspec that points Jupyter at a Skyward compute.

One ``kernel.json`` per compute, naming the ``skyward`` provisioner and the
compute it should attach to. The ``argv`` in it is inert — a custom provisioner
never substitutes ``{connection_file}`` — and everything that matters is in the
``kernel_provisioner`` metadata.

Installing goes through ``jupyter_client``'s ``KernelSpecManager`` so the spec
lands wherever that Jupyter looks; ``directory`` bypasses it and writes the tree
directly, which is how a test reads back what was written.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

KERNEL_NAME_PREFIX = "skyward-"

__all__ = ["KERNEL_NAME_PREFIX", "install_kernelspec", "kernel_json", "kernel_name", "remove_kernelspec"]


def kernel_name(compute: str) -> str:
    """The kernel name a compute installs under."""
    return f"{KERNEL_NAME_PREFIX}{compute}"


def kernel_json(compute: str, url: str = "") -> dict[str, Any]:
    """Build the ``kernel.json`` binding a kernel to a compute.

    Parameters
    ----------
    compute
        The compute to attach to, by name or by id.
    url
        The daemon to reach it through. Empty leaves the provisioner to resolve
        ``SKYWARD_URL``, or to run the daemon in the Jupyter process.

    Returns
    -------
    dict[str, Any]
        The kernelspec content.
    """
    config: dict[str, str] = {"compute": compute} | ({"url": url} if url else {})
    return {
        "argv": ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"Skyward ({compute})",
        "language": "python",
        "interrupt_mode": "message",
        "metadata": {
            "kernel_provisioner": {
                "provisioner_name": "skyward",
                "config": config,
            },
        },
    }


def install_kernelspec(compute: str, url: str = "", directory: Path | None = None) -> str:
    """Install the kernelspec for a compute and return its kernel name.

    Parameters
    ----------
    compute
        The compute to attach to, by name or by id.
    url
        The daemon URL to record in the spec.
    directory
        Write the spec tree here instead of installing it user-level.
    """
    name = kernel_name(compute)
    content = json.dumps(kernel_json(compute, url), indent=2)

    if directory is not None:
        spec = directory / name
        spec.mkdir(parents=True, exist_ok=True)
        (spec / "kernel.json").write_text(content)
        return name

    from jupyter_client.kernelspec import KernelSpecManager

    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / name
        spec.mkdir()
        (spec / "kernel.json").write_text(content)
        KernelSpecManager().install_kernel_spec(str(spec), kernel_name=name, user=True)
    return name


def remove_kernelspec(compute: str, directory: Path | None = None) -> str:
    """Remove a compute's kernelspec and return the kernel name that went.

    Parameters
    ----------
    compute
        The compute the spec was installed for.
    directory
        Remove from this tree instead of the user-level location.
    """
    name = kernel_name(compute)

    if directory is not None:
        shutil.rmtree(directory / name)
        return name

    from jupyter_client.kernelspec import KernelSpecManager

    KernelSpecManager().remove_kernel_spec(name)
    return name
