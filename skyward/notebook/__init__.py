"""Skyward remote Jupyter kernel — client-side package.

A local Jupyter, remote kernel: every cell runs on the Skyward head node.
Client-side only — not part of the ``import skyward as sky`` public surface.
"""

from __future__ import annotations

from .kernelspec import install_kernelspec, kernel_json, remove_kernelspec
from .provisioner import SkywardKernelProvisioner

__all__ = ["SkywardKernelProvisioner", "install_kernelspec", "kernel_json", "remove_kernelspec"]
