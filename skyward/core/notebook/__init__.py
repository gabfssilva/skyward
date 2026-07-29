"""The Skyward Jupyter kernel — client side.

A local Jupyter with its kernel on a Skyward machine: the kernelspec names the
compute, the provisioner starts the kernel there and bridges its channels back.
Client-side only, and not part of ``import skyward as sky`` — a machine runs
functions and has no business importing a kernel provisioner to do it.
"""

from __future__ import annotations

from skyward.core.notebook.kernelspec import install_kernelspec, kernel_json, kernel_name, remove_kernelspec
from skyward.core.notebook.provisioner import SkywardKernelProvisioner

__all__ = ["SkywardKernelProvisioner", "install_kernelspec", "kernel_json", "kernel_name", "remove_kernelspec"]
