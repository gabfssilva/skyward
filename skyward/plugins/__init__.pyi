"""Composable plugin system for extending pool behavior."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal

from skyward.api.model import Cluster
from skyward.api.runtime import InstanceInfo
from skyward.api.spec import Image

class Plugin:
    """Composable hook system for extending pool behavior.

    Plugins attach to six lifecycle points via a frozen builder API.
    Pass plugins at pool construction: ``ComputePool(..., plugins=[...])``.

    Use ``Plugin.create("name")`` and chain ``.with_*()`` methods, or
    use the built-in factories (``sky.plugins.torch()``, etc.).

    Examples
    --------
    >>> # Built-in plugins
    >>> with sky.ComputePool(
    ...     provider=sky.AWS(),
    ...     accelerator="A100",
    ...     plugins=[sky.plugins.torch(), sky.plugins.keras()],
    ... ) as pool:
    ...     result = train(data) >> pool

    >>> # Custom plugin
    >>> plugin = (
    ...     Plugin.create("my-plugin")
    ...     .with_image_transform(add_deps)
    ...     .with_decorator(timing_wrapper)
    ... )
    """

    @property
    def name(self) -> str:
        """Plugin identifier."""
        ...
    @staticmethod
    def create(name: str) -> Plugin:
        """Create an empty plugin with a name.

        Parameters
        ----------
        name
            Plugin identifier.

        Returns
        -------
        Plugin
            Empty plugin ready for hook attachment via ``.with_*()`` methods.
        """
        ...
    def with_image_transform(
        self, transform: Callable[[Image, Cluster[Any]], Image],
    ) -> Plugin:
        """Attach an image transform hook.

        Parameters
        ----------
        transform
            Receives ``(Image, Cluster)`` and returns a modified ``Image``.
            Called before bootstrap to inject dependencies.

        Returns
        -------
        Plugin
            New plugin instance with the transform attached.
        """
        ...
    def with_bootstrap(
        self, factory: Callable[[Cluster[Any]], tuple[Any, ...]],
    ) -> Plugin:
        """Attach a bootstrap factory hook.

        Parameters
        ----------
        factory
            Receives ``Cluster`` and returns shell ops to run after
            the image-driven bootstrap phases.

        Returns
        -------
        Plugin
            New plugin instance with the bootstrap factory attached.
        """
        ...
    def with_decorator(
        self, decorate: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> Plugin:
        """Attach a per-task decorator hook.

        Parameters
        ----------
        decorate
            Standard Python decorator applied to each ``@sky.function``
            at execution time on the remote worker.

        Returns
        -------
        Plugin
            New plugin instance with the decorator attached.
        """
        ...
    def with_around_app(
        self, around: Callable[[InstanceInfo], AbstractContextManager[None]],
    ) -> Plugin:
        """Attach a worker lifecycle hook.

        Parameters
        ----------
        around
            Context manager entered once in the main worker process.

        Returns
        -------
        Plugin
            New plugin instance with the lifecycle hook attached.
        """
        ...
    def with_around_process(
        self, around: Callable[[InstanceInfo], AbstractContextManager[None]],
    ) -> Plugin:
        """Attach a subprocess lifecycle hook.

        Parameters
        ----------
        around
            Context manager entered once per subprocess when
            ``executor="process"``.

        Returns
        -------
        Plugin
            New plugin instance with the subprocess hook attached.
        """
        ...
    def with_around_client(
        self, around: Callable[..., AbstractContextManager[None]],
    ) -> Plugin:
        """Attach a client-side lifecycle hook.

        Parameters
        ----------
        around
            Context manager receiving ``(ComputePool, Cluster)`` and
            entered at pool ``__enter__`` on the client.

        Returns
        -------
        Plugin
            New plugin instance with the client hook attached.
        """
        ...

# ── Plugin factories ──────────────────────────────────────────

def torch(
    *,
    backend: Literal["nccl", "gloo"] | None = None,
    cuda: str = "cu128",
    version: str = "latest",
    vision: str | None = None,
    audio: str | None = None,
) -> Plugin:
    """PyTorch distributed plugin.

    Install PyTorch with the specified CUDA backend and configure
    ``torch.distributed`` environment variables on each node.

    Parameters
    ----------
    backend
        Distributed backend. ``None`` auto-detects (NCCL for GPU,
        Gloo for CPU).
    cuda
        CUDA compute platform (e.g., ``"cu128"``, ``"cu124"``).
    version
        PyTorch version. ``"latest"`` uses the newest stable release.
    vision
        torchvision version. ``None`` skips installation.
    audio
        torchaudio version. ``None`` skips installation.
    """
    ...

def jax(*, cuda: str = "cu124") -> Plugin:
    """JAX plugin.

    Install JAX with GPU support and configure multi-node communication.

    Parameters
    ----------
    cuda
        CUDA compute platform.
    """
    ...

def keras(*, backend: Literal["jax", "torch", "tensorflow"] = "jax") -> Plugin:
    """Keras 3 plugin.

    Install Keras and set the ``KERAS_BACKEND`` environment variable.

    Parameters
    ----------
    backend
        Keras backend framework.
    """
    ...

def cuml(*, cuda: str = "cu12") -> Plugin:
    """RAPIDS cuML plugin.

    Install cuML for GPU-accelerated scikit-learn-compatible algorithms.

    Parameters
    ----------
    cuda
        CUDA compute platform.
    """
    ...

def huggingface(token: str) -> Plugin:
    """HuggingFace Hub plugin.

    Set ``HF_TOKEN`` environment variable for authenticated model access.

    Parameters
    ----------
    token
        HuggingFace API token.
    """
    ...

def joblib(*, version: str | None = None) -> Plugin:
    """Joblib parallel backend plugin.

    Install joblib and register Skyward's distributed backend.

    Parameters
    ----------
    version
        Joblib version. ``None`` uses the latest.
    """
    ...

def sklearn(*, version: str | None = None) -> Plugin:
    """scikit-learn plugin.

    Install scikit-learn on remote workers.

    Parameters
    ----------
    version
        scikit-learn version. ``None`` uses the latest.
    """
    ...

def mig(profile: str) -> Plugin:
    """NVIDIA MIG (Multi-Instance GPU) partitioning plugin.

    Configure GPU partitioning during bootstrap.

    Parameters
    ----------
    profile
        MIG profile string (e.g., ``"3g.40gb"``, ``"1g.10gb"``).
    """
    ...

def mps(
    *,
    active_thread_percentage: int | None = None,
    pinned_memory_limit: str | None = None,
) -> Plugin:
    """NVIDIA MPS (Multi-Process Service) plugin.

    Enable GPU sharing between concurrent tasks via MPS.

    Parameters
    ----------
    active_thread_percentage
        Percentage of GPU threads available to each client.
    pinned_memory_limit
        Per-client pinned memory limit (e.g., ``"4G"``).
    """
    ...

# ── Convenience constructors ─────────────────────────────────

def around_app(
    name: str,
    around: Callable[[InstanceInfo], AbstractContextManager[None]],
) -> Plugin:
    """Create a plugin with only a worker lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_app(around)``.
    """
    ...

def around_client(
    name: str,
    around: Callable[..., AbstractContextManager[None]],
) -> Plugin:
    """Create a plugin with only a client-side lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_client(around)``.
    """
    ...

def around_process(
    name: str,
    around: Callable[[InstanceInfo], AbstractContextManager[None]],
) -> Plugin:
    """Create a plugin with only a subprocess lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_process(around)``.
    """
    ...

__all__ = [
    "Plugin",
    "torch",
    "jax",
    "keras",
    "cuml",
    "huggingface",
    "joblib",
    "sklearn",
    "mig",
    "mps",
    "around_app",
    "around_client",
    "around_process",
]
