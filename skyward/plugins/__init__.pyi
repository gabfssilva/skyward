"""Composable plugin system for extending pool behavior."""

from typing import Literal

from skyward.api.plugin import Plugin as Plugin
from skyward.api.plugin import around_app as around_app
from skyward.api.plugin import around_client as around_client
from skyward.api.plugin import around_process as around_process

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
