"""Plugin type for declarative third-party integrations.

A ``Plugin`` bundles environment setup, bootstrap ops, worker lifecycle
hooks, client-side hooks, and per-task wrapping into a single composable
unit.  Built-in plugins (``sky.plugins.torch``, ``.keras``, etc.) are
constructed from this type, and users can create custom plugins via
the ``Plugin.create("name").with_*()`` builder pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.pool import Pool
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image
    from skyward.providers.bootstrap import Op

type ImageTransform[S] = Callable[[Image, Cluster[S]], Image]
"""Hook that modifies the ``Image`` before bootstrap script generation.

Receives the current image and cluster metadata, returns a modified
copy.  Typically used to inject provider-specific packages or env vars.
"""

type BootstrapFactory[S] = Callable[[Cluster[S]], tuple[Op, ...]]
"""Hook that produces extra shell ops appended after bootstrap phases.

Receives the cluster state and returns a tuple of shell operations
(strings, callables, or nested lists) that are appended to the
bootstrap script.
"""

type TaskDecorator[**P, R] = Callable[[Callable[P, R]], Callable[P, R]]
"""Classic Python decorator applied to each ``@sky.function`` at execution time.

Wraps the user's function on the remote worker before it runs.
Useful for injecting runtime context, error handling, or profiling.
"""

type AppLifecycle = Callable[["InstanceInfo"], AbstractContextManager[None]]
"""Worker lifecycle context manager, entered once in the main worker process.

Receives ``InstanceInfo`` and returns a context manager whose
``__enter__`` runs at worker startup and ``__exit__`` at shutdown.
"""

type ProcessLifecycle = Callable[["InstanceInfo"], AbstractContextManager[None]]
"""Subprocess lifecycle context manager, entered once per executor subprocess.

Only relevant when ``executor="process"``.  Lazy — enters on the first
task execution in each subprocess, after env vars are propagated.
"""

type ClientLifecycle[S] = Callable[["Pool", "Cluster[S]"], AbstractContextManager[None]]
"""Client-side lifecycle context manager, entered at pool ``__enter__``.

Receives the pool and cluster state.  Runs on the client machine
(not on remote workers).  Useful for setting up local distributed
backends or registering cleanup handlers.
"""


@dataclass(frozen=True, slots=True)
class LaunchContext:
    """Context available to launcher transform hooks.

    Parameters
    ----------
    node_id
        Index of this node in the cluster (0 = head).
    num_nodes
        Total number of nodes in the cluster.
    head_addr
        IP address of the head node.
    casty_port
        Port for the Casty actor system.
    private_ip
        Private IP of this node.
    python_bin
        Path to the Python binary inside the remote venv.
    venv_dir
        Path to the remote venv directory.
    concurrency
        Number of workers per node.
    executor
        Execution backend (``"thread"`` or ``"process"``).
    """

    node_id: int
    num_nodes: int
    head_addr: str
    casty_port: int
    private_ip: str
    python_bin: str
    venv_dir: str
    concurrency: int
    executor: str


@dataclass(frozen=True, slots=True)
class LaunchCommand:
    """Structured worker launch command.

    Parameters
    ----------
    launcher
        Wrapper tool that launches the worker process
        (e.g. ``"accelerate launch --num_machines 4"``).
        Empty string means no wrapper — Python runs the entrypoint directly.
    entrypoint
        Worker script or ``-c`` invocation.
    args
        CLI arguments for the worker (``--node-id``, ``--port``, etc.).
    """

    launcher: str = ""
    entrypoint: str = ""
    args: tuple[str, ...] = ()

    def render(self) -> str:
        """Render to a single shell command string."""
        parts = filter(None, (self.launcher, self.entrypoint, " ".join(self.args)))
        return " ".join(parts)


type LauncherTransform = Callable[[LaunchCommand, LaunchContext], LaunchCommand]
"""Hook that transforms the worker launch command.

Receives the default ``LaunchCommand`` and ``LaunchContext``, returns a
modified command.  Typically used to prepend a process wrapper like
``accelerate launch`` or ``torchrun``.
"""


@dataclass(frozen=True, slots=True)
class Plugin:
    """Declarative third-party integration bundle.

    Bundles environment setup, bootstrap ops, worker lifecycle hooks,
    client-side hooks, and per-task wrapping into a single composable unit.

    Parameters
    ----------
    name
        Plugin identifier.
    transform
        (Image, Cluster[S]) -> Image transformer. Receives current Image and
        cluster metadata, returns modified copy.
    bootstrap
        Factory that receives Cluster[S] and returns extra shell ops appended
        after Image-driven bootstrap phases.
    decorate
        Classic Python decorator: (fn) -> fn. Wraps each @sky.function
        function at execution time on the remote worker.
    around_app
        Worker lifecycle context manager: InstanceInfo -> ContextManager[None].
        Entered once in the main worker process.
    around_process
        Subprocess lifecycle context manager: InstanceInfo -> ContextManager[None].
        Entered once per subprocess when executor="process". Lazy -- enters on
        the first task execution in each subprocess, after env vars are propagated.
    around_client
        Client lifecycle context manager: (Pool, Cluster[S]) -> ContextManager[None].
    launcher
        (LaunchCommand, LaunchContext) -> LaunchCommand transformer. Wraps the
        worker launch command with a process launcher like accelerate or torchrun.
    """

    name: str
    transform: ImageTransform[Any] | None = None
    bootstrap: BootstrapFactory[Any] | None = None
    decorate: TaskDecorator | None = None
    around_app: AppLifecycle | None = None
    around_process: ProcessLifecycle | None = None
    around_client: ClientLifecycle[Any] | None = None
    launcher: LauncherTransform | None = None

    @staticmethod
    def create(name: str) -> Plugin:
        """Create an empty plugin with just a name.

        Use the ``with_*`` builder methods to attach hooks.

        Parameters
        ----------
        name
            Plugin identifier.

        Returns
        -------
        Plugin
            Empty plugin ready for hook attachment.

        Examples
        --------
        >>> plugin = (
        ...     Plugin.create("my-plugin")
        ...     .with_image_transform(add_deps)
        ...     .with_decorator(wrap_fn)
        ... )
        """
        return Plugin(name=name)

    def with_image_transform[S](self, transform: ImageTransform[S]) -> Plugin:
        """Attach an image transform hook.

        Parameters
        ----------
        transform
            ``(Image, Cluster[S]) -> Image`` that modifies the image
            before bootstrap.

        Returns
        -------
        Plugin
            New plugin instance with the transform attached.
        """
        return replace(self, transform=transform)

    def with_bootstrap[S](self, factory: BootstrapFactory[S]) -> Plugin:
        """Attach a bootstrap factory hook.

        Parameters
        ----------
        factory
            ``Cluster[S] -> tuple[Op, ...]`` returning extra shell ops
            appended after image-driven bootstrap phases.

        Returns
        -------
        Plugin
            New plugin instance with the bootstrap factory attached.
        """
        return replace(self, bootstrap=factory)

    def with_decorator[**P, R](self, decorate: TaskDecorator[P, R]) -> Plugin:
        """Attach a per-task decorator hook.

        Parameters
        ----------
        decorate
            Classic Python decorator ``(fn) -> fn`` applied to each
            ``@sky.function`` at execution time on the remote worker.

        Returns
        -------
        Plugin
            New plugin instance with the decorator attached.
        """
        return replace(self, decorate=decorate)

    def with_around_app(self, around: AppLifecycle) -> Plugin:
        """Attach a worker lifecycle hook.

        Parameters
        ----------
        around
            ``InstanceInfo -> ContextManager[None]`` entered once
            in the main worker process.

        Returns
        -------
        Plugin
            New plugin instance with the lifecycle hook attached.
        """
        return replace(self, around_app=around)

    def with_around_process(self, around: ProcessLifecycle) -> Plugin:
        """Attach a subprocess lifecycle hook.

        Parameters
        ----------
        around
            ``InstanceInfo -> ContextManager[None]`` entered once
            per subprocess when ``executor="process"``.

        Returns
        -------
        Plugin
            New plugin instance with the subprocess hook attached.
        """
        return replace(self, around_process=around)

    def with_around_client[S](self, around: ClientLifecycle[S]) -> Plugin:
        """Attach a client-side lifecycle hook.

        Parameters
        ----------
        around
            ``(Pool, Cluster[S]) -> ContextManager[None]``
            entered on the client at pool ``__enter__``.

        Returns
        -------
        Plugin
            New plugin instance with the client hook attached.
        """
        return replace(self, around_client=around)

    def with_launcher(self, launcher: LauncherTransform) -> Plugin:
        """Attach a worker launch command transform.

        Parameters
        ----------
        launcher
            ``(LaunchCommand, LaunchContext) -> LaunchCommand`` that wraps
            the worker startup with a process launcher (e.g. ``accelerate
            launch``, ``torchrun``).

        Returns
        -------
        Plugin
            New plugin instance with the launcher hook attached.
        """
        return replace(self, launcher=launcher)


def chain_decorators[**P, R](
    fn: Callable[P, R],
    decorators: list[TaskDecorator[P, R]],
) -> Callable[P, R]:
    """Chain multiple plugin decorators around a function.

    Applies decorators in reverse order so that the first decorator
    in the list becomes the outermost wrapper (i.e., runs first).
    An empty list returns *fn* unchanged.

    Parameters
    ----------
    fn
        The function to wrap.
    decorators
        Decorators to apply, ordered outermost-first.

    Returns
    -------
    Callable[P, R]
        The fully-wrapped function.
    """
    return reduce(lambda f, d: d(f), reversed(decorators), fn)

def around_app(name: str, around: AppLifecycle) -> Plugin:
    """Create a plugin with only a worker lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_app(around)``.

    Parameters
    ----------
    name
        Plugin identifier.
    around
        Worker lifecycle context manager.

    Returns
    -------
    Plugin
        Plugin with the ``around_app`` hook set.
    """
    return Plugin(name=name, around_app=around)

def around_client[S](name: str, around: ClientLifecycle[S]) -> Plugin:
    """Create a plugin with only a client-side lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_client(around)``.

    Parameters
    ----------
    name
        Plugin identifier.
    around
        Client-side lifecycle context manager.

    Returns
    -------
    Plugin
        Plugin with the ``around_client`` hook set.
    """
    return Plugin(name=name, around_client=around)

def around_process(name: str, around: ProcessLifecycle) -> Plugin:
    """Create a plugin with only a subprocess lifecycle hook.

    Shortcut for ``Plugin.create(name).with_around_process(around)``.

    Parameters
    ----------
    name
        Plugin identifier.
    around
        Subprocess lifecycle context manager.

    Returns
    -------
    Plugin
        Plugin with the ``around_process`` hook set.
    """
    return Plugin(name=name, around_process=around)


class FsdpConfig(TypedDict, total=False):
    """FSDP section of the accelerate config.

    Keys omit the ``fsdp_`` prefix — the plugin maps them to the
    correct ``FSDP_*`` environment variables that accelerate reads.
    """

    sharding_strategy: (
        Literal[
            "FULL_SHARD",
            "SHARD_GRAD_OP",
            "NO_SHARD",
            "HYBRID_SHARD",
            "HYBRID_SHARD_ZERO2",
        ]
        | str
    )
    auto_wrap_policy: Literal["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"] | str
    transformer_layer_cls_to_wrap: str
    backward_prefetch: Literal["BACKWARD_PRE", "BACKWARD_POST"] | str
    state_dict_type: (
        Literal["FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"] | str
    )
    forward_prefetch: bool
    use_orig_params: bool
    cpu_ram_efficient_loading: bool
    sync_module_states: bool
    offload_params: bool
    min_num_params: int
    activation_checkpointing: bool


class DeepSpeedConfig(TypedDict, total=False):
    """DeepSpeed section of the accelerate config.

    Keys omit prefixes — the plugin maps them to the correct
    ``ACCELERATE_DEEPSPEED_*`` environment variables.
    """

    config_file: str
    zero_stage: Literal[0, 1, 2, 3] | int
    offload_optimizer_device: Literal["none", "cpu", "nvme"] | str
    offload_param_device: Literal["none", "cpu", "nvme"] | str
    offload_optimizer_nvme_path: str
    offload_param_nvme_path: str
    gradient_accumulation_steps: int
    gradient_clipping: float
    zero3_save_16bit_model: bool


class AccelerateConfig(TypedDict, total=False):
    """Accelerate config in the same structure as the YAML produced by ``accelerate config``.

    Skyward injects topology fields automatically (``num_machines``,
    ``machine_rank``, ``main_process_ip``, ``main_process_port``,
    ``compute_environment``).  ``distributed_type`` is inferred from
    the presence of ``fsdp`` or ``deepspeed`` when not
    set explicitly.

    ``num_processes`` means **total** across all nodes (matching
    accelerate semantics).  Defaults to the cluster node count
    (1 GPU per node).
    """

    distributed_type: Literal["FSDP", "DEEPSPEED", "MULTI_GPU", "NO"] | str
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] | str
    num_processes: int
    num_cpu_threads_per_process: int
    gpu_ids: str
    same_network: bool
    debug: bool
    downcast_bf16: str
    rdzv_backend: str

    fsdp: FsdpConfig
    deepspeed: DeepSpeedConfig
