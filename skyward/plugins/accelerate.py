"""Hugging Face Accelerate, its distributed environment already set by the first task.

Not ``accelerate launch``. That wrapper spawns the training process itself, one per
rank, and exits when it returns — which is the opposite of a worker that is spawned
once and lives to take many tasks. So the plugin does what the launcher would have
done and no more: it sets the environment ``accelerate`` reads and forms the group,
in the process that runs the task, before that task runs.

The variable that matters most is set first because ``accelerate`` reads it earliest.
``TrainingArguments.__post_init__`` touches ``self.device`` before it looks at
``ACCELERATE_USE_FSDP``, and that first touch constructs the ``PartialState``
singleton and freezes the distributed type. If the FSDP flag is not already in the
environment by then the singleton locks to plain multi-GPU and FSDP never turns on.
Setting the whole environment before the first task is what keeps that from happening.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image
from skyward.runtime.api import Info

PORT = 29500

_lock = threading.Lock()
_joined = False
"""Whether this process has already formed the group. Process-global, because a
process has one default group and forming it twice is an error, not a no-op."""


_FSDP_KEY_MAP: dict[str, str] = {
    "sharding_strategy": "FSDP_SHARDING_STRATEGY",
    "auto_wrap_policy": "FSDP_AUTO_WRAP_POLICY",
    "transformer_layer_cls_to_wrap": "FSDP_TRANSFORMER_CLS_TO_WRAP",
    "backward_prefetch": "FSDP_BACKWARD_PREFETCH",
    "state_dict_type": "FSDP_STATE_DICT_TYPE",
    "offload_params": "FSDP_OFFLOAD_PARAMS",
    "sync_module_states": "FSDP_SYNC_MODULE_STATES",
    "use_orig_params": "FSDP_USE_ORIG_PARAMS",
    "cpu_ram_efficient_loading": "FSDP_CPU_RAM_EFFICIENT_LOADING",
    "forward_prefetch": "FSDP_FORWARD_PREFETCH",
    "activation_checkpointing": "FSDP_ACTIVATION_CHECKPOINTING",
    "min_num_params": "FSDP_MIN_NUM_PARAMS",
}

_DEEPSPEED_KEY_MAP: dict[str, str] = {
    "zero_stage": "ACCELERATE_DEEPSPEED_ZERO_STAGE",
    "offload_optimizer_device": "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE",
    "offload_param_device": "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE",
    "offload_optimizer_nvme_path": "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH",
    "offload_param_nvme_path": "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH",
    "zero3_save_16bit_model": "ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL",
    "config_file": "ACCELERATE_DEEPSPEED_CONFIG_FILE",
    "gradient_accumulation_steps": "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
    "gradient_clipping": "ACCELERATE_GRADIENT_CLIPPING",
}


class Accelerate(Plugin, frozen=True):
    """Install accelerate, and set its distributed environment before the first task.

    The rendezvous is rank zero, because ``accelerate`` insists on being told where
    the main process is — the compute has no head, and this is the convention that
    satisfies a library that believes it does.

    The group is formed in the process that runs the task, and on that process's
    first task, not at worker start. ``init_process_group`` is a collective — every
    node blocks in it until the last one arrives — and the node that arrives there
    must be the one that will run the collective code afterwards. Under a subprocess
    executor that is the child, not the worker; forming it in the worker would leave
    the child holding a group it never joined. Doing it on the first task, once and
    under a lock, is what lets the same plugin serve either executor.

    It composes with :class:`~skyward.plugins.torch.Torch`: the group is only formed
    when nobody has formed one yet, so a user may list both and pay for the group once.

    Attributes
    ----------
    config : dict[str, Any]
        The accelerate settings, in the same shape as the YAML ``accelerate config``
        writes. ``fsdp`` turns on FSDP, ``deepspeed`` turns on DeepSpeed,
        ``mixed_precision`` sets the dtype; topology (rank, world size, address) is
        injected from the node, not read from here. ``backend`` picks the process
        group backend, defaulting to ``nccl``.
    """

    kind: ClassVar[str] = "accelerate"
    collective: ClassVar[bool] = True

    config: dict[str, Any] = {}

    def image(self, image: Image) -> Image:
        from msgspec.structs import replace

        return replace(image, pip=(*image.pip, "accelerate"))

    def run[T](self, call: Callable[[], T], info: Info) -> T:
        _join(self.config, info)
        return call()


def _accelerate_env(config: dict[str, Any]) -> dict[str, str]:
    """The ``ACCELERATE_*``/``FSDP_*`` variables accelerate reads, from the config."""
    env: dict[str, str] = {}

    match config.get("fsdp"), config.get("deepspeed"):
        case dict() as fsdp, _:
            env["ACCELERATE_USE_FSDP"] = "true"
            for key, value in fsdp.items():
                env[_FSDP_KEY_MAP.get(key) or str(key).upper()] = _as_env(value)
        case _, dict() as deepspeed:
            env["ACCELERATE_USE_DEEPSPEED"] = "true"
            for key, value in deepspeed.items():
                env[_DEEPSPEED_KEY_MAP.get(key) or str(key).upper()] = _as_env(value)
        case _:
            pass

    if mixed_precision := config.get("mixed_precision"):
        env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    return env


def _as_env(value: Any) -> str:
    match value:
        case bool():
            return str(value).lower()
        case _:
            return str(value)


def _join(config: dict[str, Any], info: Info) -> None:
    """Set the accelerate environment and form the default group, once for this process.

    The double check keeps the import and the collective off the fast path once the
    group is up: every task after the first sees the flag and calls straight through.
    A single node is nobody to rendezvous with, so it is left as it is.
    """
    global _joined
    if _joined or info.nodes < 2:
        return

    with _lock:
        if _joined:
            return

        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = info.head
        os.environ["MASTER_PORT"] = str(PORT)
        os.environ["WORLD_SIZE"] = str(info.nodes)
        os.environ["RANK"] = str(info.rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["NODE_RANK"] = str(info.rank)
        os.environ.update(_accelerate_env(config))

        backend = config.get("backend", "nccl")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://", rank=info.rank, world_size=info.nodes)
        _joined = True
