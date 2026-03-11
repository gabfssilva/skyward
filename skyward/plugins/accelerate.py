"""Hugging Face Accelerate plugin — distributed environment setup.

Sets up the distributed environment for accelerate-based training
(FSDP, DeepSpeed, mixed precision) by configuring environment
variables and initializing the torch process group.

Unlike ``accelerate launch``, this plugin works with Skyward's
long-running worker architecture — it sets env vars via
``around_process`` so that ``PartialState()`` (created lazily by
``TrainingArguments``) detects the correct distributed type.

Key insight: ``TrainingArguments.__post_init__`` accesses
``self.device`` **before** setting ``ACCELERATE_USE_FSDP``.  That
triggers ``PartialState()`` creation.  If the env var is missing at
that point the singleton locks to ``MULTI_GPU`` and FSDP never
activates.  This plugin ensures the var is present before any task
runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skyward.api.plugin import AccelerateConfig, Plugin

if TYPE_CHECKING:
    from skyward.api.model import Cluster
    from skyward.api.runtime import InstanceInfo
    from skyward.api.spec import Image


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


def _fsdp_env(fsdp_config: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {"ACCELERATE_USE_FSDP": "true"}
    for key, value in fsdp_config.items():
        env_key = _FSDP_KEY_MAP.get(key, key.upper())
        match value:
            case bool():
                env[env_key] = str(value).lower()
            case _:
                env[env_key] = str(value)
    return env


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


def _deepspeed_env(ds_config: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {"ACCELERATE_USE_DEEPSPEED": "true"}
    for key, value in ds_config.items():
        env_key = _DEEPSPEED_KEY_MAP.get(key, key.upper())
        match value:
            case bool():
                env[env_key] = str(value).lower()
            case _:
                env[env_key] = str(value)
    return env


def accelerate(config: AccelerateConfig | None = None) -> Plugin:
    """Hugging Face Accelerate plugin.

    Sets up the distributed environment for accelerate-based training
    (FSDP, DeepSpeed, mixed precision) by configuring environment
    variables and initializing the torch process group.

    The ``config`` dict mirrors the YAML structure produced by
    ``accelerate config``.  Skyward injects topology automatically
    (``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``).

    Parameters
    ----------
    config
        Accelerate settings.  Pass ``fsdp_config`` to enable FSDP,
        ``deepspeed_config`` for DeepSpeed.  Topology fields are
        injected from the cluster.
    """
    resolved: dict[str, Any] = dict(config or {})

    def transform(image: Image, _cluster: Cluster[Any]) -> Image:
        return replace(image, pip=(*image.pip, "accelerate"))

    @contextmanager
    def around(info: InstanceInfo) -> Iterator[None]:
        import torch as _torch
        import torch.distributed as dist

        from skyward.observability.logger import logger

        log = logger.bind(plugin="accelerate")

        if info.total_nodes < 2:
            yield
            return

        env: dict[str, str] = {
            "MASTER_ADDR": info.head_addr,
            "MASTER_PORT": str(info.head_port),
            "WORLD_SIZE": str(info.total_nodes),
            "RANK": str(info.node),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "NODE_RANK": str(info.node),
        }

        fsdp_config = resolved.get("fsdp")
        deepspeed_config = resolved.get("deepspeed")

        if fsdp_config:
            env.update(_fsdp_env(fsdp_config))
        elif deepspeed_config:
            env.update(_deepspeed_env(deepspeed_config))

        mixed_precision = resolved.get("mixed_precision")
        if mixed_precision:
            env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

        for key, value in env.items():
            os.environ[key] = value

        backend = "nccl" if _torch.cuda.is_available() else "gloo"  # type: ignore[reportAttributeAccessIssue]
        log.info(
            "Initializing process group: backend={be}, rank={rank}, world_size={ws}",
            be=backend, rank=info.node, ws=info.total_nodes,
        )
        dist.init_process_group(backend=backend, init_method="env://")
        try:
            yield
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    return (
        Plugin.create("accelerate")
        .with_image_transform(transform)
        .with_around_process(around)
    )
