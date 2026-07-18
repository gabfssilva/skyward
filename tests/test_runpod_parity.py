"""RunPod keeps the v1 knobs: the factory carries them and the adapter honors them.

The pool passes a provider config through untouched, so the two halves that matter
are testable without a cloud: the factory turns keyword arguments into the config
dict the daemon stores, and the adapter reads that dict back into deploy decisions.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from skyward.protocol.schemas import ComputeSpec, Image, NodeBounds, ProviderRef, Spec
from skyward.providers.runpod import (
    _NVIDIA_VARIANT,
    DEFAULT_IMAGE,
    ENTRYPOINT,
    RunPodProvider,
    _cuda_range,
    _deploy_input,
    _select_image_candidates,
)
from skyward.sdk.provider import RunPod

LEGACY_BINDING = {
    "prefix": "skyward-cmp_1-",
    "image": "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
    "public_key": "ssh-ed25519 AAAA",
    "gpu_type_id": "NVIDIA A100",
    "gpu_count": 1,
    "cloud_type": "SECURE",
    "data_center_id": None,
    "container_disk_gb": 50,
    "market": "on_demand",
    "bid_per_gpu": None,
}


def _spec(image: Image = Image()) -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="runpod")),),
        nodes=NodeBounds(desired=1),
        image=image,
    )


def _adapter(**config: object) -> RunPodProvider:
    return RunPodProvider.create("prv_1", "runpod", {"api_key": "k"}, config)


def test_the_factory_carries_every_knob_into_config() -> None:
    provider = RunPod(
        api_key="k",
        cloud_type="community",
        container_image="ghcr.io/me/img:1",
        container_disk_gb=100,
        volume_gb=40,
        data_center_ids="EU-RO-1",
        country_codes="US",
        ports=("22/tcp", "8888/http"),
        bid_multiplier=1.5,
    )

    assert provider.config["cloud_type"] == "community"
    assert provider.config["container_image"] == "ghcr.io/me/img:1"
    assert provider.config["container_disk_gb"] == 100
    assert provider.config["volume_gb"] == 40
    assert provider.config["data_center_ids"] == ("EU-RO-1",), "a lone id becomes a one-tuple"
    assert provider.config["country_codes"] == ("US",)
    assert provider.config["ports"] == ("22/tcp", "8888/http")
    assert provider.config["bid_multiplier"] == 1.5


def test_global_stays_a_mode_not_a_member() -> None:
    assert RunPod().config["data_center_ids"] == "global"


def test_pool_image_wins_over_the_provider_override() -> None:
    adapter = _adapter(container_image="ghcr.io/me/override:1")
    assert adapter._image(_spec(Image(base="user/pinned:2"))) == "user/pinned:2"


def test_container_image_beats_the_base_family() -> None:
    adapter = _adapter(base_image="runpod-pytorch", container_image="ghcr.io/me/img:1")
    assert adapter._image(_spec()) == "ghcr.io/me/img:1"


def test_base_image_family_selects_its_image() -> None:
    adapter = _adapter(base_image="runpod-pytorch")
    assert "runpod/pytorch" in adapter._image(_spec())


def test_nothing_specified_falls_back_to_default() -> None:
    adapter = _adapter()
    assert adapter._image(_spec()) == DEFAULT_IMAGE


def test_excluded_countries_drop_out_of_the_allow_list() -> None:
    adapter = _adapter(country_codes=("US", "CA", "DE"), exclude_country_codes=("CA",))
    assert adapter._countries() == ("US", "DE")


def test_data_center_global_means_no_pin() -> None:
    assert (_adapter(data_center_ids="global"))._data_center() is None
    assert (_adapter(data_center_ids=("EU-RO-1", "US-TX-3")))._data_center() == "EU-RO-1"


def test_a_binding_from_before_the_knobs_still_launches() -> None:
    deploy, _, _ = _deploy_input(LEGACY_BINDING, "on_demand")
    assert deploy["volumeInGb"] == 0, "a legacy binding defaults the volume instead of raising"
    assert deploy["ports"] == "22/tcp"
    assert "countryCode" not in deploy and "containerRegistryAuthId" not in deploy


def test_a_full_binding_carries_its_knobs_into_the_deploy() -> None:
    deploy, mutation, key = _deploy_input({
        **LEGACY_BINDING,
        "bid_per_gpu": 0.5,
        "volume_gb": 40,
        "volume_mount_path": "/data",
        "ports": "22/tcp,8888/http",
        "country_code": "US",
        "registry_auth_id": "ra_1",
        "min_download_mbps": 100,
    }, "spot")
    assert deploy["volumeInGb"] == 40
    assert deploy["volumeMountPath"] == "/data"
    assert deploy["ports"] == "22/tcp,8888/http"
    assert deploy["countryCode"] == "US"
    assert deploy["containerRegistryAuthId"] == "ra_1"
    assert deploy["minDownload"] == 100
    assert deploy["bidPerGpu"] == 0.5
    assert key == "podRentInterruptable"


def _env(deploy: dict[str, object]) -> dict[str, str]:
    return {entry["key"]: entry["value"] for entry in deploy["env"]}  # type: ignore[index,union-attr]


def test_the_deadswitch_survives_the_bash_c_wrapping() -> None:
    assert "'" not in ENTRYPOINT, "a single quote would break bash -c '<entrypoint>'"
    assert "INSTANCE_TIMEOUT" in ENTRYPOINT
    assert "runpodctl remove pod" in ENTRYPOINT and "kill 1" in ENTRYPOINT, "full removal, kill as fallback"


def test_the_timeout_travels_as_an_env_var() -> None:
    deploy, _, _ = _deploy_input({**LEGACY_BINDING, "ttl": 3600}, "on_demand")
    assert _env(deploy)["INSTANCE_TIMEOUT"] == "3600"


def test_a_binding_without_a_ttl_disables_the_deadswitch() -> None:
    deploy, _, _ = _deploy_input(LEGACY_BINDING, "on_demand")
    assert _env(deploy)["INSTANCE_TIMEOUT"] == "0", "no ttl means the pod never self-terminates"


def test_the_ttl_reaches_the_binding() -> None:
    assert _spec().ttl == 600, "the compute carries a default ttl the provider can read"


def test_country_is_drawn_from_the_allowed_set() -> None:
    binding = {**LEGACY_BINDING, "countries": ["US", "DE", "FR"]}
    seen = {_deploy_input(binding, "on_demand")[0]["countryCode"] for _ in range(30)}
    assert seen and seen <= {"US", "DE", "FR"}


def test_image_candidates_are_tried_in_order() -> None:
    binding = {**LEGACY_BINDING, "image_candidates": ["img:a", "img:b"]}
    assert _deploy_input(binding, "on_demand", "img:b")[0]["imageName"] == "img:b"


def test_cuda_range_comes_from_the_catalog() -> None:
    assert _cuda_range("RTX 3090") == ("11.1", "13.1")
    assert _cuda_range(None) == (None, None)


def test_image_selection_picks_the_newest_patch_within_range() -> None:
    tags = [
        "13.2.0-cudnn-runtime-ubuntu24.04",
        "12.8.1-cudnn-runtime-ubuntu24.04",
        "12.8.0-cudnn-runtime-ubuntu24.04",
        "12.8.0-devel-ubuntu24.04",
        "11.8.0-cudnn-runtime-ubuntu24.04",
    ]
    result = _select_image_candidates(tags, (12, 4), (13, 1), "newest", "nvidia/cuda", _NVIDIA_VARIANT)
    assert result == ("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",), "range clips 13.2 and 11.8, cudnn-runtime wins, newest patch"


@pytest.mark.asyncio
async def test_release_sweeps_the_pods_that_carry_the_prefix() -> None:
    deleted: list[str] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, data: object = None) -> None:
            self._data = data

        def raise_for_status(self) -> None: ...

        def json(self) -> object:
            return self._data

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> bool:
            return False

        async def get(self, url: str, headers: object = None) -> FakeResponse:
            return FakeResponse([
                {"id": "pod_other", "name": "skyward-cmp_2-aaaa"},
                {"id": "pod_mine", "name": "skyward-cmp_1-bbbb"},
            ])

        async def delete(self, url: str, headers: object = None) -> FakeResponse:
            deleted.append(url.rsplit("/", 1)[-1])
            return FakeResponse()

    with patch("skyward.providers.runpod.httpx.AsyncClient", return_value=FakeClient()):
        await _adapter().release({"prefix": "skyward-cmp_1-"})

    assert deleted == ["pod_mine"], "only the leftover pod named after this compute is swept"
