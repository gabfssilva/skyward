"""RunPod keeps the v1 knobs: the factory carries them and the adapter honors them.

The pool passes a provider config through untouched, so the two halves that matter
are testable without a cloud: the factory turns keyword arguments into the config
dict the daemon stores, and the adapter reads that dict back into deploy decisions.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.provider import Machine
from skyward.shared.schemas import ComputeSpec, Image, NodeBounds, ProviderRef, Spec
from skyward.providers.runpod import (
    _NVIDIA_VARIANT,
    DEFAULT_IMAGE,
    ENTRYPOINT,
    RunPodProvider,
    _cuda_range,
    _deploy_input,
    _machine,
    _select_image_candidates,
)
from skyward.core.provider import RunPod

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


def _spec(image: Image | None = None) -> ComputeSpec:
    return ComputeSpec(
        specs=(Spec(provider=ProviderRef(kind="runpod")),),
        nodes=NodeBounds(desired=1),
        image=image or Image(),
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
    deploy = _deploy_input(LEGACY_BINDING, "on_demand")
    assert deploy["ports"] == ["22/tcp"]
    assert "mounts" not in deploy and "registry" not in deploy


def test_a_full_binding_carries_its_knobs_into_the_deploy() -> None:
    deploy = _deploy_input({
        **LEGACY_BINDING,
        "volume_gb": 40,
        "volume_mount_path": "/data",
        "ports": "22/tcp,8888/http",
        "country_code": "US",
        "registry_auth_id": "ra_1",
        "min_download_mbps": 100,
    }, "on_demand")
    assert deploy["mounts"] == {"persistent": {"size": 40, "path": "/data"}}
    assert deploy["ports"] == ["22/tcp", "8888/http"]
    assert deploy["registry"] == "ra_1"
    assert "countryCodes" not in deploy and "minDownloadMbps" not in deploy


def _env(deploy: dict[str, object]) -> dict[str, str]:
    env = deploy["env"]
    assert isinstance(env, dict)
    return {str(key): str(value) for key, value in env.items()}


def test_the_deadswitch_survives_the_bash_c_wrapping() -> None:
    assert "'" not in ENTRYPOINT, "a single quote would break bash -c '<entrypoint>'"
    assert "INSTANCE_TIMEOUT" in ENTRYPOINT
    assert "runpodctl remove pod" in ENTRYPOINT and "kill 1" in ENTRYPOINT, "full removal, kill as fallback"


def test_the_timeout_travels_as_an_env_var() -> None:
    deploy = _deploy_input({**LEGACY_BINDING, "ttl": 3600}, "on_demand")
    assert _env(deploy)["INSTANCE_TIMEOUT"] == "3600"


def test_a_binding_without_a_ttl_disables_the_deadswitch() -> None:
    deploy = _deploy_input(LEGACY_BINDING, "on_demand")
    assert _env(deploy)["INSTANCE_TIMEOUT"] == "0", "no ttl means the pod never self-terminates"


def test_the_ttl_reaches_the_binding() -> None:
    assert _spec().ttl == 600, "the compute carries a default ttl the provider can read"


def test_country_is_drawn_from_the_allowed_set() -> None:
    binding = {**LEGACY_BINDING, "countries": ["US", "DE", "FR"]}
    assert "countryCodes" not in _deploy_input(binding, "on_demand")


def test_image_candidates_are_tried_in_order() -> None:
    binding = {**LEGACY_BINDING, "image_candidates": ["img:a", "img:b"]}
    assert _deploy_input(binding, "on_demand", "img:b")["image"] == "img:b"


def test_the_pod_payload_matches_the_official_rest_openapi() -> None:
    deploy = _deploy_input(LEGACY_BINDING, "on_demand")

    assert deploy["gpu"] == {"id": "NVIDIA A100", "count": 1}
    assert deploy["cloud"] == "SECURE"
    assert deploy["args"] == f"bash -c '{ENTRYPOINT}'"
    assert "gpuTypeIds" not in deploy and "interruptible" not in deploy


def test_network_volume_uses_the_v2_mount_shape() -> None:
    deploy = _deploy_input({
        **LEGACY_BINDING,
        "network_volume_id": "vol_1",
        "volume_mount_path": "/data",
    }, "on_demand")
    assert deploy["mounts"] == {"network": [{"volumeId": "vol_1", "path": "/data"}]}


def test_v2_rejects_spot_until_the_api_exposes_it() -> None:
    with pytest.raises(CapabilityMismatchError, match="REST v2.*spot"):
        _deploy_input(LEGACY_BINDING, "spot")


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
async def test_deploy_uses_the_official_rest_pods_endpoint() -> None:
    posted: list[tuple[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        posted.append((str(request.url), request.read().decode()))
        return httpx.Response(201, json={"id": "pod_1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _adapter()._deploy(client, LEGACY_BINDING, "on_demand")

    assert not isinstance(result, Exception)
    assert result.id == "pod_1"
    assert posted[0][0] == "https://api.runpod.io/v2/pods"
    assert '"gpu":{"id":"NVIDIA A100","count":1}' in str(posted[0][1])


@pytest.mark.asyncio
async def test_registry_credentials_use_the_official_rest_endpoint() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.runpod.io/v2/registries"
        return httpx.Response(200, json={"registries": [{"id": "auth_1", "name": "docker hub"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await _adapter(registry_auth="docker hub")._registry_auth_id(client) == "auth_1"


@pytest.mark.asyncio
async def test_catalog_uses_v2_and_preserves_secure_and_community_prices() -> None:
    requested_clouds: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None: ...

        def json(self) -> object:
            return {"gpus": [{
                "id": "NVIDIA A100",
                "name": "A100",
                "memory": 80,
                "secure": True,
                "community": True,
                "price": {"secure": 2.0, "community": 1.5},
                "maxCount": {"secure": 1, "community": 1},
                "availability": "HIGH",
            }]}

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> bool:
            return False

        async def get(
            self,
            url: str,
            params: dict[str, str],
            headers: object = None,
        ) -> FakeResponse:
            assert url == "https://api.runpod.io/v2/catalog/gpus"
            requested_clouds.append(params["cloud"])
            return FakeResponse()

    with patch("skyward.providers.runpod.httpx.AsyncClient", return_value=FakeClient()):
        offers = [offer async for offer in _adapter(cloud_type="all").offers()]

    assert requested_clouds == ["SECURE", "COMMUNITY"]
    assert [(offer.specific["cloud_type"], offer.on_demand_price) for offer in offers] == [
        ("SECURE", 2.0),
        ("COMMUNITY", 1.5),
    ]
    assert all(offer.spot_price is None for offer in offers)


@pytest.mark.asyncio
async def test_country_filter_resolves_v2_datacenter_ids() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.runpod.io/v2/catalog/datacenters"
        return httpx.Response(200, json={"dataCenters": [
            {"id": "US-TX-3"},
            {"id": "EU-RO-1"},
            {"id": "CA-MTL-1"},
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        centers = await _adapter(country_codes=("US", "RO", "CA"))._data_centers(client)

    assert centers == ("US-TX-3", "EU-RO-1", "CA-MTL-1")


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
            return FakeResponse({"pods": [
                {"id": "pod_other", "name": "skyward-cmp_2-aaaa"},
                {"id": "pod_mine", "name": "skyward-cmp_1-bbbb"},
            ]})

        async def delete(self, url: str, headers: object = None) -> FakeResponse:
            deleted.append(url.rsplit("/", 1)[-1])
            return FakeResponse()

    with patch("skyward.providers.runpod.httpx.AsyncClient", return_value=FakeClient()):
        await _adapter().release({"prefix": "skyward-cmp_1-"})

    assert deleted == ["pod_mine"], "only the leftover pod named after this compute is swept"


def test_v2_running_pod_maps_its_live_ssh_and_private_dns() -> None:
    machine = _machine({
        "id": "pod_1",
        "status": "RUNNING",
        "runtime": {"ports": [{"private": 22, "public": 43122, "ip": "1.2.3.4"}]},
        "globalNetworking": {
            "enabled": True,
            "ip": "10.0.0.2",
            "internalDns": "pod_1.runpod.internal",
        },
    })
    assert machine == Machine(
        id="pod_1",
        state="running",
        host="1.2.3.4",
        port=43122,
        private_host="pod_1.runpod.internal",
    )


def test_v2_terminal_pods_are_absent() -> None:
    assert _machine({"id": "pod_1", "status": "TERMINATED"}) is None
