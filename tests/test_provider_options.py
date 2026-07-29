from skyward.sdk.provider import (
    AWS,
    GCP,
    Container,
    Hyperstack,
    JarvisLabs,
    Lambda,
    MassedCompute,
    Novita,
    RunPod,
    Scaleway,
    TensorDock,
    VastAI,
    Verda,
    Vultr,
)


def test_provider_factories_expose_the_v1_provisioning_options() -> None:
    assert AWS(
        region="r",
        ami="ami",
        ubuntu_version="u",
        subnet_id="subnet",
        security_group_id="sg",
        instance_profile_arn="profile",
        username="user",
        instance_timeout=1,
        request_timeout=2,
        allocation_strategy="lowest-price",
        exclude_burstable=True,
    ).config == {
        "regions": ("r",),
        "ami": "ami",
        "ubuntu_version": "u",
        "subnet_id": "subnet",
        "security_group_id": "sg",
        "instance_profile_arn": "profile",
        "username": "user",
        "instance_timeout": 1,
        "request_timeout": 2,
        "allocation_strategy": "lowest-price",
        "exclude_burstable": True,
    }
    assert GCP(
        zone="z",
        network="network",
        subnet="subnet",
        disk_size_gb=3,
        disk_type="disk",
        instance_timeout=4,
        service_account="service",
        thread_pool_size=5,
    ).config == {
        "zones": ("z",),
        "network": "network",
        "subnet": "subnet",
        "disk_gb": 3,
        "disk_type": "disk",
        "instance_timeout": 4,
        "service_account": "service",
        "thread_pool_size": 5,
    }
    assert Hyperstack(
        api_key="key",
        region="r",
        image="image",
        network_optimised=True,
        network_optimised_regions=("r",),
        object_storage_region="storage",
        object_storage_endpoint="endpoint",
        instance_timeout=6,
        request_timeout=7,
        teardown_timeout=8,
        teardown_poll_interval=9,
    ).config == {
        "region": "r",
        "image": "image",
        "network_optimised": True,
        "network_optimised_regions": ("r",),
        "object_storage_region": "storage",
        "object_storage_endpoint": "endpoint",
        "instance_timeout": 6,
        "request_timeout": 7,
        "teardown_timeout": 8,
        "teardown_poll_interval": 9,
    }
    assert JarvisLabs(
        api_key="key",
        region="r",
        template="template",
        storage_gb=10,
        instance_timeout=11,
        thread_pool_size=12,
    ).config == {
        "region": "r",
        "template": "template",
        "storage_gb": 10,
        "instance_timeout": 11,
        "thread_pool_size": 12,
    }
    assert MassedCompute(api_key="key", image_id=13, request_timeout=14).config == {
        "image_id": 13,
        "request_timeout": 14,
    }
    assert Novita(
        api_key="key",
        cluster_id="cluster",
        rootfs_size=15,
        docker_image="image",
        min_cuda_version="cuda",
        request_timeout=16,
    ).config == {
        "cluster_id": "cluster",
        "rootfs_size": 15,
        "docker_image": "image",
        "min_cuda_version": "cuda",
        "request_timeout": 16,
    }


def test_provider_factories_expose_the_v1_selection_and_network_options() -> None:
    assert RunPod(api_key="key", cluster_mode="instant", cpu_clock="5c").config["cluster_mode"] == "instant"
    assert RunPod(api_key="key", cluster_mode="instant", cpu_clock="5c").config["cpu_clock"] == "5c"
    assert TensorDock(
        api_key="key",
        api_token="token",
        location="country",
        tier=2,
        storage_gb=17,
        operating_system="os",
        instance_timeout=18,
        request_timeout=19,
        min_ram_gb=20,
        min_vcpus=21,
    ).config == {
        "location": "country",
        "tier": 2,
        "storage_gb": 17,
        "operating_system": "os",
        "instance_timeout": 18,
        "request_timeout": 19,
        "min_ram_gb": 20,
        "min_vcpus": 21,
    }
    assert VastAI(
        api_key="key",
        min_reliability=0.8,
        verified_only=False,
        min_cuda=12.4,
        geolocation="US",
        bid_multiplier=1.3,
        instance_timeout=22,
        request_timeout=23,
        docker_image="image",
        disk_gb=24,
        overlay_timeout=25,
        require_direct_port=True,
        min_inet_down=26,
        min_inet_up=27,
    ).config == {
        "min_reliability": 0.8,
        "verified_only": False,
        "min_cuda": 12.4,
        "geolocation": "US",
        "bid_multiplier": 1.3,
        "instance_timeout": 22,
        "request_timeout": 23,
        "docker_image": "image",
        "disk_gb": 24,
        "overlay_timeout": 25,
        "direct_port": True,
        "min_inet_down": 26,
        "min_inet_up": 27,
    }
    assert Verda(
        client_id="id",
        client_secret="secret",
        region="r",
        ssh_key_id="ssh",
        image="image",
        cuda="cuda",
        instance_timeout=28,
        request_timeout=29,
    ).config == {
        "region": "r",
        "ssh_key_id": "ssh",
        "image": "image",
        "cuda": "cuda",
        "instance_timeout": 28,
        "request_timeout": 29,
    }


def test_provider_factories_expose_the_remaining_v1_options() -> None:
    assert Scaleway(
        secret_key="secret",
        project_id="project",
        zone="zone",
        image="image",
        instance_timeout=30,
        request_timeout=31,
    ).config == {
        "zones": ("zone",),
        "image": "image",
        "instance_timeout": 30,
        "request_timeout": 31,
    }
    assert Vultr(
        api_key="key",
        mode="bare-metal",
        region="r",
        os_id=32,
        instance_timeout=33,
        request_timeout=34,
    ).config == {
        "mode": "bare-metal",
        "region": "r",
        "os_id": 32,
        "instance_timeout": 33,
        "request_timeout": 34,
    }
    assert Container(
        image="image",
        ssh_user="user",
        binary="podman",
        container_prefix="prefix",
        network="network",
    ).config == {
        "image": "image",
        "ssh_user": "user",
        "binary": "podman",
        "container_prefix": "prefix",
        "network": "network",
    }
    assert Lambda(api_key="key", region="r", request_timeout=35).config == {
        "region": "r",
        "request_timeout": 35,
    }
