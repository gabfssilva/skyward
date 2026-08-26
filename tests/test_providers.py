"""What the user writes about an account, and what the daemon gets back.

A provider is written once and read twice: the client takes it apart into the
credentials and the config a provider row is made of, and the adapter puts it
back together on the other side of the wire. These are the two halves agreeing.
"""

from pathlib import Path

import httpx
import msgspec
import pytest

import skyward as sky
from skyward.core.provider import resolve
from skyward.providers.registry import REGISTRY
from skyward.providers.runpod import RunPodProvider, _deploy_input, _machine
from skyward.shared import providers
from skyward.shared.providers import Provider

pytestmark = pytest.mark.local

ACCOUNTS = tuple(
    value
    for value in vars(providers).values()
    if isinstance(value, type) and issubclass(value, Provider) and value is not Provider
)


@pytest.fixture(autouse=True)
def nowhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No account of the machine running the tests answers for anything."""
    for account in ACCOUNTS:
        for variable in providers.variables(account).values():
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-file"))


def describe_writing_a_provider() -> None:
    def it_keeps_the_secrets_out_of_the_config() -> None:
        credentials, config = resolve(sky.AWS(access_key_id="AKIA", secret_access_key="s3cret", region="eu-west-1"))

        assert credentials == {"access_key_id": "AKIA", "secret_access_key": "s3cret"}
        assert config["region"] == "eu-west-1"
        assert not credentials.keys() & config.keys(), "a field is a secret or a setting, never both"
        assert "name" not in config, "the alias names the row rather than living in it"

    def it_takes_the_environment_only_for_what_was_left_unset(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "from-the-environment")
        assert resolve(sky.RunPod())[0] == {"api_key": "from-the-environment"}
        assert resolve(sky.RunPod(api_key="written-down"))[0] == {"api_key": "written-down"}

    def describe_when_the_environment_is_silent() -> None:
        def it_falls_back_to_the_aws_credentials_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
            shared = tmp_path / "credentials"
            shared.write_text("[default]\naws_access_key_id = AKIAFILE\naws_secret_access_key = filesecret\n")
            monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(shared))

            assert resolve(sky.AWS())[0] == {"access_key_id": "AKIAFILE", "secret_access_key": "filesecret"}

    def it_takes_one_region_or_several() -> None:
        assert sky.AWS(region="eu-west-1").regions == ("eu-west-1",)
        assert sky.AWS(region=("eu-west-1", "us-east-2")).regions == ("eu-west-1", "us-east-2")


def describe_handing_a_provider_to_the_daemon() -> None:
    @pytest.mark.parametrize("account", ACCOUNTS, ids=lambda account: account.kind)
    def it_comes_back_as_the_very_object_that_was_written(account: type[Provider]) -> None:
        written = account()
        credentials, config = resolve(written)
        wire = msgspec.json.decode(msgspec.json.encode({**credentials, **config}))

        assert msgspec.convert(wire, account) == written

    def it_is_refused_at_the_wire_when_a_setting_is_not_one_of_the_allowed() -> None:
        _, config = resolve(sky.RunPod())

        with pytest.raises(msgspec.ValidationError):
            msgspec.convert({**config, "cloud_type": "whatever"}, sky.RunPod)


def describe_the_adapters() -> None:
    def each_one_has_the_account_the_user_writes_for_it() -> None:
        kinds = {account.kind for account in ACCOUNTS}

        assert set(REGISTRY) <= kinds, "an adapter with no account is one nobody can configure"


def describe_a_pod_the_cloud_refuses_to_deploy() -> None:
    """RunPod answers problem+json, and the two refusals that matter read alike without it."""

    binding = {
        "prefix": "skyward-cmp_1-",
        "image": "runpod/base:1.0.0",
        "gpu_type_id": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "container_disk_gb": 50,
        "public_key": "ssh-ed25519 AAAA",
    }

    async def _refused(status: int, body: object) -> Exception:
        provider = RunPodProvider.create("prv_runpod", "runpod", {"api_key": "not-a-key"}, {})
        transport = httpx.MockTransport(lambda _: httpx.Response(status, json=body))

        async with httpx.AsyncClient(transport=transport) as client:
            answer = await provider._deploy(client, binding, "on_demand")

        assert isinstance(answer, Exception), "a refusal is what a deploy hands back, not a machine"
        return answer

    async def it_says_which_of_the_two_refusals_it_was() -> None:
        out_of_stock = await _refused(400, {"detail": "There are no longer any instances available with the requested specifications.", "status": 400})
        nonsense = await _refused(422, {"detail": "Unknown GPU type: NVIDIA GeForce RTX 9090", "status": 422})

        assert "no longer any instances available" in str(out_of_stock), "the one worth trying again"
        assert "Unknown GPU type" in str(nonsense), "the one that never will be"

    async def it_falls_back_to_what_was_written_when_it_is_not_problem_json() -> None:
        answer = await _refused(502, "upstream is down")

        assert "upstream is down" in str(answer)


def describe_a_pod_as_runpod_reports_it() -> None:
    """The pods listing is the only thing the daemon has to decide a machine is reachable."""

    def it_is_reachable_once_a_public_port_is_published_for_its_ssh() -> None:
        machine = _machine({
            "id": "4vkil6xz1jd8tr",
            "name": "skyward-cmp_a275b416888f-5ac8247f",
            "status": "RUNNING",
            "runtime": None,
            "publicIp": None,
            "portMappings": None,
            "globalNetworking": {"enabled": False},
            "ssh": {
                "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "4vkil6xz1jd8tr-64411c41", "command": "ssh ..."},
                "direct": {"host": "69.30.119.250", "port": 10465, "username": "root", "command": "ssh ..."},
            },
        })

        assert machine is not None
        assert machine.state == "running"
        assert (machine.host, machine.port) == ("69.30.119.250", 10465)

    def it_is_asked_for_with_the_flag_that_publishes_the_port() -> None:
        deploy = _deploy_input(
            {
                "prefix": "skyward-cmp_1-",
                "image": "runpod/base:1.0.0",
                "gpu_type_id": "NVIDIA GeForce RTX 4090",
                "gpu_count": 1,
                "cloud_type": "SECURE",
                "container_disk_gb": 50,
                "public_key": "ssh-ed25519 AAAA",
            },
            "on_demand",
        )

        assert deploy["startSsh"] is True, "without it runpod publishes no port and the node is unreachable"
        assert "22/tcp" in deploy["ports"], "and the flag alone is not enough"
        assert deploy["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA", "the key stays the compute's, not the account's"

    def it_is_still_pending_while_the_running_pod_has_none() -> None:
        machine = _machine({
            "id": "jj4obfs25b0xig",
            "status": "RUNNING",
            "runtime": None,
            "globalNetworking": {"enabled": False},
            "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "x", "command": "ssh ..."}, "direct": None},
        })

        assert machine is not None
        assert machine.state == "pending", "the proxy is an interactive shell, not somewhere a node can be bootstrapped"
        assert machine.host is None

    @pytest.mark.parametrize("status", ["EXITED", "ERROR", "TERMINATED"])
    def it_is_reported_gone_once_it_has_stopped(status: str) -> None:
        assert _machine({"id": "jj4obfs25b0xig", "status": status}) is None
