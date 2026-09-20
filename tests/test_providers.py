"""What the user writes about an account, and what the daemon gets back.

A provider is written once and read twice: the client takes it apart into the
credentials and the config a provider row is made of, and the adapter puts it
back together on the other side of the wire. These are the two halves agreeing.
"""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import msgspec
import pytest
from salad_cloud_sdk.net.transport.api_error import ApiError

import skyward as sky
from skyward.core.provider import resolve
from skyward.providers.registry import REGISTRY
from skyward.providers.runpod import RunPodProvider, _deploy_input, _machine
from skyward.providers.salad import SaladProvider
from skyward.server.application import market
from skyward.server.persistence.db import connect
from skyward.server.persistence.providers import ProviderStore
from skyward.shared import providers
from skyward.shared.providers import Provider
from skyward.shared.schemas import ComputeSpec, Image, NodeBounds, Offer, Page, ProviderCreate, ProviderRef, Spec, Worker

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


def describe_the_adapter_the_store_hands_out() -> None:
    async def _stored(tmp_path: Path) -> tuple[ProviderStore, str]:
        await connect(tmp_path / "skyward.sqlite")
        store = ProviderStore()
        provider = await store.create(ProviderCreate(name="fakey", kind="fake", credentials={}, config={"region": "here"}))
        return store, provider.id

    async def it_is_built_once_and_reused(tmp_path: Path) -> None:
        store, provider = await _stored(tmp_path)

        first = await store.adapter(provider)

        assert await store.adapter(provider) is first, "an adapter holds a session worth keeping"
        assert await store.adapter("fakey") is first, "by name or by id, it is the same account"

    async def it_is_rebuilt_once_the_account_changes(tmp_path: Path) -> None:
        store, provider = await _stored(tmp_path)
        before = await store.adapter(provider)

        await store.update(provider, ProviderCreate(name="fakey", kind="fake", credentials={}, config={"region": "elsewhere"}))
        after = await store.adapter(provider)

        assert after is not before, "an adapter built from the old config would keep answering for it"
        assert await store.adapter(provider) is after


def describe_a_salad_container_group_nobody_wrote_down() -> None:
    """Salad has no tags, and a launch records nothing: the name is the whole record.

    The window between a group existing and the daemon writing it down is a launch
    long — minutes, on a provider that is asked to wait for an allocation — so a
    group has to be findable without the binding, or it is a container that bills
    until somebody notices it in the portal.
    """

    class _Salad:
        """The project as Salad has it: some of this compute's groups, and another's."""

        def __init__(self, instances: dict[str, str], pulled: float, stopped: frozenset[str] = frozenset()) -> None:
            self._instances = instances
            self._pulled = pulled
            self._stopped = stopped
            self.deleted: list[str] = []
            self.started: list[str] = []
            self.quota_full = False

        async def list_container_groups(self, organization: str, project: str) -> Any:
            mine = [
                SimpleNamespace(
                    name=name,
                    networking=SimpleNamespace(dns=f"{name}.salad.cloud"),
                    current_state=SimpleNamespace(status="stopped" if name in self._stopped else "deploying"),
                )
                for name in self._instances
            ]
            return SimpleNamespace(items=[*mine, SimpleNamespace(name="skyward-cmp-2-bbbb", networking=None, current_state=None)])

        async def start_container_group(self, organization: str, project: str, name: str) -> None:
            self.started.append(name)
            if self.quota_full:
                raise ApiError("Bad Request", 400, SimpleNamespace(body={"type": "replicas_quota_exceeded", "status": 400}))

        async def list_container_group_instances(self, organization: str, project: str, name: str) -> Any:
            state = self._instances[name]
            return SimpleNamespace(instances=[SimpleNamespace(state=state, ready=False, pulling_progress=self._pulled)] if state else [])

        async def delete_container_group(self, organization: str, project: str, name: str) -> None:
            self.deleted.append(name)

    def _salad(instances: dict[str, str], pulled: float = 0.43, stopped: frozenset[str] = frozenset()) -> tuple[SaladProvider, _Salad]:
        provider = SaladProvider.create("prv_salad", "salad", {"api_key": "not-a-key"}, {"organization": "org", "project": "proj"})
        groups = _Salad(instances, pulled, stopped)
        provider._sdk = SimpleNamespace(container_groups=groups)
        return provider, groups

    async def it_is_still_one_of_the_computes_machines() -> None:
        provider, _ = _salad({"skyward-cmp-1-aaaa": "downloading"})

        observed = await provider.machines({"compute_id": "cmp_1"})

        assert set(observed) == {"skyward-cmp-1-aaaa"}, "found by the compute in its name, not by the binding"
        assert observed["skyward-cmp-1-aaaa"].state == "pending"
        assert observed["skyward-cmp-1-aaaa"].progress == "downloading", "a pull is progress, and the deadline is measured against it"
        assert observed["skyward-cmp-1-aaaa"].completion == 0.43, "how far into the pull, as a number a bar can be drawn from"

    async def it_says_it_is_waiting_when_salad_has_allocated_nothing() -> None:
        provider, _ = _salad({"skyward-cmp-1-aaaa": ""})

        observed = await provider.machines({"compute_id": "cmp_1"})

        assert observed["skyward-cmp-1-aaaa"].progress == "waiting for salad to allocate a machine", "one answer, so the deadline runs"
        assert observed["skyward-cmp-1-aaaa"].completion is None, "there is no fraction of a machine that was never allocated"

    async def it_is_started_when_salad_created_it_stopped() -> None:
        """Salad's autostart is a start it schedules after creating the group, and one it sometimes never sends."""
        provider, groups = _salad({"skyward-cmp-1-aaaa": ""}, stopped=frozenset({"skyward-cmp-1-aaaa"}))

        observed = await provider.machines({"compute_id": "cmp_1"})

        assert groups.started == ["skyward-cmp-1-aaaa"], "asked for again, since the one asked for at creation never happened"
        assert observed["skyward-cmp-1-aaaa"].state == "pending"
        assert observed["skyward-cmp-1-aaaa"].progress == "starting", "a start is movement, so the deadline is measured against it"

    async def it_says_why_when_salad_will_not_start_it() -> None:
        provider, groups = _salad({"skyward-cmp-1-aaaa": ""}, stopped=frozenset({"skyward-cmp-1-aaaa"}))
        groups.quota_full = True

        observed = await provider.machines({"compute_id": "cmp_1"})

        assert observed["skyward-cmp-1-aaaa"].progress == "salad refused to start it: replicas_quota_exceeded", "the reason, on the node"

    async def it_reports_the_pull_as_a_fraction_whichever_way_salad_says_it() -> None:
        """Salad sends a fraction where its own sdk promises a percentage."""
        provider, _ = _salad({"skyward-cmp-1-aaaa": "downloading"}, pulled=43)

        observed = await provider.machines({"compute_id": "cmp_1"})

        assert observed["skyward-cmp-1-aaaa"].completion == 0.43, "43 out of a hundred and 0.43 of one are the same pull"

    async def it_is_taken_down_with_the_compute() -> None:
        provider, groups = _salad({"skyward-cmp-1-aaaa": "downloading"})

        await provider.release({"compute_id": "cmp_1"})

        assert groups.deleted == ["skyward-cmp-1-aaaa"], "everything of this compute's, and nothing of anybody else's"


def describe_what_a_salad_account_may_buy() -> None:
    """Salad prices its GPU classes and nothing else, and it sells more than it prices.

    A CPU-only container group is the same request without a GPU class, and until it
    was offered there was no way to buy one: every Salad offer carried a GPU class,
    so a compute that named no accelerator matched GPU classes alone and a node meant
    to run sklearn came up holding a GPU.

    What it does *not* cost is the surprise. A GPU class bills no vCPU or RAM beside
    it, so its quoted price is the whole node, and Salad's cheapest classes undercut
    a CPU-only node of the same size. Which of the two shelves a compute is sold from
    is therefore the price's answer, not this module's.
    """

    def _classes(*specs: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    id_=spec.get("id", "gpu-1"),
                    name=spec.get("name", "RTX 3060"),
                    gpu_count=1,
                    max_storage=50 * 1024**3,
                    min_vcpu=spec.get("min_vcpu", 1),
                    max_vcpu=spec.get("max_vcpu", 0),
                    min_ram=spec.get("min_ram", 1024),
                    max_ram=spec.get("max_ram", 0),
                    gpu_class_type="gpu",
                    prices=[SimpleNamespace(price=spec.get("price", "0.047"), priority="low")],
                )
                for spec in specs or ({},)
            ]
        )

    def _salad(classes: SimpleNamespace, **config: Any) -> SaladProvider:
        provider = SaladProvider.create("prv_salad", "salad", {"api_key": "not-a-key"}, {"organization": "org", "project": "proj", **config})

        async def list_gpu_classes(organization: str) -> SimpleNamespace:
            return classes

        provider._sdk = SimpleNamespace(organization_data=SimpleNamespace(list_gpu_classes=list_gpu_classes))
        return provider

    async def _offered(provider: SaladProvider) -> tuple[Offer, ...]:
        return tuple([offer async for offer in provider.offers()])

    async def _shelf(provider: SaladProvider) -> Any:
        offers = await _offered(provider)

        class Shelf:
            async def list(self, **query: Any) -> Page[Offer]:
                return Page(items=offers, total=len(offers))

        return Shelf()

    def _spec(**wanted: Any) -> ComputeSpec:
        return ComputeSpec(
            specs=(Spec(provider=ProviderRef(kind="salad", name="salad"), **wanted),),
            nodes=NodeBounds(initial=1),
            image=Image(python="3.13"),
            worker=Worker(concurrency=1, executor="thread"),
        )

    async def _requested(provider: SaladProvider, offer: Offer) -> tuple[Mapping[str, Any], Any]:
        """The binding an offer leads to, and the resources Salad is asked for with it."""
        binding = await provider.initialize("cmp_1", _spec(), offer, "on_demand", "ssh-ed25519 AAAA")
        container = provider._group_body("skyward-cmp-1-aaaa", binding).container
        assert container is not None
        return binding, container.resources

    async def it_sells_a_node_without_a_gpu_to_a_compute_that_asked_for_none() -> None:
        """What the CPU-only shelf is for: before it, this compute could only be sold a GPU."""
        provider = _salad(_classes(), cpus=8, memory_gb=16)

        bought, _ = await market.pick(_spec(cpus=4, memory_gb=8), await _shelf(provider))

        assert bought.accelerator is None, "a CPU node is cheaper than the GPU class, and now there is one to buy"
        assert (bought.cpus, bought.memory_gb) == (4, 8), "the smallest published size that satisfies the spec"
        assert bought.price == pytest.approx(4 * 0.005 + 8 * 0.001), "vCPU-hours and GB-hours at the account's rates"

    async def it_does_not_sell_a_gpu_just_because_it_is_cheaper() -> None:
        """Salad throws the vCPUs and RAM in with a GPU class, so a GPU can undercut the CPU node.

        A cheap class is a cheaper 8x16 than 8x16 of CPU is, and while the market picks
        the cheapest offer that fits, "fits" no longer includes an accelerator nobody
        asked for. Paying less is not the only cost: a GPU class places on the machines
        that have that GPU, which is a smaller pool than the one a CPU node runs on.
        """
        provider = _salad(_classes({"price": "0.015"}), cpus=8, memory_gb=16)

        bought, _ = await market.pick(_spec(cpus=8, memory_gb=16), await _shelf(provider))

        assert bought.accelerator is None, "a spec that named no accelerator is not sold one at any price"
        assert bought.price == pytest.approx(8 * 0.005 + 16 * 0.001), "the CPU node, dearer than the GPU class it was not sold"

    async def it_still_sells_the_gpu_to_a_compute_that_wants_one() -> None:
        provider = _salad(_classes(), cpus=8, memory_gb=16)

        bought, _ = await market.pick(_spec(accelerator="rtx3060"), await _shelf(provider))

        assert bought.accelerator == "rtx-3060", "the CPU sizes are alongside the GPU classes, not instead of them"
        assert bought.price == 0.047, "a GPU class quotes the whole node: Salad bills no vCPU or RAM beside it"

    async def it_offers_every_cpu_size_salad_would_take() -> None:
        """Salad sizes a container group from the request, so there is no size to leave out."""
        provider = _salad(_classes(), cpus=4, memory_gb=16)

        sizes = {(offer.cpus, offer.memory_gb) for offer in await _offered(provider) if offer.accelerator is None}

        assert {(5, 7), (4, 4), (16, 32), (1, 1)} <= sizes, "an arbitrary size is one Salad takes, so it is one to buy"
        assert len(sizes) == 16 * 32, "all of them: a size off the shelf is a size nobody can ask for"

    async def it_prices_every_cpu_size_by_what_it_is_made_of() -> None:
        provider = _salad(_classes(), vcpu_price=0.005, memory_price=0.001)

        priced = {(offer.cpus, offer.memory_gb): offer.price for offer in await _offered(provider) if offer.accelerator is None}

        assert priced[(4, 4)] == pytest.approx(0.024), "four vCPU-hours and four GB-hours"
        assert priced[(16, 32)] == pytest.approx(0.112), "and the largest is the same arithmetic"

    async def it_leaves_out_a_gpu_class_the_account_is_too_small_for() -> None:
        """A floor above the account's size is a class that would be bought and then refused at creation."""
        provider = _salad(_classes({"id": "big", "name": "H100", "min_vcpu": 16}), cpus=4, memory_gb=16)

        offered = await _offered(provider)

        assert not [offer for offer in offered if offer.accelerator is not None], "nothing to buy that cannot be created"
        assert offered, "the CPU sizes are unaffected by a GPU class nobody can have"

    async def it_holds_a_gpu_class_to_its_own_ceiling() -> None:
        provider = _salad(_classes({"max_vcpu": 2, "max_ram": 8192}), cpus=8, memory_gb=16)

        gpu = next(offer for offer in await _offered(provider) if offer.accelerator is not None)

        assert (gpu.cpus, gpu.memory_gb) == (2, 8), "the class's ceiling trims the account's size rather than being exceeded"

    async def it_creates_the_group_at_the_size_that_was_bought() -> None:
        provider = _salad(_classes(), cpus=16, memory_gb=16)
        offer = next(offer for offer in await _offered(provider) if (offer.cpus, offer.memory_gb) == (2, 4))

        binding, resources = await _requested(provider, offer)

        assert (binding["cpu"], binding["memory"]) == (2, 4096), "the offer sized the node, not the account's ceiling"
        assert (resources.cpu, resources.memory) == (2, 4096)
        assert not hasattr(resources, "gpu_classes"), "no gpu class is asked for, which is what makes it a CPU-only group"

    async def it_asks_for_the_gpu_class_of_a_gpu_offer() -> None:
        provider = _salad(_classes({"id": "gpu-uuid"}), cpus=4, memory_gb=16)
        offer = next(offer for offer in await _offered(provider) if offer.accelerator is not None)

        _, resources = await _requested(provider, offer)

        assert resources.gpu_classes == ["gpu-uuid"], "the class travels from the offer to the request that creates the group"


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
            with pytest.raises(Exception) as refusal:
                await provider._deploy(client, binding, "on_demand", "nod-1")

        return refusal.value

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
            "name": "skyward-cmp_a275b416888f-nod-5ac8247f",
            "status": "RUNNING",
            "runtime": None,
            "publicIp": None,
            "portMappings": None,
            "globalNetworking": {"enabled": False},
            "ssh": {
                "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "4vkil6xz1jd8tr-64411c41", "command": "ssh ..."},
                "direct": {"host": "69.30.119.250", "port": 10465, "username": "root", "command": "ssh ..."},
            },
        }, "skyward-cmp_a275b416888f-")

        assert machine is not None
        assert machine.state == "running"
        assert (machine.host, machine.port) == ("69.30.119.250", 10465)
        assert machine.node == "nod-5ac8247f", "the claim it was launched under, read back off its name"

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
            "nod-1",
        )

        assert deploy["name"] == "skyward-cmp_1-nod-1", "the claim rides in the only field runpod lets an adapter read back"
        assert deploy["startSsh"] is True, "without it runpod publishes no port and the node is unreachable"
        assert "22/tcp" in deploy["ports"], "and the flag alone is not enough"
        assert deploy["env"]["SKYWARD_PUBLIC_KEY"] == "ssh-ed25519 AAAA", "the compute's key travels in a variable of its own"
        assert "PUBLIC_KEY" not in deploy["env"], "PUBLIC_KEY is runpod's, and a second writer on it costs the daemon its access"
        assert '"$PUBLIC_KEY" "$SKYWARD_PUBLIC_KEY"' in deploy["args"], "the node trusts the account's keys and the compute's"

    def it_is_still_pending_while_the_running_pod_has_none() -> None:
        machine = _machine({
            "id": "jj4obfs25b0xig",
            "status": "RUNNING",
            "runtime": None,
            "globalNetworking": {"enabled": False},
            "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "x", "command": "ssh ..."}, "direct": None},
        }, "skyward-cmp_1-")

        assert machine is not None
        assert machine.state == "pending", "the proxy is an interactive shell, not somewhere a node can be bootstrapped"
        assert machine.host is None

    @pytest.mark.parametrize("status", ["EXITED", "ERROR", "TERMINATED"])
    def it_is_reported_gone_once_it_has_stopped(status: str) -> None:
        assert _machine({"id": "jj4obfs25b0xig", "status": status}, "skyward-cmp_1-") is None
