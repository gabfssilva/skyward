"""The operational knobs, from the user's word to the value the runtime reads.

Nothing here provisions. The point of the defaults is that a pool built without
options is the pool that existed before options did, so most of what follows is an
equality against the constant the runtime used to hard-code.
"""

from __future__ import annotations

import inspect

import msgspec
import pytest

import skyward
from skyward.shared.provider import Machine
from skyward.server.application.reconciler import IDLE_SECONDS
from skyward.shared import codec
from skyward.shared.schemas import ComputeSpec, Image
from skyward.shared.schemas import Options as OptionsRef
from skyward.worker.api import Info
from skyward.server.application.node import WORKER_TIMEOUT, Node
from skyward.server.application.source import Source
from skyward.server.application.ssh import RETRY_DELAY, SshChannel
from skyward.core import spec as spec_module
from skyward.core.compute import DELETE_TIMEOUT, READY_TIMEOUT
from skyward.core.spec import HealthChecker, Options

pytestmark = pytest.mark.unit


def test_health_checker_is_part_of_the_public_spec_surface() -> None:
    assert getattr(spec_module, "HealthChecker", None) is not None


def test_health_checker_defaults_match_the_v1_contract() -> None:
    checker = HealthChecker(lambda _: True)

    assert checker.interval == 30.0
    assert checker.timeout == 15.0
    assert checker.consecutive_failures == 3
    assert checker.initial_delay == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval", 0.0),
        ("timeout", 0.0),
        ("consecutive_failures", 0),
        ("initial_delay", -0.1),
    ],
)
def test_health_checker_rejects_invalid_limits(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        HealthChecker(lambda _: True, **{field: value})


def test_health_checker_is_exported_from_the_sdk_facade() -> None:
    assert skyward.HealthChecker is HealthChecker


def test_health_checker_reaches_the_wire_with_its_predicate() -> None:
    checker = HealthChecker(
        lambda info: info.rank == 2,
        interval=4.0,
        timeout=1.5,
        consecutive_failures=5,
        initial_delay=2.0,
    )

    wire = spec_of(built(Options(health_checker=checker))).options
    fn = codec.loads(wire.health_function)

    assert fn(Info(node="n", compute="c", rank=2, peers=("a", "b", "c")))
    assert wire.health_interval == 4.0
    assert wire.health_timeout == 1.5
    assert wire.health_failures == 5
    assert wire.health_initial_delay == 2.0


def test_cluster_mode_reaches_the_wire_without_resolving_provider_defaults() -> None:
    assert spec_of(built(Options())).options.cluster is None
    assert spec_of(built(Options(cluster=False))).options.cluster is False
    assert spec_of(built(Options(cluster=True))).options.cluster is True


def built(options: Options) -> skyward.Compute:
    return skyward.Compute(provider=skyward.Container(), options=options)


def spec_of(pool: skyward.Compute) -> ComputeSpec:
    return pool._spec  # noqa: SLF001


def test_the_defaults_are_the_values_the_runtime_hard_coded():
    """Each default is the constant it replaced, so adding the knob moves nothing."""
    dialing = inspect.signature(SshChannel.__init__).parameters
    options = Options()

    assert options.ssh_timeout == dialing["connect_timeout"].default
    assert options.max_provision_attempts == dialing["reconnect_attempts"].default
    assert options.provision_retry_delay == RETRY_DELAY
    assert options.worker_timeout == WORKER_TIMEOUT
    assert options.autoscale_idle_timeout == IDLE_SECONDS
    assert options.ready_timeout == READY_TIMEOUT
    assert options.shutdown_timeout == DELETE_TIMEOUT


def test_the_knobs_nothing_reads_yet_are_unset():
    """``0`` is how both say "no limit", which is what the daemon does today."""
    assert (Options().autoscale_cooldown, Options().default_compute_timeout) == (0.0, 0.0)


def test_a_pool_that_asks_for_nothing_sends_the_wire_defaults():
    assert spec_of(skyward.Compute(provider=skyward.Container())).options == OptionsRef()


def test_the_friendly_names_land_on_the_wire_fields():
    spec = spec_of(built(Options(ssh_timeout=5.0, provision_retry_delay=0.5, max_provision_attempts=2)))

    assert spec.options.ssh_connect_timeout == 5.0
    assert spec.options.ssh_retry_delay == 0.5
    assert spec.options.ssh_reconnect_attempts == 2


def test_the_daemon_knobs_ride_the_spec():
    spec = spec_of(
        built(
            Options(
                worker_timeout=9.0,
                autoscale_idle_timeout=8.0,
                autoscale_cooldown=7.0,
                default_compute_timeout=6.0,
            ),
        ),
    )

    assert spec.options == OptionsRef(
        worker_timeout=9.0,
        autoscale_idle_timeout=8.0,
        autoscale_cooldown=7.0,
        default_compute_timeout=6.0,
    )


def test_the_session_timeouts_stay_in_the_client():
    """They say how long this process waits for its own pool; the daemon has no use for them."""
    pool = built(Options(ready_timeout=1.0, shutdown_timeout=2.0))

    assert (pool._ready_timeout, pool._shutdown_timeout) == (1.0, 2.0)  # noqa: SLF001
    assert spec_of(pool).options == OptionsRef()


def test_a_spec_with_default_options_round_trips_unchanged():
    spec = spec_of(skyward.Compute(provider=skyward.Container()))

    assert msgspec.json.decode(msgspec.json.encode(spec), type=ComputeSpec) == spec


def test_a_spec_carrying_options_round_trips_unchanged():
    spec = spec_of(built(Options(ssh_timeout=5.0, worker_timeout=9.0)))

    assert msgspec.json.decode(msgspec.json.encode(spec), type=ComputeSpec) == spec


def test_the_node_hands_its_channel_what_the_options_asked_for():
    """The knobs are only worth carrying if they reach the thing that dials."""
    node = Node(
        Machine(id="mch", state="running", host="127.0.0.1", user="root"),
        compute="cmp",
        private_key="",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=lambda *_: None,
        sample=lambda *_: None,
        phase=lambda *_: None,
        options=OptionsRef(ssh_connect_timeout=7.0, ssh_reconnect_attempts=4, ssh_retry_delay=0.25, worker_timeout=3.0),
    )

    assert node._ssh._connect_timeout == 7.0  # noqa: SLF001
    assert node._ssh._reconnect_attempts == 4  # noqa: SLF001
    assert node._ssh._retry_delay == 0.25  # noqa: SLF001
    assert node._worker_timeout == 3.0  # noqa: SLF001


def test_a_node_given_no_options_dials_the_way_it_always_did():
    node = Node(
        Machine(id="mch", state="running", host="127.0.0.1", user="root"),
        compute="cmp",
        private_key="",
        image=Image(),
        source=Source(arguments=("skyward",)),
        listener=lambda *_: None,
        output=lambda *_: None,
        sample=lambda *_: None,
        phase=lambda *_: None,
    )

    assert node._ssh._retry_delay == RETRY_DELAY  # noqa: SLF001
    assert node._worker_timeout == WORKER_TIMEOUT  # noqa: SLF001
