"""How the AWS adapter buys: many launches, one fleet, one zone.

EC2 throttles ``RunInstances`` by the call, and a pool opens by asking for every
machine at once. The adapter gathers the launches that arrive within a window into
a single fleet, and the zone is whatever the compute's machines already live in —
or, with none alive, whatever that fleet picks.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

from skyward.providers.aws import COMPUTE_TAG, AWSProvider
from skyward.shared.errors import CapabilityMismatchError
from skyward.shared.providers import AWS

pytestmark = pytest.mark.local

SUBNETS = {"us-east-1a": "subnet-a", "us-east-1b": "subnet-b"}
BINDING: Mapping[str, Any] = {
    "compute_id": "cmp_1",
    "region": "us-east-1",
    "instance_type": "c6gd.2xlarge",
    "image": "ami-1",
    "key_name": "skyward-cmp_1",
    "security_group_id": "sg-1",
    "subnets": SUBNETS,
    "user": "ubuntu",
    "disk_gb": 100,
    "instance_profile_arn": None,
    "allocation_strategy": None,
}


class Pages:
    def __init__(self, instances: list[dict[str, Any]]) -> None:
        self._instances = instances

    def paginate(self, **_: object) -> AsyncIterator[dict[str, Any]]:
        async def pages() -> AsyncIterator[dict[str, Any]]:
            yield {"Reservations": [{"Instances": list(self._instances)}]}

        return pages()


class EC2:
    """An EC2 that sells up to ``stock`` machines per fleet, in the first subnet it is offered."""

    def __init__(self, stock: int = 100) -> None:
        self.stock = stock
        self.fleets: list[dict[str, Any]] = []
        self.instances: list[dict[str, Any]] = []
        self.selling = asyncio.Event()
        self.selling.set()
        self.lagging = False

    async def create_launch_template(self, **_: object) -> dict[str, Any]:
        return {"LaunchTemplate": {"LaunchTemplateId": "lt-1"}}

    async def delete_launch_template(self, **_: object) -> None:
        return None

    async def create_fleet(self, **request: Any) -> dict[str, Any]:
        self.fleets.append(request)
        await self.selling.wait()
        wanted = request["TargetCapacitySpecification"]["TotalTargetCapacity"]
        subnet = request["LaunchTemplateConfigs"][0]["Overrides"][0]["SubnetId"]
        zone = next(zone for zone, candidate in SUBNETS.items() if candidate == subnet)
        sold = [f"i-{len(self.instances) + index}" for index in range(min(wanted, self.stock))]
        self.instances.extend(
            {
                "InstanceId": instance,
                "State": {"Name": "pending"},
                "Placement": {"AvailabilityZone": zone},
                "Tags": [{"Key": COMPUTE_TAG, "Value": "cmp_1"}],
            }
            for instance in sold
        )
        errors = [] if len(sold) == wanted else [{"ErrorCode": "InsufficientInstanceCapacity", "ErrorMessage": "none left"}]
        return {"Instances": [{"InstanceIds": sold, "LaunchTemplateAndOverrides": {"Overrides": {"SubnetId": subnet}}}], "Errors": errors}

    async def create_tags(self, Resources: list[str], Tags: list[dict[str, str]]) -> None:  # noqa: N803
        for instance in self.instances:
            if instance["InstanceId"] in Resources:
                instance["Tags"] = [*instance["Tags"], *Tags]

    def get_paginator(self, _: str) -> Pages:
        return Pages([] if self.lagging else self.instances)


def adapter(ec2: EC2) -> AWSProvider:
    provider = AWSProvider("prv_1", "aws", "AKIA", "s3cret", AWS(launch_window=0.05))

    @asynccontextmanager
    async def client(_: str) -> AsyncIterator[EC2]:
        yield ec2

    provider._ec2 = client  # type: ignore[method-assign]
    return provider


def describe_launching_on_aws() -> None:
    async def it_buys_the_launches_of_one_window_as_one_fleet() -> None:
        ec2 = EC2()
        aws = adapter(ec2)

        machines = await asyncio.gather(*(aws.launch(BINDING, "spot", f"nod_{index}") for index in range(8)))

        assert len(ec2.fleets) == 1
        assert ec2.fleets[0]["TargetCapacitySpecification"]["SpotTargetCapacity"] == 8
        assert len({machine.id for machine in machines}) == 8
        assert [machine.node for machine in machines] == [f"nod_{index}" for index in range(8)]

    async def it_claims_each_machine_for_the_node_it_was_bought_for() -> None:
        ec2 = EC2()
        aws = adapter(ec2)

        bought = await asyncio.gather(*(aws.launch(BINDING, "spot", f"nod_{index}") for index in range(3)))
        observed = await aws.machines(BINDING)

        assert {machine.id: machine.node for machine in bought} == {machine.id: machine.node for machine in observed.values()}

    async def it_lets_the_first_fleet_pick_the_zone_and_the_rest_follow_it() -> None:
        ec2 = EC2()
        aws = adapter(ec2)

        await aws.launch(BINDING, "spot", "nod_0")
        ec2.instances[0]["Placement"]["AvailabilityZone"] = "us-east-1b"
        await aws.launch(BINDING, "spot", "nod_1")

        offered = [[override["SubnetId"] for override in fleet["LaunchTemplateConfigs"][0]["Overrides"]] for fleet in ec2.fleets]
        assert offered == [["subnet-a", "subnet-b"], ["subnet-b"]]

    async def it_follows_the_zone_it_just_bought_in_while_ec2_does_not_list_the_machine_yet() -> None:
        ec2 = EC2()
        ec2.lagging = True
        aws = adapter(ec2)

        await aws.launch(BINDING, "spot", "nod_0")
        await aws.launch(BINDING, "spot", "nod_1")

        offered = [[override["SubnetId"] for override in fleet["LaunchTemplateConfigs"][0]["Overrides"]] for fleet in ec2.fleets]
        assert offered == [["subnet-a", "subnet-b"], ["subnet-a"]]

    async def it_offers_a_multi_subnet_compute_every_subnet_whatever_is_alive() -> None:
        ec2 = EC2()
        aws = adapter(ec2)
        spread = {**BINDING, "multi_subnet": True}

        await aws.launch(spread, "spot", "nod_0")
        await aws.launch(spread, "spot", "nod_1")

        offered = [[override["SubnetId"] for override in fleet["LaunchTemplateConfigs"][0]["Overrides"]] for fleet in ec2.fleets]
        assert offered == [["subnet-a", "subnet-b"], ["subnet-a", "subnet-b"]]
        assert not ec2.fleets[1]["SpotOptions"]["SingleAvailabilityZone"]

    async def it_refuses_the_launches_a_short_fleet_did_not_cover() -> None:
        ec2 = EC2(stock=2)
        aws = adapter(ec2)

        outcomes = await asyncio.gather(*(aws.launch(BINDING, "spot", f"nod_{index}") for index in range(3)), return_exceptions=True)

        assert [type(outcome).__name__ for outcome in outcomes] == ["Machine", "Machine", "CapabilityMismatchError"]
        assert isinstance(outcomes[2], CapabilityMismatchError) and "InsufficientInstanceCapacity" in str(outcomes[2])

    async def it_keeps_each_market_in_a_fleet_of_its_own() -> None:
        ec2 = EC2()
        aws = adapter(ec2)

        await asyncio.gather(aws.launch(BINDING, "spot", "nod_0"), aws.launch(BINDING, "on_demand", "nod_1"))

        assert [fleet["TargetCapacitySpecification"]["DefaultTargetCapacityType"] for fleet in ec2.fleets] == ["spot", "on-demand"]

    async def it_does_not_report_a_machine_it_is_still_claiming() -> None:
        ec2 = EC2()
        ec2.selling.clear()
        aws = adapter(ec2)

        buying = asyncio.create_task(aws.launch(BINDING, "spot", "nod_0"))
        while not ec2.fleets:
            await asyncio.sleep(0.01)
        observing = asyncio.create_task(aws.machines(BINDING))
        await asyncio.sleep(0.05)
        assert not observing.done(), "a machine seen before its claim is on it is a stray, and strays are terminated"

        ec2.selling.set()
        machine = await buying
        assert {found.node for found in (await observing).values()} == {machine.node}
