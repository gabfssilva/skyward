from dataclasses import dataclass
from typing import Literal, Callable, Protocol

type ProviderName = Literal["AWS", "Digital Ocean", "Verda", "Vast.ai"]

type InstanceStatus = Literal["started", "provisioning", "bootstrapping", "running", "stopping", "stopped"]

class Transport(Protocol):
    async def run(self, *command: str) -> tuple[int, tuple[str, ...]]:
        pass

@dataclass
class Instance:
    id: str
    provider: ProviderName
    status: InstanceStatus


class Node:
    id: str
    current: Instance
    history: list[Instance]


class Cluster:
    id: str
    nodes: list[Node]


class Provider:
    name: ProviderName