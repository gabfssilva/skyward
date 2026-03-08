from dataclasses import dataclass

from casty import SpyEvent


@dataclass(frozen=True, slots=True)
class LocalOutput:
    line: str
    stream: str = "stdout"


type ConsoleInput = SpyEvent | LocalOutput
