"""``sky compute sync`` — a folder on a node, mirrored into one here.

Two endpoints the daemon already had, and nothing else: ``exec`` to ask the
machine what is in the folder, and the file download to bring each file down.
The comparison is made here, and what it is made against is the local folder
itself — a file is current when its size and its modification time are the ones
the node reported, which is why a file brought down is given the node's time
rather than the time it arrived. There is no record of the last pass to keep,
lose or disagree with.

A file is brought down whole. One that changed by a byte is read again from the
start, which is the price of needing nothing on the machine but ``find``.

One direction, and nothing is ever removed here: a file deleted on the node stays
in the mirror. A mirror that deletes is one mistyped path away from emptying a
folder that was never a mirror.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated

import httpx
import msgspec
from cyclopts import Parameter

from skyward.api.v1 import ComputeResource
from skyward.cli import compute_app
from skyward.cli._client import Work, call
from skyward.core.client import Client
from skyward.core.errors import SkywardError
from skyward.shared.observability import notice

LISTING = "find {path} -type f -printf '%s %T@ %P\\n'"
"""Every file under the folder, as its size, its modification time and its path within it."""


@compute_app.command(name="sync")
def sync_path(
    ref: str,
    remote: str,
    local: Path,
    *,
    node: Annotated[str, Parameter(name="--node", help="Which node to mirror: a rank, or all")] = "0",
    watch: Annotated[bool, Parameter(negative="", help="Keep mirroring until interrupted, or until the compute is gone")] = False,
    interval: Annotated[float, Parameter(help="Seconds between one pass of --watch and the next")] = 10.0,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
) -> None:
    """Mirror a folder on one of the compute's nodes into a local one.

    One pass, bringing down what is missing here or differs from the node's copy
    in size or modification time. Nothing is removed locally.

    With ``--node all`` every node is mirrored into a folder named by its rank,
    because four machines hold four folders and merging them would let one
    overwrite another.

    ``--watch`` repeats the pass every ``--interval`` seconds. A pass that fails
    is reported and tried again; a compute that no longer exists ends it. A file
    the node is still writing comes down as it was at that moment, and again on
    the next pass.
    """
    if node != "all" and not node.isdigit():
        raise SystemExit(f"--node takes 'all' or a rank, not {node!r}")

    def once(*, quiet: bool) -> None:
        for pulled in _pass(partial(_mirror, ref, remote, local, node), url):
            if pulled.files or not quiet:
                sys.stdout.write(f"{pulled.into}  {pulled.files} files  {pulled.size} bytes\n")
                sys.stdout.flush()

    if not watch:
        try:
            return once(quiet=False)
        except PassError as missed:
            raise SystemExit(missed.reason) from None

    try:
        while True:
            try:
                once(quiet=True)
            except PassError as missed:
                if missed.final:
                    raise SystemExit(missed.reason) from None
                notice("WARNING", missed.reason)
            time.sleep(interval)
    except KeyboardInterrupt:
        with suppress(PassError):
            once(quiet=True)


class Result(msgspec.Struct, frozen=True):
    """What one node said. Restated for the reason ``compute.Result`` is."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One file on the node: where it is under the folder, and what tells a copy of it apart."""

    path: str
    size: int
    modified: int
    """Nanoseconds, the resolution both ends keep it at."""

    def differs(self, local: Path) -> bool:
        try:
            held = (local / self.path).stat()
        except FileNotFoundError:
            return True
        return (held.st_size, held.st_mtime_ns) != (self.size, self.modified)


@dataclass(frozen=True, slots=True)
class Pulled:
    """What one pass brought into one folder."""

    into: Path
    files: int
    size: int


class PassError(Exception):
    """A pass that did not happen, and whether a later one could."""

    def __init__(self, reason: str, *, final: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.final = final


def _pass(work: Work[tuple[Pulled, ...]], url: str | None) -> tuple[Pulled, ...]:
    try:
        return call(work, url=url)
    except SkywardError as refused:
        raise PassError(f"{refused.code}: {refused.message}", final=refused.code == "not_found") from None
    except httpx.HTTPError as dropped:
        raise PassError(f"the pass was cut short: {dropped}") from None


async def _mirror(ref: str, remote: str, local: Path, node: str, client: Client) -> tuple[Pulled, ...]:
    compute = await client.call("GET", f"/v1/computes/{ref}", ComputeResource)
    if compute.status.state in ("deleting", "deleted"):
        raise PassError(f"compute {compute.id} is {compute.status.state}", final=True)
    if node != "all":
        return (await _pull(client, ref, remote, local, int(node)),)

    ranks = sorted(held.rank for held in compute.nodes if held.state == "ready")
    return tuple(await asyncio.gather(*(_pull(client, ref, remote, local / str(rank), rank) for rank in ranks)))


async def _pull(client: Client, ref: str, remote: str, local: Path, rank: int) -> Pulled:
    said = await client.call(
        "POST",
        f"/v1/computes/{ref}/exec",
        dict[str, Result],
        command=LISTING.format(path=shlex.quote(remote)),
        node=rank,
    )
    (listing,) = said.values()
    if listing.exit_code:
        raise PassError(listing.stderr.strip() or f"{remote} could not be listed on rank {rank}")

    changed = [entry for entry in map(_entry, listing.stdout.splitlines()) if entry.differs(local)]
    for entry in changed:
        await _fetch(client, ref, f"{remote.rstrip('/')}/{entry.path}", local / entry.path, entry.modified, rank)
    return Pulled(local, len(changed), sum(entry.size for entry in changed))


async def _fetch(client: Client, ref: str, remote: str, local: Path, modified: int, rank: int) -> None:
    """One file, written beside where it belongs and moved there whole.

    Anybody reading the mirror while it is being filled sees the old file or the
    new one, never the first half of the new one.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    arriving = local.with_name(f"{local.name}.arriving")
    with arriving.open("wb") as sink:
        async for chunk in client.download(f"/v1/computes/{ref}/files/content", path=remote, node=rank):
            sink.write(chunk)
    os.utime(arriving, ns=(modified, modified))
    arriving.replace(local)


def _entry(line: str) -> Entry:
    size, modified, path = line.split(" ", 2)
    seconds, _, fraction = modified.partition(".")
    return Entry(path, int(size), int(seconds) * 1_000_000_000 + int(fraction[:9].ljust(9, "0")))


__all__ = ["sync_path"]
