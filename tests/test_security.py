"""Who a worker lets in.

The port a worker answers on runs whatever it is handed, so the certificate is the
whole door. What follows is that door: the authority a compute mints, the identity
it signs for one member, and who completes a round trip through it.

The two ends are asymmetric in code and not in fact — a node is dialled by the
daemon and by the other nodes, so the same identity has to serve both roles. That
is what makes these cases worth having: a certificate that authenticates one
direction and not the other passes every unit test and fails the moment two
machines meet.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from pathlib import Path

import casty
import pytest
from casty import TLS
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.x509 import ExtendedKeyUsage, load_pem_x509_certificate
from cryptography.x509.oid import ExtendedKeyUsageOID

from skyward.server.application.mock import SPEC
from skyward.server.application.runtimes import Runtime
from skyward.server.application.source import Source
from skyward.server.persistence.computes import ComputeStore, Infrastructure
from skyward.server.persistence.db import connect
from skyward.server.persistence.events import EventStore
from skyward.server.persistence.tables import ComputeRow
from skyward.shared import tls
from skyward.shared.schemas import ComputeCreate
from skyward.worker import worker

pytestmark = pytest.mark.local


def material(directory: Path, identity: tls.Identity) -> TLS:
    """What casty wants of an identity: three files, and the paths to them."""
    directory.mkdir(parents=True, exist_ok=True)
    certificate, key, authority = directory / "member.crt", directory / "member.key", directory / "ca.crt"
    certificate.write_text(identity.certificate)
    key.write_text(identity.key)
    authority.write_text(identity.authority)
    return TLS(cert=str(certificate), key=str(key), ca=str(authority), require_client_cert=True)


def free() -> int:
    """A port nobody is on, for a cluster that lives one test long."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def reaches(door: TLS, caller: ssl.SSLContext) -> bool:
    """Whether a caller holding that context completes a round trip through the door."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.readexactly(4))
        await writer.drain()
        writer.close()

    listener = await asyncio.start_server(echo, "127.0.0.1", 0, ssl=door.server_context())
    async with listener:
        port = listener.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=caller)
            writer.write(b"ping")
            await writer.drain()
            answer = await reader.readexactly(4)
            writer.close()
            return answer == b"ping"
        except (OSError, asyncio.IncompleteReadError):
            return False


def describe_the_authority_a_compute_mints() -> None:
    def it_signs_the_identities_it_issues() -> None:
        compute = tls.authority()
        member = tls.identity(compute, "nod_c19e40")

        certificate = load_pem_x509_certificate(member.certificate.encode())
        authority = load_pem_x509_certificate(compute.certificate.encode())

        signing = authority.public_key()
        assert isinstance(signing, EllipticCurvePublicKey), "an authority is minted on a curve"
        signing.verify(certificate.signature, certificate.tbs_certificate_bytes, ECDSA(SHA256()))
        assert certificate.issuer == authority.subject
        assert member.authority == compute.certificate

    def it_issues_one_identity_for_both_ends_of_a_connection() -> None:
        member = tls.identity(tls.authority(), "nod_c19e40")
        certificate = load_pem_x509_certificate(member.certificate.encode())

        usage = certificate.extensions.get_extension_for_class(ExtendedKeyUsage).value
        assert ExtendedKeyUsageOID.SERVER_AUTH in usage
        assert ExtendedKeyUsageOID.CLIENT_AUTH in usage

    def it_is_a_different_authority_every_time() -> None:
        assert tls.authority().certificate != tls.authority().certificate


def describe_reaching_a_worker() -> None:
    async def it_admits_a_member_of_the_same_compute(tmp_path: Path) -> None:
        compute = tls.authority()
        door = material(tmp_path / "worker", tls.identity(compute, "nod_c19e40"))
        caller = material(tmp_path / "daemon", tls.identity(compute, "cmp_7f3a1c"))

        assert await reaches(door, caller.client_context())

    async def it_refuses_a_member_of_another_compute(tmp_path: Path) -> None:
        door = material(tmp_path / "worker", tls.identity(tls.authority(), "nod_c19e40"))
        stranger = material(tmp_path / "stranger", tls.identity(tls.authority(), "nod_c19e40"))

        assert not await reaches(door, stranger.client_context())

    async def it_refuses_a_caller_that_shows_no_certificate(tmp_path: Path) -> None:
        compute = tls.authority()
        door = material(tmp_path / "worker", tls.identity(compute, "nod_c19e40"))

        anonymous = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        anonymous.check_hostname = False
        anonymous.verify_mode = ssl.CERT_REQUIRED
        anonymous.load_verify_locations(str(tmp_path / "worker" / "ca.crt"))

        assert not await reaches(door, anonymous)


def describe_the_authority_a_compute_keeps() -> None:
    async def it_outlives_the_daemon_that_minted_it(tmp_path: Path) -> None:
        await connect(tmp_path / "skyward.sqlite")
        store = ComputeStore(EventStore())
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="mint")
        minted = tls.authority()

        await store.bind(compute.id, Infrastructure(private_key="ssh", authority=minted))

        assert (await store.infrastructure(compute.id)).authority == minted

    async def it_is_added_to_a_database_that_predates_it(tmp_path: Path) -> None:
        database = tmp_path / "skyward.sqlite"
        await connect(database)
        store = ComputeStore(EventStore())
        compute, _ = await store.create(ComputeCreate(spec=SPEC), idempotency_key="older")
        await ComputeRow.raw("ALTER TABLE computes DROP COLUMN authority").run()

        await connect(database)

        assert (await store.infrastructure(compute.id)).authority is None


def describe_the_cluster_a_compute_forms() -> None:
    """The two ends as they are actually built: the worker from its environment, the daemon from the authority."""

    def _given(directory: Path, member: tls.Identity, monkeypatch: pytest.MonkeyPatch) -> None:
        """Put material on the machine the way the node does, and point the worker at it."""
        handed = material(directory, member)
        monkeypatch.setenv("SKYWARD_TLS_CERT", handed.cert)
        monkeypatch.setenv("SKYWARD_TLS_KEY", handed.key)
        monkeypatch.setenv("SKYWARD_TLS_CA", str(handed.ca))

    async def it_admits_a_daemon_that_holds_its_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        compute = tls.authority()
        _given(tmp_path / "node", tls.identity(compute, "nod_c19e40"), monkeypatch)
        daemon = material(tmp_path / "daemon", tls.identity(compute, "cmp_7f3a1c"))
        port = free()

        system = await casty.start(f"127.0.0.1:{port}", cluster_name="cmp_7f3a1c", tls=worker.material())
        try:
            client = await casty.connect([f"127.0.0.1:{port}"], cluster_name="cmp_7f3a1c", tls=daemon)
            assert [member.addr for member in client.members()] == [f"127.0.0.1:{port}"]
            await client.close()
        finally:
            await system.close()

    async def it_refuses_a_caller_signed_by_another(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _given(tmp_path / "node", tls.identity(tls.authority(), "nod_c19e40"), monkeypatch)
        stranger = material(tmp_path / "stranger", tls.identity(tls.authority(), "cmp_7f3a1c"))
        port = free()

        system = await casty.start(f"127.0.0.1:{port}", cluster_name="cmp_7f3a1c", tls=worker.material())
        try:
            async with asyncio.timeout(10):
                with pytest.raises(ssl.SSLError):
                    await casty.connect([f"127.0.0.1:{port}"], cluster_name="cmp_7f3a1c", tls=stranger)
        finally:
            await system.close()

    def it_comes_up_without_material_on_a_compute_that_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
        for variable in ("SKYWARD_TLS_CERT", "SKYWARD_TLS_KEY", "SKYWARD_TLS_CA"):
            monkeypatch.delenv(variable, raising=False)

        assert worker.material() is None


def describe_the_material_the_daemon_writes() -> None:
    async def it_is_written_once_and_taken_away_with_the_compute(tmp_path: Path) -> None:
        runtime = Runtime("cmp_7f3a1c", Source(arguments=("skyward",)), "a private key", True, tls.authority())

        daemon = runtime._material()
        assert daemon is not None
        directory = Path(daemon.cert).parent
        assert runtime._material() is daemon, "a second dial reuses the identity of the first"
        assert (directory / "daemon.key").stat().st_mode & 0o077 == 0, "nobody else on this machine reads the key"

        await runtime.close()

        assert not directory.exists()

    def it_writes_nothing_for_a_compute_without_an_authority() -> None:
        runtime = Runtime("cmp_7f3a1c", Source(arguments=("skyward",)), "a private key")

        assert runtime._material() is None
