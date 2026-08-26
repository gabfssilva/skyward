"""The certificates a compute's cluster is made of.

A worker's port runs whatever it is handed, so the only thing that can stand between
it and whoever dials it is who is allowed to speak. Every member of a compute — each
of its nodes, and the daemon that drives them — carries an identity signed by that
compute's own authority, and casty admits nobody else.

Casty checks the authority and not the name: a member is what signed it, not the
address it answers on. So nothing here carries a subject alternative name — a node is
reached on its private address by its peers and through a tunnel by the daemon, and a
name that changes with the route would prove nothing about the machine.

Nothing here touches disk and nothing here reads a compute. The authority is minted
once, when the compute is bound, and is persisted beside the SSH key it is the sibling
of; identities are issued from it on the way to a member, and the two ends that need
files write their own.

``cryptography`` is imported inside the functions. This module sits in the floor every
node imports, and a node issues nothing: it is handed three files and gives their paths
to the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from msgspec import Struct

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey, EllipticCurvePublicKey
    from cryptography.x509 import Certificate, CertificateBuilder, Name

AUTHORITY_DAYS = 397
"""Longer than any compute lives, short enough to be a certificate and not a fixture."""

IDENTITY_DAYS = 90
"""A member's certificate is not rotated: a compute that outlives this is a machine
that has been paid for since April, and the expiry is the cheaper of the two alarms."""


class Authority(Struct, frozen=True):
    """A compute's own certificate authority, in PEM.

    Persisted with the compute for the reason the SSH key is: the daemon that
    reconnects to a fleet is not the daemon that provisioned it, and an authority
    that lived in memory would leave every running worker unreachable by whoever
    comes next.
    """

    certificate: str
    key: str


@dataclass(frozen=True, slots=True)
class Identity:
    """What one member presents, and what it verifies the other members against."""

    certificate: str
    key: str
    authority: str


def authority() -> Authority:
    """Mint the authority a compute's members are signed by."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import NameOID

    key = _key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "skyward")])
    certificate = (
        _certificate(name, name, key.public_key(), AUTHORITY_DAYS)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    return Authority(certificate=_pem(certificate), key=_secret(key))


def identity(authority: Authority, name: str) -> Identity:
    """Issue one member's certificate, good for both ends of a connection.

    Both ends because a node is dialled by the daemon and by its peers, and dials
    them back: the same certificate authenticates it as the server and as the client,
    which is what mutual authentication between equals means.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    signing = _load(authority.key)
    issuer = x509.load_pem_x509_certificate(authority.certificate.encode())

    key = _key()
    certificate = (
        _certificate(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]), issuer.subject, key.public_key(), IDENTITY_DAYS)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                key_cert_sign=False,
                crl_sign=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(signing, hashes.SHA256())
    )
    return Identity(certificate=_pem(certificate), key=_secret(key), authority=authority.certificate)


def _key() -> EllipticCurvePrivateKey:
    from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key

    return generate_private_key(SECP256R1())


def _load(pem: str) -> EllipticCurvePrivateKey:
    """The authority's key, as the curve it was minted on.

    Every authority here is P-256, and the check is what turns that from a habit
    into something the signing call can rely on.
    """
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(pem.encode(), password=None)
    if not isinstance(key, EllipticCurvePrivateKey):
        raise ValueError(f"a skyward authority is an elliptic curve key, not {type(key).__name__}")
    return key


def _certificate(subject: Name, issuer: Name, public_key: EllipticCurvePublicKey, days: int) -> CertificateBuilder:
    from cryptography import x509

    now = datetime.now(UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days))
    )


def _pem(certificate: Certificate) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding

    return certificate.public_bytes(Encoding.PEM).decode()


def _secret(key: EllipticCurvePrivateKey) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    return key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
