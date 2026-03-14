"""TLS certificate generation for Casty cluster mTLS.

Generates an ECDSA P-256 certificate authority (persisted to ~/.skyward/tls/)
and issues short-lived node and client certificates on demand.
"""
from __future__ import annotations

import datetime
import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cryptography import x509

if TYPE_CHECKING:
    from casty.remote.tls import Config
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, EllipticCurvePrivateKey, generate_private_key
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_DEFAULT_TLS_DIR = Path.home() / ".skyward" / "tls"
_CA_VALIDITY_DAYS = 3650
_NODE_VALIDITY_DAYS = 90


@dataclass(frozen=True, slots=True)
class CertificateAuthority:
    cert: x509.Certificate
    key: EllipticCurvePrivateKey
    cert_pem: bytes


def ensure_ca(*, tls_dir: Path = _DEFAULT_TLS_DIR) -> CertificateAuthority:
    tls_dir.mkdir(parents=True, exist_ok=True)
    cert_path = tls_dir / "ca.crt"
    key_path = tls_dir / "ca.key"

    if cert_path.exists() and key_path.exists():
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        assert isinstance(key, EllipticCurvePrivateKey)
        return CertificateAuthority(cert=cert, key=key, cert_pem=cert_path.read_bytes())

    key = generate_private_key(SECP256R1())
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Skyward CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Skyward"),
    ])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=_CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    key_path.chmod(0o600)

    return CertificateAuthority(cert=cert, key=key, cert_pem=cert_pem)


def issue_node_cert(ca: CertificateAuthority, ip: str) -> tuple[bytes, bytes]:
    key = generate_private_key(SECP256R1())
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"skyward-node-{ip}"),
    ])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=_NODE_VALIDITY_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address(ip)),
            ]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, key_cert_sign=False,
                crl_sign=False, data_encipherment=False,
                key_agreement=False, encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )
        .sign(ca.key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def issue_client_config(ca: CertificateAuthority) -> Config:
    import ssl
    import tempfile

    from casty.remote.tls import Config

    cert_pem, key_pem = issue_node_cert(ca, "127.0.0.1")
    with tempfile.NamedTemporaryFile(suffix=".crt") as cert_f, \
         tempfile.NamedTemporaryFile(suffix=".key") as key_f, \
         tempfile.NamedTemporaryFile(suffix=".crt") as ca_f:
        cert_f.write(cert_pem)
        cert_f.flush()
        key_f.write(key_pem)
        key_f.flush()
        ca_f.write(ca.cert_pem)
        ca_f.flush()

        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cert_f.name, key_f.name)
        server_context.verify_mode = ssl.CERT_REQUIRED
        server_context.load_verify_locations(ca_f.name)

        client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_context.load_cert_chain(cert_f.name, key_f.name)
        client_context.load_verify_locations(ca_f.name)
        client_context.check_hostname = False

        return Config(server_context=server_context, client_context=client_context)
