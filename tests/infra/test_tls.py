"""Tests for TLS certificate generation."""
from __future__ import annotations

import ssl
import tempfile
from pathlib import Path

import pytest

from skyward.infra.tls import CertificateAuthority, ensure_ca, issue_client_config, issue_node_cert


class TestEnsureCA:
    def test_generates_ca_when_missing(self, tmp_path: Path) -> None:
        ca = ensure_ca(tls_dir=tmp_path)
        assert isinstance(ca, CertificateAuthority)
        assert (tmp_path / "ca.crt").exists()
        assert (tmp_path / "ca.key").exists()

    def test_loads_existing_ca(self, tmp_path: Path) -> None:
        ca1 = ensure_ca(tls_dir=tmp_path)
        ca2 = ensure_ca(tls_dir=tmp_path)
        assert ca1.cert_pem == ca2.cert_pem

    def test_ca_uses_ecdsa_p256(self, tmp_path: Path) -> None:
        from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1

        ca = ensure_ca(tls_dir=tmp_path)
        assert isinstance(ca.key.curve, SECP256R1)


class TestIssueNodeCert:
    def test_generates_cert_with_ip_san(self, tmp_path: Path) -> None:
        ca = ensure_ca(tls_dir=tmp_path)
        cert_pem, key_pem = issue_node_cert(ca, "10.0.0.1")
        assert b"BEGIN CERTIFICATE" in cert_pem
        assert b"BEGIN EC PRIVATE KEY" in key_pem

    def test_cert_has_correct_san(self, tmp_path: Path) -> None:
        from cryptography.x509 import IPAddress, SubjectAlternativeName, load_pem_x509_certificate

        ca = ensure_ca(tls_dir=tmp_path)
        cert_pem, _ = issue_node_cert(ca, "10.0.0.5")
        cert = load_pem_x509_certificate(cert_pem)
        san = cert.extensions.get_extension_for_class(SubjectAlternativeName)
        ips = [str(ip) for ip in san.value.get_values_for_type(IPAddress)]
        assert "10.0.0.5" in ips

    def test_cert_signed_by_ca(self, tmp_path: Path) -> None:
        from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
        from cryptography.hazmat.primitives.hashes import SHA256
        from cryptography.x509 import load_pem_x509_certificate

        ca = ensure_ca(tls_dir=tmp_path)
        cert_pem, _ = issue_node_cert(ca, "10.0.0.1")
        cert = load_pem_x509_certificate(cert_pem)
        ca.cert.public_key().verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            ECDSA(SHA256()),
        )


class TestIssueClientConfig:
    def test_returns_tls_config(self, tmp_path: Path) -> None:
        from casty.remote.tls import Config

        ca = ensure_ca(tls_dir=tmp_path)
        config = issue_client_config(ca)
        assert isinstance(config, Config)
        assert isinstance(config.server_context, ssl.SSLContext)
        assert isinstance(config.client_context, ssl.SSLContext)
