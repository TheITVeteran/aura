"""core/security/tls_local.py
──────────────────────────
Local self-signed TLS for the LAN surface.

Why: browsers require a secure context for getUserMedia — phone
microphone capture over plain HTTP is impossible no matter what the
server allows. A locally generated self-signed certificate (accepted
once on the phone) makes https://<lan-ip>:8000 a secure context, which
combined with the owner-granted per-device voice scope opens the phone
voice lane.

The certificate is generated once, keyed 0600, SANs covering localhost
and the host's current LAN addresses, 2-year validity, regenerated when
expired or when the LAN address set changes. Enabled only when
AURA_ENABLE_TLS=1 — the desktop app's plain-HTTP loopback default is
untouched otherwise.
"""
from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import socket
from pathlib import Path

from core.config import get_config
from core.runtime.errors import record_degradation

logger = logging.getLogger("Security.TLSLocal")

_TLS_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)


def tls_enabled() -> bool:
    return os.environ.get("AURA_ENABLE_TLS", "").strip().lower() in {"1", "true", "yes", "on"}


def tls_dir() -> Path:
    return Path(get_config().paths.data_dir) / "security" / "tls"


def _lan_ips() -> list[str]:
    addresses: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # route-selection only
            primary = probe.getsockname()[0]
            if primary:
                addresses.append(primary)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = info[4][0]
            if candidate not in addresses:
                addresses.append(candidate)
    except OSError:
        pass
    return addresses


def ensure_local_certificate() -> tuple[Path, Path] | None:
    """Create (or reuse) the local self-signed cert. Returns
    (cert_path, key_path), or None when generation is impossible."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        directory = tls_dir()
        cert_path, key_path = directory / "aura_local.crt", directory / "aura_local.key"
        wanted_ips = sorted(set(_lan_ips()) | {"127.0.0.1"})

        if cert_path.exists() and key_path.exists():
            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            not_after = certificate.not_valid_after_utc
            current_ips = {
                str(ip) for ip in certificate.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                ).value.get_values_for_type(x509.IPAddress)
            }
            fresh = not_after > datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
            if fresh and set(wanted_ips) <= current_ips:
                return cert_path, key_path
            logger.info("Regenerating local TLS cert (expiring or LAN set changed)")

        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Aura Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aura"),
        ])
        san = x509.SubjectAlternativeName(
            [x509.DNSName("localhost")]
            + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in wanted_ips]
        )
        now = datetime.datetime.now(datetime.UTC)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=730))
            .add_extension(san, critical=False)
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        directory.mkdir(parents=True, exist_ok=True)
        key_bytes = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        # Key first, restrictive before content lands anywhere readable.
        key_path.touch(mode=0o600, exist_ok=True)
        os.chmod(key_path, 0o600)
        key_path.write_bytes(key_bytes)
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        logger.info("Local TLS certificate ready (SANs: %s)", ", ".join(wanted_ips))
        return cert_path, key_path
    except _TLS_ERRORS as exc:
        record_degradation("security.tls_local", exc)
        logger.error("Local TLS certificate unavailable: %s", exc)
        return None
