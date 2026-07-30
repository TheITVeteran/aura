import base64
import hashlib
import json
import os
from pathlib import Path

from core.security.emergency_protocol import (
    MAX_VAULT_SNAPSHOTS,
    EmergencyProtocol,
)


def _protocol(tmp_path: Path) -> EmergencyProtocol:
    protocol = EmergencyProtocol()
    protocol._get_vault_path = lambda: tmp_path / "vault"
    protocol._get_machine_id = lambda: "machine-id"
    protocol._collect_state = lambda: {"identity_hint": "Aura", "value": 42}
    return protocol


def test_emergency_snapshot_uses_authenticated_vault_format(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)

    snapshot_path = protocol.take_snapshot_now()

    assert snapshot_path is not None
    snapshot_bytes = snapshot_path.read_bytes()
    assert snapshot_bytes.startswith(b"AURA_VAULT_V2:")
    assert b'"value": 42' not in snapshot_bytes
    assert protocol.recover_from_snapshot(snapshot_path) == {"identity_hint": "Aura", "value": 42}


def test_emergency_decrypt_reads_legacy_stream_format(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    key = protocol._derive_encryption_key()
    data = b'{"identity_hint": "Aura"}'
    blocks = (len(data) + 31) // 32
    expanded = b"".join(hashlib.sha256(key + counter.to_bytes(4, "big")).digest() for counter in range(blocks))
    encoded = b"AURA_VAULT_V1:" + base64.b64encode(
        bytes(a ^ b for a, b in zip(data, expanded[: len(data)], strict=True))
    )

    assert protocol._decrypt(encoded, key) == data


def test_emergency_snapshot_rotation_uses_file_age(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    for index in range(MAX_VAULT_SNAPSHOTS + 3):
        path = vault / f"snapshot_{index}.enc"
        path.write_bytes(str(index).encode())
        os.utime(path, (index, index))

    protocol._rotate_vault_snapshots()

    remaining = sorted(vault.glob("snapshot_*.enc"), key=lambda path: path.stat().st_mtime)
    assert len(remaining) == MAX_VAULT_SNAPSHOTS
    assert remaining[0].name == "snapshot_3.enc"


def test_emergency_recovery_returns_none_for_unknown_snapshot(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    snapshot = tmp_path / "snapshot_unknown.enc"
    snapshot.write_bytes(b"unknown")

    assert protocol.recover_from_snapshot(snapshot) is None


def test_emergency_recovery_manifest_points_to_snapshot(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)

    snapshot_path = protocol.take_snapshot_now()

    manifest = json.loads((tmp_path / "vault" / "recovery_manifest.json").read_text(encoding="utf-8"))
    assert snapshot_path is not None
    assert manifest["snapshot_file"] == snapshot_path.name
    assert manifest["identity_hint"] == "Aura"


def test_emergency_signal_retains_structured_threat_evidence(monkeypatch) -> None:
    protocol = EmergencyProtocol()
    monkeypatch.setattr(protocol, "_log_threat", lambda _signal: None)
    monkeypatch.setattr(protocol, "_respond_to_threat", lambda: None)

    protocol.flag_threat(
        "immune:network_sentinel",
        "corroborated novel device",
        0.65,
        evidence={
            "ip": "192.168.1.99",
            "mac": "de:ad:be:ef:00:99",
            "interface": "en0",
        },
        threat_class="intrusion",
    )

    signal = protocol._signals[-1]
    assert signal.threat_class == "intrusion"
    assert signal.evidence == {
        "ip": "192.168.1.99",
        "mac": "de:ad:be:ef:00:99",
        "interface": "en0",
    }
