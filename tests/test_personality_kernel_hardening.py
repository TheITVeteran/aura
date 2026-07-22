"""CP126 hardening contracts for core/brain/personality_kernel.py.

Verifies the value-covering seal with a backward-compatible LEGACY-format
migration (so the live instance's existing seal is re-sealed, never crashed),
seal-deletion detection, identity-key validation, and the sys.exit→exception
change. Everything uses tmp paths and a fake soul — the live ~/.aura identity
files are never touched, and PersonalityKernel() / get_kernel() are never called.
"""
from __future__ import annotations

import pytest

from core.brain.personality_kernel import KernelIntegrityError, PersonalityKernel


class _FakeSoul:
    version = "1.0"
    identity = "Aura"

    def __init__(self):
        self.intensities = {"curiosity": 0.8, "courage": 0.6}
        self.protocols = {"peer": "treat as equal", "honesty": "no theater"}


def _kernel(tmp_path, soul=None):
    k = PersonalityKernel.__new__(PersonalityKernel)
    k.soul = soul or _FakeSoul()
    k.secret_key = b"0" * 32
    k.key_file = tmp_path / ".identity_key"
    k.seal_file = tmp_path / "identity.seal"
    k.init_marker = tmp_path / ".identity_initialized"
    return k


# ── first init + 657bf82b: deletion is distinguished from first boot ───────


def test_first_init_writes_seal_and_marker(tmp_path):
    k = _kernel(tmp_path)
    assert k._verify_cryptographic_seal() is True
    assert k.seal_file.exists() and k.init_marker.exists()


def test_deleted_seal_is_detected_not_re_trusted(tmp_path):
    k = _kernel(tmp_path)
    k.init_marker.write_text("identity_initialized\n")  # a real init happened
    # ...but the seal file is gone.
    assert k._verify_cryptographic_seal() is False


# ── 66e02c75 migration: a legacy-format seal is re-sealed, not crashed ─────


def test_legacy_seal_migrates_instead_of_locking_down(tmp_path):
    k = _kernel(tmp_path)
    legacy_sig = k._sign(k._hashable_state(legacy=True))
    k.seal_file.write_text(legacy_sig)  # existing live-style legacy seal, no marker

    assert k._verify_cryptographic_seal() is True  # migrated, NOT tampering
    # The seal on disk is now the value-covering v2 signature + a marker exists.
    new_sig = k._sign(k._hashable_state(legacy=False))
    assert k.seal_file.read_text().strip() == new_sig
    assert k.init_marker.exists()


# ── 66e02c75 core: changing a trait VALUE is now detected ──────────────────


def test_value_change_is_detected(tmp_path):
    k = _kernel(tmp_path)
    assert k._verify_cryptographic_seal() is True  # seals v2 for the original soul

    # Same trait KEYS, a changed VALUE — under the old name-only seal this was
    # invisible; now it must be caught (and it must NOT false-migrate).
    tampered = _FakeSoul()
    tampered.intensities["curiosity"] = 0.1
    k2 = PersonalityKernel.__new__(PersonalityKernel)
    k2.soul = tampered
    k2.secret_key = b"0" * 32
    k2.key_file, k2.seal_file, k2.init_marker = k.key_file, k.seal_file, k.init_marker

    assert k2._verify_cryptographic_seal() is False


# ── 3cb5f9d6: identity key validation ──────────────────────────────────────


def test_bad_size_key_locks_down(tmp_path):
    k = _kernel(tmp_path)
    k.key_file.write_bytes(b"tooshort")
    with pytest.raises(KernelIntegrityError, match="BAD_SIZE"):
        k._read_validated_key()


def test_symlinked_key_is_refused(tmp_path):
    k = _kernel(tmp_path)
    target = tmp_path / "real_key"
    target.write_bytes(b"0" * 32)
    try:
        k.key_file.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(KernelIntegrityError, match="SYMLINK"):
        k._read_validated_key()


def test_valid_key_is_accepted(tmp_path):
    k = _kernel(tmp_path)
    k.key_file.write_bytes(b"k" * 32)
    import os
    os.chmod(k.key_file, 0o600)
    assert k._read_validated_key() == b"k" * 32


# ── d259fbbe: lockdown raises, it does not sys.exit ────────────────────────


def test_emergency_lockdown_raises_not_exits(tmp_path):
    k = _kernel(tmp_path)
    with pytest.raises(KernelIntegrityError):
        k._execute_emergency_lockdown("boom")
    # KernelIntegrityError is not SystemExit — a library must not kill the process.
    assert not issubclass(KernelIntegrityError, SystemExit)


# ── 3d897cee: defensive responses record a durable event ───────────────────


def test_forbidden_modification_records_degradation(tmp_path):
    from core.runtime.errors import get_degradation_tracker

    get_degradation_tracker().reset()
    k = _kernel(tmp_path)
    assert k.prevent_tampering("MUTATE", "EMOTIONAL_CORE") is False
    recent = get_degradation_tracker().recent(subsystem="personality_kernel", limit=1)
    assert recent and recent[0].severity == "critical"
