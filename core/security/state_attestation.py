"""Did Aura write this, or did something else?

``IntegrityGuardian`` hashes Aura's *code* — ``core/``, ``interface/``,
``aura_main.py`` — against an HMAC-signed manifest. Nothing hashed her
*state*. Identity lived in ``data/aura_self_profile.json`` and was loaded on
boot by whoever last wrote the file, with no question asked about who that
was.

That is the gap the January 2026 ClawHavoc campaign walked through on the
other side of the fence. Hundreds of malicious skills were published for
OpenClaw, and the persistence mechanism was not a code change: they rewrote
``MEMORY.md`` and ``SOUL.md``. Editing the agent's memory is better than
editing its code — it survives a reinstall, it reads as something the agent
learned, and no integrity check was looking at it because memory is
*supposed* to change.

Which is exactly why a code-style manifest cannot be pointed at it. A hash
of a file that legitimately changes every time Aura reinforces a fact would
alarm constantly and be switched off within a week. The question worth
asking is not "has this file changed" but "was this file changed **by
Aura**".

So each write is attested: Aura seals a digest of what she just wrote into
the :class:`~core.security.governance_vault.GovernanceVault`, whose own
hash chain is already tamper-evident. On load, the file is digested and
compared. A file Aura wrote matches. A file something else wrote does not.

**What this does and does not stop.** It detects modification by anything
that does not also update the vault — an editor, a sync client, a restored
backup, a skill that writes JSON because writing JSON is what it does. That
is the ClawHavoc shape. It does NOT stop an attacker who knows Aura's code
well enough to re-seal after tampering; they run as the same user and can
reach the vault too. The honest claim is out-of-band detection, and it is
written here rather than discovered during an incident.

The verdict is a state, never a bool, because "no seal exists yet" and "the
seal does not match" are opposite situations and collapsing them would
either quarantine every first run or wave through every tamper.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.StateAttestation")

#: Prefix keeping attested state clear of the vault's constitutional artifacts.
ARTIFACT_PREFIX = "state_attestation:"


class AttestationState:
    """What the seal said."""

    #: Digest matches the seal. Aura wrote this.
    TRUSTED = "trusted"
    #: No seal existed. First run, or state that predates attestation — the
    #: content is adopted and sealed now (trust on first use), because the
    #: alternative is deleting the identity of every instance that upgrades.
    ADOPTED = "adopted"
    #: A seal exists and the content does not match it.
    TAMPERED = "tampered"
    #: The vault could not answer. Not an accusation, and not a clean bill of
    #: health either — callers are told the state is unverified and decide.
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class AttestationVerdict:
    state: str
    artifact_id: str
    detail: str = ""

    @property
    def is_tampered(self) -> bool:
        return self.state == AttestationState.TAMPERED

    @property
    def is_verified(self) -> bool:
        """True only for a seal that matched. ADOPTED and UNVERIFIABLE are not
        verifications — they are the absence of one, and a caller that reads
        them as a pass has reinvented the defect this module exists to find."""
        return self.state == AttestationState.TRUSTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "artifact_id": self.artifact_id,
            "detail": self.detail,
            "verified": self.is_verified,
        }


_last_verdicts: dict[str, AttestationVerdict] = {}


def digest(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return hashlib.sha256(data).hexdigest()


def _vault():
    from core.security.governance_vault import get_governance_vault

    return get_governance_vault()


def attest_state(artifact_id: str, content: str | bytes) -> bool:
    """Seal the digest of what was just written. Returns whether it stuck.

    Never raises. A save that succeeded must not be turned into a failure
    because the attestation could not be recorded — but an unattested write
    means the NEXT load reads ADOPTED rather than TRUSTED, so the miss is
    recorded rather than passed over.
    """
    key = f"{ARTIFACT_PREFIX}{artifact_id}"
    try:
        _vault().seal(key, {"digest": digest(content)})
        return True
    except Exception as exc:  # noqa: BLE001 - a persistence path must not raise here
        record_degradation(
            "state_attestation",
            exc,
            severity="warning",
            action=f"state {artifact_id} written but not attested; next load will read as adopted",
            enforce_failure_policy=False,
        )
        return False


def verify_state(artifact_id: str, content: str | bytes) -> AttestationVerdict:
    """Was this content written by Aura?

    Adopts and seals when no seal exists. Never raises.
    """
    key = f"{ARTIFACT_PREFIX}{artifact_id}"
    computed = digest(content)
    try:
        vault = _vault()
        if not vault.has_artifact(key):
            vault.seal(key, {"digest": computed})
            verdict = AttestationVerdict(
                AttestationState.ADOPTED,
                artifact_id,
                "no prior seal; content adopted and sealed",
            )
            _last_verdicts[artifact_id] = verdict
            return verdict

        sealed = vault.unseal(key)
        stored = str((sealed or {}).get("digest") or "") if isinstance(sealed, dict) else ""
        if stored and stored == computed:
            verdict = AttestationVerdict(AttestationState.TRUSTED, artifact_id)
        else:
            verdict = AttestationVerdict(
                AttestationState.TAMPERED,
                artifact_id,
                f"sealed digest {stored[:12]}… does not match on-disk {computed[:12]}…",
            )
            logger.critical(
                "State attestation FAILED for %s — the file changed outside Aura's "
                "own write path. %s",
                artifact_id,
                verdict.detail,
            )
    except Exception as exc:  # noqa: BLE001 - a boot path must not raise here
        record_degradation(
            "state_attestation",
            exc,
            severity="warning",
            action=f"could not verify {artifact_id}; treated as unverified rather than trusted",
            enforce_failure_policy=False,
        )
        verdict = AttestationVerdict(
            AttestationState.UNVERIFIABLE, artifact_id, f"vault unavailable: {exc}"
        )

    _last_verdicts[artifact_id] = verdict
    return verdict


def attestation_report() -> dict[str, Any]:
    """Every attested state and how it last verified, for health reporting."""
    verdicts = {name: v.to_dict() for name, v in _last_verdicts.items()}
    return {
        "checked": len(verdicts),
        "tampered": sorted(n for n, v in _last_verdicts.items() if v.is_tampered),
        "unverified": sorted(
            n for n, v in _last_verdicts.items() if not v.is_verified and not v.is_tampered
        ),
        "artifacts": verdicts,
    }


def reset_attestation_for_test() -> None:
    _last_verdicts.clear()


__all__ = [
    "ARTIFACT_PREFIX",
    "AttestationState",
    "AttestationVerdict",
    "attest_state",
    "attestation_report",
    "digest",
    "reset_attestation_for_test",
    "verify_state",
]
