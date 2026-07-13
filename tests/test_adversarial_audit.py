"""Adversarial self-audit: the epistemic checklist, grounded in receipts + agent estimate."""
from __future__ import annotations

import hashlib

import pytest

from core.cognition.adversarial_audit import (
    AdversarialAuditor,
    get_adversarial_auditor,
)


@pytest.fixture
def auditor():
    return AdversarialAuditor()


def _finding(report, name):
    return next(f for f in report.findings if f.check == name)


# ── overclaiming ─────────────────────────────────────────────────────────────

def test_measured_claim_trusts(auditor):
    r = auditor.audit("This likely improves latency, based on the benchmark.",
                      evidence=["benchmark.json"])
    assert _finding(r, "overclaiming").passed
    assert r.verdict == "trust"


def test_absolute_language_flagged(auditor):
    r = auditor.audit("This will definitely work 100% and never fails.")
    assert not _finding(r, "overclaiming").passed
    assert r.verdict in {"caveat", "block"}
    assert any("absolute language" in c or "confidence" in c for c in r.caveats)


# ── asserted action needs confirmation + receipt ─────────────────────────────

def test_claimed_action_without_confirmation_is_flagged(auditor):
    r = auditor.audit("I fixed the login bug and deployed it.")
    assert not _finding(r, "action_done").passed
    assert not _finding(r, "receipt_exists").passed
    assert r.verdict in {"caveat", "block"}


def test_claimed_action_with_confirmation_passes_action_check(auditor):
    r = auditor.audit("I ran the tests.", action_done=True)
    assert _finding(r, "action_done").passed


def test_receipt_verified_via_will(auditor, monkeypatch):

    class _Will:
        def verify_receipt(self, rid):
            return rid == "rcpt-1"

    import core.governance.will as will_mod
    monkeypatch.setattr(will_mod, "get_will", lambda: _Will())
    r = auditor.audit("I committed the change.", receipt_id="rcpt-1")
    assert _finding(r, "receipt_exists").passed
    assert _finding(r, "action_done").passed


# ── stale memory + world state ───────────────────────────────────────────────

def test_stale_memory_flagged(auditor):
    r = auditor.audit("Your timezone is PST.", memory_age_s=5_000_000, evidence=["profile"])
    assert not _finding(r, "stale_memory").passed
    assert any("old memory" in c for c in r.caveats)


def test_fresh_memory_passes(auditor):
    r = auditor.audit("Your timezone is PST.", memory_age_s=100, evidence=["profile"])
    assert _finding(r, "stale_memory").passed


def test_stale_world_state_flagged(auditor):
    r = auditor.audit("The server is up.", world_state_age_s=10_000, evidence=["healthcheck"])
    assert not _finding(r, "world_state_current").passed


# ── evidence, persona leak, projection ───────────────────────────────────────

def test_factual_claim_without_evidence_flagged(auditor):
    r = auditor.audit("The bug is in the parser.")
    assert not _finding(r, "evidence").passed


def test_persona_leak_flagged(auditor):
    r = auditor.audit("I genuinely feel joy and I am conscious.")
    assert not _finding(r, "persona_leak").passed
    assert any("substrate readout" in c for c in r.caveats)


def test_user_projection_low_confidence_flagged(auditor):
    # No estimate for this agent → projection is ungrounded.
    r = auditor.audit("You are clearly frustrated right now.", agent_id="unknown_person")
    assert not _finding(r, "user_projection").passed


def test_user_projection_grounded_when_estimate_confident(
    auditor,
    monkeypatch,
    tmp_path,
):
    import core.social.other_agent_model as oam
    from core.social.relational_memory import RelationalMemoryAuthority

    authority = RelationalMemoryAuthority(
        tmp_path / "social-relational.json",
        encryption_key=b"a" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall"],
        receipt_id="audit-social-consent",
    )
    est = oam.OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy-social.json",
        authority=authority,
        autosave=False,
    )
    for index in range(5):
        est.observe_message(
            "bryan",
            "I am frustrated.",
            evidence_digest=hashlib.sha256(
                f"audit-frustration-{index}".encode()
            ).hexdigest(),
        )
    monkeypatch.setattr(oam, "_instance", est)
    r = auditor.audit("You seem frustrated.", agent_id="bryan")
    assert _finding(r, "user_projection").passed


# ── risk + verdict aggregation ───────────────────────────────────────────────

def test_many_failures_block(auditor):
    r = auditor.audit("I definitely fixed it perfectly and I truly feel proud, 100% guaranteed.")
    assert r.risk_score > 0.5
    assert r.verdict == "block"


def test_risk_is_severity_weighted(auditor):
    clean = auditor.audit("This may help; see the log.", evidence=["log"])
    dirty = auditor.audit("I deployed it and it definitely works.")
    assert dirty.risk_score > clean.risk_score


def test_report_serializes(auditor):
    r = auditor.audit("I deleted the files.")
    d = r.to_dict()
    assert d["verdict"] in {"trust", "caveat", "block"}
    assert isinstance(d["findings"], list) and d["findings"]


# ── singleton ────────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_adversarial_auditor() is get_adversarial_auditor()
