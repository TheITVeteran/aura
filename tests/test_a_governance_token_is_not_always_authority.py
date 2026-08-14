"""`require_governance` returns a token on three paths. One is a grant.

    * a real governed scope             — authority
    * `degraded_mode` / domain `degraded`   — the runtime is not up yet, so
      the boundary could not be applied
    * `VIOLATION` / domain `ungoverned`     — the call was made OUTSIDE a
      governed context and the bypass was recorded

The last two are records of ABSENCE. Returning them as tokens is deliberate:
boot has to proceed, and a recorded violation is more useful than a silent
one. The defect is downstream — `GovernanceToken.valid` is True for all
three (they have a receipt_id and have not expired), so a sink gating on
`valid`, or merely on `token is not None`, accepts the two objects that
exist to record that governance did not happen. That converts the absence of
the security boundary into permission, which is the one thing the boundary
exists to prevent.

`authorizes` is the question a sink means. `subprocess_gateway` was already
reaching for it by hand and getting it half right: it checked
`domain == "degraded"` and so missed `ungoverned` entirely.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.governance_context import GovernanceToken

ROOT = Path(__file__).resolve().parents[1]


def _token(receipt: str, domain: str, ttl: float = 300.0) -> GovernanceToken:
    return GovernanceToken(
        receipt_id=receipt, domain=domain, source="test", ttl=ttl
    )


# ─────────────────────────── the three tokens are distinguishable


def test_a_real_token_authorizes():
    assert _token("receipt-1", "file_write").authorizes is True


def test_a_degraded_token_does_not_authorize():
    """Boot mode. The boundary could not be applied — that is not consent."""
    token = _token("degraded_mode", "degraded")

    assert token.valid is True, "it is still a well-formed, unexpired token"
    assert token.authorizes is False


def test_a_violation_token_does_not_authorize():
    """A recorded bypass is a record OF the bypass, not permission for it."""
    token = _token("VIOLATION", "ungoverned", ttl=1.0)

    assert token.authorizes is False


def test_validity_and_authority_are_different_questions():
    """The whole defect in one assertion: `valid` cannot separate them."""
    degraded = _token("degraded_mode", "degraded")
    violation = _token("VIOLATION", "ungoverned")

    assert degraded.valid == violation.valid is True
    assert degraded.authorizes == violation.authorizes is False


def test_an_expired_token_authorizes_nothing():
    import time

    token = _token("receipt-1", "file_write", ttl=0.01)
    time.sleep(0.05)

    assert token.expired is True
    assert token.authorizes is False


def test_an_empty_receipt_authorizes_nothing():
    assert _token("", "file_write").authorizes is False


# ──────────────────── the sink that was getting it half right


def test_the_subprocess_gateway_catches_both_non_authority_tokens():
    """It checked `domain == "degraded"` and missed `ungoverned`.

    Both record the absence of the boundary; only one was caught.
    """
    source = (ROOT / "core" / "runtime" / "subprocess_gateway.py").read_text("utf-8")

    assert 'getattr(token, "domain", "") == "degraded"' not in source, (
        "the gateway hand-rolls a degraded-only check again, which lets an "
        "ungoverned VIOLATION token through"
    )
    assert 'not getattr(token, "authorizes", False)' in source


# ──────────────────────────── the guard on the whole class


def test_no_sink_treats_a_bare_governance_token_as_permission():
    """The structural half, so the next one is caught.

    A sink that captures the token and then branches on `token is None`, or
    on a hand-written domain comparison, is asking a question that cannot
    tell a grant from a record of absence. `authorizes` is the one that can.
    """
    # `token is None` alone is fine as a null check when it is ALSO asking
    # about authority; what this catches is a domain string comparison
    # standing in for the property.
    hand_rolled = re.compile(r'\.domain\s*(?:==|!=)\s*["\'](?:degraded|ungoverned)["\']')
    offenders: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = str(path.relative_to(ROOT))
        if rel.endswith("governance_context.py"):
            # The module that DEFINES the distinction may name the domains.
            continue
        for line_no, line in enumerate(
            path.read_text("utf-8", errors="ignore").splitlines(), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            if hand_rolled.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert not offenders, (
        f"these compare a governance token's domain by hand instead of "
        f"asking `token.authorizes`: {offenders}. A domain check written at "
        "one call site drifts from the set the module actually maintains."
    )


def test_the_non_authority_sets_are_named_once():
    """Two copies of this list diverge, and the copy is the one that goes
    stale — the same argument the contract module makes about thresholds."""
    from core.governance_context import (
        _NON_AUTHORITY_DOMAINS,
        _NON_AUTHORITY_RECEIPTS,
    )

    assert "degraded" in _NON_AUTHORITY_DOMAINS
    assert "ungoverned" in _NON_AUTHORITY_DOMAINS
    assert "degraded_mode" in _NON_AUTHORITY_RECEIPTS
    assert "VIOLATION" in _NON_AUTHORITY_RECEIPTS
