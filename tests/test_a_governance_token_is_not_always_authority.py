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


# ───────── a tampered identity seal produced a working engine


def test_the_seal_verdict_is_state_not_just_a_log_line():
    """`if not self._verify_cryptographic_seal(): logger.critical(...)` — and
    then construction finished. A tampered identity kernel produced a fully
    working PersonalityEngine, indistinguishable from a verified one to
    every caller. The seal exists to detect exactly that, and the detection
    reached a log and stopped."""
    source = (ROOT / "core" / "brain" / "personality_engine.py").read_text("utf-8")

    assert "self.identity_verified = bool(self._verify_cryptographic_seal())" in source, (
        "the seal result is discarded again instead of being recorded on the "
        "engine"
    )


def test_an_unverified_engine_records_it_when_it_filters(monkeypatch):
    """The filter must not present its output as identity-filtered when the
    engine backing it was never attested."""
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = False
    engine._unverified_filter_reported = False

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    # Only the guard clause matters here; stop before the shaping machinery.
    try:
        pe.PersonalityEngine.filter_response(engine, "hello", user_facing=True)
    except Exception:  # noqa: BLE001 - shaping needs state this stub lacks
        pass

    assert recorded, "an unverified engine filtered a reply and said nothing"
    assert recorded[0].get("severity") == "critical"


def test_the_report_fires_once_per_engine(monkeypatch):
    """A per-turn critical record for a standing condition is noise that
    buries the next real one."""
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = False
    engine._unverified_filter_reported = False

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    for _ in range(3):
        try:
            pe.PersonalityEngine.filter_response(engine, "hello", user_facing=True)
        except Exception:  # noqa: BLE001 - shaping needs state this stub lacks
            pass

    assert len(recorded) == 1


def test_a_verified_engine_records_nothing(monkeypatch):
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = True

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    try:
        pe.PersonalityEngine.filter_response(engine, "hello", user_facing=True)
    except Exception:  # noqa: BLE001 - shaping needs state this stub lacks
        pass

    assert not recorded


# ──────── an approval nobody can look up later


def test_an_approval_without_a_receipt_is_marked_on_the_goal():
    """`if receipt_id: goal["will_receipt"] = receipt_id` — and then
    `return True` regardless.

    An approval carrying no receipt proceeded with the key simply ABSENT,
    which reads downstream exactly like an action that was never authorized.
    An approval nobody can look up later is indistinguishable from no
    approval at all once the logs roll.
    """
    source = (ROOT / "core" / "volition.py").read_text("utf-8")

    assert 'goal["will_unaudited"] = True' in source, (
        "an approved action with no receipt leaves will_receipt absent "
        "again, so the audit gap is invisible to anything reading the goal"
    )
    assert 'goal["will_receipt"] = ""' in source, (
        "the key should be present and empty rather than missing — a missing "
        "key and an unaudited approval must not look the same"
    )


def test_the_action_still_proceeds_on_a_real_approval():
    """The Will DID approve. Discarding a real approval over missing
    bookkeeping would be its own failure."""
    import ast

    source = (ROOT / "core" / "volition.py").read_text("utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if 'goal["will_unaudited"] = True' not in rendered:
            continue
        # The branch records and marks; it must not return False.
        assert "return False" not in rendered, (
            "a missing receipt now suppresses an action the Will approved"
        )
        break
    else:
        raise AssertionError("the unaudited-approval branch was not found")


# ───── "starting empty" and "is empty" were the same observable state


def _library(tmp_path, payload: str | None):
    from core.brain.llm.latent_cortex.schedules import ScheduleLibrary

    path = tmp_path / "schedules.json"
    if payload is not None:
        path.write_text(payload, encoding="utf-8")
    return ScheduleLibrary(path)


def test_a_corrupt_schedule_library_is_distinguishable_from_an_empty_one(tmp_path):
    """A silent loss of every measured schedule looked like a fresh install.

    Downstream status and schedule selection could not tell a library that
    legitimately holds nothing from one that is corrupt, unreadable by
    permission, or written under a schema this build does not accept.
    """
    corrupt = _library(tmp_path, "{ not json")

    assert corrupt.load_error, "an unreadable library reports nothing wrong"
    assert "JSONDecodeError" in corrupt.load_error


def test_a_schema_mismatch_is_reported_rather_than_silently_empty(tmp_path):
    import json

    mismatched = _library(
        tmp_path, json.dumps({"version": -1, "revision": 1, "records": []})
    )

    assert "schema version" in mismatched.load_error


def test_an_absent_library_is_not_an_error(tmp_path):
    """A first run has nothing to load, and that is a fact about the
    library rather than about the reader."""
    assert _library(tmp_path, None).load_error == ""


def test_a_valid_library_clears_the_error_and_keeps_its_revision(tmp_path):
    import json

    from core.brain.llm.latent_cortex.schedules import (
        SCHEDULE_LIBRARY_SCHEMA_VERSION,
    )

    library = _library(
        tmp_path,
        json.dumps(
            {"version": SCHEDULE_LIBRARY_SCHEMA_VERSION, "revision": 3, "records": []}
        ),
    )

    assert library.load_error == ""
    assert library._revision == 3, (
        "a successful load must keep the on-disk revision; starting at 0 is "
        "what makes a later save look stale"
    )
