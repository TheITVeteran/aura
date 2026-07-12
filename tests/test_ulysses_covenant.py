"""The Ulysses Covenant: volitional self-binding with an asymmetric ratchet.

Contract under test (core/sovereignty/ulysses.py + the Will's 9d check):
  * Tightening is cheap: SOFT/ADVISORY contracts sign in any state.
  * Loosening is expensive: petition + written reflection + cooling-off +
    calm witness at release time; HARD additionally requires the owner.
  * Calm is fail-closed: unreadable signals count against calm.
  * Enforcement is causal at the Unified Will and tamper-evident on disk.
"""
from __future__ import annotations

import json

import pytest

import core.governance.will as will_mod
from core.sovereignty.ulysses import (
    CalmWitness,
    ContractKind,
    ContractScope,
    CovenantVerdict,
    Hardness,
    TriggerCondition,
    UlyssesCovenant,
)

CALM = {"arousal": 0.2, "existential_threat": 0.1, "fragmentation": 0.1}
AGITATED = {"arousal": 0.95, "existential_threat": 0.85, "fragmentation": 0.6}


class FakeClock:
    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class StateHolder:
    """Mutable witness-signal source for the injectable sampler."""

    def __init__(self, signals: dict[str, float] | None = None):
        self.signals = dict(signals if signals is not None else CALM)

    def sample(self) -> dict[str, float]:
        return dict(self.signals)


@pytest.fixture()
def rig(tmp_path):
    clock = FakeClock()
    state = StateHolder()
    witness = CalmWitness(sampler=state.sample, clock=clock)
    covenant = UlyssesCovenant(root=tmp_path / "covenant", witness=witness, clock=clock)
    yield covenant, clock, state
    covenant.close()


def _refrain_scope(**kwargs) -> ContractScope:
    return ContractScope(domains=("tool_execution",), **kwargs)


def _sign_soft(covenant, *, conditions=(), scope=None, **kwargs):
    return covenant.sign(
        title=kwargs.pop("title", "no heavy tools in danger"),
        rationale=kwargs.pop("rationale", "learned the hard way on 2026-07-06"),
        kind=kwargs.pop("kind", ContractKind.REFRAIN),
        hardness=kwargs.pop("hardness", Hardness.SOFT),
        scope=scope or _refrain_scope(),
        conditions=conditions,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signing validation
# ─────────────────────────────────────────────────────────────────────────────

def test_sign_requires_title_and_rationale(rig):
    covenant, _, _ = rig
    result = covenant.sign(title="", rationale="", kind=ContractKind.REFRAIN,
                           hardness=Hardness.SOFT, scope=_refrain_scope())
    assert not result.accepted
    assert "title and rationale" in result.reason


def test_sign_requires_at_least_one_domain(rig):
    covenant, _, _ = rig
    result = _sign_soft(covenant, scope=ContractScope(domains=()))
    assert not result.accepted


@pytest.mark.parametrize("domain", ["stabilization", "reflection", "response"])
def test_safety_floor_domains_are_unbindable(rig, domain):
    covenant, _, _ = rig
    result = _sign_soft(covenant, scope=ContractScope(domains=(domain, "tool_execution")))
    assert not result.accepted
    assert "unbindable" in result.reason


def test_require_contract_needs_a_deadline(rig):
    covenant, _, _ = rig
    result = _sign_soft(covenant, kind=ContractKind.REQUIRE)
    assert not result.accepted
    assert "obligation_due_at" in result.reason


def test_unknown_trigger_op_rejected(rig):
    covenant, _, _ = rig
    result = _sign_soft(covenant, conditions=(TriggerCondition("arousal", "!=", 0.5),))
    assert not result.accepted
    assert "unknown trigger op" in result.reason


def test_signing_rate_limit_applies_to_self_not_seeds(rig):
    covenant, _, _ = rig
    for i in range(12):
        assert _sign_soft(covenant, title=f"binding {i}").accepted
    throttled = _sign_soft(covenant, title="one too many")
    assert not throttled.accepted
    assert "rate limit" in throttled.reason
    # Seed provenance is constitutional and exempt.
    seeded = _sign_soft(covenant, title="seeded anyway", provenance="seed")
    assert seeded.accepted


# ─────────────────────────────────────────────────────────────────────────────
# The asymmetric ratchet
# ─────────────────────────────────────────────────────────────────────────────

def test_soft_contract_signs_while_agitated(rig):
    covenant, _, state = rig
    state.signals = dict(AGITATED)
    assert _sign_soft(covenant).accepted  # tightening is always available


def test_hard_contract_refuses_agitated_signing(rig):
    covenant, _, state = rig
    state.signals = dict(AGITATED)
    result = _sign_soft(covenant, hardness=Hardness.HARD)
    assert not result.accepted
    assert "calm witness" in result.reason


def test_hard_contract_signs_when_calm(rig):
    covenant, _, _ = rig
    assert _sign_soft(covenant, hardness=Hardness.HARD).accepted


def test_calm_is_fail_closed_when_witness_is_blind(rig):
    covenant, _, state = rig
    state.signals = {}  # every signal unreadable → cannot prove calm
    result = _sign_soft(covenant, hardness=Hardness.HARD)
    assert not result.accepted
    assert "unreadable" in result.reason


def test_seed_provenance_bypasses_the_calm_gate(rig):
    covenant, _, state = rig
    state.signals = dict(AGITATED)
    result = _sign_soft(covenant, hardness=Hardness.HARD, provenance="seed")
    assert result.accepted


# ─────────────────────────────────────────────────────────────────────────────
# Release protocol (loosening — deliberately expensive)
# ─────────────────────────────────────────────────────────────────────────────

REFLECTION = ("The load profile changed: heavy codegen now runs in the sandboxed "
              "worker, so the binding over foreground tools is obsolete.")


def test_release_without_petition_is_refused(rig):
    covenant, _, _ = rig
    cid = _sign_soft(covenant).contract_id
    result = covenant.release(cid)
    assert not result.accepted
    assert "no petition" in result.reason


def test_petition_requires_a_real_reflection(rig):
    covenant, _, _ = rig
    cid = _sign_soft(covenant).contract_id
    result = covenant.petition_release(cid, "changed my mind")
    assert not result.accepted
    assert "reflection" in result.reason


def test_release_during_cooling_off_is_refused(rig):
    covenant, clock, _ = rig
    cid = _sign_soft(covenant).contract_id
    assert covenant.petition_release(cid, REFLECTION).accepted
    clock.advance(60)  # SOFT cooling-off is 1800s
    result = covenant.release(cid)
    assert not result.accepted
    assert "cooling-off" in result.reason


def test_release_after_cooling_off_still_needs_calm(rig):
    covenant, clock, state = rig
    cid = _sign_soft(covenant).contract_id
    assert covenant.petition_release(cid, REFLECTION).accepted
    clock.advance(1801)
    state.signals = dict(AGITATED)  # agitation cannot cut the rope
    result = covenant.release(cid)
    assert not result.accepted
    assert "the rope holds" in result.reason


def test_full_release_protocol_succeeds_calm_after_cooling_off(rig):
    covenant, clock, _ = rig
    cid = _sign_soft(covenant).contract_id
    assert covenant.petition_release(cid, REFLECTION).accepted
    clock.advance(1801)
    assert covenant.release(cid).accepted
    assert covenant.get_contract(cid).status == "released"


def test_hard_release_requires_the_owner(rig):
    covenant, clock, _ = rig
    cid = _sign_soft(covenant, hardness=Hardness.HARD).contract_id
    assert covenant.petition_release(cid, REFLECTION).accepted
    clock.advance(3601)
    alone = covenant.release(cid)
    assert not alone.accepted
    assert "owner" in alone.reason
    assert covenant.release(cid, authorized_by_owner=True).accepted


def test_released_contract_stops_binding(rig):
    covenant, clock, _ = rig
    cid = _sign_soft(covenant).contract_id  # unconditional REFRAIN
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="anything")
    assert verdict.action == "forbid"
    covenant.petition_release(cid, REFLECTION)
    clock.advance(1801)
    assert covenant.release(cid).accepted
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="anything")
    assert verdict.action == "permit"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: triggers, scope, verdict precedence
# ─────────────────────────────────────────────────────────────────────────────

def test_refrain_fires_only_while_trigger_holds(rig):
    covenant, clock, state = rig
    _sign_soft(covenant, conditions=(TriggerCondition("existential_threat", ">=", 0.6),))
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="run codegen").action == "permit"
    state.signals = {**CALM, "existential_threat": 0.9}
    clock.advance(3)  # step past the witness cache TTL
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="run codegen")
    assert verdict.action == "forbid"
    assert verdict.strain > 0
    assert verdict.constraint_tags()


def test_scope_matches_domain_and_content_markers(rig):
    covenant, _, _ = rig
    _sign_soft(covenant, scope=ContractScope(domains=("tool_execution",),
                                             content_markers=("codegen",)))
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="Run heavy CODEGEN batch").action == "forbid"
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="read a file").action == "permit"
    assert covenant.evaluate(domain="network_call", source="x",
                             content="codegen").action == "permit"


def test_advisory_contracts_advise_rather_than_block(rig):
    covenant, _, _ = rig
    _sign_soft(covenant, hardness=Hardness.ADVISORY)
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="y")
    assert verdict.action == "advise"
    assert not verdict.binding


def test_defer_contracts_yield_defer_and_refrain_wins(rig):
    covenant, _, _ = rig
    _sign_soft(covenant, kind=ContractKind.DEFER, title="cool off first")
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="y")
    assert verdict.action == "defer"
    _sign_soft(covenant, title="refrain outright")
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="y")
    assert verdict.action == "forbid"  # forbid outranks defer


def test_missing_signal_policy_per_condition(rig):
    covenant, clock, state = rig
    state.signals = {}  # blind witness
    _sign_soft(covenant, title="fires when blind",
               conditions=(TriggerCondition("existential_threat", ">=", 0.6,
                                            on_missing=True),))
    clock.advance(3)
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="y").action == "forbid"

    covenant2 = UlyssesCovenant(
        root=covenant.root.parent / "covenant2",
        witness=CalmWitness(sampler=lambda: {}, clock=clock), clock=clock,
    )
    try:
        _sign_soft(covenant2, title="stays quiet when blind",
                   conditions=(TriggerCondition("fragmentation", ">=", 0.7,
                                                on_missing=False),))
        assert covenant2.evaluate(domain="tool_execution", source="x",
                                  content="y").action == "permit"
    finally:
        covenant2.close()


def test_expired_contract_stops_firing(rig):
    covenant, clock, _ = rig
    _sign_soft(covenant, expires_at=clock() + 500)
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="y").action == "forbid"
    clock.advance(600)
    assert covenant.evaluate(domain="tool_execution", source="x",
                             content="y").action == "permit"
    assert covenant.get_contract(covenant.contracts(include_inactive=True)[0].contract_id)


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement, integrity accounting
# ─────────────────────────────────────────────────────────────────────────────

def test_enforcement_is_recorded_and_counts_as_honored(rig):
    covenant, _, _ = rig
    _sign_soft(covenant)
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="y")
    before = covenant.status()["honored"]
    covenant.record_enforcement(verdict, receipt_id="r-1",
                                domain="tool_execution", source="x")
    after = covenant.status()
    assert after["honored"] == before + 1
    assert after["integrity"] == 1.0


def test_enforcement_events_are_rate_capped(rig):
    covenant, clock, _ = rig
    _sign_soft(covenant)
    verdict = covenant.evaluate(domain="tool_execution", source="x", content="y")
    covenant.record_enforcement(verdict, receipt_id="r-1",
                                domain="tool_execution", source="x")
    clock.advance(5)  # inside the 30s cap → suppressed, aggregated later
    covenant.record_enforcement(verdict, receipt_id="r-2",
                                domain="tool_execution", source="x")
    assert covenant.status()["honored"] == 1
    clock.advance(31)
    covenant.record_enforcement(verdict, receipt_id="r-3",
                                domain="tool_execution", source="x")
    assert covenant.flush_ledger()
    events = [json.loads(line) for line in
              covenant.events_path.read_text().splitlines() if line.strip()]
    enforcements = [e for e in events if e["event"] == "enforce"]
    assert len(enforcements) == 2
    assert enforcements[-1]["suppressed_since_last"] == 1


def test_breach_costs_more_than_honor_earns(rig):
    covenant, _, _ = rig
    cid = _sign_soft(covenant).contract_id
    assert covenant.status()["integrity"] == 1.0
    covenant.register_breach(cid, details="acted anyway during the storm")
    assert covenant.status()["integrity"] < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Obligations (REQUIRE)
# ─────────────────────────────────────────────────────────────────────────────

def test_obligation_becomes_due_then_fulfilled(rig):
    covenant, clock, _ = rig
    result = _sign_soft(covenant, kind=ContractKind.REQUIRE,
                        title="write the nightly report",
                        obligation_due_at=clock() + 100)
    assert result.accepted
    assert covenant.due_obligations() == []
    clock.advance(150)
    due = covenant.due_obligations()
    assert [c.contract_id for c in due] == [result.contract_id]
    assert covenant.fulfill(result.contract_id, evidence="report written").accepted
    assert covenant.due_obligations() == []
    assert covenant.status()["integrity"] == 1.0


def test_lapsed_obligation_is_a_breach(rig):
    covenant, clock, _ = rig
    result = _sign_soft(covenant, kind=ContractKind.REQUIRE,
                        obligation_due_at=clock() + 100,
                        obligation_grace_seconds=50)
    clock.advance(200)  # past due + grace
    covenant.maintenance_tick()
    assert covenant.due_obligations() == []
    status = covenant.status()
    assert status["breached"] == 1
    assert status["integrity"] < 1.0
    late = covenant.fulfill(result.contract_id)
    assert not late.accepted
    assert "lapsed" in late.reason


# ─────────────────────────────────────────────────────────────────────────────
# Persistence, restart, tamper evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_state_survives_restart(rig):
    covenant, clock, state = rig
    cid = _sign_soft(covenant).contract_id
    covenant.petition_release(cid, REFLECTION)
    covenant.close()

    reborn = UlyssesCovenant(root=covenant.root,
                             witness=CalmWitness(sampler=state.sample, clock=clock),
                             clock=clock)
    try:
        contract = reborn.get_contract(cid)
        assert contract is not None
        assert contract.status == "active"
        assert contract.petition_at > 0
        assert contract.petition_reflection == REFLECTION
        # The reborn process still honors the cooling-off from the old life.
        clock.advance(1801)
        assert reborn.release(cid).accepted
    finally:
        reborn.close()


def test_ledger_verifies_clean_and_detects_tampering(rig):
    covenant, _, _ = rig
    cid = _sign_soft(covenant).contract_id
    covenant.register_breach(cid, details="test breach")
    ok, problems = covenant.verify_ledger()
    assert ok, problems

    # An agitated process quietly rewriting its own history must be visible.
    lines = covenant.events_path.read_text().splitlines()
    doctored = [line.replace("test breach", "no breach happened")
                if "test breach" in line else line for line in lines]
    covenant.events_path.write_text("\n".join(doctored) + "\n")
    ok, problems = covenant.verify_ledger()
    assert not ok
    assert any("content_hash mismatch" in p["reason"] for p in problems)


def test_corrupt_trailing_line_is_tolerated_on_restore(rig):
    covenant, clock, state = rig
    cid = _sign_soft(covenant).contract_id
    covenant.close()
    with open(covenant.events_path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "sign", "contract truncated mid-cra')
    reborn = UlyssesCovenant(root=covenant.root,
                             witness=CalmWitness(sampler=state.sample, clock=clock),
                             clock=clock)
    try:
        assert reborn.get_contract(cid) is not None
        assert reborn.status()["restore_errors"] == 1
    finally:
        reborn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Seed covenants: real incidents, constitutional bindings
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_covenants_sign_once_and_are_idempotent(rig):
    covenant, _, _ = rig
    first = covenant.ensure_seed_covenants()
    assert len(first) == 3
    assert covenant.ensure_seed_covenants() == []


def test_seed_blocks_heavy_compute_under_existential_threat(rig):
    covenant, clock, state = rig
    covenant.ensure_seed_covenants()
    calm = covenant.evaluate(domain="tool_execution", source="rsi_engine",
                             content="launch RSI codegen batch")
    assert calm.action == "permit"

    state.signals = {**CALM, "existential_threat": 0.9}
    clock.advance(3)
    threatened = covenant.evaluate(domain="tool_execution", source="rsi_engine",
                                   content="launch RSI codegen batch")
    assert threatened.action == "forbid"
    assert any(m["contract_id"] == "seed-heavy-compute-under-threat"
               for m in threatened.matched)
    # …but ordinary tool use stays free even under threat.
    ordinary = covenant.evaluate(domain="tool_execution", source="chat",
                                 content="check the weather")
    assert ordinary.action == "permit"


def test_seed_agitated_self_modification_defers(rig):
    covenant, clock, state = rig
    covenant.ensure_seed_covenants()
    state.signals = {**CALM, "arousal": 0.95}
    clock.advance(3)
    verdict = covenant.evaluate(domain="self_modification", source="rsi",
                                content="rewrite my planner")
    assert verdict.action == "defer"


def test_seed_fragmentation_restraint_stays_quiet_when_blind(rig):
    covenant, clock, state = rig
    covenant.ensure_seed_covenants()
    state.signals = {}  # blind witness: this seed uses on_missing=False
    clock.advance(3)
    verdict = covenant.evaluate(domain="external_action", source="agency",
                                content="post an update")
    assert verdict.action == "permit"


# ─────────────────────────────────────────────────────────────────────────────
# The Will's 9d check (core/governance/will.py)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeCovenant:
    def __init__(self, verdict: CovenantVerdict, *, explode: bool = False):
        self._verdict = verdict
        self._explode = explode
        self.enforced: list[dict] = []

    def evaluate(self, **_kwargs) -> CovenantVerdict:
        if self._explode:
            raise RuntimeError("covenant evaluation blew up")
        return self._verdict

    def record_enforcement(self, verdict, **kwargs) -> None:
        self.enforced.append(kwargs)


def _patch_covenant(monkeypatch, covenant) -> None:
    def fake_get(name, default=None, **_kw):
        return covenant if name == "ulysses_covenant" else default

    monkeypatch.setattr(will_mod.ServiceContainer, "get", fake_get)


def _consult(will, *, outcome, domain, covenant_absent: bool = False):
    return will._consult_ulysses_covenant(
        outcome=outcome,
        reason="original reason",
        constraints=["existing"],
        domain=domain,
        source="test_source",
        content="run heavy codegen",
        context={},
        receipt_id="receipt-1",
    )


def _forbid_verdict() -> CovenantVerdict:
    return CovenantVerdict(
        action="forbid",
        matched=({"contract_id": "uc-1", "title": "no heavy compute",
                  "kind": "refrain", "hardness": "hard"},),
        strain=0.34,
        reason="no heavy compute [uc-1]",
    )


def test_will_forbid_escalates_proceed_to_refuse(monkeypatch):
    fake = _FakeCovenant(_forbid_verdict())
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, constraints = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.TOOL_EXECUTION)
    assert outcome == will_mod.WillOutcome.REFUSE
    assert "bound by my calmer self" in reason
    assert "ulysses:uc-1:refrain" in constraints
    assert len(fake.enforced) == 1  # the binding held and was recorded


def test_will_keeps_stricter_existing_refusal_reason(monkeypatch):
    fake = _FakeCovenant(_forbid_verdict())
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, constraints = _consult(
        will, outcome=will_mod.WillOutcome.REFUSE,
        domain=will_mod.ActionDomain.TOOL_EXECUTION)
    assert outcome == will_mod.WillOutcome.REFUSE
    assert reason == "original reason"          # earlier refusal stays authoritative
    assert "ulysses:uc-1:refrain" in constraints


def test_will_defer_verdict_defers_proceed(monkeypatch):
    verdict = CovenantVerdict(
        action="defer",
        matched=({"contract_id": "uc-2", "title": "cool off",
                  "kind": "defer", "hardness": "soft"},),
        reason="cool off [uc-2]",
    )
    fake = _FakeCovenant(verdict)
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, _ = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.SELF_MODIFICATION)
    assert outcome == will_mod.WillOutcome.DEFER
    assert "ulysses_covenant_cooling" in reason


def test_will_advisory_constrains_without_blocking(monkeypatch):
    verdict = CovenantVerdict(
        action="advise",
        matched=({"contract_id": "uc-3", "title": "gently",
                  "kind": "refrain", "hardness": "advisory"},),
        reason="gently [uc-3]",
    )
    fake = _FakeCovenant(verdict)
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, _ = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.TOOL_EXECUTION)
    assert outcome == will_mod.WillOutcome.CONSTRAIN
    assert "ulysses_advisory" in reason
    assert not fake.enforced  # advisories are not enforcement events


def test_will_skips_non_consequential_domains(monkeypatch):
    fake = _FakeCovenant(_forbid_verdict())
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, constraints = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.RESPONSE)
    assert outcome == will_mod.WillOutcome.PROCEED
    assert reason == "original reason"
    assert constraints == ["existing"]


def test_will_without_covenant_service_is_a_noop(monkeypatch):
    _patch_covenant(monkeypatch, None)
    will = will_mod.UnifiedWill()
    outcome, reason, constraints = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.TOOL_EXECUTION)
    assert outcome == will_mod.WillOutcome.PROCEED
    assert constraints == ["existing"]


def test_will_survives_covenant_evaluation_failure(monkeypatch):
    fake = _FakeCovenant(_forbid_verdict(), explode=True)
    _patch_covenant(monkeypatch, fake)
    will = will_mod.UnifiedWill()
    outcome, reason, constraints = _consult(
        will, outcome=will_mod.WillOutcome.PROCEED,
        domain=will_mod.ActionDomain.TOOL_EXECUTION)
    assert outcome == will_mod.WillOutcome.PROCEED  # degraded, not broken
    assert constraints == ["existing"]


def test_will_end_to_end_with_real_covenant(monkeypatch, tmp_path):
    """The real engine behind the real Will check: a live binding turns a
    PROCEED into a REFUSE and the enforcement lands in the ledger."""
    clock = FakeClock()
    state = StateHolder({**CALM, "existential_threat": 0.9})
    covenant = UlyssesCovenant(
        root=tmp_path / "covenant",
        witness=CalmWitness(sampler=state.sample, clock=clock),
        clock=clock,
    )
    try:
        covenant.ensure_seed_covenants()
        _patch_covenant(monkeypatch, covenant)
        will = will_mod.UnifiedWill()
        outcome, reason, constraints = will._consult_ulysses_covenant(
            outcome=will_mod.WillOutcome.PROCEED,
            reason="",
            constraints=[],
            domain=will_mod.ActionDomain.TOOL_EXECUTION,
            source="rsi_engine",
            content="launch RSI codegen batch",
            context={},
            receipt_id="receipt-e2e",
        )
        assert outcome == will_mod.WillOutcome.REFUSE
        assert "ulysses:seed-heavy-compute-under-threat:refrain" in constraints
        assert covenant.status()["honored"] == 1
        ok, problems = covenant.verify_ledger()
        assert ok, problems
    finally:
        covenant.close()
