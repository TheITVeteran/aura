"""Contract tests for the SOTA deepening wave across the triad fusions.

Lean: serializable + replayable certificates, universal checker registry.
Hyperon: rule-based PLN chaining (abduction/induction), Hebbian co-focus.
Salt: grains, check.predicate / remedy.callable modules, health wiring,
and the cross-fusion trust leg (baseline replays the theorem ledger).
"""
from __future__ import annotations

import pytest

from core.knowledge.atomspace import (
    HEBBIAN,
    AtomSpace,
    InferenceRule,
    Link,
    TruthValue,
    Variable,
    concept,
    implication,
    reset_atomspace_for_test,
    standard_pln_rules,
)
from core.reasoning.natural_deduction import CertStep, parse, prove
from core.reasoning.proof_kernel import (
    check_certificate,
    get_theorem_ledger,
    prove_certified_text,
    registered_checkers,
    reset_theorem_ledger_for_test,
)
from core.runtime.homeostate import (
    HomeostateEngine,
    StateSpec,
    grains,
    register_check_predicate,
    register_remedy,
    reset_homeostate_for_test,
)


@pytest.fixture(autouse=True)
def _fresh_singletons():
    reset_theorem_ledger_for_test()
    reset_atomspace_for_test()
    reset_homeostate_for_test()
    yield
    reset_theorem_ledger_for_test()
    reset_atomspace_for_test()
    reset_homeostate_for_test()


# ── Lean wave: portability + replay + universal checkers ──────────────────

def test_certificate_serialization_round_trips_and_still_checks():
    proof = prove([parse("A"), parse("A -> B")], parse("B"))
    assert proof.certificate is not None
    encoded = proof.certificate.to_dict()
    decoded = CertStep.from_dict(encoded)
    assert decoded == proof.certificate
    verdict = check_certificate(
        "analytic_tableau", [parse("A"), parse("A -> B")], parse("B"), decoded
    )
    assert verdict.verified


def test_ledger_replay_reverifies_recorded_theorems():
    prove_certified_text(["A", "A -> B"], "B")
    prove_certified_text(["P | Q", "~P"], "Q")
    result = get_theorem_ledger().replay()
    assert result["checked"] == 2
    assert result["ok"] and result["failed"] == []


def test_ledger_replay_catches_tampered_store():
    prove_certified_text(["A", "A -> B"], "B")
    ledger = get_theorem_ledger()
    # Corrupt one stored certificate hash (simulated bit-rot / tamper).
    for entry in ledger._replayable.values():
        entry["certificate_sha256"] = "0" * 64
    result = ledger.replay()
    assert not result["ok"]
    assert result["failed"]


def test_checker_registry_is_universal():
    assert "analytic_tableau" in registered_checkers()
    verdict = check_certificate("no_such_method", [], parse("A"), None)
    assert not verdict.verified
    assert "no kernel checker" in verdict.reason


# ── Hyperon wave: PLN rules + Hebbian ─────────────────────────────────────

def test_abduction_derives_shared_consequence_link():
    space = AtomSpace(sti_fund=1000.0)
    space.add(implication(concept("rain"), concept("wet")), TruthValue(0.9, 10))
    space.add(implication(concept("sprinkler"), concept("wet")), TruthValue(0.9, 10))
    derived = space.forward_chain(max_derivations=8, focus_only=False)
    pairs = {
        (d.outgoing[0], d.outgoing[1])
        for d in derived
        if isinstance(d, Link) and d.atype == "Implication"
    }
    assert (concept("rain"), concept("sprinkler")) in pairs or (
        concept("sprinkler"),
        concept("rain"),
    ) in pairs


def test_induction_derives_shared_antecedent_link():
    space = AtomSpace(sti_fund=1000.0)
    space.add(implication(concept("cat"), concept("furry")), TruthValue(0.9, 10))
    space.add(implication(concept("cat"), concept("whiskered")), TruthValue(0.9, 10))
    derived = space.forward_chain(max_derivations=8, focus_only=False)
    pairs = {
        (d.outgoing[0], d.outgoing[1])
        for d in derived
        if isinstance(d, Link) and d.atype == "Implication"
    }
    assert (concept("furry"), concept("whiskered")) in pairs or (
        concept("whiskered"),
        concept("furry"),
    ) in pairs


def test_custom_rules_are_first_class():
    space = AtomSpace(sti_fund=1000.0)
    space.add(Link("Inheritance", (concept("dog"), concept("mammal"))), TruthValue(0.95, 10))
    space.add(Link("Inheritance", (concept("mammal"), concept("animal"))), TruthValue(0.95, 10))
    a, b, c = Variable("a"), Variable("b"), Variable("c")
    transitivity = InferenceRule(
        name="inheritance.transitivity",
        premises=(Link("Inheritance", (a, b)), Link("Inheritance", (b, c))),
        conclusion=Link("Inheritance", (a, c)),
        tv_fn=lambda s, bind, tvs: TruthValue(
            tvs[0].strength * tvs[1].strength, min(tvs[0].count, tvs[1].count) * 0.9
        ),
    )
    derived = space.apply_rules([transitivity], focus_only=False)
    assert Link("Inheritance", (concept("dog"), concept("animal"))) in derived


def test_hebbian_links_form_between_cofocused_atoms():
    space = AtomSpace(sti_fund=1000.0)
    a, b = concept("coffee"), concept("morning")
    space.add(a)
    space.add(b)
    space.stimulate(a, 50.0)
    space.stimulate(b, 40.0)
    formed = space.form_hebbian_links()
    assert any(link.atype == HEBBIAN and set(link.outgoing) == {a, b} for link in formed)
    # Repeated co-focus strengthens the association (revision raises count).
    tv_first = space.get_tv(formed[0])
    space.form_hebbian_links()
    tv_second = space.get_tv(formed[0])
    assert tv_second.count > tv_first.count


def test_spreading_prefers_hebbian_associates():
    space = AtomSpace(sti_fund=10_000.0, spread_fraction=0.5)
    hub = concept("hub")
    friend, stranger = concept("friend"), concept("stranger")
    # Two structurally identical neighbors...
    space.add(implication(hub, friend))
    space.add(implication(hub, stranger))
    # ...but one has a learned Hebbian association with the hub.
    space.add(Link(HEBBIAN, tuple(sorted((hub, friend), key=str))), TruthValue(1.0, 10.0))
    space.stimulate(hub, 100.0)
    space.spread_importance()
    assert space.get_av(friend).sti > space.get_av(stranger).sti


def test_tick_reports_hebbian_formation():
    space = AtomSpace(sti_fund=1000.0)
    a, b = concept("x"), concept("y")
    space.add(a)
    space.add(b)
    space.stimulate(a, 50.0)
    space.stimulate(b, 40.0)
    stats = space.tick()
    assert stats["hebbian_formed"] >= 1.0


# ── Salt wave: grains, universal modules, health, cross-fusion replay ─────

def test_grains_report_host_facts():
    facts = grains()
    assert facts["python_version"]
    assert facts["pid"] > 0
    assert facts["os"] in ("darwin", "linux", "windows")


def test_check_predicate_and_remedy_modules():
    hits: list[str] = []
    register_check_predicate("triad_test_check", lambda: True)

    def remedy(test: bool):
        if not test:
            hits.append("repaired")
        return {"result": True, "changes": {"repaired": not test}, "comment": "ok"}

    register_remedy("triad_test_remedy", remedy)
    engine = HomeostateEngine()
    engine.define("hs", [
        StateSpec(id="probe", fn="check.predicate", args={"name": "triad_test_check"}),
        StateSpec(id="fix", fn="remedy.callable", args={"name": "triad_test_remedy"}),
    ])
    dry = engine.apply("hs", test=True)
    assert dry.ok and hits == []                     # dry-run touched nothing
    wet = engine.apply("hs")
    assert wet.ok and hits == ["repaired"]


def test_apply_marks_subsystem_health():
    from core.runtime.errors import get_subsystem_registry

    engine = HomeostateEngine()
    engine.define("hs", [StateSpec(id="probe", fn="check.predicate", args={"name": "missing"})])
    engine.apply("hs")
    health = get_subsystem_registry().get("homeostate.hs")
    assert health is not None and health.status == "degraded"

    register_check_predicate("triad_health_ok", lambda: True)
    engine.define("hs2", [
        StateSpec(id="probe", fn="check.predicate", args={"name": "triad_health_ok"})
    ])
    engine.apply("hs2")
    health2 = get_subsystem_registry().get("homeostate.hs2")
    assert health2 is not None and health2.status == "healthy"


def test_runtime_baseline_includes_kernel_replay_leg():
    from core.runtime.homeostate import get_homeostate_engine

    prove_certified_text(["A", "A -> B"], "B")       # something to replay
    engine = get_homeostate_engine()
    report = engine.apply("runtime_baseline", test=True)
    ids = [r.id for r in report.results]
    assert "proof_kernel_replay" in ids


def test_beacon_events_carry_grains():
    from core.runtime.errors import record_degradation
    from core.runtime.homeostate import DegradationBeacon

    beacon = DegradationBeacon(window_s=300.0, threshold=2, cooldown_s=600.0)
    for _ in range(2):
        record_degradation(
            "triad_grains_beacon_subsystem",
            RuntimeError("synthetic"),
            severity="warning",
            action="test fixture",
        )
    events = [e for e in beacon.poll_once() if e["subsystem"] == "triad_grains_beacon_subsystem"]
    assert events and events[0]["grains"]["pid"] > 0
