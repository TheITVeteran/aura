"""Contract tests for harvest wave 2.

Lean/mathlib: certified linear arithmetic (Farkas witnesses through the
universal kernel registry, ledger replay codec) and kernel-certified
logical-equivalence belief merging. Salt: per-state retry, orchestration,
scheduled convergence. Hyperon: backward-chaining explanations.
"""
from __future__ import annotations

import asyncio
from fractions import Fraction

import pytest

from core.reasoning.linear_arithmetic import (
    FarkasCertificate,
    check_farkas,
    check_feasible,
    find_farkas,
    parse_constraint,
    prove_linear,
)
from core.reasoning.proof_kernel import (
    get_theorem_ledger,
    registered_checkers,
    reset_theorem_ledger_for_test,
)
from core.runtime.homeostate import (
    HomeostateEngine,
    ScheduledConvergence,
    StateSpec,
    reset_homeostate_for_test,
)


@pytest.fixture(autouse=True)
def _fresh():
    reset_theorem_ledger_for_test()
    reset_homeostate_for_test()
    yield
    reset_theorem_ledger_for_test()
    reset_homeostate_for_test()


# ── linear arithmetic: parsing ────────────────────────────────────────────

def test_parse_constraint_normalizes_forms():
    (c,) = parse_constraint("2*x + 3*y <= 5")
    assert c.rhs == 5 and not c.strict
    (d,) = parse_constraint("x - y > 1")            # → -x + y < -1
    assert d.strict and d.rhs == -1
    eq = parse_constraint("x = 4")
    assert len(eq) == 2                              # two inequalities


def test_parse_rejects_nonlinear_and_disequality():
    with pytest.raises(ValueError):
        parse_constraint("x*y <= 1")
    with pytest.raises(ValueError):
        parse_constraint("x != 3")


# ── linear arithmetic: search + kernel check ──────────────────────────────

def test_infeasible_system_yields_verified_farkas_witness():
    lp = check_feasible(["x >= 3", "x <= 2"])
    assert lp.provable and lp.verified
    assert "contradiction" in lp.verdict.reason


def test_feasible_system_is_not_proved():
    lp = check_feasible(["x >= 1", "x <= 2", "y >= 0"])
    assert not lp.provable and lp.certificate is None


def test_entailment_transitive_bound():
    # x ≤ y, y ≤ z ⊢ x ≤ z
    lp = prove_linear(["x <= y", "y <= z"], "x <= z")
    assert lp.provable and lp.verified
    assert lp.verdict.uses_goal_negation


def test_entailment_with_scaling_and_strictness():
    # 2x + y ≤ 4, y ≥ 0 ⊢ x ≤ 2 ; strict premise gives strict conclusion
    lp = prove_linear(["2*x + y <= 4", "y >= 0"], "x <= 2")
    assert lp.provable and lp.verified
    strict = prove_linear(["x < 1"], "x < 2")
    assert strict.provable and strict.verified


def test_non_entailment_is_honest():
    lp = prove_linear(["x <= 5"], "x <= 4")
    assert not lp.provable


def test_axiom_audit_excludes_unused_linear_premises():
    lp = prove_linear(["x <= y", "y <= z", "w <= 100"], "x <= z")
    assert lp.verified
    assert not any("w" in p for p in lp.verdict.used_premises)


def test_kernel_rejects_forged_farkas_certificates():
    cons = parse_constraint("x >= 3") + parse_constraint("x <= 2")
    good = find_farkas(cons)
    assert good is not None and check_farkas(cons, good).verified
    # Negative multiplier: rejected.
    bad = FarkasCertificate(((0, Fraction(-1)), (1, Fraction(1))))
    assert not check_farkas(cons, bad).verified
    # Non-cancelling combination: rejected.
    lopsided = FarkasCertificate(((0, Fraction(2)), (1, Fraction(1))))
    assert not check_farkas(cons, lopsided).verified
    # Satisfiable combination presented as refutation: rejected.
    feasible = parse_constraint("x <= 5") + parse_constraint("-x <= 5")
    sat = FarkasCertificate(((0, Fraction(1)), (1, Fraction(1))))
    assert not check_farkas(feasible, sat).verified


def test_farkas_method_registered_and_replayable():
    assert "farkas_linear" in registered_checkers()
    prove_linear(["x <= y", "y <= z"], "x <= z")
    ledger = get_theorem_ledger()
    assert ledger.stats()["theorems"] >= 1
    result = ledger.replay()
    assert result["ok"] and result["checked"] >= 1


def test_symbolic_bridge_prove_linear_live():
    from core.reasoning.symbolic_bridge import SymbolicBridge

    res = SymbolicBridge().prove_linear(["x <= y", "y <= z"], "x <= z")
    assert res.ok and res.result is True
    assert "kernel-verified" in res.proof_trace
    not_entailed = SymbolicBridge().prove_linear(["x <= 5"], "x <= 4")
    assert not_entailed.ok and not_entailed.result is False


def test_symbolic_bridge_solve_constraints_prefers_certified_path():
    from core.reasoning.symbolic_bridge import SymbolicBridge

    res = SymbolicBridge().solve_constraints(["x >= 3", "x <= 2"])
    assert res.ok
    assert res.engine == "farkas_linear"
    assert "kernel-verified" in res.proof_trace


# ── kernel-certified equivalence merge ────────────────────────────────────

@pytest.mark.asyncio
async def test_contrapositive_claims_merge_into_one_belief(tmp_path):
    from core.epistemics.belief_revision import BeliefRevisionEngine

    engine = BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))
    first = await engine.process_new_claim(
        "if the power fails then the backup starts", "world", "tool_result", 0.9
    )
    assert first["ok"] and not first.get("updated")
    count_before = len(engine.beliefs)
    # Contrapositive: logically equivalent, differently worded.
    second = await engine.process_new_claim(
        "if the backup does not start then the power does not fail",
        "world",
        "tool_result",
        0.9,
    )
    assert second["ok"] and second.get("updated") is True
    assert len(engine.beliefs) == count_before        # merged, not duplicated


@pytest.mark.asyncio
async def test_non_equivalent_claims_stay_separate(tmp_path):
    from core.epistemics.belief_revision import BeliefRevisionEngine

    engine = BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))
    await engine.process_new_claim("the reactor is stable", "world", "tool_result", 0.9)
    before = len(engine.beliefs)
    await engine.process_new_claim("the reactor is not stable", "world", "tool_result", 0.9)
    assert len(engine.beliefs) == before + 1          # negation is NOT the same belief


# ── Salt wave 2 ───────────────────────────────────────────────────────────

def test_state_retry_succeeds_on_later_attempt():
    attempts: list[int] = []

    def flaky(test=False, watch_triggered=False, **_):
        attempts.append(1)
        ok = len(attempts) >= 3
        return {"result": ok, "changes": {}, "comment": "ok" if ok else "flaky"}

    engine = HomeostateEngine()
    engine.registry.register("t.flaky", flaky)
    engine.define("hs", [StateSpec(id="f", fn="t.flaky", retries=3)])
    report = engine.apply("hs")
    assert report.ok
    assert len(attempts) == 3
    assert "retry" in report.results[0].comment


def test_dry_run_never_retries():
    attempts: list[int] = []

    def flaky(test=False, watch_triggered=False, **_):
        attempts.append(1)
        return {"result": False, "changes": {}, "comment": "flaky"}

    engine = HomeostateEngine()
    engine.registry.register("t.flaky", flaky)
    engine.define("hs", [StateSpec(id="f", fn="t.flaky", retries=5)])
    engine.apply("hs", test=True)
    assert len(attempts) == 1


def test_orchestrate_runs_in_order_and_stops_on_failure():
    order: list[str] = []

    def make(name: str, ok: bool):
        def fn(test=False, watch_triggered=False, **_):
            order.append(name)
            return {"result": ok, "changes": {}, "comment": ""}
        return fn

    engine = HomeostateEngine()
    engine.registry.register("t.a", make("a", True))
    engine.registry.register("t.b", make("b", False))
    engine.registry.register("t.c", make("c", True))
    engine.define("stage_a", [StateSpec(id="a", fn="t.a")])
    engine.define("stage_b", [StateSpec(id="b", fn="t.b")])
    engine.define("stage_c", [StateSpec(id="c", fn="t.c")])
    reports = engine.orchestrate(["stage_a", "stage_b", "stage_c"])
    assert order == ["a", "b"]                        # c never ran
    assert list(reports) == ["stage_a", "stage_b"]
    assert not reports["stage_b"].ok


@pytest.mark.asyncio
async def test_scheduled_convergence_reapplies_highstate():
    runs: list[float] = []

    def counting(test=False, watch_triggered=False, **_):
        runs.append(1.0)
        return {"result": True, "changes": {}, "comment": ""}

    engine = HomeostateEngine()
    engine.registry.register("t.count", counting)
    engine.define("hs", [StateSpec(id="c", fn="t.count")])
    scheduler = ScheduledConvergence(engine, "hs", interval_s=0.05)
    scheduler.start()
    try:
        for _ in range(60):
            if scheduler.runs >= 2:
                break
            await asyncio.sleep(0.05)
        assert scheduler.runs >= 2
        assert len(runs) >= 2
    finally:
        await scheduler.stop()


# ── Hyperon wave 2: backward chaining ─────────────────────────────────────

def test_explain_finds_supporting_chain():
    from core.knowledge.atomspace import AtomSpace, TruthValue, concept, implication

    space = AtomSpace(sti_fund=1000.0)
    space.add(implication(concept("rain"), concept("wet")), TruthValue(0.9, 10))
    space.add(implication(concept("wet"), concept("slippery")), TruthValue(0.8, 10))
    target = implication(concept("rain"), concept("slippery"))
    explanations = space.explain(target)
    assert explanations, "expected a supporting chain"
    top = explanations[0]
    assert top["hops"] == 2
    assert any("rain" in step for step in top["chain"])
    assert 0.0 < top["strength"] <= 1.0


def test_explain_respects_depth_bound():
    from core.knowledge.atomspace import AtomSpace, TruthValue, concept, implication

    space = AtomSpace(sti_fund=1000.0)
    for i in range(6):
        space.add(implication(concept(f"n{i}"), concept(f"n{i + 1}")), TruthValue(0.9, 10))
    target = implication(concept("n0"), concept("n6"))
    assert space.explain(target, max_depth=3) == []   # 6 hops > bound
    assert space.explain(target, max_depth=6)         # reachable within bound
