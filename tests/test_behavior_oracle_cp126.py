"""CP126 contract tests for the semantic behavior oracle."""
from __future__ import annotations

import pytest

from core.architect.behavior_oracle import SemanticBehaviorOracle
from core.architect.models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
    MutationTier,
    RefactorPlan,
    RefactorStep,
    SemanticSurface,
)

PASSING = {
    "safe_boot": "passed",
    "changed_modules_import": "passed",
    "critical_tests": "passed",
}
REL = "pkg/mod.py"


def _graph(
    *,
    symbols=(("pkg.mod.api", "api", "function", ("self", "x")),),
    signatures=None,
    effects=(),
    calls=(),
    surfaces=(),
    registrations=(),
    receipts=0,
    receipt_paths=0,
) -> ArchitectureGraph:
    graph = ArchitectureGraph(root=".")
    signatures = signatures or {}
    for qualified, name, kind, args in symbols:
        graph.add_node(
            ArchitectureNode(
                id=f"{kind}:{qualified}",
                kind=kind,
                name=name,
                path=REL,
                qualified_name=qualified,
                metadata={
                    "args": tuple(args),
                    "signature": signatures.get(qualified, {"args": tuple(args)}),
                    "effects": tuple(effects),
                    "decorators": (),
                },
            )
        )
    for target in calls:
        graph.add_edge(ArchitectureEdge(source=f"file:{REL}", target=target, kind="calls", path=REL))
    for target in registrations:
        graph.add_edge(ArchitectureEdge(source=f"file:{REL}", target=target, kind="calls", path=REL))
    graph.semantic_surfaces[REL] = tuple(surfaces)
    graph.metrics["runtime_receipts"] = receipts
    graph.metrics["runtime_receipt_paths"] = receipt_paths
    return graph


def _plan(tier: MutationTier, obligations=("behavior_equivalence",)) -> RefactorPlan:
    return RefactorPlan(
        id="p",
        objective="o",
        risk_tier=tier,
        affected_files=(REL,),
        affected_symbols=(),
        semantic_surfaces=(),
        steps=(
            RefactorStep(
                id="s",
                description="edit",
                operation="replace_file",
                target_path=REL,
                new_content="VALUE = 1\n",
            ),
        ),
        proof_obligations=tuple(obligations),
        expected_smell_reduction=(),
        expected_behavior_delta="equivalent",
        promotion_eligible=True,
    )


def _verdict(tier, before, after, statuses=None):
    return SemanticBehaviorOracle().evaluate(
        _plan(tier), before, after, dict(statuses or PASSING)
    )


# --- 169305bf: T0/T1 are checked, not waved through -----------------------


@pytest.mark.parametrize("tier", [MutationTier.T0_SYNTAX_STYLE, MutationTier.T1_CLEANUP])
def test_low_tiers_are_actually_examined(tier):
    before = _graph()
    after = _graph(symbols=())

    verdict = _verdict(tier, before, after)

    assert verdict.equivalent is False
    assert verdict.evidence["contract"] == "strict_equivalence"
    assert any("pkg.mod.api" in reason for reason in verdict.regressions)


def test_a_clean_low_tier_change_still_passes():
    graph = _graph()

    verdict = _verdict(MutationTier.T1_CLEANUP, graph, _graph())

    assert verdict.equivalent is True


def test_low_tier_call_graph_changes_are_regressions():
    before = _graph(calls=("helper", "other"))
    after = _graph(calls=("helper", "replacement"))

    verdict = _verdict(MutationTier.T1_CLEANUP, before, after)

    assert verdict.equivalent is False
    assert any("call graph changed" in reason for reason in verdict.regressions)


def test_low_tier_effect_removal_is_a_regression():
    before = _graph(effects=("memory_write",))
    after = _graph(effects=())

    verdict = _verdict(MutationTier.T1_CLEANUP, before, after)

    assert any("protected effect removed" in reason for reason in verdict.regressions)


# --- 04927872: unavailable evidence is unproven, not passed ---------------


def test_missing_critical_tests_block_a_t2_change():
    graph = _graph()
    statuses = {**PASSING, "critical_tests": "not_available"}

    verdict = _verdict(MutationTier.T2_REFACTOR, graph, _graph(), statuses)

    assert verdict.equivalent is False
    assert any("unavailable" in reason for reason in verdict.regressions)


def test_absent_critical_tests_block_a_t3_change():
    graph = _graph()
    statuses = {"safe_boot": "passed", "changed_modules_import": "passed"}

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, graph, _graph(), statuses)

    assert verdict.equivalent is False
    assert verdict.evidence["critical_tests"] == "missing"


def test_unavailable_evidence_is_tolerated_but_recorded_at_t1():
    graph = _graph()
    statuses = {
        "safe_boot": "BOOT_HARNESS_UNAVAILABLE",
        "changed_modules_import": "passed",
        "critical_tests": "not_available",
    }

    verdict = _verdict(MutationTier.T1_CLEANUP, graph, _graph(), statuses)

    assert verdict.equivalent is True
    assert len(verdict.evidence["unproven"]) == 2


def test_a_genuinely_failed_test_blocks_every_tier():
    graph = _graph()
    statuses = {**PASSING, "critical_tests": "failed"}

    verdict = _verdict(MutationTier.T1_CLEANUP, graph, _graph(), statuses)

    assert verdict.equivalent is False
    assert any("did not pass" in reason for reason in verdict.regressions)


# --- 1978de68: the contract covers methods and full signatures ------------


def test_public_methods_are_part_of_the_contract():
    before = _graph(symbols=(("pkg.mod.Worker.run", "run", "method", ("self",)),))
    after = _graph(symbols=())

    verdict = _verdict(MutationTier.T2_REFACTOR, before, after)

    assert verdict.equivalent is False
    assert any("pkg.mod.Worker.run" in reason for reason in verdict.regressions)


def test_private_class_methods_are_not_public_surface():
    before = _graph(symbols=(("pkg.mod._Internal.run", "run", "method", ("self",)),))
    after = _graph(symbols=())

    assert _verdict(MutationTier.T2_REFACTOR, before, after).equivalent is True


def test_a_default_value_change_is_a_signature_change():
    args = ("self", "x")
    before = _graph(signatures={"pkg.mod.api": {"args": args, "defaults": ("1",)}})
    after = _graph(signatures={"pkg.mod.api": {"args": args, "defaults": ("2",)}})

    verdict = _verdict(MutationTier.T2_REFACTOR, before, after)

    assert any("signatures changed" in reason for reason in verdict.regressions)


def test_a_return_annotation_change_is_a_signature_change():
    args = ("self", "x")
    before = _graph(signatures={"pkg.mod.api": {"args": args, "returns": "int"}})
    after = _graph(signatures={"pkg.mod.api": {"args": args, "returns": "str"}})

    assert _verdict(MutationTier.T2_REFACTOR, before, after).equivalent is False


def test_a_keyword_only_parameter_change_is_detected():
    args = ("self",)
    before = _graph(signatures={"pkg.mod.api": {"args": args, "kwonly": ("strict:bool",)}})
    after = _graph(signatures={"pkg.mod.api": {"args": args, "kwonly": ()}})

    assert _verdict(MutationTier.T2_REFACTOR, before, after).equivalent is False


def test_the_graph_builder_records_a_full_signature():
    import ast

    from core.architect.code_graph import _FileVisitor

    source = "def f(a, b=1, *args, c: int = 2, **kw) -> str:\n    return ''\n"
    visitor = _FileVisitor.__new__(_FileVisitor)
    visitor._name = lambda node: (
        getattr(node, "id", None) or getattr(node, "attr", None) or ""
    )
    signature = visitor._signature(ast.parse(source).body[0])

    assert signature["vararg"] == "args"
    assert signature["kwarg"] == "kw"
    assert signature["kwonly"] == ("c:int",)
    assert signature["defaults"] == ("1",)
    assert signature["kw_defaults"] == ("2",)
    assert signature["returns"] == "str"
    assert signature["is_async"] is False


# --- dadd9e6d: the call graph is compared -------------------------------


def test_removed_call_paths_are_a_regression_at_t2():
    before = _graph(calls=("audit_log", "helper"))
    after = _graph(calls=("helper",))

    verdict = _verdict(MutationTier.T2_REFACTOR, before, after)

    assert verdict.equivalent is False
    assert any("call paths removed" in reason for reason in verdict.regressions)


def test_added_calls_alone_do_not_fail_a_behavioral_tier():
    before = _graph(calls=("helper",))
    after = _graph(calls=("helper", "new_helper"))

    assert _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after).equivalent is True


# --- 880aae0e: a plan string cannot waive removal ------------------------


def test_generic_caller_migration_text_does_not_waive_removal():
    before = _graph()
    after = _graph(symbols=())
    plan = _plan(MutationTier.T2_REFACTOR, obligations=("caller migration is complete",))

    verdict = SemanticBehaviorOracle().evaluate(plan, before, after, dict(PASSING))

    assert verdict.equivalent is False


def test_an_obligation_naming_the_symbol_waives_removal():
    before = _graph()
    after = _graph(symbols=())
    plan = _plan(
        MutationTier.T2_REFACTOR,
        obligations=("caller migration for pkg.mod.api verified",),
    )

    verdict = SemanticBehaviorOracle().evaluate(plan, before, after, dict(PASSING))

    assert verdict.equivalent is True


def test_a_remaining_caller_blocks_removal_even_with_the_obligation():
    before = _graph()
    after = _graph(symbols=())
    after.add_edge(
        ArchitectureEdge(source="file:other.py", target="mod.api", kind="calls", path="other.py")
    )
    plan = _plan(MutationTier.T2_REFACTOR, obligations=("caller migration for api done",))

    verdict = SemanticBehaviorOracle().evaluate(plan, before, after, dict(PASSING))

    assert verdict.equivalent is False
    assert any("still called from" in reason for reason in verdict.regressions)


# --- 62eb172e: T3 cannot evade the protected checks ----------------------


def test_t3_cannot_drop_a_protected_surface():
    before = _graph(surfaces=(SemanticSurface.AUTHORITY_GOVERNANCE,))
    after = _graph(surfaces=())

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after)

    assert verdict.equivalent is False
    assert any("protected semantic surfaces disappeared" in r for r in verdict.regressions)


def test_t3_cannot_reduce_receipt_coverage():
    before = _graph(receipts=10, receipt_paths=4)
    after = _graph(receipts=3, receipt_paths=4)

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after)

    assert any("receipt coverage decreased" in reason for reason in verdict.regressions)


def test_t3_cannot_reduce_receipt_path_coverage():
    before = _graph(receipts=3, receipt_paths=9)
    after = _graph(receipts=3, receipt_paths=1)

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after)

    assert any("path coverage decreased" in reason for reason in verdict.regressions)


def test_t3_cannot_change_service_registrations():
    before = _graph(registrations=("register_service",))
    after = _graph(registrations=())

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after)

    assert any("service registration contract changed" in r for r in verdict.regressions)


def test_increased_receipt_coverage_is_an_improvement():
    before = _graph(receipts=1)
    after = _graph(receipts=5)

    verdict = _verdict(MutationTier.T3_BEHAVIORAL_IMPROVEMENT, before, after)

    assert verdict.equivalent is True
    assert "runtime receipt coverage increased" in verdict.improvements
