"""Grade any reconstructed artifact against its own plan.

This is the piece that makes the lane an engine rather than a builder for two
programs. Nothing here knows what the target is. It takes the source she wrote
and the plan she wrote, and asks the same questions of both:

* does it stand up as code — parses, imports quietly, contains no stubs;
* does it reproduce her own worked examples;
* do her own invariants hold;
* and, through the plan's adapter, does it *behave* — the pieces move, illegal
  actions are refused, it can end, and there is something to look at.

The adapter is what generalises the behavioural half. Every interactive program
has a starting state, a set of things a user may do, an effect for doing one,
and a view — checkers, 2048, a spreadsheet, an editor. Given four function
names, "do the pieces move?" is answerable without knowing which program it is.

A target with no adapter is still graded on code quality, examples and
invariants; it simply is not claimed to be interactive. Silence about a
property is better than a claim nothing checked.
"""
from __future__ import annotations

from typing import Any

from core.self_improvement.artifact_quality import (
    QualityFinding,
    QualityReport,
    check_import_is_quiet,
    check_it_can_end,
    check_it_is_interactive,
    check_it_refuses_the_illegal,
    check_source_is_finished,
    check_there_is_something_to_look_at,
)
from core.self_improvement.reconstruction_plan import ReconstructionPlan

_RECOVERABLE = (
    RuntimeError, AttributeError, TypeError, ValueError, KeyError, IndexError,
    ZeroDivisionError, StopIteration, RecursionError, OSError,
)


def _check_entry_points(namespace: dict[str, Any], plan: ReconstructionPlan) -> list[QualityFinding]:
    missing = [name for name in plan.entry_points if not callable(namespace.get(name))]
    if missing:
        return [
            QualityFinding(
                "entry points",
                f"the plan promises {', '.join(missing[:5])} and the module does not define "
                f"{'them' if len(missing) > 1 else 'it'}",
            )
        ]
    return []


def _check_worked_examples(
    namespace: dict[str, Any], plan: ReconstructionPlan
) -> tuple[list[QualityFinding], list[str]]:
    """Her own examples, run against what she wrote."""
    if not plan.worked_examples:
        return [], []
    failed: list[str] = []
    passed = 0
    for example in plan.worked_examples:
        fn = namespace.get(example.entry_point)
        if not callable(fn):
            failed.append(f"{example.entry_point} is not callable")
            continue
        try:
            actual = fn(example.argument)
        except _RECOVERABLE as exc:
            failed.append(f"{example.entry_point}({example.argument!r}) raised {type(exc).__name__}")
            continue
        if actual == example.expected:
            passed += 1
        else:
            failed.append(
                f"{example.entry_point}: expected {example.expected!r}, got {actual!r}"
            )
    if failed:
        return [
            QualityFinding(
                "worked examples",
                f"{passed}/{len(plan.worked_examples)} reproduced; first failure — {failed[0]}",
            )
        ], []
    return [], [f"{passed}/{len(plan.worked_examples)} worked examples reproduced"]


def _check_invariants(
    namespace: dict[str, Any], plan: ReconstructionPlan
) -> tuple[list[QualityFinding], list[str]]:
    """Properties that must hold for every input, not just the examples."""
    if not plan.invariants:
        return [], []
    findings: list[QualityFinding] = []
    held = 0
    for invariant in plan.invariants:
        scope: dict[str, Any] = {
            name: value for name, value in namespace.items() if not name.startswith("__")
        }
        scope.update(invariant.bindings or {})
        try:
            outcome = eval(invariant.expression, {"__builtins__": {}}, scope)  # noqa: S307
        except _RECOVERABLE as exc:
            findings.append(
                QualityFinding(
                    "invariants",
                    f"{invariant.description!r} could not be evaluated: {type(exc).__name__}: {exc}",
                )
            )
            continue
        except NameError as exc:
            findings.append(
                QualityFinding("invariants", f"{invariant.description!r} references {exc}")
            )
            continue
        if outcome:
            held += 1
        else:
            findings.append(
                QualityFinding("invariants", f"{invariant.description!r} does not hold")
            )
    evidence = [f"{held}/{len(plan.invariants)} invariants hold"] if held else []
    return findings, evidence


def _behavioural_checks(
    namespace: dict[str, Any], plan: ReconstructionPlan
) -> tuple[list[QualityFinding], list[str]]:
    """The adapter turns "is it a working program?" into four function calls."""
    adapter = plan.adapter
    if adapter is None:
        return [], ["no adapter declared; behaviour was not claimed either way"]

    def _initial(ns: dict[str, Any]) -> Any:
        return ns[adapter.initial_state]()

    def _legal(ns: dict[str, Any], state: Any) -> list[Any]:
        return list(ns[adapter.legal_actions](state) or [])

    def _apply(ns: dict[str, Any], state: Any, action: Any) -> Any:
        return ns[adapter.apply_action](state, action)

    def _describe(state: Any) -> str:
        try:
            return repr(state)[:4000]
        except _RECOVERABLE:
            return str(id(state))

    def _illegal(_ns: dict[str, Any], _state: Any) -> list[Any]:
        return list(adapter.illegal_action_examples)

    def _to_completion(ns: dict[str, Any]) -> tuple[bool, str]:
        state = _initial(ns)
        for step in range(400):
            actions = _legal(ns, state)
            if not actions:
                return True, f"reached a terminal state after {step} actions"
            state = _apply(ns, state, actions[0])
        return False, "no terminal state within 400 actions"

    findings: list[QualityFinding] = []
    evidence: list[str] = []

    part_findings, part_evidence = check_it_is_interactive(
        namespace,
        initial_state=_initial,
        legal_actions=_legal,
        apply_action=_apply,
        describe_state=_describe,
        min_effective_actions=adapter.min_effective_actions,
    )
    findings.extend(part_findings)
    evidence.extend(part_evidence)

    if adapter.illegal_action_examples:
        part_findings, part_evidence = check_it_refuses_the_illegal(
            namespace,
            initial_state=_initial,
            illegal_actions=_illegal,
            apply_action=_apply,
            describe_state=_describe,
        )
        findings.extend(part_findings)
        evidence.extend(part_evidence)

    if adapter.expects_terminal_state:
        part_findings, part_evidence = check_it_can_end(
            namespace, play_to_completion=_to_completion
        )
        findings.extend(part_findings)
        evidence.extend(part_evidence)

    if adapter.render:
        part_findings, part_evidence = check_there_is_something_to_look_at(
            namespace,
            initial_state=_initial,
            render_names=(adapter.render,),
            min_chars=adapter.min_render_chars,
        )
        findings.extend(part_findings)
        evidence.extend(part_evidence)
    return findings, evidence


def grade_artifact(source: str, plan: ReconstructionPlan) -> QualityReport:
    """Everything that can be checked about this artifact, without knowing it.

    Ordered so the cheapest and most decisive checks run first: source that does
    not parse cannot be imported, and a module that will not import cannot be
    played.
    """
    report = QualityReport()

    report.findings.extend(check_source_is_finished(source))
    if not report.passed:
        return report

    namespace, findings = check_import_is_quiet(source)
    report.findings.extend(findings)
    if namespace is None:
        return report

    report.findings.extend(_check_entry_points(namespace, plan))
    if not report.passed:
        return report

    for part_findings, part_evidence in (
        _check_worked_examples(namespace, plan),
        _check_invariants(namespace, plan),
        _behavioural_checks(namespace, plan),
    ):
        report.findings.extend(part_findings)
        report.evidence.extend(part_evidence)
    return report


__all__ = ["grade_artifact"]
