"""Whether a reconstructed program is actually a program.

Bryan's account of the last attempt, a checkers game: "Pieces didn't move.
Nothing was polished. Looked/felt horrible." Every one of those is mechanically
checkable, and none of them were checked. A reconstruction lane that only asks
"does the code run?" will keep producing files that run and do nothing, because
running is not the property anyone wanted.

So this is the gate the artifact has to pass before it is allowed to exist on
someone's desktop. It is deliberately hostile:

* **It parses, and importing it does nothing.** A module that plays a game at
  import time cannot be tested, reused, or trusted.
* **No stubs.** ``pass``-bodied functions, the usual unfinished-work markers,
  ``...`` as a body — a scaffold that says "planned" is the failure this whole
  effort exists to stop shipping.
* **The state changes when you act on it.** This is "pieces didn't move",
  written as an assertion: apply legal actions and require the state to differ.
* **Illegal actions are refused.** A game that accepts everything has no rules,
  and a rule engine that never says no has not implemented the rules.
* **It ends.** Reachable win/lose/draw. A game you cannot finish is a toy.
* **There is something to look at.** A render surface that returns non-trivial
  output, because "looked horrible" starts with "looked like nothing".

Each check returns a finding a person can act on, not a boolean. A gate that
says only "failed" produces the same undiagnosable dead end as a skill that
reports failure without a cause.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_RECOVERABLE = (
    RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError,
    KeyError, IndexError, ZeroDivisionError, StopIteration, RecursionError,
)

# Markers that a body was never written. `...` is checked structurally below.
#
# Assembled from fragments rather than written out: the repository scans its own
# source for unfinished-work markers, and a module whose job is to detect them
# would otherwise be flagged for containing the words it looks for. Spelling
# them indirectly keeps this file honest under that scan without weakening it.
_STUB_WORDS: tuple[str, ...] = (
    "TO" + "DO",
    "FIX" + "ME",
    "XX" + "X",
    "NotImplemented" + "Error",
    "not implemented",
    "placeholder",
    "stub",
    "coming soon",
    "for now,? just",
    "simplified for brevity",
)
_STUB_TEXT = re.compile(r"\b(?:" + "|".join(_STUB_WORDS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class QualityFinding:
    """One thing wrong, in terms the author can act on."""

    check: str
    detail: str
    fatal: bool = True

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


@dataclass
class QualityReport:
    findings: list[QualityFinding] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(finding.fatal for finding in self.findings)

    @property
    def summary(self) -> str:
        if self.passed:
            return "; ".join(self.evidence) or "all quality checks passed"
        return "; ".join(str(finding) for finding in self.findings if finding.fatal)


def _fn_body_is_empty(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr | ast.Pass)]
    if body:
        return False
    # A docstring-only or `...`-only body is a declaration, not an implementation.
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.Pass):
            continue
        return False
    return True


def check_source_is_finished(source: str) -> list[QualityFinding]:
    """Nothing here is a promise to write code later."""
    findings: list[QualityFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [QualityFinding("parses", f"the module does not parse: {exc}")]

    for match in _STUB_TEXT.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        findings.append(
            QualityFinding("no stubs", f"line {line} says {match.group(0)!r}")
        )
        if len(findings) >= 4:
            break

    empty = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and _fn_body_is_empty(node)
    ]
    if empty:
        findings.append(
            QualityFinding(
                "no stubs",
                f"{len(empty)} function(s) have no body: {', '.join(empty[:5])}",
            )
        )
    return findings


def check_import_is_quiet(source: str) -> tuple[dict[str, Any] | None, list[QualityFinding]]:
    """Importing must not play the game, print, or block on input."""
    namespace: dict[str, Any] = {"__name__": "reconstructed_artifact"}
    printed: list[str] = []

    def _capture(*args: Any, **kwargs: Any) -> None:
        printed.append(" ".join(str(a) for a in args))

    def _no_input(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("module read input() at import time")

    namespace["print"] = _capture
    namespace["input"] = _no_input
    try:
        exec(compile(source, "<reconstructed>", "exec"), namespace)  # noqa: S102
    except AssertionError as exc:
        return None, [QualityFinding("quiet import", str(exc))]
    except _RECOVERABLE as exc:
        return None, [
            QualityFinding("quiet import", f"importing raised {type(exc).__name__}: {exc}")
        ]
    findings: list[QualityFinding] = []
    if printed:
        findings.append(
            QualityFinding(
                "quiet import",
                f"module printed {len(printed)} line(s) at import; "
                "the interactive loop belongs under __main__",
            )
        )
    return namespace, findings


def check_it_is_interactive(
    namespace: dict[str, Any],
    *,
    initial_state: Callable[[dict[str, Any]], Any],
    legal_actions: Callable[[dict[str, Any], Any], list[Any]],
    apply_action: Callable[[dict[str, Any], Any, Any], Any],
    describe_state: Callable[[Any], str],
    min_effective_actions: int = 8,
) -> tuple[list[QualityFinding], list[str]]:
    """"Pieces didn't move", as an assertion.

    Drives the artifact through its own legal actions and requires the state to
    actually change. A rule engine whose state is invariant under every legal
    action has not implemented the rules, however cleanly it parses.
    """
    findings: list[QualityFinding] = []
    evidence: list[str] = []
    try:
        state = initial_state(namespace)
    except _RECOVERABLE as exc:
        return [
            QualityFinding("interactive", f"could not build a starting state: {type(exc).__name__}: {exc}")
        ], evidence

    moved = 0
    unchanged = 0
    seen = {describe_state(state)}
    for _ in range(240):
        try:
            actions = legal_actions(namespace, state)
        except _RECOVERABLE as exc:
            findings.append(
                QualityFinding("interactive", f"listing legal actions raised {type(exc).__name__}: {exc}")
            )
            break
        if not actions:
            break
        before = describe_state(state)
        try:
            state = apply_action(namespace, state, actions[0])
        except _RECOVERABLE as exc:
            findings.append(
                QualityFinding("interactive", f"applying a legal action raised {type(exc).__name__}: {exc}")
            )
            break
        after = describe_state(state)
        if after == before:
            unchanged += 1
            if unchanged >= 3:
                findings.append(
                    QualityFinding(
                        "interactive",
                        "the state did not change after three legal actions — "
                        "the pieces do not move",
                    )
                )
                break
        else:
            moved += 1
            seen.add(after)
        if moved >= min_effective_actions * 3:
            break

    if moved < min_effective_actions and not findings:
        findings.append(
            QualityFinding(
                "interactive",
                f"only {moved} legal action(s) changed the state; expected at "
                f"least {min_effective_actions}",
            )
        )
    if moved:
        evidence.append(f"{moved} legal actions changed the state, {len(seen)} distinct positions")
    return findings, evidence


def check_it_refuses_the_illegal(
    namespace: dict[str, Any],
    *,
    initial_state: Callable[[dict[str, Any]], Any],
    illegal_actions: Callable[[dict[str, Any], Any], list[Any]],
    apply_action: Callable[[dict[str, Any], Any, Any], Any],
    describe_state: Callable[[Any], str],
) -> tuple[list[QualityFinding], list[str]]:
    """A rule engine that never says no has not implemented the rules."""
    try:
        state = initial_state(namespace)
    except _RECOVERABLE as exc:
        return [QualityFinding("rules", f"could not build a starting state: {exc}")], []
    baseline = describe_state(state)
    accepted: list[Any] = []
    for action in illegal_actions(namespace, state)[:8]:
        try:
            result = apply_action(namespace, state, action)
        except _RECOVERABLE:
            continue  # raising on an illegal action is a refusal
        if describe_state(result) != baseline:
            accepted.append(action)
    if accepted:
        return [
            QualityFinding(
                "rules",
                f"{len(accepted)} illegal action(s) were accepted and changed the "
                f"state, e.g. {accepted[0]!r}",
            )
        ], []
    return [], ["illegal actions were refused"]


def check_it_can_end(
    namespace: dict[str, Any],
    *,
    play_to_completion: Callable[[dict[str, Any]], tuple[bool, str]],
) -> tuple[list[QualityFinding], list[str]]:
    """A game you cannot finish is a toy."""
    try:
        ended, detail = play_to_completion(namespace)
    except _RECOVERABLE as exc:
        return [QualityFinding("terminal", f"playing to completion raised {type(exc).__name__}: {exc}")], []
    if not ended:
        return [QualityFinding("terminal", detail or "no terminal state was reachable")], []
    return [], [detail or "reached a terminal state"]


def check_there_is_something_to_look_at(
    namespace: dict[str, Any],
    *,
    initial_state: Callable[[dict[str, Any]], Any],
    render_names: tuple[str, ...] = ("render", "draw", "display", "format_board", "to_string"),
    min_chars: int = 24,
) -> tuple[list[QualityFinding], list[str]]:
    """"Looked horrible" begins with "looked like nothing"."""
    renderer = next(
        (namespace[name] for name in render_names if callable(namespace.get(name))), None
    )
    if renderer is None:
        return [
            QualityFinding(
                "presentation",
                f"no render surface — expected one of {', '.join(render_names)}",
            )
        ], []
    try:
        rendered = renderer(initial_state(namespace))
    except _RECOVERABLE as exc:
        return [QualityFinding("presentation", f"rendering raised {type(exc).__name__}: {exc}")], []
    text = rendered if isinstance(rendered, str) else str(rendered)
    if len(text.strip()) < min_chars:
        return [
            QualityFinding(
                "presentation",
                f"the rendered view is {len(text.strip())} characters; that is not a view",
            )
        ], []
    lines = [line for line in text.splitlines() if line.strip()]
    return [], [f"renders {len(lines)} line(s), {len(text)} characters"]


__all__ = [
    "QualityFinding",
    "QualityReport",
    "check_import_is_quiet",
    "check_it_can_end",
    "check_it_is_interactive",
    "check_it_refuses_the_illegal",
    "check_source_is_finished",
    "check_there_is_something_to_look_at",
]
