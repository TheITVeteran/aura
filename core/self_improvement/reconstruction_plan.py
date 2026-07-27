"""A reconstruction plan any target can have, written by her, not by us.

The first version of this lane carried a registry: a hand-written oracle per
program, 2048 and checkers. That is a builder for two programs wearing the
clothes of an engine. Nothing about it generalises, and the moment the target
is anything else there is no grader at all — which is how a checkers board
whose pieces did not move got written to someone's desktop.

The general move is to make the *acceptance criteria* part of the
reconstruction rather than part of the tool. For any target she can research,
she first produces a plan:

* **decomposition** — what the thing is made of, in components, so gaps are
  filled deliberately instead of hoped over;
* **worked examples** — concrete input/output pairs she derived from research
  and reasoning, which become the differential battery;
* **invariants** — properties that must hold for every input, expressed as
  Python expressions, which catch what examples miss;
* **an adapter** — which of her own functions start the thing, enumerate what
  a user can do, apply one of those, and draw it.

The adapter is the load-bearing part, because it lets a gate that knows nothing
about the target still ask "do the pieces move?". That question is the same for
checkers, 2048, a spreadsheet and a text editor: state, actions, effect, view.

The plan is validated before a line is synthesised. A plan with no examples and
no invariants cannot grade anything, and a reconstruction that cannot be graded
is a guess with a filename.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, KeyError, IndexError)

# An invariant is an expression, not a program. No imports, no calls to
# anything but the artifact's own entry points, no attribute access into
# dunders — it is graded in a namespace we control and must not be a way to run
# arbitrary code at plan time.
_FORBIDDEN_INVARIANT_NODES = (
    ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.Delete,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
    ast.Global, ast.Nonlocal, ast.Raise, ast.Try, ast.With, ast.AsyncWith,
)


@dataclass(frozen=True)
class Component:
    """One part of the target, and what it is responsible for."""

    name: str
    responsibility: str
    entry_point: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "responsibility": self.responsibility,
            "entry_point": self.entry_point,
        }


@dataclass(frozen=True)
class WorkedExample:
    """One input and the output it must produce, with why it is known."""

    entry_point: str
    argument: Any
    expected: Any
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "argument": self.argument,
            "expected": self.expected,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Invariant:
    """A property that must hold, as an expression over the artifact."""

    description: str
    expression: str
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "expression": self.expression,
            "bindings": self.bindings,
        }


@dataclass(frozen=True)
class PlayAdapter:
    """How to drive the artifact without knowing what it is.

    Four names, and every interactive program has them under some spelling:
    what a fresh one looks like, what a user may do to it, what doing that
    produces, and how it is shown. With these, "do the pieces move?" is
    answerable for any target.
    """

    initial_state: str
    legal_actions: str
    apply_action: str
    render: str = ""
    illegal_action_examples: tuple[Any, ...] = ()
    min_effective_actions: int = 8
    # Not every program ends. A game must reach a terminal state; a calculator,
    # an editor and a spreadsheet must not be marked broken for continuing to
    # accept input. Likewise a board needs a substantial view and an empty
    # stack legitimately renders as "[]" — so the bar is declared by the plan
    # rather than assumed from games, which is how a general engine avoids
    # becoming a game grader wearing a general name.
    expects_terminal_state: bool = False
    min_render_chars: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state,
            "legal_actions": self.legal_actions,
            "apply_action": self.apply_action,
            "render": self.render,
            "illegal_action_examples": list(self.illegal_action_examples),
            "min_effective_actions": self.min_effective_actions,
            "expects_terminal_state": self.expects_terminal_state,
            "min_render_chars": self.min_render_chars,
        }


@dataclass(frozen=True)
class ReconstructionPlan:
    """Everything needed to build a target and to know whether it worked."""

    target: str
    summary: str
    components: tuple[Component, ...]
    entry_points: tuple[str, ...]
    worked_examples: tuple[WorkedExample, ...]
    invariants: tuple[Invariant, ...]
    adapter: PlayAdapter | None
    research_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "summary": self.summary,
            "components": [component.to_dict() for component in self.components],
            "entry_points": list(self.entry_points),
            "worked_examples": [example.to_dict() for example in self.worked_examples],
            "invariants": [invariant.to_dict() for invariant in self.invariants],
            "adapter": self.adapter.to_dict() if self.adapter else None,
            "research_notes": list(self.research_notes),
        }


def invariant_is_safe(expression: str) -> tuple[bool, str]:
    """An invariant is an expression over the artifact, not a second program."""
    text = str(expression or "").strip()
    if not text:
        return False, "empty expression"
    if len(text) > 600:
        return False, "expression is too long to be a property"
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        return False, f"does not parse as an expression: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_INVARIANT_NODES):
            return False, f"{type(node).__name__} is not allowed in an invariant"
        if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
            return False, "dunder access is not allowed in an invariant"
        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile", "open", "__import__"}:
            return False, f"{node.id} is not allowed in an invariant"
    return True, ""


def validate_plan(plan: ReconstructionPlan) -> list[str]:
    """Problems that would make this plan ungradeable. Empty means usable.

    Refusing an ungradeable plan before synthesis is the whole point: without
    it, the lane produces something, cannot say whether it is right, and says
    it is right anyway.
    """
    problems: list[str] = []
    if not plan.target.strip():
        problems.append("the plan names no target")
    if not plan.entry_points:
        problems.append("no entry points, so nothing can be called or graded")
    if not plan.components:
        problems.append("no decomposition — gaps cannot be filled deliberately")

    if not plan.worked_examples and not plan.invariants:
        problems.append(
            "no worked examples and no invariants: nothing would distinguish a "
            "faithful reconstruction from a plausible one"
        )
    for example in plan.worked_examples:
        if example.entry_point not in plan.entry_points:
            problems.append(
                f"example targets {example.entry_point!r}, which is not an entry point"
            )
    for invariant in plan.invariants:
        ok, why = invariant_is_safe(invariant.expression)
        if not ok:
            problems.append(f"invariant {invariant.description!r} rejected: {why}")

    if plan.adapter is not None:
        for role, name in (
            ("initial_state", plan.adapter.initial_state),
            ("legal_actions", plan.adapter.legal_actions),
            ("apply_action", plan.adapter.apply_action),
        ):
            if not str(name or "").strip():
                problems.append(f"the adapter does not say which function is {role}")
            elif name not in plan.entry_points:
                problems.append(f"adapter {role} {name!r} is not an entry point")
    return problems


def plan_from_payload(payload: Any) -> tuple[ReconstructionPlan | None, list[str]]:
    """Parse a plan she produced. Returns (plan, problems).

    Tolerant about shape and strict about substance: a missing optional field is
    fine, a malformed grading criterion is not.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, [f"plan is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["plan must be a JSON object"]

    try:
        components = tuple(
            Component(
                name=str(item.get("name") or "").strip(),
                responsibility=str(item.get("responsibility") or "").strip(),
                entry_point=str(item.get("entry_point") or "").strip(),
            )
            for item in (payload.get("components") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
        entry_points = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in (payload.get("entry_points") or [])
                if str(name).strip()
            )
        )
        examples = tuple(
            WorkedExample(
                entry_point=str(item.get("entry_point") or "").strip(),
                argument=item.get("argument"),
                expected=item.get("expected"),
                provenance=str(item.get("provenance") or "").strip(),
            )
            for item in (payload.get("worked_examples") or [])
            if isinstance(item, dict)
        )
        invariants = tuple(
            Invariant(
                description=str(item.get("description") or "").strip(),
                expression=str(item.get("expression") or "").strip(),
                bindings=dict(item.get("bindings") or {}),
            )
            for item in (payload.get("invariants") or [])
            if isinstance(item, dict)
        )
        raw_adapter = payload.get("adapter")
        adapter = None
        if isinstance(raw_adapter, dict):
            adapter = PlayAdapter(
                initial_state=str(raw_adapter.get("initial_state") or "").strip(),
                legal_actions=str(raw_adapter.get("legal_actions") or "").strip(),
                apply_action=str(raw_adapter.get("apply_action") or "").strip(),
                render=str(raw_adapter.get("render") or "").strip(),
                illegal_action_examples=tuple(raw_adapter.get("illegal_action_examples") or ()),
                min_effective_actions=int(raw_adapter.get("min_effective_actions") or 8),
                expects_terminal_state=bool(raw_adapter.get("expects_terminal_state", False)),
                min_render_chars=int(raw_adapter.get("min_render_chars") or 2),
            )
        plan = ReconstructionPlan(
            target=str(payload.get("target") or "").strip(),
            summary=str(payload.get("summary") or "").strip(),
            components=components,
            entry_points=entry_points,
            worked_examples=examples,
            invariants=invariants,
            adapter=adapter,
            research_notes=tuple(
                str(note).strip()
                for note in (payload.get("research_notes") or [])
                if str(note).strip()
            ),
        )
    except _RECOVERABLE as exc:
        return None, [f"plan could not be read: {type(exc).__name__}: {exc}"]

    return plan, validate_plan(plan)


__all__ = [
    "Component",
    "Invariant",
    "PlayAdapter",
    "ReconstructionPlan",
    "WorkedExample",
    "invariant_is_safe",
    "plan_from_payload",
    "validate_plan",
]
