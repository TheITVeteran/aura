"""Strict terminal reason and model-generated disclosure contract for RLC episodes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256

SCHEMA = "aura.rlc.terminal_disposition.v1"
POLICY_VERSION = "spark-053.v1"

VERIFIED_CONVERGENCE = "verified_convergence"
LOW_VALUE = "low_value_of_further_compute"
COMPUTE_BUDGET = "compute_budget_exhausted"
WALL_BUDGET = "wall_clock_budget_exhausted"
RECURRENCE_BUDGET = "recurrence_budget_exhausted"
IRREDUCIBLE_UNCERTAINTY = "irreducible_uncertainty"
VERIFIED_ANSWER = "verified_answer"
EXTERNAL_ACTION_READY = "external_action_ready"
PLANNED_DEPTH_COMPLETE = "planned_depth_complete"
STABILITY_CONTAINMENT = "stability_containment"
INTERRUPTED = "interrupted"
UNCLASSIFIED_TERMINATION = "unclassified_termination"

_REASON_PRECEDENCE = (
    IRREDUCIBLE_UNCERTAINTY,
    WALL_BUDGET,
    COMPUTE_BUDGET,
    RECURRENCE_BUDGET,
    LOW_VALUE,
    VERIFIED_CONVERGENCE,
    VERIFIED_ANSWER,
    EXTERNAL_ACTION_READY,
    STABILITY_CONTAINMENT,
    INTERRUPTED,
    PLANNED_DEPTH_COMPLETE,
    UNCLASSIFIED_TERMINATION,
)

_INSTRUCTIONS = {
    VERIFIED_CONVERGENCE: (
        "The recurrent state reached a verified stable point. Answer in your own words "
        "from the stable result, preserve calibrated uncertainty, and do not narrate "
        "internal stopping mechanics unless the user asks."
    ),
    LOW_VALUE: (
        "Measured evidence says more recurrent computation is unlikely to improve this "
        "answer enough to justify its cost. Answer in your own words from the best-supported "
        "result and disclose any material unresolved limitation."
    ),
    COMPUTE_BUDGET: (
        "The computation allocation ended before uncertainty was fully resolved. Respond in "
        "your own words, clearly distinguish established findings from unfinished reasoning, "
        "and do not present the result as complete or certain."
    ),
    WALL_BUDGET: (
        "The available reasoning time ended before uncertainty was fully resolved. Respond in "
        "your own words, clearly distinguish established findings from unfinished reasoning, "
        "and state what further work would resolve it."
    ),
    RECURRENCE_BUDGET: (
        "The planned recurrent depth was exhausted without verified resolution. Give only the "
        "best bounded answer in your own words, disclose the unresolved part, and identify the "
        "next discriminating check when one is known."
    ),
    IRREDUCIBLE_UNCERTAINTY: (
        "Available evidence cannot presently distinguish the live alternatives. Do not invent "
        "an answer. In your own words, state what remains uncertain, which missing evidence "
        "matters, and what observation or action could resolve it."
    ),
    VERIFIED_ANSWER: (
        "A verifier-backed answer is available. Answer directly in your own words, preserve "
        "the evidence boundary, and do not add unsupported certainty."
    ),
    EXTERNAL_ACTION_READY: (
        "A verified external action is ready for the governed execution lane. In your own "
        "words, describe only the action and evidence actually authorized; do not claim that "
        "an effect occurred until its execution receipt exists."
    ),
    PLANNED_DEPTH_COMPLETE: (
        "The planned reasoning program completed. Answer in your own words from the resulting "
        "evidence and make any material uncertainty explicit."
    ),
    STABILITY_CONTAINMENT: (
        "The recurrent path was contained after instability and restored to its best verified "
        "state. Answer in your own words only from that restored evidence and disclose any "
        "material limitation caused by the containment."
    ),
    INTERRUPTED: (
        "Reasoning was interrupted. Do not imply completion. In your own words, state what was "
        "completed, what remains unresolved, and whether a retry is needed."
    ),
    UNCLASSIFIED_TERMINATION: (
        "The reasoning path ended without enough evidence to certify why. Do not claim a fully "
        "resolved answer; respond in your own words with the supported portion and disclose the "
        "uncertainty."
    ),
}

_DISCLOSURE_REASONS = frozenset(
    {
        LOW_VALUE,
        COMPUTE_BUDGET,
        WALL_BUDGET,
        RECURRENCE_BUDGET,
        IRREDUCIBLE_UNCERTAINTY,
        STABILITY_CONTAINMENT,
        INTERRUPTED,
        UNCLASSIFIED_TERMINATION,
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_sha256(tokens: Sequence[int]) -> str:
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("terminal-disposition token sequence is invalid")
    return canonical_sha256(list(tokens))


def _bridge_token_sha256(tokens: Sequence[int]) -> str:
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("terminal-disposition bridge token sequence is invalid")
    raw = json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
    return _sha256_bytes(raw)


def _terminal_action(trace: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    if not trace:
        return "", "", {}
    row = trace[-1]
    decision = row.get("decision")
    state = row.get("state_signal")
    if not isinstance(decision, Mapping) or not isinstance(state, Mapping):
        raise ValueError("terminal action trace is malformed")
    action = decision.get("action")
    mode = decision.get("mode")
    if not isinstance(action, str) or not isinstance(mode, str):
        raise ValueError("terminal action decision is malformed")
    return action, mode, dict(state)


def _selected_loop_branch(loop_stability: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = loop_stability.get("selected_branch")
    branches = loop_stability.get("branches")
    if type(selected) is not int or not isinstance(branches, list) or selected not in range(len(branches)):
        raise ValueError("terminal disposition lacks selected loop evidence")
    branch = branches[selected]
    if not isinstance(branch, Mapping):
        raise ValueError("terminal disposition loop branch is malformed")
    return branch


def _last_learned_stop(halting: Mapping[str, Any], selected_branch: int) -> Mapping[str, Any] | None:
    branches = halting.get("branches")
    if not isinstance(branches, list) or selected_branch not in range(len(branches)):
        return None
    decisions = branches[selected_branch].get("decisions")
    if not isinstance(decisions, list):
        return None
    return next(
        (
            row
            for row in reversed(decisions)
            if isinstance(row, Mapping) and row.get("halt") is True
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    disposition: str
    reason: str
    requires_disclosure: bool
    instruction: str
    evidence: dict[str, Any]


def classify_terminal_disposition(
    *,
    halting_reason: str,
    halting: Mapping[str, Any],
    loop_stability: Mapping[str, Any],
    cognitive_action_trace: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> TerminalDecision:
    """Classify one terminal state from independently committed public evidence."""

    if not isinstance(halting_reason, str) or not halting_reason:
        raise ValueError("terminal halting reason is missing")
    action, mode, state = _terminal_action(cognitive_action_trace)
    branch = _selected_loop_branch(loop_stability)
    selected = int(loop_stability["selected_branch"])
    stop = _last_learned_stop(halting, selected)
    budget_exhausted = budget.get("exhausted") is True
    elapsed = budget.get("elapsed_s")
    wall_limit = budget.get("wall_clock_s")
    wall_exhausted = bool(
        budget_exhausted
        and isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and isinstance(wall_limit, (int, float))
        and not isinstance(wall_limit, bool)
        and float(elapsed) >= float(wall_limit)
    )

    if action == "abstain" and mode == "irreducible_abstain":
        reason, disposition = IRREDUCIBLE_UNCERTAINTY, "abstain"
    elif wall_exhausted:
        reason, disposition = WALL_BUDGET, "bounded_answer"
    elif budget_exhausted:
        reason, disposition = COMPUTE_BUDGET, "bounded_answer"
    elif mode == "budget_abstain":
        reason, disposition = RECURRENCE_BUDGET, "abstain"
    elif mode == "budget_stop" or "budget" in halting_reason or "max_steps" in halting_reason:
        reason, disposition = RECURRENCE_BUDGET, "bounded_answer"
    elif (
        stop is not None
        and halting.get("head_was_causal") is True
        and stop.get("reason") == "learned_stop"
        and stop.get("evidence_ready") is True
        and isinstance(stop.get("features"), Mapping)
        and float(stop["features"].get("expected_net_value", 1.0)) <= 0.0
    ):
        reason, disposition = LOW_VALUE, "answer"
    elif halting_reason.startswith("converged"):
        transitions = branch.get("transitions")
        if (
            not isinstance(transitions, list)
            or not transitions
            or transitions[-1].get("fixed_point_candidate") is not True
            or loop_stability.get("all_finite") is not True
            or loop_stability.get("all_accepted_states_anchor_bounded") is not True
        ):
            raise ValueError("convergence lacks fixed-point and stability evidence")
        reason, disposition = VERIFIED_CONVERGENCE, "answer"
    elif action == "answer" and mode == "verified_stop" and state.get("answer_verified") is True:
        reason, disposition = VERIFIED_ANSWER, "answer"
    elif action == "execute" and mode == "verified_execute" and state.get("answer_verified") is True:
        reason, disposition = EXTERNAL_ACTION_READY, "execute"
    elif "diverg" in halting_reason or halting_reason.endswith("_reverted"):
        reason, disposition = STABILITY_CONTAINMENT, "bounded_answer"
    elif "cancel" in halting_reason or "interrupt" in halting_reason:
        reason, disposition = INTERRUPTED, "defer"
    elif halting_reason in {"fixed_depth", "schedule_complete", "value_controller_answer"}:
        reason, disposition = PLANNED_DEPTH_COMPLETE, "answer"
    else:
        reason, disposition = UNCLASSIFIED_TERMINATION, "bounded_answer"

    evidence = {
        "halting_reason": halting_reason,
        "halting_receipt_sha256": halting.get("receipt_sha256", ""),
        "loop_stability_sha256": loop_stability.get("receipt_sha256", ""),
        "cognitive_action_trace_sha256": canonical_sha256(list(cognitive_action_trace)),
        "budget_at_decision": dict(budget),
        "budget_snapshot_sha256": canonical_sha256(dict(budget)),
        "selected_branch": selected,
        "selected_action": action,
        "selected_mode": mode,
        "uncertainty": state.get("uncertainty"),
        "verifier_score": state.get("verifier_score"),
        "budget_remaining_fraction": state.get("budget_remaining_fraction"),
        "selected_final_residual": branch.get("final_residual"),
        "learned_stop_features_sha256": (
            stop.get("features_sha256") if stop is not None else ""
        ),
    }
    return TerminalDecision(
        disposition=disposition,
        reason=reason,
        requires_disclosure=reason in _DISCLOSURE_REASONS,
        instruction=_INSTRUCTIONS[reason],
        evidence=evidence,
    )


def terminal_instruction_texts() -> tuple[str, ...]:
    return tuple(_INSTRUCTIONS[reason] for reason in _REASON_PRECEDENCE)


def finalize_terminal_disposition_receipt(
    decision: TerminalDecision,
    *,
    instruction_tokens: Sequence[int],
    full_bridge_tokens: Sequence[int],
    output_tokens: Sequence[int],
    output_text: str,
    output_source: str,
) -> dict[str, Any]:
    if decision.reason not in _REASON_PRECEDENCE or _INSTRUCTIONS[decision.reason] != decision.instruction:
        raise ValueError("terminal decision identity is invalid")
    if not isinstance(output_text, str) or output_source not in {
        "resident_model_decode",
        "resident_model_repair",
        "substrate_model_decode",
    }:
        raise ValueError("terminal output provenance is invalid")
    instruction_applied = bool(instruction_tokens)
    if output_source != "substrate_model_decode" and not instruction_applied:
        raise ValueError("resident terminal output lacks its language instruction")
    language = {
        "source": output_source,
        "model_generated": True,
        "instruction_applied": instruction_applied,
        "instruction": decision.instruction,
        "instruction_sha256": _sha256_bytes(decision.instruction.encode("utf-8")),
        "instruction_token_count": len(instruction_tokens),
        "instruction_tokens": list(instruction_tokens),
        "instruction_tokens_sha256": _token_sha256(instruction_tokens),
        "full_bridge_token_count": len(full_bridge_tokens),
        "full_bridge_tokens": list(full_bridge_tokens),
        "full_bridge_tokens_sha256": _bridge_token_sha256(full_bridge_tokens),
        "output_token_count": len(output_tokens),
        "output_tokens_sha256": _token_sha256(output_tokens),
        "output_text_sha256": _sha256_bytes(output_text.encode("utf-8")),
    }
    payload = {
        "schema": SCHEMA,
        "policy_version": POLICY_VERSION,
        "reason_precedence": list(_REASON_PRECEDENCE),
        "disposition": decision.disposition,
        "reason": decision.reason,
        "requires_disclosure": decision.requires_disclosure,
        "evidence": decision.evidence,
        "language": language,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_terminal_disposition_receipt(
    value: Any,
    *,
    halting_reason: str,
    halting: Mapping[str, Any],
    loop_stability: Mapping[str, Any],
    cognitive_action_trace: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    output_tokens: Sequence[int],
    output_text: str,
    full_bridge_tokens_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "policy_version",
        "reason_precedence",
        "disposition",
        "reason",
        "requires_disclosure",
        "evidence",
        "language",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("terminal-disposition receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("terminal-disposition receipt commitment mismatch")
    embedded_evidence = value.get("evidence")
    decision_budget = (
        embedded_evidence.get("budget_at_decision")
        if isinstance(embedded_evidence, Mapping)
        else None
    )
    if not isinstance(decision_budget, Mapping):
        raise ValueError("terminal-disposition decision budget is missing")
    for field in ("max_layer_apps", "wall_clock_s"):
        if decision_budget.get(field) != budget.get(field):
            raise ValueError("terminal-disposition decision budget identity differs")
    decision_spent = decision_budget.get("spent_layer_apps")
    final_spent = budget.get("spent_layer_apps")
    decision_elapsed = decision_budget.get("elapsed_s")
    final_elapsed = budget.get("elapsed_s")
    max_apps = decision_budget.get("max_layer_apps")
    wall_limit = decision_budget.get("wall_clock_s")
    if (
        type(decision_spent) is not int
        or type(final_spent) is not int
        or type(max_apps) is not int
        or decision_spent < 0
        or decision_spent > max_apps
        or decision_spent > final_spent
        or not isinstance(decision_elapsed, (int, float))
        or isinstance(decision_elapsed, bool)
        or not isinstance(final_elapsed, (int, float))
        or isinstance(final_elapsed, bool)
        or not isinstance(wall_limit, (int, float))
        or isinstance(wall_limit, bool)
        or float(decision_elapsed) < 0.0
        or float(decision_elapsed) > float(final_elapsed) + 0.001
        or type(decision_budget.get("exhausted")) is not bool
        or (
            decision_budget["exhausted"]
            is not (
                decision_spent >= max_apps
                or float(decision_elapsed) >= float(wall_limit)
            )
        )
        or (
            decision_budget["exhausted"] is True
            and budget.get("exhausted") is not True
        )
    ):
        raise ValueError("terminal-disposition decision budget is not monotonic")
    expected = classify_terminal_disposition(
        halting_reason=halting_reason,
        halting=halting,
        loop_stability=loop_stability,
        cognitive_action_trace=cognitive_action_trace,
        budget=decision_budget,
    )
    language = value.get("language")
    if (
        value["schema"] != SCHEMA
        or value["policy_version"] != POLICY_VERSION
        or value["reason_precedence"] != list(_REASON_PRECEDENCE)
        or value["disposition"] != expected.disposition
        or value["reason"] != expected.reason
        or value["requires_disclosure"] is not expected.requires_disclosure
        or value["evidence"] != expected.evidence
        or not isinstance(language, dict)
        or set(language)
        != {
            "source",
            "model_generated",
            "instruction_applied",
            "instruction",
            "instruction_sha256",
            "instruction_token_count",
            "instruction_tokens",
            "instruction_tokens_sha256",
            "full_bridge_token_count",
            "full_bridge_tokens",
            "full_bridge_tokens_sha256",
            "output_token_count",
            "output_tokens_sha256",
            "output_text_sha256",
        }
        or language["source"]
        not in {
            "resident_model_decode",
            "resident_model_repair",
            "substrate_model_decode",
        }
        or language["model_generated"] is not True
        or type(language["instruction_applied"]) is not bool
        or language["instruction"] != expected.instruction
        or language["instruction_sha256"]
        != _sha256_bytes(expected.instruction.encode("utf-8"))
        or type(language["instruction_token_count"]) is not int
        or language["instruction_token_count"] < 0
        or not isinstance(language["instruction_tokens"], list)
        or language["instruction_token_count"] != len(language["instruction_tokens"])
        or language["instruction_tokens_sha256"]
        != _token_sha256(language["instruction_tokens"])
        or language["instruction_applied"]
        is not (language["instruction_token_count"] > 0)
        or (
            language["source"] != "substrate_model_decode"
            and language["instruction_applied"] is not True
        )
        or type(language["full_bridge_token_count"]) is not int
        or language["full_bridge_token_count"] < language["instruction_token_count"]
        or not isinstance(language["full_bridge_tokens"], list)
        or language["full_bridge_token_count"] != len(language["full_bridge_tokens"])
        or language["full_bridge_tokens_sha256"]
        != _bridge_token_sha256(language["full_bridge_tokens"])
        or (
            language["instruction_applied"]
            and language["full_bridge_tokens"][-language["instruction_token_count"] :]
            != language["instruction_tokens"]
        )
        or language["full_bridge_tokens_sha256"] != full_bridge_tokens_sha256
        or language["output_token_count"] != len(output_tokens)
        or language["output_tokens_sha256"] != _token_sha256(output_tokens)
        or language["output_text_sha256"] != _sha256_bytes(output_text.encode("utf-8"))
    ):
        raise ValueError("terminal-disposition identity or output binding is invalid")
    return value


__all__ = [
    "COMPUTE_BUDGET",
    "IRREDUCIBLE_UNCERTAINTY",
    "LOW_VALUE",
    "RECURRENCE_BUDGET",
    "TerminalDecision",
    "VERIFIED_CONVERGENCE",
    "WALL_BUDGET",
    "classify_terminal_disposition",
    "finalize_terminal_disposition_receipt",
    "terminal_instruction_texts",
    "validate_terminal_disposition_receipt",
]
