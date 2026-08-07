#!/usr/bin/env python3
"""Certify a nonclaiming resident-32B RLC directional campaign.

The ordinary paired grader correctly refuses a statistical gain claim for a
small directional campaign.  This verifier applies the separate preregistered
advance rule: valid positive directional evidence may open a powered,
externally custodied campaign, but can never establish a reasoning, frontier,
or production claim by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    read_canonical_json,
    read_strict_json,
)
from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS  # noqa: E402
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_RLC,
    BASE_VANILLA,
)
from tools.verify_paired_campaign_evidence import (  # noqa: E402
    VERDICT_SCHEMA as INDEPENDENT_VERDICT_SCHEMA,
)
from tools.verify_paired_campaign_evidence import (  # noqa: E402
    _open_journal_readonly,
    _regenerate_tasks,
    verify_campaign_evidence,
)
from tools.verify_recurrence_v2_smoke import _activation  # noqa: E402

SCHEMA = "aura.latent_cortex.directional_gate.v1"
EXPECTED_ARMS = frozenset({BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC})
EXPECTED_RULES = (
    "all_planned_cells_commit_and_replay",
    "adapter_rlc_total_correct_strictly_exceeds_adapter_vanilla",
    "adapter_rlc_total_correct_strictly_exceeds_base_rlc",
    "adapter_rlc_interaction_strictly_exceeds_base_rlc_interaction",
    "adapter_vanilla_output_is_byte_identical_to_base_vanilla_per_task",
    "adapter_vanilla_total_correct_is_not_below_base_vanilla",
    "base_rlc_has_zero_recurrence_adapter_activity",
    "adapter_rlc_has_positive_scoped_recurrence_adapter_activity",
    "adapter_rlc_changes_at_least_one_first_logit_digest",
    "raw_terminal_output_policy_is_symmetric_and_unedited",
    "independent_evidence_and_receipts_validate",
)


class DirectionalGateError(RuntimeError):
    """Stable fail-closed directional-gate error."""


def _fail(reason: str) -> Never:
    raise DirectionalGateError(reason)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != payload:
            _fail("directional_verdict_output_conflict")
        return
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("directional_verdict_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _counter(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _minmax(values: Sequence[float | int], *, digits: int = 6) -> list[float | int]:
    if not values:
        return []
    low, high = min(values), max(values)
    if all(type(value) is int for value in values):
        return [int(low), int(high)]
    return [round(float(low), digits), round(float(high), digits)]


def _validate_plan(plan: CampaignPlan) -> tuple[int, int]:
    document = plan.to_dict()
    metadata = document.get("metadata")
    cells = document.get("cells")
    if not isinstance(metadata, Mapping) or not isinstance(cells, list):
        _fail("directional_plan_shape_invalid")
    tasks = metadata.get("task_manifest", {}).get("tasks")
    execution = metadata.get("execution_config")
    if not isinstance(tasks, list) or not isinstance(execution, Mapping):
        _fail("directional_plan_contract_missing")
    if metadata.get("claim_eligible") is not False:
        _fail("directional_plan_must_be_nonclaiming")
    if set(execution.get("domains", ())) != set(FRONTIER_DOMAINS):
        _fail("directional_domain_coverage_invalid")
    task_domains = Counter(task.get("domain") for task in tasks if isinstance(task, Mapping))
    if set(task_domains) != set(FRONTIER_DOMAINS) or len(set(task_domains.values())) != 1:
        _fail("directional_tasks_not_domain_balanced")
    definitions = [cell.get("definition") for cell in cells if isinstance(cell, Mapping)]
    arms = {
        definition.get("arm")
        for definition in definitions
        if isinstance(definition, Mapping)
    }
    if arms != EXPECTED_ARMS or len(definitions) != len(cells):
        _fail("directional_arm_contract_invalid")
    expected_cells = len(tasks) * len(EXPECTED_ARMS)
    if len(cells) != expected_cells:
        _fail("directional_cell_matrix_invalid")
    policy = execution.get("response_contract_policy")
    effective = execution.get("effective_rlc_config")
    adapter_spec = execution.get("adapter_execution_spec")
    if (
        not isinstance(policy, Mapping)
        or policy.get("applies_identically_to_all_decode_arms") is not True
        or policy.get("causal_attribution_rule") != "raw_terminal_decode_all_arms"
        or policy.get("output_editing") is not False
        or policy.get("rlc_answer_replacement_enabled") is not False
        or execution.get("vanilla_fallback_allowed") is not False
        or not isinstance(effective, Mapping)
        or effective.get("allow_vanilla_fallback") is not False
        or effective.get("answer_replacement_enabled") is not False
        or effective.get("decode_bridge_policy") != "none"
        or not isinstance(adapter_spec, Mapping)
        or adapter_spec.get("decode_bridge_policy") != "none"
    ):
        _fail("directional_output_symmetry_policy_invalid")
    return len(tasks), len(cells)


def _replacement_retained(receipt: Mapping[str, Any]) -> bool:
    replacement = receipt.get("answer_replacement")
    if replacement in (None, {}):
        return True
    if not isinstance(replacement, Mapping):
        return False
    accepted = replacement.get("accepted_output")
    baseline = replacement.get("baseline_decode")
    return bool(
        replacement.get("answer_selection_effect") == "retained"
        and isinstance(accepted, Mapping)
        and isinstance(baseline, Mapping)
        and accepted.get("source") == "baseline_decode"
        and accepted.get("text_sha256") == baseline.get("text_sha256")
        and accepted.get("tokens_sha256") == baseline.get("tokens_sha256")
    )


def _summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tasks_by_id: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        definition = record.get("definition")
        result = record.get("result")
        if not isinstance(definition, Mapping) or not isinstance(result, Mapping):
            _fail("directional_record_invalid")
        task_id = definition.get("task_id")
        arm = definition.get("arm")
        if (
            not isinstance(task_id, str)
            or task_id not in tasks_by_id
            or arm not in EXPECTED_ARMS
            or arm in rows[task_id]
        ):
            _fail("directional_record_identity_invalid")
        rows[task_id][str(arm)] = record
    if set(rows) != set(tasks_by_id) or any(
        set(task_rows) != EXPECTED_ARMS for task_rows in rows.values()
    ):
        _fail("directional_arm_matrix_incomplete")

    arm_summary: dict[str, Any] = {}
    base_activation = {"calls": 0, "adapted_positions": 0, "observed_positions": 0}
    adapter_activation = dict(base_activation)
    causal_digest_changes = 0
    ordinary_exact_match = True
    raw_outputs_retained = True
    for arm in (BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC):
        scored: list[bool] = []
        score_reasons: list[str] = []
        output_lengths: list[int] = []
        generated_tokens: list[int] = []
        terminations: list[str] = []
        for task_id in sorted(rows):
            result = rows[task_id][arm]["result"]
            text = result.get("text")
            if not isinstance(text, str) or result.get("arm") != arm:
                _fail("directional_output_invalid")
            score = tasks_by_id[task_id].score(text)
            scored.append(bool(score.correct))
            score_reasons.append(str(score.reason))
            output_lengths.append(len(text))
            receipt = result.get("episode_receipt")
            if arm in {BASE_RLC, ADAPTER_RLC}:
                if not isinstance(receipt, Mapping):
                    _fail("directional_episode_receipt_missing")
                generated = receipt.get("decode_generated_tokens")
                if type(generated) is int:
                    generated_tokens.append(generated)
                terminations.append(str(receipt.get("decode_termination")))
                raw_outputs_retained = raw_outputs_retained and _replacement_retained(receipt)
            else:
                if receipt not in ({}, None):
                    _fail("directional_ordinary_episode_receipt_present")
                terminations.append("ordinary_generation")
        arm_summary[arm] = {
            "correct": sum(scored),
            "total": len(scored),
            "score_reasons": _counter(score_reasons),
            "output_chars_minmax": _minmax(output_lengths),
            "decode_generated_tokens_minmax": _minmax(generated_tokens),
            "decode_terminations": _counter(terminations),
        }

    for task_rows in rows.values():
        base_v = task_rows[BASE_VANILLA]["result"]
        adapter_v = task_rows[ADAPTER_VANILLA]["result"]
        base_r = task_rows[BASE_RLC]["result"]
        adapter_r = task_rows[ADAPTER_RLC]["result"]
        ordinary_exact_match = ordinary_exact_match and (
            base_v.get("text") == adapter_v.get("text")
            and base_v.get("output_sha256") == adapter_v.get("output_sha256")
        )
        base_receipt = base_r.get("episode_receipt")
        adapter_receipt = adapter_r.get("episode_receipt")
        if not isinstance(base_receipt, Mapping) or not isinstance(adapter_receipt, Mapping):
            _fail("directional_recurrence_receipt_missing")
        base_current = _activation(
            base_receipt.get("recurrence_adapter"),
            expected_active=False,
        )
        adapter_current = _activation(
            adapter_receipt.get("recurrence_adapter"),
            expected_active=True,
        )
        for key in base_activation:
            base_activation[key] += int(base_current[key])
            adapter_activation[key] += int(adapter_current[key])
        if base_receipt.get("first_logits_digest") != adapter_receipt.get(
            "first_logits_digest"
        ):
            causal_digest_changes += 1

    mechanics = {
        "ordinary_generation_exact_match": ordinary_exact_match,
        "raw_terminal_outputs_retained": raw_outputs_retained,
        "base_recurrence_adapter_activation": base_activation,
        "adapter_recurrence_adapter_activation": adapter_activation,
        "causal_first_logit_digest_changes": causal_digest_changes,
    }
    return arm_summary, mechanics


def _evaluate_rules(
    *,
    arm_summary: Mapping[str, Mapping[str, Any]],
    mechanics: Mapping[str, Any],
    expected_cells: int,
    committed_cells: int,
    replayed_cells: int,
    independent_valid: bool,
) -> dict[str, bool]:
    base_activation = mechanics.get("base_recurrence_adapter_activation")
    adapter_activation = mechanics.get("adapter_recurrence_adapter_activation")
    base_zero = isinstance(base_activation, Mapping) and all(
        base_activation.get(key) == 0
        for key in ("calls", "adapted_positions", "observed_positions")
    )
    adapter_positive = isinstance(adapter_activation, Mapping) and (
        type(adapter_activation.get("calls")) is int
        and type(adapter_activation.get("adapted_positions")) is int
        and type(adapter_activation.get("observed_positions")) is int
        and adapter_activation["calls"] > 0
        and adapter_activation["adapted_positions"] > 0
        and adapter_activation["observed_positions"]
        >= adapter_activation["adapted_positions"]
    )
    adapter_delta = (
        arm_summary[ADAPTER_RLC]["correct"] - arm_summary[ADAPTER_VANILLA]["correct"]
    )
    base_delta = arm_summary[BASE_RLC]["correct"] - arm_summary[BASE_VANILLA]["correct"]
    rules = {
        EXPECTED_RULES[0]: committed_cells == expected_cells == replayed_cells,
        EXPECTED_RULES[1]: arm_summary[ADAPTER_RLC]["correct"]
        > arm_summary[ADAPTER_VANILLA]["correct"],
        EXPECTED_RULES[2]: arm_summary[ADAPTER_RLC]["correct"]
        > arm_summary[BASE_RLC]["correct"],
        EXPECTED_RULES[3]: adapter_delta > base_delta,
        EXPECTED_RULES[4]: mechanics.get("ordinary_generation_exact_match") is True,
        EXPECTED_RULES[5]: arm_summary[ADAPTER_VANILLA]["correct"]
        >= arm_summary[BASE_VANILLA]["correct"],
        EXPECTED_RULES[6]: base_zero,
        EXPECTED_RULES[7]: adapter_positive,
        EXPECTED_RULES[8]: type(mechanics.get("causal_first_logit_digest_changes")) is int
        and mechanics["causal_first_logit_digest_changes"] > 0,
        EXPECTED_RULES[9]: mechanics.get("raw_terminal_outputs_retained") is True,
        EXPECTED_RULES[10]: independent_valid,
    }
    if tuple(rules) != EXPECTED_RULES:
        _fail("directional_rule_order_invalid")
    return rules


def _diagnoses(rules: Mapping[str, bool], arm_summary: Mapping[str, Any]) -> list[str]:
    diagnoses: list[str] = []
    if not rules[EXPECTED_RULES[0]]:
        diagnoses.append("campaign_matrix_or_replay_incomplete")
    if not rules[EXPECTED_RULES[1]] or not rules[EXPECTED_RULES[2]]:
        diagnoses.append("recurrence_adapter_failed_directional_gain_gate")
    if not rules[EXPECTED_RULES[3]]:
        diagnoses.append("positive_adapter_rlc_interaction_not_observed")
    if not rules[EXPECTED_RULES[4]] or not rules[EXPECTED_RULES[5]]:
        diagnoses.append("adapter_vanilla_regression_or_execution_asymmetry")
    if not rules[EXPECTED_RULES[6]] or not rules[EXPECTED_RULES[7]]:
        diagnoses.append("recurrence_adapter_scope_or_activation_invalid")
    if not rules[EXPECTED_RULES[8]]:
        diagnoses.append("recurrence_adapter_causal_logit_effect_not_observed")
    if not rules[EXPECTED_RULES[9]]:
        diagnoses.append("output_replacement_or_editing_detected")
    if not rules[EXPECTED_RULES[10]]:
        diagnoses.append("independent_evidence_invalid")
    marker_failures = sum(
        int(summary.get("score_reasons", {}).get("final_answer_marker_count_invalid", 0))
        for summary in arm_summary.values()
    )
    if marker_failures:
        diagnoses.append("decode_response_contract_failure")
    return diagnoses


def verify(
    *,
    campaign_dir: Path,
    independent_verdict_path: Path,
    contamination_trust_root: Path,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    plan_path = campaign_dir / "plan.json"
    plan = CampaignPlan.from_dict(read_canonical_json(plan_path, role="directional_plan"))
    expected_tasks, expected_cells = _validate_plan(plan)
    # The independent verifier intentionally emits human-reviewable indented
    # JSON. Its bytes are hashed below and its complete semantics must equal an
    # independent recomputation, so canonical whitespace is not a trust input.
    supplied = read_strict_json(
        independent_verdict_path.expanduser().resolve(strict=True),
        role="directional_independent_verdict",
    )
    recomputed = verify_campaign_evidence(
        campaign_dir,
        contamination_trust_root=str(contamination_trust_root.expanduser().resolve(strict=True)),
    )
    independent_valid = bool(
        supplied == recomputed
        and supplied.get("schema") == INDEPENDENT_VERDICT_SCHEMA
        and supplied.get("passed") is True
        and supplied.get("failures") == []
        and supplied.get("committed_records") == expected_cells
        and supplied.get("task_count") == expected_tasks
        and supplied.get("production_semantic_grade_sha256")
        == supplied.get("independent_semantic_grade_sha256")
        and supplied.get("published_verdict") == supplied.get("recomputed_verdict")
        and supplied.get("published_verdict") == supplied.get("independent_verdict")
    )
    tasks, generation = _regenerate_tasks(plan, campaign_dir)
    tasks_by_id = {task.task_id: task for task in tasks}
    with _open_journal_readonly(campaign_dir / "campaign.jsonl", plan) as journal:
        records = journal.committed_records()
    arm_summary, mechanics = _summarize_records(records, tasks_by_id=tasks_by_id)
    replayed_cells = int(supplied.get("committed_records", -1))
    rules = _evaluate_rules(
        arm_summary=arm_summary,
        mechanics=mechanics,
        expected_cells=expected_cells,
        committed_cells=len(records),
        replayed_cells=replayed_cells,
        independent_valid=independent_valid,
    )
    advance = all(rules.values())
    material = {
        "schema": SCHEMA,
        "claim_scope": "resident_32b_directional_gate_only",
        "passed": True,
        "evidence_valid": independent_valid,
        "directional_gate_passed": advance,
        "decision": (
            "advance_to_powered_external_campaign"
            if advance
            else "repair_and_preregister_directional_revision"
        ),
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "production_activation_authorized": False,
        "static_weight_fusion_authorized": False,
        "required_next_gate": (
            "powered_external_campaign" if advance else "directional_repair"
        ),
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": _file_sha(plan_path),
        "independent_verdict_sha256": _file_sha(independent_verdict_path),
        "task_count": expected_tasks,
        "cell_count": expected_cells,
        "generation": generation,
        "advance_rules": rules,
        "arm_results": arm_summary,
        "mechanics": mechanics,
        "diagnoses": _diagnoses(rules, arm_summary),
        "validator_identity": {
            "directional_gate_sha256": _file_sha(Path(__file__).resolve()),
            "independent_evidence_verifier_sha256": _file_sha(
                REPO_ROOT / "tools/verify_paired_campaign_evidence.py"
            ),
            "independent_scoring_sha256": _file_sha(
                REPO_ROOT / "tools/independent_paired_campaign_scoring.py"
            ),
            "frontier_task_scorer_sha256": _file_sha(
                REPO_ROOT / "core/brain/llm/latent_cortex/frontier_tasks.py"
            ),
        },
    }
    return {**material, "verdict_sha256": _sha(material)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--independent-verdict", type=Path, required=True)
    parser.add_argument("--contamination-trust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = verify(
            campaign_dir=args.campaign_dir,
            independent_verdict_path=args.independent_verdict,
            contamination_trust_root=args.contamination_trust_root,
        )
        _write_once(args.output, verdict)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            f"verify_latent_cortex_directional_gate: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
