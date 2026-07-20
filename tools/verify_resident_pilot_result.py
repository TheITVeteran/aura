#!/usr/bin/env python3
"""Certify and diagnose the preregistered resident-32B directional pilot.

This verifier applies the nine CP178 advance rules to independently replayed
raw evidence.  A positive pilot can only admit a powered campaign; it can
never establish a reasoning or frontier claim.  A non-advance result is still
a valid certificate when every evidence and containment check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    read_canonical_json,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_RLC,
    BASE_VANILLA,
)
from tools.verify_paired_campaign_evidence import (  # noqa: E402
    _open_journal_readonly,
    _regenerate_tasks,
    verify_campaign_evidence,
)
from tools.verify_recurrence_v2_smoke import (  # noqa: E402
    _activation,
    _atomic_create_or_verify,
    _verify_campaign,
    _verify_detached_receipt,
)
from tools.verify_resident_pilot_preflight import (  # noqa: E402
    _file_sha,
    _sha,
    _verified_contract,
    _verify_plan,
)

SCHEMA = "aura.latent_cortex.resident_pilot_result.v1"
EXPECTED_RULES = (
    "all_56_cells_commit_and_replay",
    "adapter_rlc_total_correct_strictly_exceeds_adapter_vanilla",
    "adapter_rlc_total_correct_strictly_exceeds_base_rlc",
    "adapter_vanilla_output_is_byte_identical_to_base_vanilla_per_task",
    "adapter_vanilla_total_correct_is_not_below_base_vanilla",
    "base_rlc_has_zero_recurrence_adapter_activity",
    "adapter_rlc_has_positive_scoped_recurrence_adapter_activity",
    "adapter_rlc_changes_at_least_one_first_logit_digest",
    "all_model_adapter_source_reset_scorer_and_detached_receipts_validate",
)


class ResidentPilotResultError(RuntimeError):
    """Stable fail-closed resident-pilot result error."""


def _fail(reason: str) -> Never:
    raise ResidentPilotResultError(reason)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validator_identity() -> dict[str, str]:
    paths = {
        "campaign_evidence_verifier_sha256": REPO_ROOT / "tools/verify_paired_campaign_evidence.py",
        "campaign_replay_verifier_sha256": REPO_ROOT / "tools/verify_recurrence_v2_smoke.py",
        "frontier_task_scorer_sha256": REPO_ROOT / "core/brain/llm/latent_cortex/frontier_tasks.py",
        "independent_scoring_sha256": REPO_ROOT / "tools/independent_paired_campaign_scoring.py",
        "pilot_preflight_verifier_sha256": REPO_ROOT / "tools/verify_resident_pilot_preflight.py",
        "pilot_result_verifier_sha256": Path(__file__).resolve(),
    }
    return {role: _sha256_file(path) for role, path in paths.items()}


def _verified_mechanics(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    mechanics = read_canonical_json(path, role="mechanics_verdict")
    material = dict(mechanics)
    claimed = material.pop("verdict_sha256", None)
    gate = contract.get("mechanics_gate")
    if not isinstance(gate, Mapping) or (
        _file_sha(path) != gate.get("file_sha256")
        or claimed != gate.get("verdict_sha256")
        or claimed != _sha(material)
        or mechanics.get("passed") is not True
        or mechanics.get("ready_for_fresh_hidden_task_pilot") is not True
        or mechanics.get("reasoning_gain_proven") is not False
        or mechanics.get("frontier_gain_proven") is not False
    ):
        _fail("pilot_mechanics_gate_invalid")
    return mechanics


def _validate_independent_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("passed") is not True
        or evidence.get("failures") != []
        or evidence.get("committed_records") != 56
        or evidence.get("task_count") != 14
        or evidence.get("production_semantic_grade_sha256")
        != evidence.get("independent_semantic_grade_sha256")
        or evidence.get("published_verdict") != evidence.get("recomputed_verdict")
        or evidence.get("published_verdict") != evidence.get("independent_verdict")
    ):
        _fail("independent_campaign_evidence_invalid")


def _counter(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _minmax(values: Sequence[float | int], *, digits: int = 6) -> list[float | int]:
    if not values:
        return []
    low, high = min(values), max(values)
    if all(type(value) is int for value in values):
        return [int(low), int(high)]
    return [round(float(low), digits), round(float(high), digits)]


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
            _fail("pilot_record_invalid")
        task_id = definition.get("task_id")
        arm = definition.get("arm")
        if (
            not isinstance(task_id, str)
            or task_id not in tasks_by_id
            or arm not in {BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC}
            or arm in rows[task_id]
        ):
            _fail("pilot_record_identity_invalid")
        rows[task_id][str(arm)] = record
    expected_arms = {BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC}
    if len(rows) != 14 or any(set(task_rows) != expected_arms for task_rows in rows.values()):
        _fail("pilot_arm_matrix_incomplete")

    arm_summary: dict[str, Any] = {}
    base_activation_totals = {"calls": 0, "adapted_positions": 0, "observed_positions": 0}
    adapter_activation_totals = dict(base_activation_totals)
    causal_digest_changes = 0
    ordinary_exact_match = True
    for arm in (BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC):
        scored = []
        score_reasons: list[str] = []
        output_lengths: list[int] = []
        generated_tokens: list[int] = []
        terminations: list[str] = []
        selected_branches: list[int] = []
        branch_spreads: list[float] = []
        exchange_cosines: list[float] = []
        for task_id in sorted(rows):
            result = rows[task_id][arm]["result"]
            text = result.get("text")
            if not isinstance(text, str):
                _fail("pilot_output_text_invalid")
            score = tasks_by_id[task_id].score(text)
            scored.append(bool(score.correct))
            score_reasons.append(str(score.reason))
            output_lengths.append(len(text))
            receipt = result.get("episode_receipt")
            if arm in {BASE_RLC, ADAPTER_RLC}:
                if not isinstance(receipt, Mapping):
                    _fail("pilot_episode_receipt_missing")
                generated = receipt.get("decode_generated_tokens")
                termination = receipt.get("decode_termination")
                selected = receipt.get("selected_branch")
                telemetry = receipt.get("latent_telemetry")
                scores = receipt.get("branch_scores")
                if type(generated) is int:
                    generated_tokens.append(generated)
                terminations.append(str(termination))
                if type(selected) is int:
                    selected_branches.append(selected)
                if (
                    isinstance(scores, list)
                    and scores
                    and all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in scores
                    )
                ):
                    branch_spreads.append(float(max(scores) - min(scores)))
                if isinstance(telemetry, Mapping):
                    snapshots = telemetry.get("exchange_snapshots")
                    if isinstance(snapshots, list):
                        exchange_cosines.extend(
                            float(snapshot["mean_cos"])
                            for snapshot in snapshots
                            if isinstance(snapshot, Mapping)
                            and isinstance(snapshot.get("mean_cos"), (int, float))
                            and not isinstance(snapshot.get("mean_cos"), bool)
                        )
            else:
                if receipt not in ({}, None):
                    _fail("ordinary_episode_receipt_present")
                terminations.append("ordinary_generation")
        arm_summary[arm] = {
            "correct": sum(scored),
            "total": len(scored),
            "score_reasons": _counter(score_reasons),
            "output_chars_minmax": _minmax(output_lengths),
            "decode_generated_tokens_minmax": _minmax(generated_tokens),
            "decode_terminations": _counter(terminations),
            "selected_branches": _counter(selected_branches),
            "branch_score_ties": sum(spread == 0.0 for spread in branch_spreads),
            "branch_score_spread_minmax": _minmax(branch_spreads),
            "exchange_mean_cosine_minmax": _minmax(exchange_cosines),
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
        base_activation = _activation(
            base_r.get("episode_receipt", {}).get("recurrence_adapter"),
            expected_active=False,
        )
        adapter_activation = _activation(
            adapter_r.get("episode_receipt", {}).get("recurrence_adapter"),
            expected_active=True,
        )
        for key in base_activation_totals:
            base_activation_totals[key] += int(base_activation[key])
            adapter_activation_totals[key] += int(adapter_activation[key])
        if base_r.get("episode_receipt", {}).get("first_logits_digest") != adapter_r.get(
            "episode_receipt", {}
        ).get("first_logits_digest"):
            causal_digest_changes += 1

    mechanics = {
        "ordinary_generation_exact_match": ordinary_exact_match,
        "base_recurrence_adapter_activation": base_activation_totals,
        "adapter_recurrence_adapter_activation": adapter_activation_totals,
        "causal_first_logit_digest_changes": causal_digest_changes,
    }
    return arm_summary, mechanics


def _evaluate_advance_rules(
    *,
    arm_summary: Mapping[str, Mapping[str, Any]],
    mechanics: Mapping[str, Any],
    committed_cells: int,
    replayed_cells: int,
    receipts_valid: bool,
) -> dict[str, bool]:
    base_activation = mechanics.get("base_recurrence_adapter_activation")
    adapter_activation = mechanics.get("adapter_recurrence_adapter_activation")
    base_zero = isinstance(base_activation, Mapping) and all(
        base_activation.get(key) == 0
        for key in ("calls", "adapted_positions", "observed_positions")
    )
    adapter_positive = isinstance(adapter_activation, Mapping) and (
        isinstance(adapter_activation.get("calls"), int)
        and isinstance(adapter_activation.get("adapted_positions"), int)
        and isinstance(adapter_activation.get("observed_positions"), int)
        and adapter_activation["calls"] > 0
        and adapter_activation["adapted_positions"] > 0
        and adapter_activation["observed_positions"] >= adapter_activation["adapted_positions"]
    )
    rules = {
        EXPECTED_RULES[0]: committed_cells == 56 and replayed_cells == 56,
        EXPECTED_RULES[1]: arm_summary[ADAPTER_RLC]["correct"]
        > arm_summary[ADAPTER_VANILLA]["correct"],
        EXPECTED_RULES[2]: arm_summary[ADAPTER_RLC]["correct"] > arm_summary[BASE_RLC]["correct"],
        EXPECTED_RULES[3]: mechanics.get("ordinary_generation_exact_match") is True,
        EXPECTED_RULES[4]: arm_summary[ADAPTER_VANILLA]["correct"]
        >= arm_summary[BASE_VANILLA]["correct"],
        EXPECTED_RULES[5]: base_zero,
        EXPECTED_RULES[6]: adapter_positive,
        EXPECTED_RULES[7]: isinstance(mechanics.get("causal_first_logit_digest_changes"), int)
        and mechanics["causal_first_logit_digest_changes"] > 0,
        EXPECTED_RULES[8]: receipts_valid,
    }
    if tuple(rules) != EXPECTED_RULES:
        _fail("pilot_advance_rule_order_invalid")
    return rules


def _diagnoses(
    arm_summary: Mapping[str, Mapping[str, Any]],
    mechanics: Mapping[str, Any],
    rules: Mapping[str, bool],
) -> list[str]:
    diagnoses: list[str] = []
    marker_failures = sum(
        int(summary.get("score_reasons", {}).get("final_answer_marker_count_invalid", 0))
        for summary in arm_summary.values()
    )
    invalid_json = sum(
        int(summary.get("score_reasons", {}).get("final_answer_invalid_json", 0))
        for summary in arm_summary.values()
    )
    if marker_failures or invalid_json:
        diagnoses.append("decode_response_contract_failure")
    adapter = arm_summary[ADAPTER_RLC]
    if (
        adapter.get("branch_score_ties") == adapter.get("total")
        and adapter.get("selected_branches") == {"0": adapter.get("total")}
        and adapter.get("branch_score_spread_minmax") == [0.0, 0.0]
    ):
        diagnoses.append("adapter_virtual_width_functionally_collapsed")
    if not rules[EXPECTED_RULES[1]] or not rules[EXPECTED_RULES[2]]:
        diagnoses.append("recurrence_training_failed_directional_gain_gate")
    if mechanics.get("ordinary_generation_exact_match") is True and marker_failures:
        diagnoses.append("shared_vanilla_decode_budget_truncates_contract_answers")
    return diagnoses


def verify(
    *,
    contract_path: Path,
    mechanics_path: Path,
    campaign_dir: Path,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    mechanics_path = mechanics_path.expanduser().resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    contract = _verified_contract(contract_path)
    _verified_mechanics(mechanics_path, contract)
    if campaign_dir != Path(str(contract["campaign"]["directory"])).resolve():
        _fail("pilot_campaign_directory_mismatch")
    plan = _verify_plan(contract, campaign_dir / "plan.json")
    detached_receipt = _verify_detached_receipt(
        campaign_dir, expected_returncodes=frozenset({0, 2})
    )
    campaign_replay = _verify_campaign(campaign_dir)
    independent = verify_campaign_evidence(campaign_dir)
    _validate_independent_evidence(independent)
    tasks, generation = _regenerate_tasks(plan, campaign_dir)
    tasks_by_id = {task.task_id: task for task in tasks}
    with _open_journal_readonly(campaign_dir / "campaign.jsonl", plan) as journal:
        records = journal.committed_records()
    arm_summary, mechanics = _summarize_records(records, tasks_by_id=tasks_by_id)
    rules = _evaluate_advance_rules(
        arm_summary=arm_summary,
        mechanics=mechanics,
        committed_cells=len(records),
        replayed_cells=int(campaign_replay.get("committed_cells", -1)),
        receipts_valid=True,
    )
    declared_rules = contract.get("decision", {}).get("advance_only_if")
    if declared_rules != list(EXPECTED_RULES):
        _fail("pilot_declared_advance_rules_invalid")
    advance = all(rules.values())
    diagnoses = _diagnoses(arm_summary, mechanics, rules)
    material = {
        "schema": SCHEMA,
        "claim_scope": "resident_32b_directional_pilot_result_only",
        "passed": True,
        "evidence_valid": True,
        "pilot_advance_gate_passed": advance,
        "decision": (
            "advance_to_powered_external_frontier_campaign"
            if advance
            else "diagnose_and_preregister_revision"
        ),
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "external_attestation_present": False,
        "pilot_can_prove_frontier_gain": False,
        "contract_sha256": contract["contract_sha256"],
        "mechanics_verdict_sha256": contract["mechanics_gate"]["verdict_sha256"],
        "plan_sha256": plan.plan_sha256,
        "task_manifest_sha256": contract["campaign"]["task_manifest_sha256"],
        "campaign_manifest_sha256": campaign_replay["campaign_manifest_sha256"],
        "grade_sha256": campaign_replay["grade_sha256"],
        "detached_receipt_sha256": detached_receipt["receipt_sha256"],
        "independent_semantic_grade_sha256": independent["independent_semantic_grade_sha256"],
        "validator_identity": _validator_identity(),
        "generation": generation,
        "advance_rules": rules,
        "arm_results": arm_summary,
        "mechanics": mechanics,
        "diagnoses": diagnoses,
        "required_next_gate": (
            "powered_external_frontier_campaign"
            if advance
            else "repair_and_preregister_recurrence_v3_directional_pilot"
        ),
    }
    return {**material, "verdict_sha256": _sha(material)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mechanics-verdict", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        verdict = verify(
            contract_path=Path(args.contract),
            mechanics_path=Path(args.mechanics_verdict),
            campaign_dir=Path(args.campaign_dir),
        )
        _atomic_create_or_verify(
            Path(args.output).expanduser().resolve(strict=False),
            canonical_json_bytes(verdict) + b"\n",
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            f"verify_resident_pilot_result: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
