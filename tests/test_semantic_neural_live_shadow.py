from __future__ import annotations

import json
from pathlib import Path

from tools.run_semantic_neural_live_shadow import (
    _canonical_sha,
    _contract_issues,
    _load_ledger_rows,
    _sanitized,
    _sha_text,
    _tasks,
    _validate_shadow_row,
)
from tools.verify_semantic_neural_live_shadow import (
    SemanticNeuralLiveShadowVerificationError,
    verify_document,
)


def _contract(**overrides):
    values = {
        "desktop_cognitive_engine_required": True,
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "full_mind_path": True,
    }
    values.update(overrides)
    return {
        "live_turn_contract": values,
        "response_confidence": "high",
        "status": "cognitive_engine",
    }


def _shadow_row(
    objective: str,
    response: str,
    *,
    family: str = "frontier_calibration",
):
    body = {
        "schema": "aura.semantic_neural_shadow.v1",
        "recorded_at": 1.0,
        "objective_sha256": _sha_text(objective),
        "qualified_answer_sha256": "q" * 64,
        "ordinary_answer_sha256": _sha_text(response),
        "qualified_object_sha256": "x" * 64,
        "ordinary_object_sha256": "x" * 64,
        "ordinary_answer_parsed": True,
        "answer_match": True,
        "qualified_gain_candidate": False,
        "ordinary_success_preserved": True,
        "family": family,
        "parser_id": "semantic_calibration_canonical.v1",
        "admission_receipt_sha256": "a" * 64,
        "activation_sha256": "b" * 64,
        "package_id": "cp568-resident-semantic-neural-shadow",
        "promotion_mode": "shadow",
        "raw_prompt_retained": False,
        "raw_answers_retained": False,
    }
    return {
        **body,
        "receipt_sha256": _canonical_sha(body),
        "persisted": True,
    }


def test_live_shadow_contract_requires_real_ordinary_authority():
    assert _contract_issues(_contract()) == []
    issues = _contract_issues(
        _contract(
            engine_think_invoked=False,
            bounded_contract_used=True,
            full_mind_path=False,
        )
    )
    assert "engine_think_invoked:False" in issues
    assert "bounded_contract_used:True" in issues
    assert "full_mind_path:False" in issues
    missing_envelope = _contract_issues(
        {"live_turn_contract": _contract()["live_turn_contract"]}
    )
    assert "response_confidence:missing" in missing_envelope
    assert "status:missing" in missing_envelope


def test_live_shadow_row_reopens_receipt_and_authority():
    objective = "Fresh calibration task."
    response = 'FINAL_ANSWER: {"posterior":"3/7"}'
    row = _shadow_row(objective, response)
    assert (
        _validate_shadow_row(
            row,
            objective=objective,
            response=response,
            family="frontier_calibration",
            activation_sha256="b" * 64,
        )
        == []
    )


def test_live_shadow_row_rejects_tampering_and_direct_promotion():
    objective = "Fresh calibration task."
    response = 'FINAL_ANSWER: {"posterior":"3/7"}'
    row = _shadow_row(objective, response)
    row["promotion_mode"] = "active"
    row["answer_match"] = False
    issues = _validate_shadow_row(
        row,
        objective=objective,
        response=response,
        family="frontier_calibration",
        activation_sha256="b" * 64,
    )
    assert "promotion_mode:'active'" in issues
    assert "receipt_sha256" in issues
    assert "match_gain_complement" in issues


def test_live_shadow_ledger_reader_honors_start_offset(tmp_path: Path):
    path = tmp_path / "shadow.jsonl"
    path.write_text(json.dumps({"old": True}) + "\n", encoding="utf-8")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"new": True}) + "\n")
    assert _load_ledger_rows(path, offset=offset) == [{"new": True}]


def test_live_shadow_summary_drops_raw_prompts_answers_and_health():
    prompt = "private prompt"
    response = "private answer"
    row = _shadow_row(prompt, response)
    document = {
        "schema": "aura.semantic_neural_live_shadow.v1",
        "seed": 1,
        "task_count": 1,
        "ordinary_correct": 1,
        "shadow_answer_matches": 1,
        "qualified_gain_candidates": 0,
        "all_requests_proven_ordinary_authority": True,
        "boot_health": {"private": "runtime detail"},
        "result_sha256": "r" * 64,
        "transcript": [
            {
                "task_id": "task-1",
                "family": "frontier_calibration",
                "depth": 1,
                "prompt": prompt,
                "response": response,
                "latency_s": 1.25,
                "ordinary_correct": True,
                "contract_issues": [],
                "shadow_issues": [],
                "shadow_row": row,
            }
        ],
    }
    summary = _sanitized(document)
    encoded = json.dumps(summary, sort_keys=True)
    assert prompt not in encoded
    assert response not in encoded
    assert "runtime detail" not in encoded
    assert summary["rows"][0]["prompt_sha256"] == _sha_text(prompt)
    assert summary["rows"][0]["response_sha256"] == _sha_text(response)


def _verified_document():
    seed = 2026081568
    transcript = []
    for task in _tasks(seed=seed, tasks_per_difficulty=1):
        response = "FINAL_ANSWER: " + json.dumps(
            task.expected,
            sort_keys=True,
            separators=(",", ":"),
        )
        shadow_row = _shadow_row(task.prompt, response, family=task.family)
        transcript.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "depth": task.depth,
                "prompt": task.prompt,
                "response": response,
                "http_status": 200,
                "latency_s": 1.0,
                "ordinary_correct": True,
                "ordinary_parsed": task.grade(response)["parsed"],
                "expected": task.grade(response)["expected"],
                "live_turn_contract": _contract()["live_turn_contract"],
                "response_confidence": "high",
                "status": "ok",
                "contract_issues": [],
                "shadow_issues": [],
                "shadow_row": shadow_row,
            }
        )
    body = {
        "schema": "aura.semantic_neural_live_shadow.v1",
        "seed": seed,
        "tasks_per_difficulty": 1,
        "task_count": len(transcript),
        "domains": [
            "coding",
            "calibration",
            "misleading_premise",
            "scientific_inference",
        ],
        "activation_sha256": "b" * 64,
        "package_id": "cp568-resident-semantic-neural-shadow",
        "promotion_mode": "shadow",
        "model_path": "/tmp/model",
        "base_url": "http://127.0.0.1:8000",
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "boot_health": {
            "ready": True,
            "launch_provenance": {"required": True, "verified": True},
        },
        "ordinary_correct": len(transcript),
        "shadow_answer_matches": len(transcript),
        "qualified_gain_candidates": 0,
        "all_requests_proven_ordinary_authority": True,
        "transcript": transcript,
    }
    return {**body, "result_sha256": _canonical_sha(body)}


def test_live_shadow_independent_verifier_regrades_every_response():
    document = _verified_document()
    verification = verify_document(document)
    assert verification["ordinary_authority_verified"] is True
    assert verification["ordinary_correct"] == document["task_count"]


def test_live_shadow_independent_verifier_rejects_aggregate_tamper():
    document = _verified_document()
    document["ordinary_correct"] -= 1
    body = {key: value for key, value in document.items() if key != "result_sha256"}
    document["result_sha256"] = _canonical_sha(body)
    try:
        verify_document(document)
    except SemanticNeuralLiveShadowVerificationError as exc:
        assert "aggregate" in str(exc)
    else:
        raise AssertionError("aggregate tamper was accepted")
