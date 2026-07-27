from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from core.learning import recurrent_sft_evaluation as evaluation
from core.learning.recurrent_sft_evaluation import EVALUATION_SOURCE_ROLES
from core.learning.recurrent_sft_falsification import CONTROL_ARMS, sha256_json
from core.learning.structured_sft import (
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
)
from tools import evaluate_recurrent_sft_falsification as evaluator_tool


def _artifacts(names: tuple[str, ...]) -> dict[str, bytes]:
    return {name: f"private-{name}".encode() for name in names}


def _holdout() -> dict:
    example = {
        "example_id": "1" * 64,
        "case_fingerprint": "2" * 64,
        "family": "structured_program",
        "target_kind": "derivation",
        "curriculum_version": "v1",
        "loss_policy": {"mask_prompt": True},
        "messages": [
            {"role": "user", "content": "compute"},
            {"role": "assistant", "content": "answer"},
        ],
        "tools": [],
        "projection": {
            "answer_evidence_basis": "not_present",
            "answer_evidence_in_input": False,
            "oracle_fields_exported_to_trainer": [],
        },
    }
    return {"examples": [example]}


def test_holdout_projection_requires_replayed_disjoint_custody(monkeypatch) -> None:
    candidate = _artifacts(STRUCTURED_SFT_CANDIDATE_FILES)
    evaluator = _artifacts(STRUCTURED_SFT_EVALUATOR_FILES)
    evaluator["holdout.private.json"] = json.dumps(_holdout()).encode()
    monkeypatch.setattr(
        evaluation,
        "validate_structured_sft_custody_pair",
        lambda *_args: {
            "holdout_example_count": 1,
            "example_id_overlap_count": 0,
            "case_fingerprint_overlap_count": 0,
            "candidate_contains_holdout_seed": False,
            "custody_report_sha256": "3" * 64,
        },
    )

    rows, custody = evaluation.evaluator_holdout_rows(candidate, evaluator)

    assert custody["custody_report_sha256"] == "3" * 64
    assert rows == [
        {
            "messages": _holdout()["examples"][0]["messages"],
            "tools": [],
            "_meta": {
                "example_id": "1" * 64,
                "case_fingerprint": "2" * 64,
                "family": "structured_program",
                "target_kind": "derivation",
                "curriculum_version": "v1",
                "loss_policy": {"mask_prompt": True},
                "projection": {
                    "answer_evidence_basis": "not_present",
                    "answer_evidence_in_input": False,
                    "oracle_fields_exported_to_trainer": [],
                },
            },
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer_evidence_in_input", True),
        ("oracle_fields_exported_to_trainer", ["oracle"]),
    ],
)
def test_holdout_projection_rejects_answer_leakage(
    monkeypatch,
    field: str,
    value: object,
) -> None:
    candidate = _artifacts(STRUCTURED_SFT_CANDIDATE_FILES)
    evaluator = _artifacts(STRUCTURED_SFT_EVALUATOR_FILES)
    holdout = _holdout()
    holdout["examples"][0]["projection"][field] = value
    evaluator["holdout.private.json"] = json.dumps(holdout).encode()
    monkeypatch.setattr(
        evaluation,
        "validate_structured_sft_custody_pair",
        lambda *_args: {
            "holdout_example_count": 1,
            "example_id_overlap_count": 0,
            "case_fingerprint_overlap_count": 0,
            "candidate_contains_holdout_seed": False,
        },
    )
    with pytest.raises(
        evaluation.RecurrentSFTEvaluationError,
        match="holdout_example_invalid",
    ):
        evaluation.evaluator_holdout_rows(candidate, evaluator)


def test_holdout_projection_allows_executed_tool_evidence_for_interpretation(
    monkeypatch,
) -> None:
    candidate = _artifacts(STRUCTURED_SFT_CANDIDATE_FILES)
    evaluator = _artifacts(STRUCTURED_SFT_EVALUATOR_FILES)
    holdout = _holdout()
    example = holdout["examples"][0]
    example["target_kind"] = "tool_result_interpretation"
    example["projection"] = {
        "answer_evidence_basis": "executed_tool_stdout",
        "answer_evidence_in_input": True,
        "oracle_fields_exported_to_trainer": [],
    }
    example["messages"].insert(
        -1,
        {"role": "tool", "content": "verified stdout"},
    )
    evaluator["holdout.private.json"] = json.dumps(holdout).encode()
    monkeypatch.setattr(
        evaluation,
        "validate_structured_sft_custody_pair",
        lambda *_args: {
            "holdout_example_count": 1,
            "example_id_overlap_count": 0,
            "case_fingerprint_overlap_count": 0,
            "candidate_contains_holdout_seed": False,
        },
    )

    rows, _custody = evaluation.evaluator_holdout_rows(candidate, evaluator)

    assert rows[0]["_meta"]["target_kind"] == "tool_result_interpretation"


def test_holdout_projection_can_use_bound_launcher_custody_without_fork(
    monkeypatch,
) -> None:
    candidate = _artifacts(STRUCTURED_SFT_CANDIDATE_FILES)
    evaluator = _artifacts(STRUCTURED_SFT_EVALUATOR_FILES)
    evaluator["holdout.private.json"] = json.dumps(_holdout()).encode()
    custody = {
        "holdout_example_count": 1,
        "example_id_overlap_count": 0,
        "case_fingerprint_overlap_count": 0,
        "candidate_contains_holdout_seed": False,
    }
    monkeypatch.setattr(
        evaluation,
        "validate_structured_sft_custody_pair",
        lambda *_args: pytest.fail("semantic replay would fork"),
    )

    rows, observed = evaluation.evaluator_holdout_rows(
        candidate,
        evaluator,
        replay_semantics=False,
        expected_custody=custody,
    )

    assert len(rows) == 1
    assert observed == custody


def _control_report() -> dict:
    arms = {}
    for arm in CONTROL_ARMS:
        body = {
            "arm": arm,
            "transform": {"bound": True},
            "adapter": {
                "filename": f"{arm}.safetensors",
                "sha256": hashlib.sha256(arm.encode()).hexdigest(),
                "size_bytes": 10,
            },
            "starting_adapter_sha256": "4" * 64,
            "optimizer": "AdamW",
            "optimizer_updates": 20,
            "sample_indices": list(range(20)),
            "sample_token_counts": [100] * 20,
            "sample_token_budget": 2000,
            "loss_trail": [1.0],
            "branch_cosine_trail": [[0.1]],
            "duration_s": 1.0,
        }
        arms[arm] = {**body, "arm_report_sha256": sha256_json(body)}
    body = {
        "schema": evaluation.CONTROL_REPORT_SCHEMA,
        "status": "completed_equal_work_negative_controls",
        "reference_authority_sha256": "a" * 64,
        "reference_checkpoint_sha256": "b" * 64,
        "model_identity_sha256": "c" * 64,
        "execution_spec_sha256": "d" * 64,
        "trainer_config_sha256": "f" * 64,
        "initial_adapter_sha256": "4" * 64,
        "equal_sample_order": True,
        "equal_per_step_token_counts": True,
        "equal_optimizer_and_hyperparameters": True,
        "identical_initial_adapter_for_all_controls": True,
        "base_weights_unchanged": True,
        "evaluator_access": False,
        "production_effect": False,
        "promotion_allowed": False,
        "reference_optimizer_updates": 20,
        "control_optimizer_updates": {arm: 20 for arm in CONTROL_ARMS},
        "arms": arms,
    }
    return {**body, "report_sha256": sha256_json(body)}


def test_control_report_binds_equal_work_and_adapters() -> None:
    report = _control_report()
    file_sha = "e" * 64
    bindings = evaluation.validate_control_report(
        report,
        report_file_sha256=file_sha,
        expected_report_file_sha256=file_sha,
        expected_authority_sha256="a" * 64,
        expected_reference_checkpoint_sha256="b" * 64,
        expected_model_identity_sha256="c" * 64,
        expected_execution_spec_sha256="d" * 64,
        expected_reference_optimizer_updates=20,
        expected_trainer_config_sha256="f" * 64,
    )
    assert set(bindings) == set(CONTROL_ARMS)
    assert bindings["syntax_only"]["filename"] == "syntax_only.safetensors"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("equal_sample_order", False),
        ("evaluator_access", True),
        ("production_effect", True),
    ],
)
def test_control_report_rejects_false_equal_work_claims(
    field: str,
    value: object,
) -> None:
    report = _control_report()
    report[field] = value
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = sha256_json(body)
    with pytest.raises(
        evaluation.RecurrentSFTEvaluationError,
        match="control_report_invalid",
    ):
        evaluation.validate_control_report(
            report,
            report_file_sha256="e" * 64,
            expected_report_file_sha256="e" * 64,
            expected_authority_sha256="a" * 64,
            expected_reference_checkpoint_sha256="b" * 64,
            expected_model_identity_sha256="c" * 64,
            expected_execution_spec_sha256="d" * 64,
            expected_reference_optimizer_updates=20,
            expected_trainer_config_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda report: report["arms"]["shuffled_traces"].__setitem__(
                "sample_indices",
                list(reversed(range(20))),
            ),
            "control_workload_invalid",
        ),
        (
            lambda report: report["arms"]["syntax_only"].__setitem__(
                "sample_token_counts",
                [99] + [100] * 19,
            ),
            "control_arm_invalid",
        ),
        (
            lambda report: report["arms"]["sham_labels"].__setitem__(
                "starting_adapter_sha256",
                "9" * 64,
            ),
            "control_arm_invalid",
        ),
        (
            lambda report: report["arms"]["sham_labels"].__setitem__(
                "optimizer",
                "SGD",
            ),
            "control_arm_invalid",
        ),
    ],
)
def test_control_report_reconstructs_equal_work(
    mutation,
    expected: str,
) -> None:
    report = _control_report()
    mutation(report)
    for arm in CONTROL_ARMS:
        arm_body = dict(report["arms"][arm])
        arm_body.pop("arm_report_sha256")
        report["arms"][arm]["arm_report_sha256"] = sha256_json(arm_body)
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = sha256_json(body)

    with pytest.raises(
        evaluation.RecurrentSFTEvaluationError,
        match=expected,
    ):
        evaluation.validate_control_report(
            report,
            report_file_sha256="e" * 64,
            expected_report_file_sha256="e" * 64,
            expected_authority_sha256="a" * 64,
            expected_reference_checkpoint_sha256="b" * 64,
            expected_model_identity_sha256="c" * 64,
            expected_execution_spec_sha256="d" * 64,
            expected_reference_optimizer_updates=20,
            expected_trainer_config_sha256="f" * 64,
        )


def test_control_report_rejects_reference_workload_drift() -> None:
    report = _control_report()
    with pytest.raises(
        evaluation.RecurrentSFTEvaluationError,
        match="control_workload_invalid",
    ):
        evaluation.validate_control_report(
            report,
            report_file_sha256="e" * 64,
            expected_report_file_sha256="e" * 64,
            expected_authority_sha256="a" * 64,
            expected_reference_checkpoint_sha256="b" * 64,
            expected_model_identity_sha256="c" * 64,
            expected_execution_spec_sha256="d" * 64,
            expected_reference_optimizer_updates=21,
            expected_trainer_config_sha256="f" * 64,
        )


def test_score_forward_uses_branch_mean_ce_and_uniform_mixture_top1() -> None:
    import mlx.core as mx

    logits_a = mx.array([[[5.0, 1.0], [1.0, 5.0]]])
    logits_b = mx.array([[[4.0, 1.0], [1.0, 4.0]]])
    loss, top1 = evaluation.score_forward(
        SimpleNamespace(branch_logits=(logits_a, logits_b)),
        [0, 1],
    )
    assert loss > 0.0
    assert top1 == [True, True]


def _observations(*, loss: float, top1: list[bool]) -> list[dict]:
    return [
        {
            "example_id": row["_meta"]["example_id"],
            "family": row["_meta"]["family"],
            "loss": loss,
            "target_top1": copy.copy(top1),
            "generated_correct": None,
        }
        for row in evaluation.build_regression_canary_rows()
    ]


def test_regression_canaries_require_every_family_to_hold() -> None:
    base = _observations(loss=1.0, top1=[True, False])
    trained = _observations(loss=0.99, top1=[True, False])
    passed = evaluation.regression_canary_verdict(base, trained)
    assert passed["passed"] is True

    trained[-1]["loss"] = 1.2
    failed = evaluation.regression_canary_verdict(base, trained)
    assert failed["passed"] is False
    assert failed["by_family"]["safety"]["passed"] is False


def test_evaluator_source_closure_is_exact_and_hash_bound() -> None:
    closure = evaluator_tool.evaluation_source_closure()
    assert closure["schema"].endswith("source_closure")
    assert len(closure["files"]) == len(EVALUATION_SOURCE_ROLES)
    assert [row["role"] for row in closure["files"]] == sorted(EVALUATION_SOURCE_ROLES)
    assert closure["closure_sha256"] == sha256_json(
        {
            "schema": closure["schema"],
            "files": closure["files"],
        }
    )


def test_adapter_switch_replaces_every_recurrent_tensor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    names = (
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.0.self_attn.q_proj.lora_b",
    )
    topology = {
        names[0]: mx.zeros((2, 3), dtype=mx.float32),
        names[1]: mx.zeros((3, 2), dtype=mx.float32),
    }
    adapter_a = {
        names[0]: mx.ones((2, 3), dtype=mx.float32),
        names[1]: mx.ones((3, 2), dtype=mx.float32),
    }
    adapter_b = {
        names[0]: mx.full((2, 3), 2.0, dtype=mx.float32),
        names[1]: mx.full((3, 2), 3.0, dtype=mx.float32),
    }
    path_a = tmp_path / "a.safetensors"
    path_b = tmp_path / "b.safetensors"
    mx.save_safetensors(str(path_a), adapter_a)
    mx.save_safetensors(str(path_b), adapter_b)

    class _Model:
        current = dict(topology)

        def load_weights(self, items, *, strict: bool) -> None:
            assert strict is False
            self.current = dict(items)

        def trainable_parameters(self):
            return self.current

    model = _Model()
    monkeypatch.setattr(
        evaluator_tool,
        "adapter_tensor_dict",
        lambda observed_model: dict(observed_model.current),
    )

    first = evaluator_tool._load_adapter(
        model,
        path_a,
        expected_topology=topology,
    )
    second = evaluator_tool._load_adapter(
        model,
        path_b,
        expected_topology=topology,
    )

    assert first != second
    assert second == evaluator_tool._tensor_fingerprint(adapter_b)
    assert all(bool(mx.array_equal(model.current[name], adapter_b[name])) for name in names)
