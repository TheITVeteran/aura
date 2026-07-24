from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.loop_core import canonical_sha256  # noqa: E402
from core.brain.llm.latent_cortex.mistake_locator import (  # noqa: E402
    LEARNED,
    UNAVAILABLE,
    MistakeLocatorRuntime,
    build_mistake_locator_receipt,
    process_branch_assessment,
    validate_mistake_locator_receipt,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.learning.mistake_locator import (  # noqa: E402
    MistakeLocatorHead,
    MistakeTransitionExample,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _examples(
    relation: str,
    prefix: str,
    domains: tuple[str, str],
) -> list[MistakeTransitionExample]:
    rows: list[MistakeTransitionExample] = []
    for trace_index in range(8):
        error_index = trace_index % 4 if trace_index < 4 else None
        trace_id = f"{prefix}-trace-{trace_index}"
        for index in range(4):
            prior = (
                trace_index * 0.01,
                index * 0.02,
                (trace_index + index) * 0.01,
            )
            delta = (4.0, -3.5, 3.0) if index == error_index else (0.03, -0.02, 0.01)
            rows.append(
                MistakeTransitionExample(
                    example_id=f"{trace_id}-{index}",
                    trace_id=trace_id,
                    task_id=f"{prefix}-task-{trace_index}",
                    domain_id=domains[trace_index % 2],
                    relation=relation,
                    mutation_family=("premise_flip" if trace_index % 2 == 0 else "operator_swap"),
                    transition_index=index,
                    transition_count=4,
                    error_index=error_index,
                    prior_hidden=prior,
                    candidate_hidden=tuple(
                        value + change for value, change in zip(prior, delta, strict=True)
                    ),
                    trace_receipt_sha256=_digest(trace_id),
                    outcome_verifier_id=f"verifier-{prefix}",
                )
            )
    return rows


def _wide_examples(
    relation: str,
    prefix: str,
    domains: tuple[str, str],
    *,
    width: int = 32,
) -> list[MistakeTransitionExample]:
    rng = np.random.default_rng(len(prefix))
    rows: list[MistakeTransitionExample] = []
    for trace_index in range(8):
        error_index = trace_index % 4 if trace_index < 4 else None
        trace_id = f"{prefix}-wide-{trace_index}"
        for index in range(4):
            prior = rng.normal(0.0, 0.03, size=width)
            delta = rng.normal(0.0, 0.01, size=width)
            if index == error_index:
                delta[:4] += np.asarray([4.0, -3.5, 3.0, -2.5])
            rows.append(
                MistakeTransitionExample(
                    example_id=f"{trace_id}-{index}",
                    trace_id=trace_id,
                    task_id=f"{prefix}-wide-task-{trace_index}",
                    domain_id=domains[trace_index % 2],
                    relation=relation,
                    mutation_family=("premise_flip" if trace_index % 2 == 0 else "operator_swap"),
                    transition_index=index,
                    transition_count=4,
                    error_index=error_index,
                    prior_hidden=tuple(float(value) for value in prior),
                    candidate_hidden=tuple(float(value) for value in prior + delta),
                    trace_receipt_sha256=_digest(trace_id),
                    outcome_verifier_id=f"verifier-{prefix}",
                )
            )
    return rows


def _calibration_examples(
    relation: str,
    prefix: str,
    domains: tuple[str, str],
    *,
    width: int = 3,
) -> list[MistakeTransitionExample]:
    rows: list[MistakeTransitionExample] = []
    for trace_index in range(48):
        error_index = (trace_index // 2) % 4 if trace_index < 36 else None
        trace_id = f"{prefix}-calibrated-{trace_index}"
        for index in range(4):
            if width == 3:
                prior = (
                    trace_index * 0.001,
                    index * 0.002,
                    (trace_index + index) * 0.001,
                )
            else:
                prior = tuple(
                    float(((trace_index * 37 + index * 11 + coordinate * 17) % 101) - 50)
                    for coordinate in range(width)
                )
            delta = [0.01 * ((coordinate % 3) - 1) for coordinate in range(width)]
            if index == error_index:
                delta[:3] = [4.0, -3.5, 3.0]
            rows.append(
                MistakeTransitionExample(
                    example_id=f"{trace_id}-{index}",
                    trace_id=trace_id,
                    task_id=f"{prefix}-calibrated-task-{trace_index}",
                    domain_id=domains[trace_index % 2],
                    relation=relation,
                    mutation_family=("premise_flip" if trace_index % 2 == 0 else "operator_swap"),
                    transition_index=index,
                    transition_count=4,
                    error_index=error_index,
                    prior_hidden=tuple(prior),
                    candidate_hidden=tuple(
                        value + change for value, change in zip(prior, delta, strict=True)
                    ),
                    trace_receipt_sha256=_digest(trace_id),
                    outcome_verifier_id=f"verifier-{prefix}",
                )
            )
    return rows


@pytest.fixture
def runtime(tmp_path: Path) -> MistakeLocatorRuntime:
    head = MistakeLocatorHead.fit(
        _examples("train", "train", ("logic", "math")),
        _examples("in_domain", "cal", ("logic", "math")),
        _examples("out_of_domain", "ood", ("code", "planning")),
        hidden_width=8,
        steps=800,
        seed=11,
    )
    path = tmp_path / "locator.json"
    digest = head.save(path)
    return MistakeLocatorRuntime.from_config(
        {
            "mode": "learned",
            "head_path": str(path),
            "head_sha256": digest,
        }
    )


@pytest.fixture
def process_runtime(tmp_path: Path) -> MistakeLocatorRuntime:
    head = MistakeLocatorHead.fit(
        _calibration_examples("train", "process-train", ("logic", "math")),
        _calibration_examples("in_domain", "process-cal", ("logic", "math")),
        _calibration_examples(
            "out_of_domain",
            "process-ood",
            ("code", "planning"),
        ),
        hidden_width=8,
        steps=800,
        seed=17,
    )
    path = tmp_path / "process-locator.json"
    digest = head.save(path)
    return MistakeLocatorRuntime.from_config(
        {
            "mode": "learned",
            "head_path": str(path),
            "head_sha256": digest,
        }
    )


def _source(transitions: list[dict]) -> dict:
    payload = {"branches": [{"transitions": transitions}]}
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _process_trace(
    runtime: MistakeLocatorRuntime,
    *,
    branch_index: int,
    error_step: int | None,
) -> tuple[list[dict], list[dict]]:
    observations = []
    transitions = []
    calibration_trace_index = 36 if error_step is None else error_step * 2
    for step in range(4):
        prior = (
            calibration_trace_index * 0.001,
            step * 0.002,
            (calibration_trace_index + step) * 0.001,
        )
        delta = (4.0, -3.5, 3.0) if step == error_step else (0.03, -0.02, 0.01)
        proposal = tuple(value + change for value, change in zip(prior, delta, strict=True))
        prior_hash = _digest(f"process-prior-{branch_index}-{step}")
        proposal_hash = _digest(f"process-proposal-{branch_index}-{step}")
        observations.append(
            runtime.observe(
                mx.array([[prior]], dtype=mx.float32),
                mx.array([[proposal]], dtype=mx.float32),
                branch_index=branch_index,
                branch_step=step,
                prior_state_sha256=prior_hash,
                proposal_state_sha256=proposal_hash,
                admitted_state_sha256=proposal_hash,
                accepted=True,
            )
        )
        transitions.append(
            {
                "branch_step": step,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": proposal_hash,
                "admitted_reasoning_sha256": proposal_hash,
                "accepted": True,
            }
        )
    return observations, transitions


def test_live_locator_reconstructs_exact_transition_and_never_steers_repair(
    runtime: MistakeLocatorRuntime,
):
    observations = []
    transitions = []
    states = [
        ((0.0, 0.0, 0.0), (0.03, -0.02, 0.01)),
        ((0.1, 0.0, 0.0), (4.1, -3.5, 3.0)),
        ((0.2, 0.0, 0.0), (0.23, -0.02, 0.01)),
    ]
    for step, (prior, admitted) in enumerate(states):
        prior_hash = _digest(f"prior-{step}")
        admitted_hash = _digest(f"admitted-{step}")
        observations.append(
            runtime.observe(
                mx.array([[prior]], dtype=mx.float32),
                mx.array([[admitted]], dtype=mx.float32),
                branch_index=0,
                branch_step=step,
                prior_state_sha256=prior_hash,
                proposal_state_sha256=admitted_hash,
                admitted_state_sha256=admitted_hash,
                accepted=True,
            )
        )
        transitions.append(
            {
                "branch_step": step,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": admitted_hash,
                "admitted_reasoning_sha256": admitted_hash,
                "accepted": True,
            }
        )
    branch = SimpleNamespace(index=0, mistake_locator_trace=observations)
    source = _source(transitions)
    receipt = build_mistake_locator_receipt(
        branches=[branch],
        runtime=runtime,
        update_acceptance=source,
        selected_branch=0,
    )
    assert receipt["mode"] == LEARNED
    assert receipt["selected_branch_candidate"] == 1
    assert receipt["localization_admitted"] is True
    assert receipt["repair_steering_authorized"] is False
    assert receipt["observation_count"] == 3


def test_unavailable_mode_emits_no_localization():
    runtime = MistakeLocatorRuntime.from_config(None)
    branch = SimpleNamespace(index=0, mistake_locator_trace=[])
    source = _source([])
    receipt = build_mistake_locator_receipt(
        branches=[branch],
        runtime=runtime,
        update_acceptance=source,
        selected_branch=0,
    )
    assert receipt["mode"] == UNAVAILABLE
    assert receipt["candidate_count"] == 0
    assert receipt["selected_branch_candidate"] is None
    assert receipt["localization_admitted"] is False


def test_process_verifier_abstains_for_unknown_domain(
    process_runtime: MistakeLocatorRuntime,
):
    observations, _transitions = _process_trace(
        process_runtime,
        branch_index=0,
        error_step=None,
    )
    assessment = process_branch_assessment(
        observations,
        runtime=process_runtime,
        domain="unseen-domain",
    )
    assert assessment["process_score"] is None
    assert assessment["selection_authority_admitted"] is False
    assert assessment["abstention_reason"] == "domain_not_calibrated"
    assert all(row["process_credit"] is None for row in assessment["transitions"])


def test_process_verifier_abstains_under_hidden_distribution_shift(
    process_runtime: MistakeLocatorRuntime,
):
    observations, _transitions = _process_trace(
        process_runtime,
        branch_index=0,
        error_step=None,
    )
    shifted = [dict(row) for row in observations]
    for row in shifted:
        row["prior_pooled_hidden"] = [value + 100.0 for value in row["prior_pooled_hidden"]]
        row["proposal_pooled_hidden"] = [value + 100.0 for value in row["proposal_pooled_hidden"]]
    assessment = process_branch_assessment(
        shifted,
        runtime=process_runtime,
        domain="logic",
    )
    assert assessment["selection_authority_admitted"] is False
    assert assessment["abstention_reason"] == "feature_distribution_shift"
    assert {row["abstention_reason"] for row in assessment["transitions"]} == {
        "feature_distribution_shift"
    }


def test_calibrated_process_scores_are_causal_and_receipt_bound(
    process_runtime: MistakeLocatorRuntime,
):
    clean_observations, clean_transitions = _process_trace(
        process_runtime,
        branch_index=0,
        error_step=None,
    )
    error_observations, error_transitions = _process_trace(
        process_runtime,
        branch_index=1,
        error_step=2,
    )
    clean = process_branch_assessment(
        clean_observations,
        runtime=process_runtime,
        domain="logic",
    )
    error = process_branch_assessment(
        error_observations,
        runtime=process_runtime,
        domain="logic",
    )
    assert clean["selection_authority_admitted"] is True
    assert error["selection_authority_admitted"] is True
    assert clean["process_score"] > error["process_score"]
    source_payload = {
        "branches": [
            {"transitions": clean_transitions},
            {"transitions": error_transitions},
        ]
    }
    source = {
        **source_payload,
        "receipt_sha256": canonical_sha256(source_payload),
    }
    receipt = build_mistake_locator_receipt(
        branches=[
            SimpleNamespace(index=0, mistake_locator_trace=clean_observations),
            SimpleNamespace(index=1, mistake_locator_trace=error_observations),
        ],
        runtime=process_runtime,
        update_acceptance=source,
        selected_branch=0,
        domain="logic",
        process_selection_used=True,
    )
    assert receipt["process_selection_authorized"] is True
    assert receipt["branches"][0]["process"]["process_score"] == clean["process_score"]
    forged = dict(receipt)
    forged["branches"] = [dict(row) for row in receipt["branches"]]
    forged["branches"][0]["process"] = dict(receipt["branches"][0]["process"])
    forged["branches"][0]["process"]["process_score"] = 0.0
    forged["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="process assessment differs"):
        validate_mistake_locator_receipt(
            forged,
            expected_runtime=process_runtime,
            update_acceptance=source,
            expected_n_branches=2,
            expected_domain="logic",
        )


def test_rejected_bad_proposal_remains_visible(runtime: MistakeLocatorRuntime):
    prior_hash = _digest("rejected-prior")
    proposal_hash = _digest("rejected-proposal")
    observation = runtime.observe(
        mx.array([[[0.0, 0.0, 0.0]]]),
        mx.array([[[4.0, -3.5, 3.0]]]),
        branch_index=0,
        branch_step=0,
        prior_state_sha256=prior_hash,
        proposal_state_sha256=proposal_hash,
        admitted_state_sha256=prior_hash,
        accepted=False,
    )
    source = _source(
        [
            {
                "branch_step": 0,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": proposal_hash,
                "admitted_reasoning_sha256": prior_hash,
                "accepted": False,
            }
        ]
    )
    receipt = build_mistake_locator_receipt(
        branches=[SimpleNamespace(index=0, mistake_locator_trace=[observation])],
        runtime=runtime,
        update_acceptance=source,
        selected_branch=0,
    )
    assert receipt["selected_branch_candidate"] == 0
    assert receipt["branches"][0]["observations"][0]["accepted"] is False


def test_rehashed_probability_lie_is_rejected(runtime: MistakeLocatorRuntime):
    prior_hash = _digest("prior")
    admitted_hash = _digest("admitted")
    observation = runtime.observe(
        mx.array([[[0.0, 0.0, 0.0]]]),
        mx.array([[[4.0, -3.5, 3.0]]]),
        branch_index=0,
        branch_step=0,
        prior_state_sha256=prior_hash,
        proposal_state_sha256=admitted_hash,
        admitted_state_sha256=admitted_hash,
        accepted=True,
    )
    source = _source(
        [
            {
                "branch_step": 0,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": admitted_hash,
                "admitted_reasoning_sha256": admitted_hash,
                "accepted": True,
            }
        ]
    )
    receipt = build_mistake_locator_receipt(
        branches=[SimpleNamespace(index=0, mistake_locator_trace=[observation])],
        runtime=runtime,
        update_acceptance=source,
        selected_branch=0,
    )
    forged = dict(receipt)
    forged["branches"] = [dict(receipt["branches"][0])]
    forged["branches"][0]["observations"] = [dict(receipt["branches"][0]["observations"][0])]
    row = forged["branches"][0]["observations"][0]
    row["error_probability"] = 0.01
    row["observation_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "observation_sha256"}
    )
    forged["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="reconstruction failed"):
        validate_mistake_locator_receipt(
            forged,
            expected_runtime=runtime,
            update_acceptance=source,
            expected_n_branches=1,
        )


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 128 for character in text][:16]

    def decode(self, ids):
        return " ".join(str(item) for item in ids)


def test_real_tiny_qwen_covers_every_admitted_transition(tmp_path: Path):
    head = MistakeLocatorHead.fit(
        _wide_examples("train", "train", ("logic", "math")),
        _wide_examples("in_domain", "cal", ("logic", "math")),
        _wide_examples("out_of_domain", "ood", ("code", "planning")),
        hidden_width=8,
        steps=800,
        seed=5,
    )
    assert head.admitted
    path = tmp_path / "wide-locator.json"
    digest = head.save(path)
    config = {
        "mode": LEARNED,
        "head_path": str(path),
        "head_sha256": digest,
    }
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=8,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=23),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=3,
                convergence_eps=1e-9,
                fixed_depth=True,
            ),
            branches=BranchConfig(n_branches=2),
            latent_opt=LatentOptConfig(enabled=False),
            decode_max_tokens=3,
            mistake_locator=config,
            allow_vanilla_fallback=False,
        ),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    assert result.ok is True
    receipt = result.receipt.mistake_locator
    transitions = [
        transition
        for branch in result.receipt.update_acceptance["branches"]
        for transition in branch["transitions"]
    ]
    assert receipt["observation_count"] == len(transitions) >= 1
    assert receipt["repair_steering_authorized"] is False
    assert (
        result.receipt.budget["resource_accounting"]["operations"]["mistake_locator_head"][
            "tensor_element_reads"
        ]
        >= len(transitions) * 64
    )
    validate_mistake_locator_receipt(
        receipt,
        expected_runtime=MistakeLocatorRuntime.from_config(config),
        update_acceptance=result.receipt.update_acceptance,
        expected_n_branches=2,
    )


def test_real_tiny_qwen_uses_only_fully_calibrated_process_selection(tmp_path: Path):
    head = MistakeLocatorHead.fit(
        _calibration_examples(
            "train",
            "wide-process-train",
            ("logic", "math"),
            width=32,
        ),
        _calibration_examples(
            "in_domain",
            "wide-process-cal",
            ("logic", "math"),
            width=32,
        ),
        _calibration_examples(
            "out_of_domain",
            "wide-process-ood",
            ("code", "planning"),
            width=32,
        ),
        hidden_width=8,
        steps=800,
        seed=29,
    )
    path = tmp_path / "wide-process-locator.json"
    digest = head.save(path)
    locator = {
        "mode": LEARNED,
        "head_path": str(path),
        "head_sha256": digest,
    }
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=8,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=31),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=3,
                convergence_eps=1e-9,
                fixed_depth=True,
            ),
            branches=BranchConfig(n_branches=2),
            latent_opt=LatentOptConfig(enabled=False),
            decode_max_tokens=3,
            mistake_locator=locator,
            allow_vanilla_fallback=False,
        ),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        domain="logic",
    )
    assert result.ok is True
    receipt = result.receipt.mistake_locator
    assert receipt["process_selection_used"] is True
    assert receipt["process_selection_authorized"] is True
    assert result.receipt.neural_uncertainty["selection_basis"] == "process_verifier"
    expected = max(
        range(2),
        key=lambda index: receipt["branches"][index]["process"]["process_score"],
    )
    assert result.receipt.selected_branch == expected
