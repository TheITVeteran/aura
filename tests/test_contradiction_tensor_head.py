"""Controlled-mutation admission tests for the RLC contradiction tensor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.brain.llm.latent_cortex.bidirectional_reflector import (
    build_bidirectional_reflector_receipt,
    observe_reflector_vectors,
    position_hidden_sketch,
)
from core.brain.llm.latent_cortex.contradiction_tensor import (
    LEARNED,
    ContradictionTensorRuntime,
    build_contradiction_tensor_receipt,
    validate_contradiction_tensor_receipt,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.learning.contradiction_tensor import (
    ContradictionCellExample,
    ContradictionTensorHead,
    contradiction_features,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _mean(rows):
    return tuple(sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0])))


def _trace_states(
    trace_index: int,
    *,
    domain_offset: float,
    invert: bool = False,
):
    transitions = 8 if trace_index % 2 == 0 else 10
    positions = 3
    has_error = trace_index < 6
    error_transition = transitions // 2 if has_error else None
    error_position = trace_index % positions if has_error else None
    prior_rows = []
    proposal_rows = []
    admitted_rows = []
    for transition in range(transitions):
        prior_positions = []
        proposal_positions = []
        for position in range(positions):
            prior = (
                0.10 * transition,
                0.20 * position,
                domain_offset,
                1.0,
                0.05 * transition,
                -0.10 * position,
                0.25,
                -0.25,
            )
            proposal = tuple(
                left + right
                for left, right in zip(
                    prior,
                    (0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.01, 0.0),
                    strict=True,
                )
            )
            is_error = transition == error_transition and position == error_position
            if is_error != invert:
                proposal = tuple(
                    left + right
                    for left, right in zip(
                        proposal,
                        (5.0, -4.0, 3.0, -2.0, 4.0, -3.0, 2.0, -1.0),
                        strict=True,
                    )
                )
            prior_positions.append(prior)
            proposal_positions.append(proposal)
        prior_rows.append(prior_positions)
        proposal_rows.append(proposal_positions)
        admitted_rows.append(proposal_positions)
    return (
        prior_rows,
        proposal_rows,
        admitted_rows,
        error_transition,
        error_position,
    )


def _split(
    relation: str,
    prefix: str,
    domains: tuple[str, str],
    *,
    invert: bool = False,
) -> list[ContradictionCellExample]:
    examples = []
    for trace_index in range(8):
        domain = domains[trace_index % len(domains)]
        (
            priors,
            proposals,
            admitted,
            error_transition,
            error_position,
        ) = _trace_states(
            trace_index,
            domain_offset=float(trace_index % 2),
            invert=invert,
        )
        transitions = len(priors)
        positions = len(priors[0])
        for transition in range(transitions):
            for position in range(positions):
                prefix_context = _mean(
                    [
                        position_hidden_sketch(admitted[index][position])
                        for index in range(transition + 1)
                    ]
                )
                suffix_context = _mean(
                    [
                        position_hidden_sketch(admitted[index][position])
                        for index in range(transition, transitions)
                    ]
                )
                features = contradiction_features(
                    position_hidden_sketch(priors[transition][position]),
                    position_hidden_sketch(proposals[transition][position]),
                    position_hidden_sketch(admitted[transition][position]),
                    position_hidden_sketch(priors[0][position]),
                    position_hidden_sketch(admitted[-1][position]),
                    prefix_context,
                    suffix_context,
                    accepted=True,
                    transition_fraction=transition / max(1, transitions - 1),
                    position_fraction=position / max(1, positions - 1),
                )
                trace_id = f"{prefix}-trace-{trace_index}"
                examples.append(
                    ContradictionCellExample(
                        example_id=(f"{trace_id}-transition-{transition}-position-{position}"),
                        trace_id=trace_id,
                        task_id=f"{prefix}-task-{trace_index}",
                        domain_id=domain,
                        relation=relation,
                        mutation_family=(
                            "sham"
                            if error_transition is None
                            else (
                                "premise_negation"
                                if trace_index % 2 == 0
                                else "operator_substitution"
                            )
                        ),
                        transition_index=transition,
                        transition_count=transitions,
                        position_index=position,
                        position_count=positions,
                        contradiction_transition_index=error_transition,
                        contradiction_position_index=error_position,
                        features=features,
                        trace_receipt_sha256=_digest(f"trace:{trace_id}"),
                        mutation_receipt_sha256=_digest(f"mutation:{trace_id}"),
                        outcome_verifier_id=f"deterministic-verifier-{prefix}",
                    )
                )
    return examples


@pytest.fixture(scope="module")
def admitted_head() -> ContradictionTensorHead:
    head = ContradictionTensorHead.fit(
        _split("train", "train", ("logic", "math")),
        _split("in_domain", "cal", ("logic", "math")),
        _split("out_of_domain", "ood", ("code", "planning")),
        hidden_width=12,
        steps=1_000,
        seed=17,
    )
    assert head.admitted
    return head


def test_admission_proves_middle_long_context_and_ood_localization(
    admitted_head: ContradictionTensorHead,
):
    manifest = admitted_head.manifest()
    assert manifest["attention_perturbation_authorized"] is False
    for relation in ("in_domain_metrics", "out_of_domain_metrics"):
        metrics = manifest[relation]
        assert metrics["middle_error_trace_count"] >= 2
        assert metrics["long_error_trace_count"] >= 1
        assert metrics["long_no_error_trace_count"] >= 1
        assert metrics["step_exact_accuracy"] >= 0.70
        assert metrics["no_error_specificity"] >= 0.75
        assert metrics["cell_auc"] >= 0.75
        assert metrics["step_auc"] >= 0.75
        assert metrics["step_brier"] <= 0.20
        assert metrics["step_ece"] <= 0.15


def test_training_requires_complete_transition_position_tensors():
    train = _split("train", "train", ("logic", "math"))
    with pytest.raises(ValueError, match="incomplete"):
        ContradictionTensorHead.fit(
            train[1:],
            _split("in_domain", "cal", ("logic", "math")),
            _split("out_of_domain", "ood", ("code", "planning")),
        )


def test_training_requires_independent_trace_and_mutation_evidence():
    train = _split("train", "train", ("logic", "math"))
    first_receipt = train[0].mutation_receipt_sha256
    corrupted = [
        replace(row, mutation_receipt_sha256=first_receipt)
        if row.trace_id == "train-trace-1"
        else row
        for row in train
    ]
    with pytest.raises(ValueError, match="duplicate trace/mutation evidence"):
        ContradictionTensorHead.fit(
            corrupted,
            _split("in_domain", "cal", ("logic", "math")),
            _split("out_of_domain", "ood", ("code", "planning")),
        )


def test_training_requires_genuinely_out_of_domain_tasks():
    out = [
        replace(row, domain_id="logic")
        if row.domain_id == "code"
        else replace(row, domain_id="math")
        for row in _split("out_of_domain", "ood", ("code", "planning"))
    ]
    with pytest.raises(ValueError, match="ID/OOD domains"):
        ContradictionTensorHead.fit(
            _split("train", "train", ("logic", "math")),
            _split("in_domain", "cal", ("logic", "math")),
            out,
        )


def test_unadmitted_ood_failure_cannot_be_loaded(tmp_path: Path):
    head = ContradictionTensorHead.fit(
        _split("train", "train", ("logic", "math")),
        _split("in_domain", "cal", ("logic", "math")),
        _split(
            "out_of_domain",
            "ood",
            ("code", "planning"),
            invert=True,
        ),
        hidden_width=12,
        steps=1_000,
        seed=17,
    )
    assert not head.admitted
    path = tmp_path / "unadmitted.json"
    path.write_bytes(
        json.dumps(
            head.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    with pytest.raises(ValueError, match="failed admission"):
        ContradictionTensorHead.load(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_pinned_artifact_round_trip_and_symlink_refusal(
    tmp_path: Path,
    admitted_head: ContradictionTensorHead,
):
    path = tmp_path / "contradiction.json"
    digest = admitted_head.save(path)
    loaded = ContradictionTensorHead.load(path, expected_sha256=digest)
    assert loaded.to_payload() == admitted_head.to_payload()

    link = tmp_path / "contradiction-link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        ContradictionTensorHead.load(link, expected_sha256=digest)


@pytest.fixture
def runtime(
    tmp_path: Path,
    admitted_head: ContradictionTensorHead,
) -> ContradictionTensorRuntime:
    path = tmp_path / "contradiction.json"
    digest = admitted_head.save(path)
    return ContradictionTensorRuntime.from_config(
        {
            "mode": "learned",
            "head_path": str(path),
            "head_sha256": digest,
        }
    )


def _reflector_source(
    *,
    trace_index: int = 0,
    future_shift: float = 0.0,
):
    priors, proposals, admitted, _, _ = _trace_states(
        trace_index,
        domain_offset=0.0,
    )
    if future_shift:
        proposals[-1][-1] = tuple(value + future_shift for value in proposals[-1][-1])
        admitted[-1][-1] = proposals[-1][-1]
    observations = []
    transitions = []
    for step, (prior_positions, proposal_positions, admitted_positions) in enumerate(
        zip(priors, proposals, admitted, strict=True)
    ):
        prior_hash = _digest(f"prior:{trace_index}:{step}:{prior_positions}")
        proposal_hash = _digest(f"proposal:{trace_index}:{step}:{proposal_positions}")
        admitted_hash = proposal_hash
        observations.append(
            observe_reflector_vectors(
                _mean(prior_positions),
                _mean(proposal_positions),
                _mean(admitted_positions),
                branch_index=0,
                branch_step=step,
                prior_state_sha256=prior_hash,
                proposal_state_sha256=proposal_hash,
                admitted_state_sha256=admitted_hash,
                accepted=True,
                prior_positions=prior_positions,
                proposal_positions=proposal_positions,
                admitted_positions=admitted_positions,
            )
        )
        transitions.append(
            {
                "branch_step": step,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": proposal_hash,
                "admitted_reasoning_sha256": admitted_hash,
                "accepted": True,
            }
        )
    source_payload = {"branches": [{"transitions": transitions}]}
    update_acceptance = {
        **source_payload,
        "receipt_sha256": canonical_sha256(source_payload),
    }
    reflector = build_bidirectional_reflector_receipt(
        branches=[SimpleNamespace(index=0, reflector_trace=observations)],
        update_acceptance=update_acceptance,
        selected_branch=0,
    )
    return reflector


def test_runtime_emits_calibrated_transition_position_tensor_without_authority(
    runtime: ContradictionTensorRuntime,
):
    reflector = _reflector_source()
    receipt = build_contradiction_tensor_receipt(
        reflector=reflector,
        runtime=runtime,
        selected_branch=0,
    )
    assert receipt["mode"] == LEARNED
    assert receipt["calibrated"] is True
    assert receipt["cell_count"] == 8 * 3
    assert receipt["selected_branch_candidate"] == {
        "transition_index": 4,
        "position_index": 0,
    }
    assert receipt["decoded_answer_consumed"] is False
    assert receipt["diagnostic_only"] is True
    assert receipt["attention_perturbation_authorized"] is False
    cell = receipt["branches"][0]["tensor"][4][0]
    assert cell["position_kind"] == "latent_workspace_sequence_position"
    assert cell["decoded_token_index"] is None
    assert set(cell["channels"]) == {
        "local_discontinuity",
        "admission_gap",
        "premise_conflict",
        "conclusion_conflict",
        "prefix_conflict",
        "suffix_conflict",
        "trajectory_conflict",
    }


def test_future_only_lesion_changes_earlier_contradiction_evidence(
    runtime: ContradictionTensorRuntime,
):
    baseline = build_contradiction_tensor_receipt(
        reflector=_reflector_source(),
        runtime=runtime,
        selected_branch=0,
    )
    changed = build_contradiction_tensor_receipt(
        reflector=_reflector_source(future_shift=3.0),
        runtime=runtime,
        selected_branch=0,
    )
    before = baseline["branches"][0]["tensor"][0][2]
    after = changed["branches"][0]["tensor"][0][2]
    assert before["features_sha256"] != after["features_sha256"]
    assert before["contradiction_probability"] != after["contradiction_probability"]


def test_rehashed_probability_lie_is_rejected(
    runtime: ContradictionTensorRuntime,
):
    reflector = _reflector_source()
    receipt = build_contradiction_tensor_receipt(
        reflector=reflector,
        runtime=runtime,
        selected_branch=0,
    )
    forged = json.loads(json.dumps(receipt))
    forged["branches"][0]["tensor"][4][0]["contradiction_probability"] = 0.01
    forged["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="reconstruction failed"):
        validate_contradiction_tensor_receipt(
            forged,
            expected_runtime=runtime,
            reflector=reflector,
            expected_n_branches=1,
        )


def test_unavailable_mode_emits_no_synthetic_probability():
    runtime = ContradictionTensorRuntime.from_config(None)
    receipt = build_contradiction_tensor_receipt(
        reflector=_reflector_source(),
        runtime=runtime,
        selected_branch=0,
    )
    assert receipt["calibrated"] is False
    assert receipt["complete_trace_consumed"] is False
    assert receipt["latent_positions_consumed"] is False
    assert receipt["cell_count"] == 0
    assert receipt["branches"][0]["tensor"] == []
    assert receipt["selected_branch_candidate"] is None


def test_real_tiny_qwen_runs_full_tensor_and_meters_work(
    tmp_path: Path,
    admitted_head: ContradictionTensorHead,
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )
    from core.brain.llm.latent_cortex.verified_best import (
        VERIFIER_OBSERVATION_SCHEMA,
    )
    from core.learning.neural_uncertainty import (
        HiddenStateCorrectnessExample,
        NeuralUncertaintyHead,
    )

    path = tmp_path / "real-qwen-contradiction.json"
    digest = admitted_head.save(path)

    def uncertainty_examples(split: str):
        rng = np.random.default_rng(31 if split == "train" else 47)
        rows = []
        index = 0
        for signal, expected_rate in (
            (-3.0, 0.05),
            (-1.5, 0.20),
            (-0.5, 0.40),
            (0.5, 0.60),
            (1.5, 0.80),
            (3.0, 0.95),
        ):
            labels = [True] * round(48 * expected_rate)
            labels += [False] * (48 - len(labels))
            rng.shuffle(labels)
            for correct in labels:
                hidden = rng.normal(0.0, 0.10, size=32)
                hidden[0] = signal + rng.normal(0.0, 0.03)
                rows.append(
                    HiddenStateCorrectnessExample(
                        example_id=f"{split}-{index}",
                        task_id=f"{split}-task-{index % 8}",
                        hidden_state=tuple(float(value) for value in hidden),
                        correct=correct,
                        state_sha256=_digest(f"{split}:state:{index}"),
                        outcome_receipt_sha256=_digest(f"{split}:outcome:{index}"),
                        outcome_verifier_id=("independent-exact-grader-v1"),
                    )
                )
                index += 1
        return rows

    uncertainty_head = NeuralUncertaintyHead.fit(
        uncertainty_examples("train"),
        uncertainty_examples("calibration"),
        hidden_width=8,
        seed=53,
        steps=500,
    )
    uncertainty_path = tmp_path / "real-qwen-uncertainty.json"
    uncertainty_digest = uncertainty_head.save(uncertainty_path)

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 128 for character in text][:16]

        def decode(self, ids):
            return " ".join(str(item) for item in ids)

    class ExactVerifier:
        def __call__(self, text):
            if text.startswith("Independent consistency check:"):
                from core.brain.llm.latent_cortex.task_verifiers import (
                    check_arithmetic_claims,
                )

                return float(check_arithmetic_claims(text)["score"])
            return 0.8

        def observe_with_bounds(self, text):
            score = self(text)
            return {
                "schema": VERIFIER_OBSERVATION_SCHEMA,
                "score": score,
                "lower_bound": score,
                "upper_bound": score,
                "sample_count": 1,
                "basis": "deterministic_exact",
                "independent": True,
                "evidence_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }

    mx.random.seed(17)
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
        Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=23),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=3,
                fixed_depth=True,
                convergence_eps=1e-9,
            ),
            branches=BranchConfig(n_branches=2),
            latent_opt=LatentOptConfig(enabled=False),
            contradiction_head={
                "mode": "learned",
                "head_path": str(path),
                "head_sha256": digest,
            },
            contradiction_perturber={
                "mode": "counterfactual",
                "replicates": 2,
            },
            uncertainty_head={
                "mode": "learned",
                "head_path": str(uncertainty_path),
                "head_sha256": uncertainty_digest,
            },
            local_exploration={
                "mode": "counterfactual",
                "min_predictive_entropy": 0.0,
                "max_stable_contradiction_probability": 1.0,
            },
            verifier_probe_max_tokens=16,
            decode_max_tokens=3,
            allow_vanilla_fallback=False,
        ),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        verifier=ExactVerifier(),
    )
    assert result.ok is True
    receipt = result.receipt.contradiction_tensor
    assert receipt["calibrated"] is True
    assert receipt["cell_count"] >= 1
    operations = result.receipt.budget["resource_accounting"]["operations"]
    assert operations["contradiction_tensor_head"]["host_scalar_ops"] > 0
    perturbation = result.receipt.contradiction_perturbation
    assert perturbation["status"] == "restored", perturbation
    assert perturbation["state_mutation_applied"] is False
    assert perturbation["rollback_proven"] is True
    assert perturbation["all_arms_equal_compute"] is True
    assert len(perturbation["arms"]) == 3
    assert operations["contradiction_perturbation_candidates"]["tensor_element_writes"] > 0
    exploration = result.receipt.local_exploration
    assert exploration["status"] == "restored", exploration
    assert exploration["state_mutation_applied"] is False
    assert exploration["rollback_proven"] is True
    assert exploration["all_candidates_equal_compute"] is True
    assert len(exploration["candidates"]) == 9
    assert exploration["target_position"] != exploration["sham_position"]
    assert operations["local_exploration_candidates"]["tensor_element_writes"] > 0
    validate_contradiction_tensor_receipt(
        receipt,
        expected_runtime=engine._resolve_contradiction_head(),
        reflector=result.receipt.bidirectional_reflector,
        expected_n_branches=2,
    )
