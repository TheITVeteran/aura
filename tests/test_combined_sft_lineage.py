from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from test_verified_replay_sft_publication import (
    DEDUP_KEY,
    PARTITION_KEY,
    _source,
)

from core.learning.combined_sft_lineage import (
    COMBINED_SFT_LINEAGE_CANDIDATE_FILES,
    COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
    CombinedSFTLineageError,
    build_combined_sft_lineage_bundle,
    revalidate_combined_sft_lineage_bundle,
    validate_combined_sft_lineage_custody,
)
from core.learning.structured_sft import (
    StructuredSFTCurriculumSpec,
    build_structured_sft_custody_bundles,
    canonical_json_bytes,
)
from core.learning.verified_replay_sft import (
    build_verified_replay_sft_custody_bundles,
    empty_reference_index,
)


@pytest.fixture(scope="module")
def lineage_inputs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("combined-lineage")
    structured = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=31,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
            max_seq_length=4096,
        ),
        holdout_seed=b"s" * 32,
    )
    protector, _buffer, store, clearances, _payloads = _source(root, count=20)
    replay = build_verified_replay_sft_custody_bundles(
        replay_store=store,
        protector=protector,
        privacy_clearances=clearances,
        partition_key=PARTITION_KEY,
        dedup_key=DEDUP_KEY,
        reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
    )
    evaluations = [
        {
            "corpus": "eval:fresh-reasoning",
            "lineage_root_sha256": "a" * 64,
            "objective": "A sealed evaluation question absent from both training sources.",
            "answer": "A sealed evaluation answer absent from both training sources.",
        },
        {
            "corpus": "eval:fresh-tools",
            "lineage_root_sha256": "b" * 64,
            "objective": "A distinct sealed tool-use evaluation request.",
            "answer": "A distinct sealed tool-use evaluation result.",
        },
    ]
    return structured, replay, evaluations


def _build(lineage_inputs, *, evaluations=None, corpora=None, key=DEDUP_KEY):
    structured, replay, defaults = lineage_inputs
    selected = defaults if evaluations is None else evaluations
    required = sorted({record["corpus"] for record in selected}) if corpora is None else corpora
    return build_combined_sft_lineage_bundle(
        structured_candidate_artifacts=structured.candidate_artifacts,
        structured_evaluator_artifacts=structured.evaluator_artifacts,
        replay_candidate_artifacts=replay.candidate_artifacts,
        replay_evaluator_artifacts=replay.evaluator_artifacts,
        external_evaluation_records=selected,
        required_evaluation_corpora=required,
        dedup_key=key,
    )


def _sha(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_combined_manifest_reconstructs_all_corpora_without_trainer_authority(
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    report = bundle.custody_report
    candidate = bundle.candidate_artifacts[COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]]
    manifest = json.loads(bundle.evaluator_artifacts[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]])

    assert report["record_count"] == 40
    assert report["trainer_ready"] is False
    assert report["training_authority"].startswith("none_pending_")
    assert manifest["augmentation_generation"] == 0
    assert manifest["future_derivatives_inherit_lineage_split"] is True
    assert manifest["required_evaluation_corpora"] == [
        "eval:fresh-reasoning",
        "eval:fresh-tools",
    ]
    assert (
        manifest["combined_semantic_index"]["same_lineage_same_split_derivatives_allowed"] is True
    )
    assert (
        manifest["combined_semantic_index"]["same_corpus_template_near_duplicates_allowed"] is True
    )
    assert b"sealed evaluation question" not in candidate
    assert b"record_id_sha256" not in candidate


def test_complete_bundle_revalidation_is_byte_identical(lineage_inputs) -> None:
    bundle = _build(lineage_inputs)
    structured, replay, evaluations = lineage_inputs
    report = revalidate_combined_sft_lineage_bundle(
        bundle=bundle,
        structured_candidate_artifacts=structured.candidate_artifacts,
        structured_evaluator_artifacts=structured.evaluator_artifacts,
        replay_candidate_artifacts=replay.candidate_artifacts,
        replay_evaluator_artifacts=replay.evaluator_artifacts,
        external_evaluation_records=evaluations,
        required_evaluation_corpora=[
            "eval:fresh-reasoning",
            "eval:fresh-tools",
        ],
        dedup_key=DEDUP_KEY,
    )
    assert report == bundle.custody_report


def test_missing_declared_evaluation_corpus_fails_closed(lineage_inputs) -> None:
    with pytest.raises(
        CombinedSFTLineageError,
        match="evaluation_coverage_incomplete",
    ):
        _build(
            lineage_inputs,
            evaluations=[lineage_inputs[2][0]],
            corpora=["eval:fresh-reasoning", "eval:fresh-tools"],
        )


def test_cross_corpus_exact_or_near_overlap_fails_closed(lineage_inputs) -> None:
    structured, replay, _evaluations = lineage_inputs
    replay_holdout = json.loads(replay.evaluator_artifacts["verified_replay_holdout.json"])
    row = replay_holdout["examples"][0]
    overlap = [
        {
            "corpus": "eval:contaminated",
            "lineage_root_sha256": "c" * 64,
            "objective": row["messages"][0]["content"],
            "answer": row["messages"][1]["content"],
        }
    ]
    with pytest.raises(CombinedSFTLineageError, match="semantic_overlap"):
        _build(lineage_inputs, evaluations=overlap)


def test_same_lineage_cannot_cross_into_evaluation_split(lineage_inputs) -> None:
    structured, replay, _evaluations = lineage_inputs
    replay_holdout = json.loads(replay.evaluator_artifacts["verified_replay_holdout.json"])
    row = replay_holdout["examples"][0]
    attacked = [
        {
            "corpus": "eval:lineage-reuse",
            "lineage_root_sha256": row["_meta"]["lineage_root_sha256"],
            "objective": "Different surface text cannot erase causal lineage reuse.",
            "answer": "This answer is also deliberately different.",
        }
    ]
    with pytest.raises(CombinedSFTLineageError, match="semantic_overlap"):
        _build(lineage_inputs, evaluations=attacked)


def test_candidate_or_evaluator_tampering_breaks_pair_binding(lineage_inputs) -> None:
    bundle = _build(lineage_inputs)
    candidate = deepcopy(bundle.candidate_artifacts)
    commitment = json.loads(candidate[COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]])
    commitment["record_count"] += 1
    candidate[COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]] = canonical_json_bytes(commitment)
    with pytest.raises(CombinedSFTLineageError, match="binding_invalid"):
        validate_combined_sft_lineage_custody(
            candidate,
            bundle.evaluator_artifacts,
        )

    evaluator = deepcopy(bundle.evaluator_artifacts)
    manifest = json.loads(evaluator[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]])
    manifest["combined_semantic_index"]["records"][0]["split"] = "holdout"
    evaluator[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]] = canonical_json_bytes(manifest)
    with pytest.raises(CombinedSFTLineageError, match="binding_invalid"):
        validate_combined_sft_lineage_custody(
            bundle.candidate_artifacts,
            evaluator,
        )


def test_duplicate_json_keys_are_rejected(lineage_inputs) -> None:
    bundle = _build(lineage_inputs)
    candidate = deepcopy(bundle.candidate_artifacts)
    name = COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]
    candidate[name] = candidate[name][:-1] + b',"trainer_ready":false}'
    with pytest.raises(CombinedSFTLineageError, match="duplicate_json_key"):
        validate_combined_sft_lineage_custody(
            candidate,
            bundle.evaluator_artifacts,
        )


def test_recomputed_plain_hashes_cannot_make_malformed_index_valid(
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    candidate = deepcopy(bundle.candidate_artifacts)
    evaluator = deepcopy(bundle.evaluator_artifacts)
    evaluator_name = COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]
    candidate_name = COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]
    manifest = json.loads(evaluator[evaluator_name])
    index = manifest["combined_semantic_index"]
    index["records"][0]["split"] = "attacker_split"
    index_body = dict(index)
    index_body.pop("index_sha256")
    index["index_sha256"] = _sha(index_body)
    manifest["combined_semantic_index_sha256"] = index["index_sha256"]
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha(manifest_body)
    evaluator[evaluator_name] = canonical_json_bytes(manifest)

    commitment = json.loads(candidate[candidate_name])
    commitment["manifest_sha256"] = manifest["manifest_sha256"]
    commitment["combined_semantic_index_sha256"] = index["index_sha256"]
    commitment_body = dict(commitment)
    commitment_body.pop("commitment_sha256")
    commitment["commitment_sha256"] = _sha(commitment_body)
    candidate[candidate_name] = canonical_json_bytes(commitment)

    with pytest.raises(
        CombinedSFTLineageError,
        match="signature_record_invalid",
    ):
        validate_combined_sft_lineage_custody(candidate, evaluator)


def test_dedup_key_change_changes_every_combined_commitment(lineage_inputs) -> None:
    first = _build(lineage_inputs)
    second = _build(lineage_inputs, key=b"different-combined-dedup-key" * 2)
    assert first.candidate_artifacts != second.candidate_artifacts
    assert first.evaluator_artifacts != second.evaluator_artifacts
    assert first.custody_report["manifest_sha256"] != second.custody_report["manifest_sha256"]
