"""Combined pre-augmentation lineage custody for all SFT and eval corpora.

The structured-synthetic and verified-replay builders each prove internal split
integrity.  This module closes the cross-corpus boundary: it reconstructs both
custody pairs, projects every split plus every declared external evaluation
corpus through one Horcrux-derived dedup domain, and emits a candidate-safe
commitment paired with an evaluator-held keyed index.  It grants no training
authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.structured_sft import (
    STRUCTURED_SFT_EVALUATOR_FILES,
    StructuredSFTCurriculumSpec,
    build_structured_sft_curriculum,
    validate_structured_sft_curriculum,
    validate_structured_sft_custody_pair,
)
from core.learning.verified_replay_sft import (
    VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
    VerifiedReplaySFTError,
    assert_semantic_signature_integrity,
    build_semantic_signature_records,
    validate_verified_replay_sft_candidate_artifacts,
    validate_verified_replay_sft_custody_pair,
)

COMBINED_SFT_LINEAGE_MANIFEST_SCHEMA: Final = "aura.rlc.combined_sft_lineage_manifest.v1"
COMBINED_SFT_LINEAGE_COMMITMENT_SCHEMA: Final = "aura.rlc.combined_sft_lineage_commitment.v1"
COMBINED_SFT_LINEAGE_CUSTODY_SCHEMA: Final = "aura.rlc.combined_sft_lineage_custody.v1"
COMBINED_SFT_LINEAGE_INDEX_SCHEMA: Final = "aura.rlc.combined_sft_lineage_keyed_index.v1"
COMBINED_SFT_LINEAGE_CANDIDATE_FILES: Final = ("combined_sft_lineage_commitment.json",)
COMBINED_SFT_LINEAGE_EVALUATOR_FILES: Final = ("combined_sft_lineage_manifest.private.json",)
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_EVALUATION_CORPORA = 1_024
_MAX_EVALUATION_RECORDS = 100_000
_MAX_COMBINED_RECORDS = 250_000
_SPLITS = ("train", "validation", "holdout")
_SIGNATURE_FIELDS = {
    "record_id_sha256",
    "corpus",
    "split",
    "lineage_token",
    "exact_token",
    "objective_token",
    "answer_token",
    "objective_character_count",
    "answer_character_count",
    "token_shingles",
    "character_shingles",
}


class CombinedSFTLineageError(RuntimeError):
    """Combined lineage evidence is incomplete, overlapping, or malformed."""


def _error(reason: str) -> CombinedSFTLineageError:
    return CombinedSFTLineageError(str(reason or "combined_sft_lineage_failed"))


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise _error(f"combined_sft_lineage_duplicate_json_key:{key}")
        value[key] = child
    return value


def _json(payload: bytes, *, code: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise _error(code)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"nonfinite:{constant}")
            ),
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CombinedSFTLineageError(code) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise _error(code)
    return value


def _artifact(
    artifacts: Mapping[str, bytes],
    names: Sequence[str],
    *,
    suffix: str,
) -> bytes:
    matches = [artifacts[name] for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise _error("combined_sft_lineage_artifact_role_invalid")
    return matches[0]


def _structured_records(
    candidate: Mapping[str, bytes],
    evaluator: Mapping[str, bytes],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    custody = validate_structured_sft_custody_pair(candidate, evaluator)
    holdout = _json(
        _artifact(evaluator, STRUCTURED_SFT_EVALUATOR_FILES, suffix="holdout.private.json"),
        code="combined_sft_lineage_structured_holdout_invalid",
    )
    try:
        seed = bytes.fromhex(holdout["holdout_seed_hex"])
        spec = StructuredSFTCurriculumSpec(**dict(holdout["curriculum_manifest"]["spec"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CombinedSFTLineageError("combined_sft_lineage_structured_holdout_invalid") from exc
    curriculum = validate_structured_sft_curriculum(
        build_structured_sft_curriculum(spec, holdout_seed=seed),
        holdout_seed=seed,
    )
    records: list[dict[str, str]] = []
    for split in _SPLITS:
        for row in curriculum["splits"][split]:
            projection = row["projection"]
            target_index = projection["target_message_index"]
            prefix = {
                "messages": row["messages"][:target_index],
                "tools": row["tools"],
            }
            target = row["messages"][target_index]
            records.append(
                {
                    "corpus": f"structured_sft:{row['family']}",
                    "split": split,
                    "lineage_root_sha256": row["case_fingerprint"],
                    "objective": canonical_json_bytes(prefix).decode("ascii"),
                    "answer": canonical_json_bytes(target).decode("ascii"),
                }
            )
    return records, {
        "candidate_package_sha256": custody["candidate_package_sha256"],
        "evaluator_package_sha256": custody["evaluator_package_sha256"],
        "custody_root_sha256": custody["custody_root_sha256"],
        "holdout_seed_commitment_sha256": custody["holdout_seed_commitment_sha256"],
        "record_count": len(records),
    }


def _replay_records(
    candidate: Mapping[str, bytes],
    evaluator: Mapping[str, bytes],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    pair = validate_verified_replay_sft_custody_pair(candidate, evaluator)
    visible = validate_verified_replay_sft_candidate_artifacts(candidate)
    holdout = _json(
        _artifact(evaluator, VERIFIED_REPLAY_SFT_EVALUATOR_FILES, suffix="holdout.json"),
        code="combined_sft_lineage_replay_holdout_invalid",
    )
    rows = [*visible["train_rows"], *visible["validation_rows"], *holdout["examples"]]
    records = []
    for row in rows:
        records.append(
            {
                "corpus": "verified_replay",
                "split": row["_meta"]["split"],
                "lineage_root_sha256": row["_meta"]["lineage_root_sha256"],
                "objective": row["messages"][0]["content"],
                "answer": row["messages"][1]["content"],
            }
        )
    candidate_manifest = pair["candidate_manifest"]
    evaluator_manifest = pair["evaluator_manifest"]
    return records, {
        "candidate_package_sha256": candidate_manifest["candidate_package_sha256"],
        "evaluator_package_sha256": evaluator_manifest["evaluator_package_sha256"],
        "custody_root_sha256": candidate_manifest["custody_root_sha256"],
        "source_store_sha256": candidate_manifest["source_store_sha256"],
        "partition_manifest_sha256": pair["partition_manifest"]["manifest_sha256"],
        "source_semantic_dedup_manifest_sha256": pair["semantic_dedup_manifest"]["manifest_sha256"],
        "privacy_manifest_sha256": pair["privacy_manifest"]["manifest_sha256"],
        "record_count": len(records),
    }


def _external_records(
    records: Sequence[Mapping[str, str]],
    *,
    required_corpora: Sequence[str],
) -> tuple[list[dict[str, str]], list[str]]:
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not 1 <= len(records) <= _MAX_COMBINED_RECORDS
        or not isinstance(required_corpora, Sequence)
        or isinstance(required_corpora, (str, bytes))
        or not 1 <= len(required_corpora) <= _MAX_EVALUATION_CORPORA
    ):
        raise _error("combined_sft_lineage_evaluation_inventory_invalid")
    required = list(required_corpora)
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip() for name in required
    ) or required != sorted(set(required)):
        raise _error("combined_sft_lineage_evaluation_inventory_invalid")
    normalized: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "corpus",
            "lineage_root_sha256",
            "objective",
            "answer",
        }:
            raise _error("combined_sft_lineage_evaluation_record_invalid")
        corpus = raw["corpus"]
        if (
            corpus not in required
            or not _is_sha(raw["lineage_root_sha256"])
            or not isinstance(raw["objective"], str)
            or not raw["objective"].strip()
            or not isinstance(raw["answer"], str)
            or not raw["answer"].strip()
        ):
            raise _error("combined_sft_lineage_evaluation_record_invalid")
        counts[corpus] += 1
        normalized.append(
            {
                "corpus": corpus,
                "split": "external_evaluation",
                "lineage_root_sha256": raw["lineage_root_sha256"],
                "objective": raw["objective"],
                "answer": raw["answer"],
            }
        )
    if set(counts) != set(required):
        raise _error("combined_sft_lineage_evaluation_coverage_incomplete")
    return normalized, required


@dataclass(frozen=True, slots=True)
class CombinedSFTLineageBundle:
    candidate_artifacts: dict[str, bytes]
    evaluator_artifacts: dict[str, bytes]
    custody_report: dict[str, Any]


def _validate_manifest_inventory(manifest: Mapping[str, Any]) -> None:
    index = manifest["combined_semantic_index"]
    records = index["records"]
    coverage = index["coverage"]
    required = manifest["required_evaluation_corpora"]
    declared_counts = manifest["record_counts"]
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= _MAX_EVALUATION_RECORDS
        or not isinstance(coverage, list)
        or coverage != sorted(set(coverage))
        or any(not isinstance(corpus, str) or not corpus for corpus in coverage)
        or not isinstance(required, list)
        or required != sorted(set(required))
        or not set(required) <= set(coverage)
        or not isinstance(declared_counts, list)
    ):
        raise _error("combined_sft_lineage_inventory_invalid")
    observed: Counter[tuple[str, str]] = Counter()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _SIGNATURE_FIELDS:
            raise _error("combined_sft_lineage_signature_record_invalid")
        token_shingles = record["token_shingles"]
        character_shingles = record["character_shingles"]
        if (
            any(
                not _is_sha(record[field])
                for field in (
                    "record_id_sha256",
                    "lineage_token",
                    "exact_token",
                    "objective_token",
                    "answer_token",
                )
            )
            or record["corpus"] not in coverage
            or record["split"] not in {*_SPLITS, "external_evaluation"}
            or type(record["objective_character_count"]) is not int
            or record["objective_character_count"] < 1
            or type(record["answer_character_count"]) is not int
            or record["answer_character_count"] < 1
            or not isinstance(token_shingles, list)
            or not isinstance(character_shingles, list)
            or token_shingles != sorted(set(token_shingles))
            or character_shingles != sorted(set(character_shingles))
            or any(not _is_sha(value) for value in token_shingles)
            or any(not _is_sha(value) for value in character_shingles)
        ):
            raise _error("combined_sft_lineage_signature_record_invalid")
        observed[(record["corpus"], record["split"])] += 1
    expected_counts = [
        {"corpus": corpus, "split": split, "count": count}
        for (corpus, split), count in sorted(observed.items())
    ]
    if (
        declared_counts != expected_counts
        or set(coverage) != {record["corpus"] for record in records}
        or any(observed.get((corpus, "external_evaluation"), 0) < 1 for corpus in required)
        or any(
            record["split"] != "external_evaluation"
            for record in records
            if record["corpus"] in required
        )
    ):
        raise _error("combined_sft_lineage_inventory_invalid")
    for binding_name, required_fields in (
        (
            "structured_binding",
            {
                "candidate_package_sha256",
                "evaluator_package_sha256",
                "custody_root_sha256",
                "holdout_seed_commitment_sha256",
                "record_count",
            },
        ),
        (
            "verified_replay_binding",
            {
                "candidate_package_sha256",
                "evaluator_package_sha256",
                "custody_root_sha256",
                "source_store_sha256",
                "partition_manifest_sha256",
                "source_semantic_dedup_manifest_sha256",
                "privacy_manifest_sha256",
                "record_count",
            },
        ),
    ):
        binding = manifest[binding_name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != required_fields
            or any(not _is_sha(value) for key, value in binding.items() if key != "record_count")
            or type(binding["record_count"]) is not int
            or binding["record_count"] < 1
        ):
            raise _error("combined_sft_lineage_source_binding_invalid")


def build_combined_sft_lineage_bundle(
    *,
    structured_candidate_artifacts: Mapping[str, bytes],
    structured_evaluator_artifacts: Mapping[str, bytes],
    replay_candidate_artifacts: Mapping[str, bytes],
    replay_evaluator_artifacts: Mapping[str, bytes],
    external_evaluation_records: Sequence[Mapping[str, str]],
    required_evaluation_corpora: Sequence[str],
    dedup_key: bytes,
) -> CombinedSFTLineageBundle:
    """Reconstruct and seal every pre-augmentation training/eval lineage."""

    structured, structured_binding = _structured_records(
        structured_candidate_artifacts,
        structured_evaluator_artifacts,
    )
    replay, replay_binding = _replay_records(
        replay_candidate_artifacts,
        replay_evaluator_artifacts,
    )
    external, required = _external_records(
        external_evaluation_records,
        required_corpora=required_evaluation_corpora,
    )
    records = [*structured, *replay, *external]
    coverage = sorted({record["corpus"] for record in records})
    try:
        signatures = build_semantic_signature_records(
            dedup_key=dedup_key,
            records=records,
            allow_same_lineage_same_split=True,
            allow_same_corpus_near_duplicates=True,
        )
    except VerifiedReplaySFTError as exc:
        raise CombinedSFTLineageError(f"combined_sft_lineage_semantic_overlap:{exc}") from exc
    index_body = {
        "schema": COMBINED_SFT_LINEAGE_INDEX_SCHEMA,
        "dedup_key_commitment_sha256": hashlib.sha256(bytes(dedup_key)).hexdigest(),
        "records": signatures,
        "record_count": len(signatures),
        "coverage": coverage,
        "same_lineage_same_split_derivatives_allowed": True,
        "same_corpus_template_near_duplicates_allowed": True,
        "cross_lineage_or_cross_split_overlap_allowed": False,
    }
    index = {**index_body, "index_sha256": _sha(index_body)}
    counts = Counter((record["corpus"], record["split"]) for record in records)
    manifest_body = {
        "schema": COMBINED_SFT_LINEAGE_MANIFEST_SCHEMA,
        "structured_binding": structured_binding,
        "verified_replay_binding": replay_binding,
        "required_evaluation_corpora": required,
        "coverage": coverage,
        "record_counts": [
            {"corpus": corpus, "split": split, "count": count}
            for (corpus, split), count in sorted(counts.items())
        ],
        "combined_semantic_index": index,
        "combined_semantic_index_sha256": index["index_sha256"],
        "dedup_key_commitment_sha256": index["dedup_key_commitment_sha256"],
        "augmentation_generation": 0,
        "future_derivatives_inherit_lineage_split": True,
        "exact_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
        "cross_split_lineage_overlap_count": 0,
        "status": "sealed_combined_pre_augmentation_quarantine",
        "trainer_ready": False,
        "training_authority": "none_pending_external_audit_and_trainer_admission",
    }
    manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    manifest_bytes = canonical_json_bytes(manifest)
    commitment_body = {
        "schema": COMBINED_SFT_LINEAGE_COMMITMENT_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "structured_candidate_package_sha256": structured_binding["candidate_package_sha256"],
        "verified_replay_candidate_package_sha256": replay_binding["candidate_package_sha256"],
        "combined_semantic_index_sha256": index["index_sha256"],
        "dedup_key_commitment_sha256": index["dedup_key_commitment_sha256"],
        "record_count": len(records),
        "evaluator_index_visible_to_trainer": False,
        "trainer_ready": False,
        "training_authority": "none_pending_external_audit_and_trainer_admission",
    }
    commitment = {**commitment_body, "commitment_sha256": _sha(commitment_body)}
    candidate = {COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]: canonical_json_bytes(commitment)}
    evaluator = {COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]: manifest_bytes}
    report = validate_combined_sft_lineage_custody(candidate, evaluator)
    return CombinedSFTLineageBundle(candidate, evaluator, report)


def validate_combined_sft_lineage_custody(
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate candidate noncontainment and evaluator manifest commitments."""

    if set(candidate_artifacts) != set(COMBINED_SFT_LINEAGE_CANDIDATE_FILES) or set(
        evaluator_artifacts
    ) != set(COMBINED_SFT_LINEAGE_EVALUATOR_FILES):
        raise _error("combined_sft_lineage_file_set_invalid")
    commitment = _json(
        candidate_artifacts[COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0]],
        code="combined_sft_lineage_commitment_invalid",
    )
    manifest = _json(
        evaluator_artifacts[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]],
        code="combined_sft_lineage_manifest_invalid",
    )
    commitment_fields = {
        "schema",
        "manifest_sha256",
        "structured_candidate_package_sha256",
        "verified_replay_candidate_package_sha256",
        "combined_semantic_index_sha256",
        "dedup_key_commitment_sha256",
        "record_count",
        "evaluator_index_visible_to_trainer",
        "trainer_ready",
        "training_authority",
        "commitment_sha256",
    }
    manifest_fields = {
        "schema",
        "structured_binding",
        "verified_replay_binding",
        "required_evaluation_corpora",
        "coverage",
        "record_counts",
        "combined_semantic_index",
        "combined_semantic_index_sha256",
        "dedup_key_commitment_sha256",
        "augmentation_generation",
        "future_derivatives_inherit_lineage_split",
        "exact_overlap_count",
        "near_duplicate_overlap_count",
        "cross_split_lineage_overlap_count",
        "status",
        "trainer_ready",
        "training_authority",
        "manifest_sha256",
    }
    if set(commitment) != commitment_fields or set(manifest) != manifest_fields:
        raise _error("combined_sft_lineage_schema_invalid")
    commitment_body = dict(commitment)
    commitment_sha = commitment_body.pop("commitment_sha256", None)
    manifest_body = dict(manifest)
    manifest_sha = manifest_body.pop("manifest_sha256", None)
    index = manifest.get("combined_semantic_index")
    key_commitment = manifest.get("dedup_key_commitment_sha256")
    index_fields = {
        "schema",
        "dedup_key_commitment_sha256",
        "records",
        "record_count",
        "coverage",
        "same_lineage_same_split_derivatives_allowed",
        "same_corpus_template_near_duplicates_allowed",
        "cross_lineage_or_cross_split_overlap_allowed",
        "index_sha256",
    }
    if (
        commitment.get("schema") != COMBINED_SFT_LINEAGE_COMMITMENT_SCHEMA
        or manifest.get("schema") != COMBINED_SFT_LINEAGE_MANIFEST_SCHEMA
        or not _is_sha(commitment_sha)
        or _sha(commitment_body) != commitment_sha
        or not _is_sha(manifest_sha)
        or _sha(manifest_body) != manifest_sha
        or commitment.get("manifest_sha256") != manifest_sha
        or not isinstance(index, Mapping)
        or set(index) != index_fields
        or index.get("schema") != COMBINED_SFT_LINEAGE_INDEX_SCHEMA
        or _sha({key: value for key, value in index.items() if key != "index_sha256"})
        != index.get("index_sha256")
        or index.get("index_sha256") != manifest.get("combined_semantic_index_sha256")
        or index.get("dedup_key_commitment_sha256") != key_commitment
        or commitment.get("combined_semantic_index_sha256") != index.get("index_sha256")
        or commitment.get("dedup_key_commitment_sha256") != key_commitment
        or commitment.get("record_count") != index.get("record_count")
        or not isinstance(index.get("records"), list)
        or index.get("record_count") != len(index.get("records", []))
        or index.get("coverage") != manifest.get("coverage")
        or index.get("same_lineage_same_split_derivatives_allowed") is not True
        or index.get("same_corpus_template_near_duplicates_allowed") is not True
        or index.get("cross_lineage_or_cross_split_overlap_allowed") is not False
        or commitment.get("structured_candidate_package_sha256")
        != manifest.get("structured_binding", {}).get("candidate_package_sha256")
        or commitment.get("verified_replay_candidate_package_sha256")
        != manifest.get("verified_replay_binding", {}).get("candidate_package_sha256")
        or commitment.get("evaluator_index_visible_to_trainer") is not False
        or manifest.get("augmentation_generation") != 0
        or manifest.get("future_derivatives_inherit_lineage_split") is not True
        or any(
            manifest.get(field) != 0
            for field in (
                "exact_overlap_count",
                "near_duplicate_overlap_count",
                "cross_split_lineage_overlap_count",
            )
        )
        or manifest.get("status") != "sealed_combined_pre_augmentation_quarantine"
        or commitment.get("trainer_ready") is not False
        or manifest.get("trainer_ready") is not False
        or commitment.get("training_authority")
        != "none_pending_external_audit_and_trainer_admission"
        or manifest.get("training_authority") != commitment.get("training_authority")
    ):
        raise _error("combined_sft_lineage_binding_invalid")
    _validate_manifest_inventory(manifest)
    try:
        assert_semantic_signature_integrity(
            index["records"],
            allow_same_lineage_same_split=True,
            allow_same_corpus_near_duplicates=True,
        )
    except (KeyError, TypeError, ValueError, VerifiedReplaySFTError) as exc:
        raise CombinedSFTLineageError("combined_sft_lineage_signature_integrity_invalid") from exc
    candidate_bytes = b"".join(candidate_artifacts.values())
    if index.get("records") and any(
        record["record_id_sha256"].encode("ascii") in candidate_bytes for record in index["records"]
    ):
        raise _error("combined_sft_lineage_evaluator_index_leaked")
    return {
        "schema": COMBINED_SFT_LINEAGE_CUSTODY_SCHEMA,
        "manifest_sha256": manifest_sha,
        "commitment_sha256": commitment_sha,
        "combined_semantic_index_sha256": index["index_sha256"],
        "record_count": index["record_count"],
        "trainer_ready": False,
        "training_authority": commitment["training_authority"],
        "status": "passed_combined_lineage_custody_no_training_authority",
    }


def revalidate_combined_sft_lineage_bundle(
    *,
    bundle: CombinedSFTLineageBundle,
    structured_candidate_artifacts: Mapping[str, bytes],
    structured_evaluator_artifacts: Mapping[str, bytes],
    replay_candidate_artifacts: Mapping[str, bytes],
    replay_evaluator_artifacts: Mapping[str, bytes],
    external_evaluation_records: Sequence[Mapping[str, str]],
    required_evaluation_corpora: Sequence[str],
    dedup_key: bytes,
) -> dict[str, Any]:
    """Reconstruct the entire bundle and require byte-identical evidence."""

    rebuilt = build_combined_sft_lineage_bundle(
        structured_candidate_artifacts=structured_candidate_artifacts,
        structured_evaluator_artifacts=structured_evaluator_artifacts,
        replay_candidate_artifacts=replay_candidate_artifacts,
        replay_evaluator_artifacts=replay_evaluator_artifacts,
        external_evaluation_records=external_evaluation_records,
        required_evaluation_corpora=required_evaluation_corpora,
        dedup_key=dedup_key,
    )
    if (
        bundle.candidate_artifacts != rebuilt.candidate_artifacts
        or bundle.evaluator_artifacts != rebuilt.evaluator_artifacts
    ):
        raise _error("combined_sft_lineage_reconstruction_mismatch")
    return validate_combined_sft_lineage_custody(
        bundle.candidate_artifacts,
        bundle.evaluator_artifacts,
    )


__all__ = [
    "COMBINED_SFT_LINEAGE_CANDIDATE_FILES",
    "COMBINED_SFT_LINEAGE_COMMITMENT_SCHEMA",
    "COMBINED_SFT_LINEAGE_CUSTODY_SCHEMA",
    "COMBINED_SFT_LINEAGE_EVALUATOR_FILES",
    "COMBINED_SFT_LINEAGE_INDEX_SCHEMA",
    "COMBINED_SFT_LINEAGE_MANIFEST_SCHEMA",
    "CombinedSFTLineageBundle",
    "CombinedSFTLineageError",
    "build_combined_sft_lineage_bundle",
    "revalidate_combined_sft_lineage_bundle",
    "validate_combined_sft_lineage_custody",
]
