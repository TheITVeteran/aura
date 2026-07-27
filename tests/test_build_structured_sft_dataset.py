from __future__ import annotations

import json
import os
import shutil
import stat

import pytest

from core.learning.structured_sft import (
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
    StructuredSFTCurriculumSpec,
    StructuredSFTError,
)
from core.runtime.file_write_gateway import FileWriteTransactionError
from tools import build_structured_sft_dataset as builder

HOLDOUT_SEED = bytes(range(32))


@pytest.fixture(scope="module")
def built_custody(tmp_path_factory):
    root = tmp_path_factory.mktemp("structured-sft")
    candidate = root / "candidate"
    evaluator = root / "evaluator"
    report = builder.build_custodied_dataset_directories(
        candidate_directory=candidate,
        evaluator_directory=evaluator,
        spec=StructuredSFTCurriculumSpec(
            seed=1618,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    return candidate, evaluator, report


def test_builder_writes_separate_private_replayable_bundles(built_custody):
    candidate, evaluator, report = built_custody

    assert report["status"] == "custody_bundles_built_and_replay_validated"
    assert report["trainer_ready"] is False
    assert report["training_authority"].startswith("none_pending_")
    assert {
        path.name for path in candidate.iterdir()
    } == {*STRUCTURED_SFT_CANDIDATE_FILES, ".aura_file_write_batch.lock"}
    assert {
        path.name for path in evaluator.iterdir()
    } == {*STRUCTURED_SFT_EVALUATOR_FILES, ".aura_file_write_batch.lock"}
    for directory, names in (
        (candidate, STRUCTURED_SFT_CANDIDATE_FILES),
        (evaluator, STRUCTURED_SFT_EVALUATOR_FILES),
    ):
        for name in names:
            assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    candidate_manifest = builder.validate_candidate_dataset_directory(candidate)
    custody = builder.validate_custody_directories(
        candidate_directory=candidate,
        evaluator_directory=evaluator,
    )
    assert (
        candidate_manifest["package_sha256"]
        == report["candidate_package_sha256"]
    )
    assert custody["custody_root_sha256"] == report["custody_root_sha256"]
    assert not (candidate / "holdout.private.json").exists()
    assert not (candidate / "curriculum.private.json").exists()


def test_validate_only_cli_reports_custody(capsys, built_custody):
    candidate, evaluator, report = built_custody

    assert (
        builder.main(
            [
                "--candidate-dir",
                str(candidate),
                "--evaluator-dir",
                str(evaluator),
                "--validate-only",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "custody_bundles_replay_validated"
    assert (
        result["candidate_package_sha256"]
        == report["candidate_package_sha256"]
    )
    assert (
        result["evaluator_package_sha256"]
        == report["evaluator_package_sha256"]
    )


def test_builder_refuses_symlinked_output_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)

    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="symlink_output_path_rejected",
    ):
        builder.validate_candidate_dataset_directory(link)


@pytest.mark.parametrize("layout", ("same", "candidate_parent", "evaluator_parent"))
def test_builder_refuses_overlapping_custody_directories(tmp_path, layout):
    if layout == "same":
        candidate = evaluator = tmp_path / "shared"
    elif layout == "candidate_parent":
        candidate = tmp_path / "candidate"
        evaluator = candidate / "evaluator"
    else:
        evaluator = tmp_path / "evaluator"
        candidate = evaluator / "candidate"

    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="custody_directories_must_be_distinct_non_nested_siblings",
    ):
        builder.build_custodied_dataset_directories(
            candidate_directory=candidate,
            evaluator_directory=evaluator,
            spec=StructuredSFTCurriculumSpec(seed=42),
            holdout_seed=HOLDOUT_SEED,
        )


def test_builder_refuses_mixed_purpose_directory(tmp_path):
    candidate = tmp_path / "candidate"
    evaluator = tmp_path / "evaluator"
    candidate.mkdir(mode=0o700)
    (candidate / "unrelated.txt").write_text("do not overwrite")

    with pytest.raises(
        FileWriteTransactionError,
        match="unexpected entries",
    ):
        builder.build_custodied_dataset_directories(
            candidate_directory=candidate,
            evaluator_directory=evaluator,
            spec=StructuredSFTCurriculumSpec(
                seed=42,
                train_cases_per_family=1,
                validation_cases_per_family=1,
                holdout_cases_per_family=1,
            ),
            holdout_seed=HOLDOUT_SEED,
        )


def test_directory_validation_refuses_artifact_tampering(
    built_custody,
    tmp_path,
):
    candidate, _evaluator, _report = built_custody
    tampered_root = tmp_path / "tampered-root"
    shutil.copytree(candidate.parent, tampered_root)
    tampered = tampered_root / candidate.name
    target = tampered / "candidate_train.jsonl"
    target.write_bytes(target.read_bytes() + b'{"messages":[]}\n')

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_candidate_replay_mismatch",
    ):
        builder.validate_candidate_dataset_directory(tampered)


def test_candidate_only_read_never_opens_evaluator_holdout(
    built_custody,
    tmp_path,
):
    candidate, evaluator, report = built_custody
    isolated_root = tmp_path / "candidate-only-root"
    shutil.copytree(candidate.parent, isolated_root)
    shutil.rmtree(isolated_root / evaluator.name)

    artifacts, attestation = (
        builder.read_candidate_dataset_directory_with_attestation(
            isolated_root / candidate.name
        )
    )

    assert set(artifacts) == set(STRUCTURED_SFT_CANDIDATE_FILES)
    assert attestation["candidate_package_sha256"] == report[
        "candidate_package_sha256"
    ]


@pytest.mark.parametrize(
    "escaped_name",
    ("../outside", "/private/tmp/outside", "nested/evaluator"),
)
def test_custody_commit_rejects_non_flat_directory_names(
    built_custody,
    tmp_path,
    escaped_name,
):
    candidate, _evaluator, _report = built_custody
    copied_root = tmp_path / "escaped-root"
    shutil.copytree(candidate.parent, copied_root)
    commit_path = copied_root / builder._CUSTODY_COMMIT_FILE
    record = json.loads(commit_path.read_text())
    body = dict(record)
    body.pop("commit_sha256")
    body["evaluator_directory"] = escaped_name
    commit_path.write_bytes(
        builder._canonical_json_bytes(builder._committed_custody_record(body))
    )

    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="custody_commit_schema_invalid",
    ):
        builder.read_candidate_dataset_directory(copied_root / candidate.name)


def test_builder_requires_sibling_custody_roots(tmp_path):
    candidate_root = tmp_path / "candidate-root"
    evaluator_root = tmp_path / "evaluator-root"
    candidate_root.mkdir(mode=0o700)
    evaluator_root.mkdir(mode=0o700)

    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="custody_directories_must_be_distinct_non_nested_siblings",
    ):
        builder.build_custodied_dataset_directories(
            candidate_directory=candidate_root / "candidate",
            evaluator_directory=evaluator_root / "evaluator",
            spec=StructuredSFTCurriculumSpec(seed=43),
            holdout_seed=HOLDOUT_SEED,
        )


def test_pair_publication_stays_fail_closed_after_second_side_failure(
    monkeypatch,
    tmp_path,
):
    candidate = tmp_path / "candidate"
    evaluator = tmp_path / "evaluator"
    spec = StructuredSFTCurriculumSpec(
        seed=44,
        train_cases_per_family=1,
        validation_cases_per_family=1,
        holdout_cases_per_family=1,
    )
    builder.build_custodied_dataset_directories(
        candidate_directory=candidate,
        evaluator_directory=evaluator,
        spec=spec,
        holdout_seed=HOLDOUT_SEED,
    )
    real_publish = builder.FileWriteTransactionError
    gateway = builder.get_file_write_gateway()
    original = gateway.write_bytes_batch_in_directory

    def fail_candidate(*args, source="unknown", **kwargs):
        if source == "structured_sft.candidate_dataset":
            raise real_publish("injected candidate publication failure")
        return original(*args, source=source, **kwargs)

    monkeypatch.setattr(
        gateway,
        "write_bytes_batch_in_directory",
        fail_candidate,
    )
    with pytest.raises(FileWriteTransactionError):
        builder.build_custodied_dataset_directories(
            candidate_directory=candidate,
            evaluator_directory=evaluator,
            spec=spec,
            holdout_seed=b"x" * 32,
        )
    commit = builder._read_custody_record(
        tmp_path / builder._CUSTODY_COMMIT_FILE
    )
    assert commit["state"] == "preparing"
    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="candidate_custody_generation_not_committed",
    ):
        builder.read_candidate_dataset_directory(candidate)


def test_holdout_seed_file_requires_owner_only_single_link(tmp_path):
    seed = tmp_path / "holdout.seed"
    seed.write_bytes(HOLDOUT_SEED)
    seed.chmod(0o644)
    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="owner_only_single_link",
    ):
        builder._read_private_holdout_seed(seed)

    seed.chmod(0o600)
    assert builder._read_private_holdout_seed(seed) == HOLDOUT_SEED
    os.link(seed, tmp_path / "holdout.seed.link")
    with pytest.raises(
        builder.CandidateDatasetBuildError,
        match="owner_only_single_link",
    ):
        builder._read_private_holdout_seed(seed)
