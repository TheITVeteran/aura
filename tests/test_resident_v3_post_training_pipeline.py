from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import run_resident_v3_post_training_pipeline as pipeline


def _config(tmp_path: Path) -> dict[str, object]:
    partial = tmp_path / "cp189/detached-partial"
    resume = tmp_path / "cp189/detached-resume"
    protocol = tmp_path / "protocol.json"
    amendment = tmp_path / "amendment.json"
    protocol.write_bytes(
        canonical_json_bytes(
            {
                "detached_execution": {"partial_run_dir": str(partial)},
                "training": {"output_dir": str(tmp_path / "adapter")},
            }
        )
        + b"\n"
    )
    amendment.write_bytes(
        canonical_json_bytes({"resume": {"run_dir": str(resume)}}) + b"\n"
    )
    return pipeline.build_config(
        protocol_path=protocol,
        amendment_path=amendment,
        output_root=tmp_path / "cp190",
        training_source_root=pipeline.REPO_ROOT,
        source_commit="1" * 40,
        seeds=[2**62 + 17, 2**62 + 19],
    )


def test_config_freezes_sources_paths_and_63_bit_seeds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    material = dict(config)
    claimed = material.pop("config_sha256")

    assert claimed == pipeline._document_sha(material)
    assert config["pilot"]["seeds"] == [2**62 + 17, 2**62 + 19]
    assert config["training_runs"]["partial_sentinel"].endswith("sentinel-partial")
    assert config["training_runs"]["resume_sentinel"].endswith("sentinel-resume")
    assert config["claim_policy"]["physical_weight_merge_allowed"] is False
    assert config["source_bindings"]["pipeline"]["sha256"]


def test_event_journal_restores_exact_hash_chain(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = pipeline.PipelineRun(config)
    first.stage = "one"
    first.event("started")
    first.event("completed", {"receipt": "a" * 64})

    resumed = pipeline.PipelineRun(config)
    assert resumed.event_sequence == 2
    assert resumed.event_head == first.event_head
    resumed.stage = "two"
    resumed.event("started")

    events = [json.loads(line) for line in resumed.journal_path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert events[2]["previous_event_sha256"] == events[1]["event_sha256"]


def test_event_journal_rejects_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = pipeline.PipelineRun(config)
    run.stage = "one"
    run.event("started")
    event = json.loads(run.journal_path.read_text())
    event["stage"] = "changed"
    run.journal_path.write_bytes(canonical_json_bytes(event) + b"\n")

    with pytest.raises(
        pipeline.ResidentV3PostTrainingPipelineError,
        match="pipeline_journal_invalid",
    ):
        pipeline.PipelineRun(config)


@pytest.mark.parametrize("seeds", [[7, 11], [2**62 + 1, 2**62 + 1]])
def test_config_rejects_weak_or_duplicate_seeds(
    tmp_path: Path, seeds: list[int]
) -> None:
    partial = tmp_path / "partial"
    protocol = tmp_path / "protocol.json"
    amendment = tmp_path / "amendment.json"
    protocol.write_bytes(
        canonical_json_bytes(
            {"detached_execution": {"partial_run_dir": str(partial)}}
        )
        + b"\n"
    )
    amendment.write_bytes(
        canonical_json_bytes({"resume": {"run_dir": str(tmp_path / "resume")}})
        + b"\n"
    )

    with pytest.raises(
        pipeline.ResidentV3PostTrainingPipelineError,
        match="config_seed_contract_invalid",
    ):
        pipeline.build_config(
            protocol_path=protocol,
            amendment_path=amendment,
            output_root=tmp_path / "out",
            training_source_root=pipeline.REPO_ROOT,
            source_commit="1" * 40,
            seeds=seeds,
        )
