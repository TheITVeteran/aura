from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import run_resident_v3_post_training_pipeline as pipeline


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    import hashlib

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


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


def test_recovery_config_binds_destination_and_all_terminal_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "resident_32b_v3_cp195"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    protocol = tmp_path / "protocol.json"
    amendment = tmp_path / "amendment.json"
    protocol.write_bytes(
        canonical_json_bytes(
            {
                "detached_execution": {"partial_run_dir": str(tmp_path / "partial")},
                "model": {"path": str(tmp_path / "model")},
                "training": {"output_dir": str(tmp_path / "old-adapter")},
            }
        )
        + b"\n"
    )
    amendment.write_bytes(
        canonical_json_bytes({"resume": {"run_dir": str(tmp_path / "old-resume")}})
        + b"\n"
    )
    migration = root / "migration.json"
    migration.write_bytes(
        canonical_json_bytes(
            {
                "protocol": _binding(protocol),
                "amendment": _binding(amendment),
                "destination": {"root": str(adapter)},
            }
        )
        + b"\n"
    )
    (root / "calibration_verdict.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        pipeline.recovery_admission.recovery,
        "_migration",
        lambda *_args, **_kwargs: ({}, {"migration_sha256": "a" * 64}),
    )

    config = pipeline.build_recovery_config(
        migration_path=migration,
        output_root=tmp_path / "post-training",
        training_source_root=pipeline.REPO_ROOT,
        source_commit="1" * 40,
        seeds=[2**62 + 101, 2**62 + 103],
    )

    assert config["recovery"]["adapter_root"] == str(adapter)
    assert config["training_runs"] == {
        "resume": str(root / "detached-resume"),
        "resume_sentinel": str(root / "sentinel-resume"),
        "recovery_controller": str(root / "detached-recovery-controller"),
        "sentinel_archive": str(root / "detached-sentinel-proof-archive"),
    }
    assert config["mechanics"]["campaign_name"].startswith("cp195-")
    material = dict(config)
    claimed = material.pop("config_sha256")
    assert claimed == pipeline._document_sha(material)


def test_recovery_mode_derives_destination_protocol_and_dispatches_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    root = tmp_path / "resident_32b_v3_cp195"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    migration = root / "migration.json"
    migration.write_text("{}", encoding="ascii")
    calibration = root / "calibration.json"
    calibration.write_text("{}", encoding="ascii")
    config["recovery"] = {
        "mode": "migration_recovery",
        "migration": _binding(migration),
        "calibration_verdict": _binding(calibration),
        "controller_verdict_path": str(root / "controller.json"),
        "operational_ring_path": str(root / "operational.jsonl"),
        "archive_run_dir": str(root / "archive-run"),
        "archive_ring_path": str(root / "archive.jsonl"),
        "archive_receipt_path": str(root / "archive-receipt.json"),
        "adapter_root": str(adapter),
    }
    config["training_runs"] = {
        "resume": str(root / "resume"),
        "resume_sentinel": str(root / "sentinel"),
        "recovery_controller": str(root / "controller"),
        "sentinel_archive": str(root / "archive-controller"),
    }
    run = pipeline.PipelineRun(config)
    monkeypatch.setattr(
        pipeline.recovery_admission,
        "_protocol",
        lambda *_args: {
            "model": {"path": str(tmp_path / "resident-model")},
            "training": {
                "output_dir": str(adapter),
                "adapter_id": "resident-32b-recurrence-v3-cp195",
            },
        },
    )
    observed: dict[str, object] = {}

    def verify(args):
        observed.update(vars(args))
        return {"claim_flags": {"adapter_freeze_eligible": True}}

    monkeypatch.setattr(pipeline.recovery_admission, "verify", verify)
    recovery = run.recovery_config()
    effective = run.effective_protocol(
        {
            "model": {"path": "old-model"},
            "training": {"output_dir": "old-adapter", "adapter_id": "old"},
        },
        recovery,
    )
    admitted = run.verify_training_admission(
        protocol_path=tmp_path / "protocol.json",
        amendment_path=tmp_path / "amendment.json",
        amendment={},
        recovery=recovery,
        output=tmp_path / "admission.json",
    )

    assert effective["training"]["output_dir"] == str(adapter)
    assert effective["model"]["path"] == str(tmp_path / "resident-model")
    assert set(run.training_run_dirs(recovery)) == {
        "resume",
        "resume_sentinel",
        "recovery_controller",
        "sentinel_archive",
    }
    assert observed["migration"] == migration
    assert admitted["claim_flags"]["adapter_freeze_eligible"] is True


def test_launcher_holds_independent_sleep_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "pipeline-config.json"
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    launched: dict[str, object] = {}

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    def popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)

    receipt = pipeline.launch_pipeline(config_path, pipeline.REPO_ROOT)

    assert launched["command"][:2] == ["/usr/bin/caffeinate", "-i"]
    assert launched["kwargs"]["start_new_session"] is True
    assert receipt["controller_pid"] == 123
