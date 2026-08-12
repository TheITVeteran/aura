from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm import unified_recurrent_shadow as runtime_shadow
from tools import materialize_unified_intrinsic_shadow_package as materializer
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    output_root = tmp_path / "packages"
    training_output = campaign / "training-output"
    training_output.mkdir(parents=True, mode=0o700)
    campaign.chmod(0o700)
    weights = campaign / "controller-source.safetensors"
    weights.write_bytes(b"controller-weights")
    weights.chmod(0o400)

    checkpoint_sha256 = __import__("hashlib").sha256(weights.read_bytes()).hexdigest()
    identity = {
        "identity_sha256": "1" * 64,
        "families": ["khop", "modular", "register_trace"],
        "task_depths": [1, 2, 4],
        "tokenizer": {"identity_sha256": "2" * 64},
        "answer_emission_contract": {"contract_sha256": "3" * 64},
    }
    receipt_body = {
        "schema": "aura.unified_intrinsic_training.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "identity": identity,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    checkpoint = SimpleNamespace(receipt=receipt, weights_path=weights)
    config = {
        "campaign_id": "fixture-campaign",
        "config_sha256": "4" * 64,
        "paths": {"training_output": str(training_output)},
        "source": {"git": {"commit": "5" * 40}},
        "model": {"manifest_sha256": "6" * 64},
    }
    completion = {
        "checkpoint": {"checkpoint_sha256": checkpoint_sha256},
        "completion_sha256": "7" * 64,
    }
    plan_body = {
        "recurrence_depths": [4],
        "seeds": [101, 102, 103],
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    reports = []
    for seed in plan["seeds"]:
        report_body = {"schema": "report", "evaluation_seed": seed}
        reports.append(
            {**report_body, "report_sha256": canonical_sha256(report_body)}
        )
    verdict_body = {
        "supported": True,
        "verdict": "SUPPORTED",
        "checkpoint_sha256": checkpoint_sha256,
        "plan_sha256": plan["plan_sha256"],
        "reports": [
            {"seed": report["evaluation_seed"], "report_sha256": report["report_sha256"]}
            for report in reports
        ],
    }
    verdict = {**verdict_body, "verdict_sha256": canonical_sha256(verdict_body)}
    monkeypatch.setattr(
        materializer,
        "_verified_evidence",
        lambda _campaign: (config, completion, plan, verdict, reports),
    )
    monkeypatch.setattr(
        materializer,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: checkpoint,
    )
    return campaign, output_root


def test_materializes_and_reopens_shadow_only_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)

    result = materializer.materialize(
        campaign,
        output_root=output_root,
        package_id="cp257-fixture",
    )
    package = Path(result["package"])
    reopened = materializer.inspect_shadow_package(package)
    runtime_inspection = runtime_shadow.inspect_shadow_package(package)
    manifest = json.loads((package / "manifest.json").read_bytes())

    assert reopened == result
    assert runtime_inspection["manifest"] == manifest
    assert manifest["mode"] == "shadow_only"
    assert manifest["serving_authority"] is False
    assert manifest["domain_contract"]["ordinary_chat_authorized"] is False
    assert manifest["domain_contract"]["arbitrary_reasoning_authorized"] is False
    assert "global_activation" in manifest["claims_not_supported"]
    assert set(path.name for path in package.iterdir()) == {
        "PACKAGE_COMPLETE.json",
        "campaign-completion.json",
        "checkpoint-complete.json",
        "controller.safetensors",
        "manifest.json",
        "replication-plan.json",
        "replication-report-01.json",
        "replication-report-02.json",
        "replication-report-03.json",
        "replication-verdict.json",
    }


def test_refuses_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    output_root.mkdir(mode=0o700)
    (output_root / "collision").mkdir(mode=0o700)

    with pytest.raises(
        materializer.UnifiedIntrinsicShadowPackageError,
        match="destination already exists",
    ):
        materializer.materialize(
            campaign,
            output_root=output_root,
            package_id="collision",
        )


def test_reopen_refuses_tampered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    result = materializer.materialize(
        campaign,
        output_root=output_root,
        package_id="tamper-fixture",
    )
    controller = Path(result["package"]) / "controller.safetensors"
    controller.chmod(0o600)
    controller.write_bytes(b"tampered-weights")
    controller.chmod(0o400)

    with pytest.raises(
        materializer.UnifiedIntrinsicShadowPackageError,
        match="artifact binding differs",
    ):
        materializer.inspect_shadow_package(Path(result["package"]))


def test_refuses_unsupported_verdict_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    original = materializer._verified_evidence

    def unsupported(path: Path):
        config, completion, plan, verdict, reports = original(path)
        return config, completion, plan, {**verdict, "supported": False}, reports

    monkeypatch.setattr(materializer, "_verified_evidence", unsupported)
    with pytest.raises(
        materializer.UnifiedIntrinsicShadowPackageError,
        match="supported replication verdict",
    ):
        materializer.materialize(
            campaign,
            output_root=output_root,
            package_id="unsupported-fixture",
        )
    assert not (output_root / "unsupported-fixture").exists()


def test_documents_are_canonical_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    result = materializer.materialize(
        campaign,
        output_root=output_root,
        package_id="canonical-fixture",
    )

    for path in Path(result["package"]).iterdir():
        assert path.stat().st_mode & 0o777 == 0o400
        if path.suffix == ".json":
            value = json.loads(path.read_bytes())
            assert path.read_bytes() == canonical_bytes(value) + b"\n"


def test_runtime_refuses_report_rebound_away_from_frozen_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    result = materializer.materialize(
        campaign,
        output_root=output_root,
        package_id="report-rebind-fixture",
    )
    package = Path(result["package"])
    report_path = package / "replication-report-01.json"
    report = json.loads(report_path.read_bytes())
    report["evaluation_seed"] = 999
    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_sha256(report_body)
    report_path.chmod(0o600)
    report_path.write_bytes(canonical_bytes(report) + b"\n")
    report_path.chmod(0o400)

    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    binding = manifest["artifacts"]["replication_reports"][0]
    binding["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    binding["size_bytes"] = report_path.stat().st_size
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest_body)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    manifest_path.chmod(0o400)

    complete_path = package / "PACKAGE_COMPLETE.json"
    complete = json.loads(complete_path.read_bytes())
    complete["manifest_sha256"] = manifest["manifest_sha256"]
    complete["manifest_file_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    complete_body = {
        key: value for key, value in complete.items() if key != "complete_sha256"
    }
    complete["complete_sha256"] = canonical_sha256(complete_body)
    complete_path.chmod(0o600)
    complete_path.write_bytes(canonical_bytes(complete) + b"\n")
    complete_path.chmod(0o400)

    with pytest.raises(
        runtime_shadow.UnifiedRecurrentShadowError,
        match="replication report commitments differ",
    ):
        runtime_shadow.inspect_shadow_package(package)
