from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx_lm.utils
import pytest

from core.brain.llm import unified_recurrent_shadow as runtime_shadow
from tools import evaluate_unified_intrinsic_checkpoint as checkpoint_evaluator
from tools import materialize_unified_intrinsic_shadow_package as materializer
from tools import unified_intrinsic_tokenization_contract as tokenization_contract
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
    canary_battery = materializer.seal_shadow_canary_battery(
        [
            {
                "task_id": "fresh-khop-1",
                "family": "khop",
                "task_depth": 2,
                "prompt_sha256": "8" * 64,
                "expected_sha256": "9" * 64,
                "public_token_ids": [1, 201, 2],
                "expected_token_ids": [12, 999],
                "max_tokens": 2,
            }
        ],
        seed=301,
        replication_plan_sha256=plan["plan_sha256"],
        replication_verdict_sha256=verdict["verdict_sha256"],
        excluded_task_ids_sha256="a" * 64,
        excluded_prompt_sha256s_sha256="b" * 64,
        generator_source_sha256s={"core/learning/example.py": "c" * 64},
    )
    monkeypatch.setattr(
        materializer,
        "_verified_evidence",
        lambda _campaign, **_kwargs: (config, completion, plan, verdict, reports),
    )
    monkeypatch.setattr(
        materializer,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        materializer,
        "_fresh_canary_battery",
        lambda *_args, **_kwargs: canary_battery,
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
    assert manifest["canary_battery_sha256"] == runtime_inspection[
        "canary_battery"
    ]["battery_sha256"]
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
        "shadow-canary-battery.json",
    }


def test_materialize_forwards_explicit_replication_evidence_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, output_root = _fixture(tmp_path, monkeypatch)
    replication_root = campaign / "resident-replication-corrected"
    replication_root.mkdir(mode=0o700)
    original = materializer._verified_evidence
    observed: list[Path | None] = []

    def verified(campaign_path: Path, *, replication_root: Path | None = None):
        observed.append(replication_root)
        return original(campaign_path, replication_root=replication_root)

    monkeypatch.setattr(materializer, "_verified_evidence", verified)

    result = materializer.materialize(
        campaign,
        output_root=output_root,
        package_id="cp266-explicit-evidence",
        replication_root=replication_root,
    )

    assert Path(result["package"]).is_dir()
    assert observed == [replication_root]


def test_replication_evidence_root_cannot_escape_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    monkeypatch.setattr(
        materializer.launcher,
        "_terminal_campaign",
        lambda *_args, **_kwargs: ({}, {}),
    )

    with pytest.raises(
        materializer.UnifiedIntrinsicShadowPackageError,
        match="strict campaign child",
    ):
        materializer._verified_evidence(  # noqa: SLF001
            campaign,
            replication_root=outside,
        )


def test_fresh_canary_generator_tokenizes_prompt_disjoint_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("fixture")
    training = [SimpleNamespace(task_id="train-1", prompt="training prompt")]
    holdout = [SimpleNamespace(task_id="holdout-1", prompt="holdout prompt")]

    class Tokenizer:
        eos_token_id = 999

        def apply_chat_template(self, messages, **_kwargs):
            return f"USER:{messages[0]['content']}\nASSISTANT:"

        def encode(self, text, **_kwargs):
            return [1, *text.encode("ascii"), 2]

    monkeypatch.setattr(
        tokenization_contract,
        "load_source_dataset",
        lambda _path: (training, holdout),
    )
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda _path: Tokenizer())

    def fresh(_identity, *, per_cell, seed, task_depth):
        assert per_cell == 1
        return [
            SimpleNamespace(
                task_id=f"fresh-{task_depth}-{seed}",
                family="khop",
                depth=task_depth,
                prompt=f"fresh prompt {task_depth} {seed}",
                answer='FINAL_ANSWER: {"value":1}',
            )
        ]

    monkeypatch.setattr(checkpoint_evaluator, "_fresh_tasks", fresh)
    plan = {"plan_sha256": "1" * 64}
    verdict = {"verdict_sha256": "2" * 64}
    reports = [
        {
            "candidates": [
                {
                    "task_id": "replication-1",
                    "prompt_sha256": "3" * 64,
                }
            ]
        }
    ]

    battery = materializer._fresh_canary_battery(
        {"paths": {"dataset": str(dataset)}},
        {
            "model": {"canonical_path": str(tmp_path)},
            "task_depths": [1, 2, 4],
            "bridge": "assistant_answer",
        },
        plan,
        verdict,
        reports,
    )

    assert battery["task_count"] == 3
    assert {row["task_depth"] for row in battery["cases"]} == {1, 2, 4}
    assert all(row["expected_token_ids"][-1] == 999 for row in battery["cases"])
    assert battery["replication_plan_sha256"] == plan["plan_sha256"]
    assert battery["replication_verdict_sha256"] == verdict["verdict_sha256"]


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

    def unsupported(path: Path, **kwargs):
        config, completion, plan, verdict, reports = original(path, **kwargs)
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
