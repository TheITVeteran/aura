from __future__ import annotations

import argparse
import json
import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.unified_recurrent_shadow_battery import (
    seal_shadow_canary_battery,
)
from tools import run_unified_recurrent_promotion_pipeline as pipeline
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256


def _config(tmp_path: Path) -> dict:
    replication = tmp_path / "replication"
    replication.mkdir()
    root = tmp_path / "promotion"
    root.mkdir()
    package_root = tmp_path / "authority" / "releases"
    return {
        "config_sha256": "c" * 64,
        "campaign": str(tmp_path),
        "campaign_id": "campaign",
        "campaign_config_sha256": "d" * 64,
        "campaign_source_commit": "s" * 40,
        "replication_root": str(replication),
        "replication_plan_sha256": "p" * 64,
        "pipeline_root": str(root),
        "package_id": "candidate",
        "package_root": str(package_root),
        "package": str(package_root / "candidate"),
        "lifecycle_output": str(root / "lifecycle"),
        "qualified_canary_output": str(root / "qualified-canary.json"),
        "completion_output": str(root / "promotion-complete.json"),
        "model_path": str(tmp_path / "model"),
        "model_manifest_sha256": "m" * 64,
        "pointer_path": str(tmp_path / "authority" / "active.json"),
        "activation_path": str(tmp_path / "authority" / "qualified-active.json"),
        "stage_timeouts": {
            "materialize": 30.0,
            "lifecycle": 30.0,
            "activate": 30.0,
        },
    }


class _Inhibitor:
    pid = 52

    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, *, timeout: float) -> None:
        assert timeout == 5.0

    def kill(self) -> None:  # pragma: no cover - wait does not time out
        raise AssertionError("unexpected kill")


def _run_arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=tmp_path / "promotion-config.json",
        poll_interval=0.01,
        controller_timeout=30.0,
        launchd_supervised=True,
    )


def _write_verdict(config: dict, verdict: dict) -> None:
    path = Path(config["replication_root"]) / "replication-verdict.json"
    path.write_bytes(canonical_bytes(verdict) + b"\n")


def _battery() -> dict:
    return seal_shadow_canary_battery(
        [
            {
                "task_id": "fresh-khop-1",
                "family": "khop",
                "task_depth": 2,
                "prompt_sha256": "7" * 64,
                "expected_sha256": "8" * 64,
                "public_token_ids": [10, 11],
                "expected_token_ids": [20],
                "max_tokens": 1,
            }
        ],
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )


def _canary(activation: dict, *, serving: bool) -> dict:
    battery = _battery()
    case = battery["cases"][0]
    expected_tokens_sha256 = canonical_sha256(case["expected_token_ids"])
    evidence = [
        {
            "index": 0,
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 2,
            "request_sha256": case["request_sha256"],
            "expected_token_ids_sha256": expected_tokens_sha256,
            "generated_token_ids_sha256": expected_tokens_sha256,
            "qualified_result_sha256": "9" * 64,
            "latency_ms": 7,
            "exact": True,
        }
    ]
    body = {
        "schema": pipeline.activation.QUALIFIED_CANARY_SCHEMA,
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "checkpoint_sha256": activation["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "battery_sha256": battery["battery_sha256"],
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "case_count": 1,
        "exact_count": 1,
        "total_latency_ms": 7,
        "maximum_latency_ms": 7,
        "evidence": evidence,
        "supported": True,
        "serving_authority": serving,
        "authority_remains_active": serving,
        "canary_authority_was_request_scoped": not serving,
        "output_exposed": False,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _activation_chain() -> tuple[dict, dict, dict, dict, dict]:
    body = {
        "schema": "aura.unified_intrinsic.qualified_activation.v2",
        "package_id": "candidate",
        "manifest_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "controller_sha256": "3" * 64,
        "pointer_sha256": "4" * 64,
        "lifecycle_result_sha256": "5" * 64,
        "canary_plan_sha256": "6" * 64,
        "candidate_canary_sha256": "",
        "qualified_canary_sha256": "",
        "families": ["khop"],
        "task_depths": [2],
        "recurrence_depth": 4,
        "mode": "qualified_canary_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": False,
    }
    candidate = {**body, "activation_sha256": canonical_sha256(body)}
    candidate_canary = _canary(candidate, serving=False)
    pending = pipeline.activation.seal_verified_qualified_activation(
        candidate,
        candidate_canary,
        expected_battery=_battery(),
    )
    canary = _canary(pending, serving=False)
    durable = pipeline.activation.seal_serving_qualified_activation(
        pending,
        canary,
        expected_battery=_battery(),
    )
    return candidate, candidate_canary, pending, canary, durable


def test_refuted_replication_never_calls_a_promotion_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    inhibitor = _Inhibitor()
    verdict = {
        "supported": False,
        "verdict": "refuted_powered_resident_replication",
        "verdict_sha256": "v" * 64,
        "plan_sha256": config["replication_plan_sha256"],
        "checkpoint_sha256": "k" * 64,
    }
    _write_verdict(config, verdict)
    states: list[str] = []
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "_start_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(
        pipeline,
        "_verify_supervision",
        lambda *_args: {"controller_pid": 51, "sleep_inhibitor_pid": 52},
    )
    monkeypatch.setattr(
        pipeline.replication,
        "status",
        lambda _args: {"complete": True, "controller": {"state": "refuted"}},
    )
    monkeypatch.setattr(pipeline.replication, "adjudicate", lambda _args: verdict)
    monkeypatch.setattr(
        pipeline,
        "_publish_status",
        lambda _config, state, _details, **_kwargs: states.append(state) or {"state": state},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_bounded_stage",
        lambda *_args, **_kwargs: pytest.fail("negative verdict called a stage"),
    )

    result = pipeline.run(_run_arguments(tmp_path))

    assert result["state"] == "refuted"
    assert result["supported"] is False
    assert states == ["refuted"]
    assert inhibitor.terminated is True


def test_promotion_independently_seals_early_refutation_from_live_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    inhibitor = _Inhibitor()
    verdict = {
        "supported": False,
        "verdict": "refuted_powered_resident_replication",
        "verdict_sha256": "v" * 64,
        "plan_sha256": config["replication_plan_sha256"],
        "checkpoint_sha256": "k" * 64,
        "adjudication_scope": "decisive_early_refutation",
    }
    states: list[str] = []
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "_start_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(pipeline, "_verify_supervision", lambda *_args: {"ok": True})
    monkeypatch.setattr(
        pipeline.replication,
        "status",
        lambda _args: {
            "complete": False,
            "controller": {"state": "running"},
            "evaluations": [
                {"seed": 1, "state": "completed"},
                {"seed": 2, "state": "failed"},
                {"seed": 3, "state": "pending"},
            ],
        },
    )

    def adjudicate(arguments: argparse.Namespace) -> dict:
        assert arguments.verdict_output == (
            Path(config["replication_root"]) / "replication-verdict.json"
        )
        _write_verdict(config, verdict)
        return verdict

    monkeypatch.setattr(pipeline.replication, "adjudicate", adjudicate)
    monkeypatch.setattr(
        pipeline,
        "_publish_status",
        lambda _config, state, _details, **_kwargs: states.append(state) or {"state": state},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_bounded_stage",
        lambda *_args, **_kwargs: pytest.fail("early refutation called a stage"),
    )

    result = pipeline.run(_run_arguments(tmp_path))

    assert result["state"] == "refuted"
    assert states == ["refuted"]
    assert inhibitor.terminated is True


def test_promotion_waits_without_adjudicating_before_evidence_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    inhibitor = _Inhibitor()
    monotonic = iter((0.0, 0.0, 31.0))
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "_start_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(pipeline, "_verify_supervision", lambda *_args: {"ok": True})
    monkeypatch.setattr(
        pipeline.replication,
        "status",
        lambda _args: {
            "complete": False,
            "controller": {"state": "waiting_for_training"},
            "evaluations": [{"seed": 1, "state": "pending"}],
        },
    )
    monkeypatch.setattr(
        pipeline.replication,
        "adjudicate",
        lambda _args: pytest.fail("adjudication ran without completed evidence"),
    )
    monkeypatch.setattr(
        pipeline,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic), sleep=lambda _seconds: None),
    )
    monkeypatch.setattr(
        pipeline,
        "_publish_status",
        lambda *_args, **_kwargs: {"state": "waiting_for_replication"},
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="timed out",
    ):
        pipeline.run(_run_arguments(tmp_path))

    assert inhibitor.terminated is True


def test_supported_replication_runs_every_gate_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    inhibitor = _Inhibitor()
    verdict = {
        "supported": True,
        "verdict": "supported_powered_resident_replication",
        "verdict_sha256": "v" * 64,
        "plan_sha256": config["replication_plan_sha256"],
        "checkpoint_sha256": "k" * 64,
        "claim_boundary": {"bounded": True},
    }
    _write_verdict(config, verdict)
    order: list[str] = []
    package = {"manifest": {"package_id": "candidate", "manifest_sha256": "m" * 64}}
    lifecycle_result = {"result_sha256": "l" * 64, "supported": True}
    activated = {
        "activation_sha256": "a" * 64,
        "canary": {"result_sha256": "q" * 64, "supported": True},
    }
    completion = {
        "completion_sha256": "z" * 64,
        "activation_sha256": "a" * 64,
    }
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "_start_sleep_inhibitor", lambda: inhibitor)
    monkeypatch.setattr(pipeline, "_verify_supervision", lambda *_args: {"ok": True})
    monkeypatch.setattr(
        pipeline.replication,
        "status",
        lambda _args: {"complete": True, "controller": {"state": "completed"}},
    )
    monkeypatch.setattr(pipeline.replication, "adjudicate", lambda _args: verdict)
    monkeypatch.setattr(
        pipeline,
        "_publish_status",
        lambda _config, state, _details, **_kwargs: {"state": state},
    )
    def run_stage(_path, _config, stage, **_kwargs):
        order.append(stage)
        return {
            "materialize": package,
            "lifecycle": lifecycle_result,
            "activate": activated,
        }[stage]

    monkeypatch.setattr(pipeline, "_run_bounded_stage", run_stage)
    monkeypatch.setattr(
        pipeline,
        "_write_completion",
        lambda *_args: order.append("complete") or completion,
    )

    result = pipeline.run(_run_arguments(tmp_path))

    assert result["state"] == "completed"
    assert result["supported"] is True
    assert order == ["materialize", "lifecycle", "activate", "complete"]
    assert inhibitor.terminated is True


def test_interrupted_lifecycle_retires_only_its_exact_pointer_and_quarantines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    output = Path(config["lifecycle_output"])
    output.mkdir()
    (output / "cold-load-01.json").write_text("partial", encoding="ascii")
    pointer_path = Path(config["pointer_path"])
    pointer_path.parent.mkdir()
    pointer_path.write_text("pointer", encoding="ascii")
    package = Path(config["package"])
    retired: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "read_shadow_pointer",
        lambda _path: {"pointer_sha256": "x" * 64},
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_shadow_pointer",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(
        pipeline,
        "deactivate_shadow_pointer",
        lambda **kwargs: retired.append(kwargs["expected_current_sha256"]),
    )
    monkeypatch.setattr(
        pipeline.lifecycle,
        "run_lifecycle",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    monkeypatch.setattr(
        pipeline.asyncio,
        "run",
        lambda _awaitable: {"supported": True, "result_sha256": "r" * 64},
    )

    result = pipeline._lifecycle_or_reopen(config)  # noqa: SLF001

    assert result["supported"] is True
    assert retired == ["x" * 64]
    assert not output.exists()
    assert len(list(output.parent.glob("lifecycle.interrupted-*"))) == 1


def test_interrupted_lifecycle_refuses_another_selected_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    output = Path(config["lifecycle_output"])
    output.mkdir()
    (output / "partial.json").write_text("partial", encoding="ascii")
    pointer_path = Path(config["pointer_path"])
    pointer_path.parent.mkdir()
    pointer_path.write_text("pointer", encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "read_shadow_pointer",
        lambda _path: {"pointer_sha256": "x" * 64},
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_shadow_pointer",
        lambda *_args, **_kwargs: tmp_path / "other-package",
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="another package",
    ):
        pipeline._lifecycle_or_reopen(config)  # noqa: SLF001


def test_launch_contract_is_source_bound_and_restart_supervised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    campaign_config = {"runtime": {"interpreter": {}}}
    monkeypatch.setattr(
        pipeline.resident,
        "_load_config",
        lambda _path: campaign_config,
    )
    monkeypatch.setattr(
        pipeline.replication,
        "_runtime_python",
        lambda _config: (
            Path("/private/runtime/bin/python"),
            {"sys_prefix": "/private/runtime", "sha256": "i" * 64},
        ),
    )
    monkeypatch.setattr(pipeline, "LAUNCH_AGENTS_ROOT", tmp_path / "agents")
    arguments = argparse.Namespace(poll_interval=7.0, controller_timeout=900.0)

    _path, payload, intent = pipeline._launch_contract(  # noqa: SLF001
        tmp_path / "promotion-config.json",
        config,
        arguments,
    )
    plist = plistlib.loads(payload)

    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["EnvironmentVariables"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert plist["ProgramArguments"][2] == "run"
    assert plist["ProgramArguments"][-1] == "--launchd-supervised"
    assert intent["config_sha256"] == config["config_sha256"]


def test_supervision_requires_launchd_parent_and_exact_caffeinate_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 41)
    monkeypatch.setattr(
        pipeline,
        "_launchd_job",
        lambda _label: {"target": "gui/501/job", "pid": 41},
    )
    monkeypatch.setattr(
        pipeline,
        "_process_row",
        lambda _pid: (41, "/usr/bin/caffeinate -dims -w 41"),
    )

    assert pipeline._verify_supervision(config, 42) == {  # noqa: SLF001
        "target": "gui/501/job",
        "controller_pid": 41,
        "sleep_inhibitor_pid": 42,
    }

    monkeypatch.setattr(pipeline, "_process_row", lambda _pid: (99, "wrong"))
    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="sleep inhibitor lineage",
    ):
        pipeline._verify_supervision(config, 42)  # noqa: SLF001


def test_install_published_prepares_and_launches_only_inside_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    capsule = tmp_path / "capsule"
    script = capsule / "tools/run_unified_recurrent_promotion_pipeline.py"
    script.parent.mkdir(parents=True)
    script.write_text("# capsule", encoding="ascii")
    config = _config(campaign)
    commit = "1" * 40
    config["source"] = {"git": {"commit": commit}}
    commands: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def run(command: list[str], **kwargs: object) -> Result:
        commands.append(command)
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        if "prepare" in command:
            return Result(json.dumps(config))
        return Result(json.dumps({"launch_sha256": "l" * 64}))

    monkeypatch.setattr(pipeline, "_published_commit", lambda _root: commit)
    monkeypatch.setattr(pipeline, "_capsule", lambda _commit: capsule)
    monkeypatch.setattr(
        pipeline.resident,
        "_load_config",
        lambda _path: {"runtime": {"interpreter": {}}},
    )
    monkeypatch.setattr(
        pipeline.replication,
        "_runtime_python",
        lambda _config: (Path(sys.executable), {"sys_prefix": sys.prefix}),
    )
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    arguments = argparse.Namespace(
        campaign=campaign,
        replication_root=Path(config["replication_root"]),
        output=Path(config["pipeline_root"]),
        package_id="candidate",
        poll_interval=11.0,
        controller_timeout=900.0,
    )

    result = pipeline.install_published(arguments)

    assert result["capsule"] == str(capsule)
    assert commands[0][1] == str(script)
    assert commands[0][2] == "prepare"
    assert commands[1][1] == str(script)
    assert commands[1][2] == "install-launchd"
    assert commands[1][3] == str(Path(config["pipeline_root"]) / "promotion-config.json")


def test_source_identity_binds_clean_detached_git_and_selected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in pipeline.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="ascii")
    commit = "2" * 40
    git = {
        "root": str(tmp_path),
        "commit": commit,
        "branch": "DETACHED",
        "identity_sha256": "g" * 64,
    }
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{commit}\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_source_git_identity",
        lambda root, *, source_commit: git,
    )
    verified: list[dict] = []
    monkeypatch.setattr(
        pipeline,
        "verify_source_git_identity",
        lambda root, identity: verified.append(identity),
    )

    identity = pipeline._source_identity(tmp_path)  # noqa: SLF001

    assert identity["git"] == git
    assert set(identity["source_sha256s"]) == set(pipeline.SOURCE_PATHS)
    assert verified == [git]


def test_pipeline_root_rejects_symlink_traversal(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (campaign / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="contains a symlink",
    ):
        pipeline._pipeline_root(campaign, campaign / "linked" / "promotion")  # noqa: SLF001


def test_existing_package_must_bind_current_replication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config["source"] = {"git": {"commit": "s" * 40}}
    verdict = {"verdict_sha256": "v" * 64, "checkpoint_sha256": "k" * 64}
    manifest = {
        "manifest_sha256": "h" * 64,
        "package_id": config["package_id"],
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["campaign_config_sha256"],
        "source_commit": config["source"]["git"]["commit"],
        "model_manifest_sha256": config["model_manifest_sha256"],
        "replication_plan_sha256": config["replication_plan_sha256"],
        "replication_verdict_sha256": "old-verdict",
        "checkpoint_sha256": verdict["checkpoint_sha256"],
    }
    monkeypatch.setattr(
        pipeline.materializer,
        "_read_document",
        lambda _path: manifest,
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="current replication",
    ):
        pipeline._verify_package_binding(  # noqa: SLF001
            config,
            {
                "package": config["package"],
                "manifest_sha256": manifest.get("manifest_sha256"),
            },
            verdict,
        )


def test_package_binds_training_epoch_not_later_promotion_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config["campaign_source_commit"] = "a" * 40
    config["source"] = {"git": {"commit": "b" * 40}}
    verdict = {"verdict_sha256": "v" * 64, "checkpoint_sha256": "k" * 64}
    manifest = {
        "manifest_sha256": "h" * 64,
        "package_id": config["package_id"],
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["campaign_config_sha256"],
        "source_commit": config["campaign_source_commit"],
        "model_manifest_sha256": config["model_manifest_sha256"],
        "replication_plan_sha256": config["replication_plan_sha256"],
        "replication_verdict_sha256": verdict["verdict_sha256"],
        "checkpoint_sha256": verdict["checkpoint_sha256"],
    }
    monkeypatch.setattr(pipeline.materializer, "_read_document", lambda _path: manifest)

    observed = pipeline._verify_package_binding(  # noqa: SLF001
        config,
        {"package": config["package"], "manifest_sha256": manifest["manifest_sha256"]},
        verdict,
    )

    assert observed["source_commit"] == "a" * 40
    assert observed["source_commit"] != config["source"]["git"]["commit"]


def test_existing_completion_must_match_current_activation_and_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    verdict = {
        "supported": True,
        "verdict_sha256": "v" * 64,
        "claim_boundary": {"bounded": True},
    }
    _candidate, candidate_canary, _pending, canary, durable = _activation_chain()
    package = {
        "manifest": {
            "package_id": "candidate",
            "manifest_sha256": durable["manifest_sha256"],
        }
    }
    lifecycle = {"supported": True, "result_sha256": "l" * 64}
    activated = {
        "active": True,
        "activation_sha256": durable["activation_sha256"],
        "canary": canary,
        "candidate_canary": candidate_canary,
    }
    path = Path(config["completion_output"])
    body = {
        "schema": pipeline.COMPLETION_SCHEMA,
        "config_sha256": config["config_sha256"],
        "replication_verdict_sha256": verdict["verdict_sha256"],
        "package_id": "candidate",
        "manifest_sha256": durable["manifest_sha256"],
        "lifecycle_result_sha256": lifecycle["result_sha256"],
        "activation_sha256": "old-activation",
        "qualified_canary_sha256": activated["canary"]["result_sha256"],
        "candidate_canary_sha256": activated["candidate_canary"]["result_sha256"],
        "supported": True,
        "serving_authority": True,
        "claim_boundary": verdict["claim_boundary"],
        "completed_at_unix": 1.0,
    }
    existing = {**body, "completion_sha256": canonical_sha256(body)}
    path.write_bytes(canonical_bytes(existing) + b"\n")
    monkeypatch.setattr(
        pipeline.activation,
        "read_qualified_activation",
        lambda _path: durable,
    )
    monkeypatch.setattr(pipeline, "_read_bound_canary_battery", lambda *_args: _battery())

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="completion receipt identity differs",
    ):
        pipeline._write_completion(  # noqa: SLF001
            config,
            verdict,
            package,
            lifecycle,
            activated,
        )


def test_installed_lineage_rejects_every_partial_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "controller_pid": 41,
        "controller_start_token": "token",
        "sleep_inhibitor_pid": 42,
    }
    job = {"pid": 41}
    monkeypatch.setattr(
        pipeline,
        "_process_row",
        lambda _pid: (41, "/usr/bin/caffeinate -dims -w 41"),
    )
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "alive",
    )
    assert pipeline._installed_lineage_is_valid(status, job) is True  # noqa: SLF001
    assert pipeline._installed_lineage_is_valid(status, {"pid": 99}) is False  # noqa: SLF001
    monkeypatch.setattr(pipeline, "_process_row", lambda _pid: (99, "wrong"))
    assert pipeline._installed_lineage_is_valid(status, job) is False  # noqa: SLF001

    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "unknown",
    )
    assert pipeline._installed_lineage_state(status, job) == "unknown"  # noqa: SLF001
    assert pipeline._installed_lineage_is_valid(status, job) is False  # noqa: SLF001
    monkeypatch.setattr(
        pipeline,
        "_process_row",
        lambda _pid: (41, "/usr/bin/caffeinate -dims -w 41"),
    )
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "dead",
    )
    monkeypatch.setattr(
        pipeline.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert pipeline._installed_lineage_is_valid(status, job) is False  # noqa: SLF001


def test_active_stage_receipt_is_authenticated_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_key", lambda _config: b"k" * 32)
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 41)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_process_start_token",
        lambda _pid: pytest.fail("stage publication re-probed controller identity"),
    )
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda pgid: [{"pid": pgid, "start_token": f"token-{pgid}"}],
    )

    receipt = pipeline._publish_active_stage(  # noqa: SLF001
        config,
        controller_pid=41,
        controller_start_token="token-41",
        stage="lifecycle",
        child_pid=42,
        child_start_token="token-42",
        result_path=Path(config["pipeline_root"]) / "result.json",
        log_path=Path(config["pipeline_root"]) / "stage.log",
        timeout_s=30.0,
    )
    assert pipeline._read_active_stage(config) == receipt  # noqa: SLF001

    attacked = dict(receipt)
    attacked["child_pid"] = 99
    pipeline._active_stage_path(config).write_bytes(  # noqa: SLF001
        canonical_bytes(attacked) + b"\n"
    )
    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="authentication failed",
    ):
        pipeline._read_active_stage(config)  # noqa: SLF001


def test_stage_cleanup_uses_authenticated_memory_when_receipt_reopen_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = {
        "group_members": [
            {"pid": 42, "start_token": "token-42"},
        ]
    }
    monkeypatch.setattr(
        pipeline,
        "_refresh_active_stage_members",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            pipeline.UnifiedRecurrentPromotionError("receipt unavailable")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda _pgid: [
            {"pid": 42, "start_token": "token-42"},
            {"pid": 44, "start_token": "token-44"},
        ],
    )

    assert pipeline._stage_cleanup_members(  # noqa: SLF001
        {},
        active,
        expected_pid=42,
        expected_start_token="token-42",
    ) == [
        {"pid": 42, "start_token": "token-42"},
        {"pid": 44, "start_token": "token-44"},
    ]


def test_exact_stage_termination_targets_only_recorded_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    alive = True

    class Process:
        def wait(self, *, timeout: float):
            assert timeout == 0.01
            return -9

    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda _pgid: (
            [{"pid": 42, "start_token": "token-42"}] if alive else []
        ),
    )

    def kill_group(pgid, sent):
        nonlocal alive
        signals.append((pgid, sent))
        if sent == pipeline.signal.SIGKILL:
            alive = False

    monkeypatch.setattr(
        pipeline.os,
        "killpg",
        kill_group,
    )

    pipeline._terminate_exact_stage_process(  # noqa: SLF001
        Process(),
        pid=42,
        start_token="token-42",
        grace_s=0.01,
    )

    assert signals == [
        (42, pipeline.signal.SIGTERM),
        (42, pipeline.signal.SIGKILL),
    ]


def test_dead_stage_leader_does_not_abandon_authenticated_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = True
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda _pgid: (
            [{"pid": 44, "start_token": "token-44"}] if alive else []
        ),
    )

    def kill_group(pgid, sent):
        nonlocal alive
        signals.append((pgid, sent))
        alive = False

    monkeypatch.setattr(pipeline.os, "killpg", kill_group)

    pipeline._terminate_exact_stage_process(  # noqa: SLF001
        None,
        pid=42,
        start_token="token-42",
        group_members=[
            {"pid": 42, "start_token": "token-42"},
            {"pid": 44, "start_token": "token-44"},
        ],
        grace_s=0.01,
    )

    assert signals == [(42, pipeline.signal.SIGTERM)]


def test_recovery_refuses_unauthenticated_process_group_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda _pgid: [{"pid": 99, "start_token": "foreign"}],
    )
    monkeypatch.setattr(
        pipeline.os,
        "killpg",
        lambda *_args: pytest.fail("foreign process group was signalled"),
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="unauthenticated member",
    ):
        pipeline._terminate_exact_stage_process(  # noqa: SLF001
            None,
            pid=42,
            start_token="token-42",
            group_members=[{"pid": 42, "start_token": "token-42"}],
            grace_s=0.01,
        )


def test_restart_never_retires_stage_owned_by_another_live_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_key", lambda _config: b"k" * 32)
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 41)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_process_start_token",
        lambda pid: f"token-{pid}",
    )
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda pgid: [{"pid": pgid, "start_token": f"token-{pgid}"}],
    )
    pipeline._publish_active_stage(  # noqa: SLF001
        config,
        controller_pid=41,
        controller_start_token="token-41",
        stage="lifecycle",
        child_pid=43,
        child_start_token="token-43",
        result_path=Path(config["pipeline_root"]) / "result.json",
        log_path=Path(config["pipeline_root"]) / "stage.log",
        timeout_s=30.0,
    )
    value = pipeline._read_active_stage(config)  # noqa: SLF001
    assert value is not None
    body = {
        key: item
        for key, item in value.items()
        if key != "hmac_sha256"
    }
    body["controller_pid"] = 42
    body["controller_start_token"] = "token-42"
    attacked = {
        **body,
        "hmac_sha256": pipeline._signature(body, b"k" * 32),  # noqa: SLF001
    }
    pipeline._active_stage_path(config).write_bytes(  # noqa: SLF001
        canonical_bytes(attacked) + b"\n"
    )
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda pid, token: "alive" if (pid, token) == (42, "token-42") else "dead",
    )
    monkeypatch.setattr(
        pipeline,
        "_terminate_exact_stage_process",
        lambda *_args, **_kwargs: pytest.fail("live controller child was terminated"),
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="owner is not proven dead: alive",
    ):
        pipeline._retire_interrupted_stage(config)  # noqa: SLF001


def test_restart_never_retires_stage_when_owner_liveness_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_key", lambda _config: b"k" * 32)
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 41)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_process_start_token",
        lambda pid: f"token-{pid}",
    )
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda pgid: [{"pid": pgid, "start_token": f"token-{pgid}"}],
    )
    pipeline._publish_active_stage(  # noqa: SLF001
        config,
        controller_pid=41,
        controller_start_token="token-41",
        stage="lifecycle",
        child_pid=43,
        child_start_token="token-43",
        result_path=Path(config["pipeline_root"]) / "result.json",
        log_path=Path(config["pipeline_root"]) / "stage.log",
        timeout_s=30.0,
    )
    value = pipeline._read_active_stage(config)  # noqa: SLF001
    assert value is not None
    body = {key: item for key, item in value.items() if key != "hmac_sha256"}
    body["controller_pid"] = 42
    body["controller_start_token"] = "token-42"
    interrupted = {
        **body,
        "hmac_sha256": pipeline._signature(body, b"k" * 32),  # noqa: SLF001
    }
    pipeline._active_stage_path(config).write_bytes(  # noqa: SLF001
        canonical_bytes(interrupted) + b"\n"
    )
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "unknown",
    )
    monkeypatch.setattr(
        pipeline,
        "_terminate_exact_stage_process",
        lambda *_args, **_kwargs: pytest.fail("unknown owner child was terminated"),
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="owner is not proven dead: unknown",
    ):
        pipeline._retire_interrupted_stage(config)  # noqa: SLF001


def test_restart_does_not_trust_reused_current_controller_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_key", lambda _config: b"k" * 32)
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 41)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_process_start_token",
        lambda pid: "current-token" if pid == 41 else f"token-{pid}",
    )
    monkeypatch.setattr(
        pipeline,
        "_process_group_members",
        lambda pgid: [{"pid": pgid, "start_token": f"token-{pgid}"}],
    )
    pipeline._publish_active_stage(  # noqa: SLF001
        config,
        controller_pid=41,
        controller_start_token="current-token",
        stage="lifecycle",
        child_pid=43,
        child_start_token="token-43",
        result_path=Path(config["pipeline_root"]) / "result.json",
        log_path=Path(config["pipeline_root"]) / "stage.log",
        timeout_s=30.0,
    )
    value = pipeline._read_active_stage(config)  # noqa: SLF001
    assert value is not None
    body = {key: item for key, item in value.items() if key != "hmac_sha256"}
    body["controller_start_token"] = "previous-token"
    interrupted = {
        **body,
        "hmac_sha256": pipeline._signature(body, b"k" * 32),  # noqa: SLF001
    }
    pipeline._active_stage_path(config).write_bytes(  # noqa: SLF001
        canonical_bytes(interrupted) + b"\n"
    )
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda pid, token: (
            "unknown" if (pid, token) == (41, "previous-token") else "dead"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_terminate_exact_stage_process",
        lambda *_args, **_kwargs: pytest.fail("reused PID child was terminated"),
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="owner is not proven dead: unknown",
    ):
        pipeline._retire_interrupted_stage(config)  # noqa: SLF001


def test_stage_parent_monitor_terminates_its_own_group_on_parent_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "dead",
    )
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 73)
    monkeypatch.setattr(pipeline.os, "getpgrp", lambda: 73)
    monkeypatch.setattr(
        pipeline.os,
        "kill",
        lambda *_args: pytest.fail("dedicated group must be terminated as a group"),
    )
    monkeypatch.setattr(
        pipeline.os,
        "killpg",
        lambda pgid, sent: signals.append((pgid, sent)),
    )

    pipeline._monitor_controller(41, "token-41")  # noqa: SLF001

    assert signals == [(73, pipeline.signal.SIGTERM)]


def test_stage_parent_monitor_does_not_signal_on_unknown_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    states = iter(["unknown", "dead"])
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: next(states),
    )
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 73)
    monkeypatch.setattr(pipeline.os, "getpgrp", lambda: 73)
    monkeypatch.setattr(
        pipeline.os,
        "killpg",
        lambda pgid, sent: signals.append((pgid, sent)),
    )

    pipeline._monitor_controller(41, "token-41")  # noqa: SLF001

    assert signals == [(73, pipeline.signal.SIGTERM)]


@pytest.mark.parametrize(
    ("owner_state", "pid_present", "expected_state"),
    [
        ("unknown", False, "unknown"),
        ("dead", True, "conflict"),
    ],
)
def test_launchd_install_never_boots_out_unproven_existing_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_state: str,
    pid_present: bool,
    expected_state: str,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "promotion-config.json"
    plist_path = tmp_path / "promotion.plist"
    intent = {
        "config_sha256": config["config_sha256"],
        "intent_sha256": "i" * 64,
    }
    status = {
        "controller_pid": 41,
        "controller_start_token": "token-41",
        "sleep_inhibitor_pid": 42,
    }
    arguments = argparse.Namespace(
        config=config_path,
        poll_interval=1.0,
        controller_timeout=30.0,
    )
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        pipeline,
        "_launch_contract",
        lambda *_args: (plist_path, b"plist", intent),
    )
    monkeypatch.setattr(pipeline, "LAUNCH_AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "atomic_write_bytes_if_absent",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(pipeline, "atomic_write_bytes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "ensure_private_directory", lambda _path: None)
    monkeypatch.setattr(
        pipeline,
        "_launchd_job",
        lambda _label: {"target": "gui/501/test", "pid": 41},
    )
    monkeypatch.setattr(pipeline, "_read_status", lambda _config: status)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: owner_state,
    )

    def probe_pid(_pid: int, _signal: int) -> None:
        if not pid_present:
            raise ProcessLookupError

    monkeypatch.setattr(
        pipeline.os,
        "kill",
        probe_pid,
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unknown controller was disturbed"),
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match=f"not proven stale: {expected_state}",
    ):
        pipeline.install_launchd(arguments)


def test_run_stage_rejects_result_outside_pipeline_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        pipeline.replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "alive",
    )
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 73)
    monkeypatch.setattr(pipeline.os, "getpgrp", lambda: 73)
    arguments = argparse.Namespace(
        config=tmp_path / "config.json",
        stage="materialize",
        result_output=tmp_path / "outside.json",
        controller_pid=41,
        controller_start_token="token-41",
    )

    with pytest.raises(
        pipeline.UnifiedRecurrentPromotionError,
        match="inside the pipeline root",
    ):
        pipeline.run_stage(arguments)


def test_active_authority_reopen_reads_real_flattened_package_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    activation_path = Path(config["activation_path"])
    activation_path.parent.mkdir(parents=True)
    activation_path.write_text("active", encoding="ascii")
    Path(config["pointer_path"]).write_text("pointer", encoding="ascii")
    canary_path = Path(config["qualified_canary_output"])
    canary_path.write_text("canary", encoding="ascii")
    candidate_canary_path = pipeline.activation._candidate_canary_path(canary_path)  # noqa: SLF001
    candidate_canary_path.write_text("candidate-canary", encoding="ascii")
    _candidate, candidate_canary, _pending, canary, durable = _activation_chain()
    observed = {
        "active": True,
        "package": config["package"],
        "activation_sha256": durable["activation_sha256"],
    }
    monkeypatch.setattr(pipeline.activation, "_status", lambda _args: observed)
    monkeypatch.setattr(
        pipeline.activation,
        "_read_lifecycle",
        lambda path: candidate_canary if path == candidate_canary_path else canary,
    )
    monkeypatch.setattr(
        pipeline.activation,
        "read_qualified_activation",
        lambda _path: durable,
    )
    monkeypatch.setattr(pipeline, "_read_bound_canary_battery", lambda *_args: _battery())
    monkeypatch.setattr(
        pipeline.materializer,
        "_read_document",
        lambda _path: {"manifest_sha256": durable["manifest_sha256"]},
    )
    monkeypatch.setattr(
        pipeline.materializer,
        "inspect_shadow_package",
        lambda *_args, **_kwargs: pytest.fail("flattened receipt must not be nested"),
    )

    result = pipeline._activate_or_reopen(config)  # noqa: SLF001

    assert result["active"] is True
    assert result["canary"] == canary
    assert result["candidate_canary"] == candidate_canary


def test_orphaned_durable_authority_is_revoked_and_recovered_in_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    activation_path = Path(config["activation_path"])
    activation_path.parent.mkdir(parents=True)
    activation_path.write_text("orphaned", encoding="ascii")
    canary_path = Path(config["qualified_canary_output"])
    candidate_path = pipeline.activation._candidate_canary_path(canary_path)  # noqa: SLF001
    canary_path.write_text("stale", encoding="ascii")
    candidate_path.write_text("stale", encoding="ascii")
    retired: list[str] = []

    monkeypatch.setattr(
        pipeline.activation,
        "read_qualified_activation",
        lambda _path: {"activation_sha256": "a" * 64},
    )

    def deactivate(**kwargs):
        retired.append(kwargs["expected_current_sha256"])
        activation_path.unlink()
        return {"activation_sha256": kwargs["expected_current_sha256"]}

    recovered = {"active": True, "activation_sha256": "b" * 64}
    monkeypatch.setattr(
        pipeline.activation,
        "deactivate_qualified_activation",
        deactivate,
    )
    monkeypatch.setattr(
        pipeline.activation,
        "_activate_verified",
        lambda _arguments: recovered,
    )

    assert pipeline._activate_or_reopen(config) == recovered  # noqa: SLF001
    assert retired == ["a" * 64]
    assert not activation_path.exists()
    assert not canary_path.exists()
    assert not candidate_path.exists()
    assert list(canary_path.parent.glob("*.interrupted-*"))
