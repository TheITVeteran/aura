from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import pytest

from tools import adjudicate_unified_intrinsic_resident_replication as replication
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256

SEEDS = replication.DEFAULT_SEEDS


def _arguments(campaign: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "campaign": campaign,
        "output": None,
        "evaluator_source_root": None,
        "verdict_output": None,
        "seeds": SEEDS,
        "per_cell": 1,
        "max_tokens": 32,
        "poll_interval": 0.01,
        "controller_timeout": 60.0,
        "launchd_supervised": True,
        "task_depths": (1, 2, 4),
        "recurrence_depths": (4,),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _config(campaign: Path) -> dict:
    executable = Path(sys.executable).absolute()
    real_executable = executable.resolve()
    return {
        "campaign_id": "resident-full",
        "config_sha256": "c" * 64,
        "paths": {"campaign_root": str(campaign)},
        "source": {
            "git": {"commit": "d" * 40, "root": str(replication.REPO_ROOT)}
        },
        "runtime": {
            "interpreter": {
                "executable": str(executable),
                "real_executable": str(real_executable),
                "sys_prefix": str(Path(sys.prefix).absolute()),
                "sha256": hashlib.sha256(real_executable.read_bytes()).hexdigest(),
                "size_bytes": executable.stat().st_size,
            }
        },
        "training": {"families": "khop,modular,register_trace"},
    }


def _report(seed: int, *, task_prefix: str | None = None) -> dict:
    tasks = 9
    prefix = task_prefix or f"seed-{seed}"
    arm_correct = {
        "base_t1": 5,
        "untrained_t1": 1,
        "trained_t1": 3,
        "untrained_t4": 1,
        "trained_t4": 9,
        "grammar_lesion_t4": 1,
        "pointer_lesion_t4": 2,
        "compiled_t4": 9,
    }
    candidates = []
    for task in range(tasks):
        for arm, correct_count in arm_correct.items():
            candidates.append(
                {
                    "task_id": f"{prefix}-task-{task}",
                    "prompt_sha256": canonical_sha256({"prefix": prefix, "task": task}),
                    "arm": arm,
                    "correct": task < correct_count,
                }
            )

    def summary(correct: int) -> dict:
        return {
            "correct": correct,
            "tasks": tasks,
            "accuracy": correct / tasks,
            "eos_stops": tasks,
        }

    body = {
        "schema": "aura.unified_intrinsic_decode_evaluation.v1",
        "checkpoint_sha256": "a" * 64,
        "evaluation_seed": seed,
        "per_cell": 1,
        "task_count": tasks,
        "task_depths": [1, 2, 4],
        "recurrence_depths": [4],
        "evaluation_source_sha256s": replication.launcher._evaluator_source_sha256s(  # noqa: SLF001
            replication.REPO_ROOT
        ),
        "arm_results": {arm: summary(correct) for arm, correct in arm_correct.items()},
        "candidates": candidates,
        "paired_training_effects": {
            "4": {
                "tasks": tasks,
                "control_arm": "untrained_t4",
                "trained_arm": "trained_t4",
                "untrained_correct": 1,
                "trained_correct": 9,
                "net_correct_gain": 8,
                "wrong_to_right": 8,
                "right_to_wrong": 0,
            }
        },
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def _neutralize_treatment(report: dict) -> dict:
    changed = copy.deepcopy(report)
    changed["arm_results"]["trained_t4"] = {
        "correct": 1,
        "tasks": 9,
        "accuracy": 1 / 9,
        "eos_stops": 9,
    }
    for candidate in changed["candidates"]:
        if candidate["arm"] == "trained_t4":
            candidate["correct"] = candidate["task_id"].endswith("task-0")
    changed["paired_training_effects"]["4"].update(
        {
            "trained_correct": 1,
            "net_correct_gain": 0,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
        }
    )
    body = {key: value for key, value in changed.items() if key != "report_sha256"}
    changed["report_sha256"] = canonical_sha256(body)
    return changed


def _installed_plan(campaign: Path) -> dict:
    root = campaign / "resident-replication"
    return {
        "schema": replication.PLAN_SCHEMA,
        "campaign_id": "resident-full",
        "campaign_root": str(campaign),
        "campaign_config_sha256": "c" * 64,
        "source_commit": "d" * 40,
        "evaluator_source_root": str(replication.REPO_ROOT),
        "evaluator_source_sha256s": replication.launcher._evaluator_source_sha256s(  # noqa: SLF001
            replication.REPO_ROOT
        ),
        "checkpoint_contract": "exact_terminal_answer_bridge_admitted_checkpoint",
        "seeds": list(SEEDS),
        "per_cell": 1,
        "task_depths": [1, 2, 4],
        "recurrence_depths": [4],
        "max_tokens": 32,
        "task_count_per_seed": 9,
        "total_tasks": 27,
        "total_candidates": 216,
        "evaluations": [{"seed": seed, "output": str(root / f"seed-{seed}")} for seed in SEEDS],
        "decision_rule": {
            "alpha_numerator": 1,
            "alpha_denominator": 100,
            "minimum_pooled_effect_numerator": 1,
            "minimum_pooled_effect_denominator": 5,
            "each_seed_positive_matched_control_gain": True,
            "zero_pooled_right_to_wrong": True,
            "compiled_exact_every_seed": True,
            "strict_aggregate_grammar_lesion_loss": True,
            "strict_aggregate_pointer_lesion_loss": True,
            "strict_aggregate_base_loss": True,
            "strict_aggregate_trained_t1_loss": True,
            "task_and_prompt_identity_disjoint_across_seeds": True,
        },
        "claim_boundary": (
            "A supported verdict proves powered multi-seed resident-32B neural "
            "transfer only on the typed recurrent task battery. It does not "
            "prove broad reasoning, frontier performance, production fusion, "
            "or a WOW Signal."
        ),
    }


def test_prepare_freezes_powered_plan_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = _config(campaign)
    monkeypatch.setattr(replication, "_campaign", lambda _args: (campaign, config))

    first = replication.prepare(_arguments(campaign))
    second = replication.prepare(_arguments(campaign))

    assert first == second
    assert first["total_tasks"] == 27
    assert first["total_candidates"] == 216
    assert first["evaluator_source_sha256s"] == (
        replication.launcher._evaluator_source_sha256s(replication.REPO_ROOT)  # noqa: SLF001
    )
    assert first["plan_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "plan_sha256"}
    )
    assert (
        campaign / "resident-replication" / "replication-plan.json"
    ).stat().st_mode & 0o777 == 0o400


def test_replication_plan_rejects_evaluator_drift_across_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    evaluator_root = tmp_path / "evaluator"
    (evaluator_root / "tools").mkdir(parents=True)
    for relative in replication.launcher.EVALUATION_SOURCE_FILES:
        path = evaluator_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="ascii")
    config = _config(campaign)
    monkeypatch.setattr(replication, "_campaign", lambda _args: (campaign, config))
    arguments = _arguments(campaign, evaluator_source_root=evaluator_root)
    replication.prepare(arguments)
    (evaluator_root / replication.launcher.EVALUATION_SOURCE_FILES[0]).write_text(
        "# changed after preregistration\n",
        encoding="ascii",
    )

    with pytest.raises(replication.ResidentReplicationError, match="source differs"):
        replication._load_plan(arguments)  # noqa: SLF001


def _install_adjudication(
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    reports: dict[int, dict],
) -> None:
    plan_body = _installed_plan(campaign)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    config = _config(campaign)
    monkeypatch.setattr(
        replication,
        "_load_plan",
        lambda _args: (campaign, config, plan),
    )
    monkeypatch.setattr(
        replication.launcher,
        "_read_document",
        lambda _path: {"checkpoint": {"complete": True, "checkpoint_sha256": "a" * 64}},
    )

    def fake_status(arguments: argparse.Namespace) -> dict:
        return {"state": "completed", "report": reports[arguments.evaluation_seed]}

    monkeypatch.setattr(replication.launcher, "status", fake_status)


def test_adjudicate_supports_disjoint_three_seed_replication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {seed: _report(seed) for seed in SEEDS}
    _install_adjudication(tmp_path, monkeypatch, reports)

    verdict = replication.adjudicate(_arguments(tmp_path))

    assert verdict["verdict"] == replication.SUPPORTED
    assert verdict["supported"] is True
    assert verdict["total_tasks"] == 27
    assert verdict["paired_effect"]["wrong_to_right"] == 24
    assert verdict["paired_effect"]["right_to_wrong"] == 0
    assert verdict["paired_effect"]["one_sided_exact_p_denominator"] == 2**24
    assert all(verdict["checks"].values())


def test_adjudicate_rejects_cross_seed_task_or_prompt_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {seed: _report(seed, task_prefix="reused") for seed in SEEDS}
    _install_adjudication(tmp_path, monkeypatch, reports)

    with pytest.raises(replication.ResidentReplicationError, match="overlap"):
        replication.adjudicate(_arguments(tmp_path))


def test_valid_negative_seed_produces_refutation_not_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {seed: _report(seed) for seed in SEEDS}
    reports[SEEDS[0]] = _neutralize_treatment(reports[SEEDS[0]])
    _install_adjudication(tmp_path, monkeypatch, reports)

    verdict = replication.adjudicate(_arguments(tmp_path))

    assert verdict["verdict"] == replication.REFUTED
    assert verdict["supported"] is False
    assert verdict["checks"]["every_seed_positive_matched_control_gain"] is False


def test_adjudicate_contains_malformed_single_seed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {seed: _report(seed) for seed in SEEDS}
    changed = copy.deepcopy(reports[SEEDS[0]])
    changed["candidates"].pop()
    body = {key: value for key, value in changed.items() if key != "report_sha256"}
    changed["report_sha256"] = canonical_sha256(body)
    reports[SEEDS[0]] = changed
    _install_adjudication(tmp_path, monkeypatch, reports)

    with pytest.raises(replication.ResidentReplicationError, match="malformed"):
        replication.adjudicate(_arguments(tmp_path))


def test_run_adjudicates_and_publishes_terminal_controller_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_body = _installed_plan(tmp_path)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    config = _config(tmp_path)
    monkeypatch.setattr(
        replication,
        "_load_plan",
        lambda _args: (tmp_path, config, plan),
    )
    monkeypatch.setattr(
        replication,
        "_campaign_is_terminal",
        lambda _campaign, _config: (
            True,
            {
                "completion_sha256": "e" * 64,
                "answer_bridge_admitted": True,
            },
        ),
    )
    monkeypatch.setattr(
        replication,
        "_verify_launchd_supervision",
        lambda *_args: {"target": "gui/501/test"},
    )
    monkeypatch.setattr(
        replication,
        "status",
        lambda _args: {"complete": True, "evaluations": []},
    )
    expected = {
        "verdict": replication.SUPPORTED,
        "supported": True,
        "verdict_sha256": "f" * 64,
    }
    monkeypatch.setattr(replication, "adjudicate", lambda _args: expected)
    published: list[tuple[str, dict]] = []

    def publish(
        _arguments: argparse.Namespace,
        _campaign: Path,
        _config: dict,
        _plan: dict,
        state: str,
        details: dict,
    ) -> dict:
        published.append((state, details))
        return {"state": state}

    monkeypatch.setattr(replication, "_publish_controller_status", publish)

    result = replication.run(_arguments(tmp_path))

    assert result["state"] == "completed"
    assert result["supported"] is True
    assert published[0][0] == "admitted"
    assert published[0][1]["answer_bridge_admitted"] is True
    assert published[-1][0] == "completed"
    assert published[-1][1]["verdict_sha256"] == "f" * 64


def test_run_stops_cleanly_when_terminal_training_never_earned_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_body = _installed_plan(tmp_path)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    config = _config(tmp_path)
    monkeypatch.setattr(
        replication,
        "_load_plan",
        lambda _args: (tmp_path, config, plan),
    )
    monkeypatch.setattr(
        replication,
        "_verify_launchd_supervision",
        lambda *_args: {"target": "gui/501/test"},
    )
    monkeypatch.setattr(
        replication,
        "_campaign_is_terminal",
        lambda _campaign, _config: (
            True,
            {
                "completion_sha256": "e" * 64,
                "answer_bridge_admitted": False,
                "admission_error": "checkpoint unavailable",
            },
        ),
    )
    monkeypatch.setattr(
        replication,
        "launch_next",
        lambda _args: pytest.fail("an evaluator must not launch without admission"),
    )
    published: list[str] = []

    def publish(*_args: object) -> dict[str, str]:
        published.append(str(_args[4]))
        return {"state": str(_args[4])}

    monkeypatch.setattr(replication, "_publish_controller_status", publish)

    result = replication.run(_arguments(tmp_path))

    assert result["state"] == "not_admitted"
    assert result["supported"] is False
    assert published == ["not_admitted"]


def test_launchd_contract_runs_process_capable_controller_with_failure_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = _config(campaign)
    plan_body = _installed_plan(campaign)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    monkeypatch.setattr(replication, "LAUNCH_AGENTS_ROOT", tmp_path / "agents")

    plist_path, plist_bytes, intent = replication._launch_contract(  # noqa: SLF001
        _arguments(campaign), campaign, config, plan
    )

    import plistlib

    plist = plistlib.loads(plist_bytes)
    assert plist_path.parent == tmp_path / "agents"
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ProgramArguments"][0] == str(Path(sys.executable).absolute())
    assert plist["EnvironmentVariables"] == {"VIRTUAL_ENV": str(Path(sys.prefix).absolute())}
    assert "--launchd-supervised" in plist["ProgramArguments"]
    assert plist["ProgramArguments"][2:4] == ["run", str(campaign)]
    assert intent["plan_sha256"] == plan["plan_sha256"]
    assert intent["controller_source_sha256"] == replication.hashlib.sha256(
        Path(replication.__file__).read_bytes()
    ).hexdigest()
    assert intent["interpreter"]["sys_prefix"] == str(Path(sys.prefix).absolute())


def test_launchd_contract_rejects_base_interpreter_in_place_of_virtualenv(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = _config(campaign)
    config["runtime"]["interpreter"]["executable"] = config["runtime"]["interpreter"][
        "real_executable"
    ]
    plan_body = _installed_plan(campaign)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}

    with pytest.raises(
        replication.ResidentReplicationError,
        match="attested virtualenv python",
    ):
        replication._launch_contract(_arguments(campaign), campaign, config, plan)  # noqa: SLF001


def test_runtime_probe_requires_mlx_and_exact_interpreter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    identity = config["runtime"]["interpreter"]

    class Result:
        returncode = 0
        stderr = ""
        stdout = replication.json.dumps(
            {
                "executable": identity["executable"],
                "prefix": identity["sys_prefix"],
                "mlx_module": "/venv/mlx/core.so",
            }
        )

    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(replication.subprocess, "run", run)

    receipt = replication._probe_runtime_python(config)  # noqa: SLF001

    assert commands[0][0] == identity["executable"]
    assert "import mlx.core as mx" in commands[0][-1]
    assert receipt["interpreter"]["sys_prefix"] == identity["sys_prefix"]
    assert len(receipt["probe_sha256"]) == 64


def test_status_authenticates_controller_heartbeat_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "resident-replication"
    root.mkdir()
    plan_body = _installed_plan(tmp_path)
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    config = _config(tmp_path)
    monkeypatch.setattr(
        replication,
        "_load_plan",
        lambda _args: (tmp_path, config, plan),
    )
    monkeypatch.setattr(replication, "_controller_key", lambda _config: b"k" * 32)
    monkeypatch.setattr(
        replication.launcher.detached,
        "_process_start_token",
        lambda _pid: "process-token",
    )
    monkeypatch.setattr(
        replication.launcher.detached,
        "_identity_state",
        lambda _pid, _token: "alive",
    )
    arguments = _arguments(tmp_path)
    replication._publish_controller_status(
        arguments,
        tmp_path,
        config,
        plan,
        "waiting_for_training",
        {"training_state": "training"},
    )

    observed = replication.status(arguments)

    assert observed["controller"]["state"] == "waiting_for_training"
    assert observed["controller_liveness"] == "alive"

    path = root / "controller-status.json"
    tampered = replication._read_canonical(path)
    tampered["state"] = "completed"
    path.chmod(0o600)
    path.write_bytes(canonical_bytes(tampered) + b"\n")
    with pytest.raises(replication.ResidentReplicationError, match="authentication"):
        replication.status(arguments)
