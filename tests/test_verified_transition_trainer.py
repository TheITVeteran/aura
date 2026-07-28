"""Exclusive mutation and receipt contracts for verified recurrent training."""

from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.grpo import GRPOConfig
from core.learning.grpo_training_state import canonical_json_bytes
from core.learning.recurrence_native_objective_v2 import (
    ExactAdjointTrajectoryConfig,
)
from core.learning.recurrent_grpo import VerifiedTrajectoryGroupConfig
from core.learning.verified_transition_trainer import (
    PreparedVerifiedTransitionGroup,
    VerifiedTransitionMutationResult,
    VerifiedTransitionTelemetry,
    apply_prepared_verified_transition_group,
    build_verified_transition_step_receipt,
    validate_verified_transition_step_receipt,
)
from tools import train_grpo


def _sha(character: str) -> str:
    return character * 64


class _Sample:
    def __init__(self, *, policy: str = _sha("a")) -> None:
        self.policy_sha256 = policy

    def receipt(self) -> dict:
        return {
            "schema": "aura.recurrent_sampling_behavior.v4",
            "episode_id": "verified-transition-trainer-sample",
            "policy_sha256": self.policy_sha256,
        }


def _answer_channel() -> dict:
    return {
        "completions": 2,
        "parseable": 2,
        "unparseable": 0,
        "correct": 1,
        "parseable_fraction": 1.0,
        "correct_fraction": 0.5,
        "grade_reasons": {"correct": 1, "incorrect": 1},
    }


def _updated_mutation() -> VerifiedTransitionMutationResult:
    manifest_sha256 = _sha("d")
    admission_sha256 = _sha("e")
    update_sha256 = _sha("f")
    update = {
        "schema": "aura.verified_transition.update_receipt.v1",
        "optimizer_update_count": 1,
        "group_admission_sha256": admission_sha256,
        "policy_before_sha256": _sha("a"),
        "policy_after_sha256": _sha("b"),
        "receipt_sha256": update_sha256,
    }
    return VerifiedTransitionMutationResult(
        campaign_sequence=0,
        group_manifest_sha256=manifest_sha256,
        optimizer_updated=True,
        structured_rewards=(1.1, -0.2),
        optimizer_admission_reason="admitted",
        reward_receipt_sha256=_sha("c"),
        group_admission_sha256=admission_sha256,
        update_receipt_sha256=update_sha256,
        update_receipt=update,
        terminal_receipt={
            "schema": "aura.verified_transition.campaign_group_terminal.v2",
            "sequence": 0,
            "group_manifest_sha256": manifest_sha256,
            "status": "updated",
            "reward_receipt_sha256": _sha("c"),
            "group_admission_sha256": admission_sha256,
            "update_receipt_sha256": update_sha256,
        },
        policy_before_sha256=_sha("a"),
        policy_after_sha256=_sha("b"),
        replay_group=SimpleNamespace(sequence=0),
    )


def test_verified_step_receipt_replays_signed_delta_rewards() -> None:
    receipt = build_verified_transition_step_receipt(
        step_number=1,
        task_id="task-1",
        sample_seed=17,
        execution_spec_sha256=_sha("d"),
        samples=(_Sample(), _Sample()),
        answer_channel=_answer_channel(),
        mutation=_updated_mutation(),
    )

    validated = validate_verified_transition_step_receipt(
        receipt,
        group_size=2,
        execution_spec_sha256=_sha("d"),
    )

    assert validated["structured_rewards"] == [1.1, -0.2]
    assert validated["step_kind"] == "verified_optimizer_update"


def test_verified_step_receipt_rejects_reward_and_policy_forgery() -> None:
    receipt = build_verified_transition_step_receipt(
        step_number=1,
        task_id="task-1",
        sample_seed=17,
        execution_spec_sha256=_sha("d"),
        samples=(_Sample(), _Sample()),
        answer_channel=_answer_channel(),
        mutation=_updated_mutation(),
    )
    reward_forgery = copy.deepcopy(receipt)
    reward_forgery["structured_rewards"][0] = 99.0
    with pytest.raises(ValueError, match="digest_mismatch"):
        validate_verified_transition_step_receipt(
            reward_forgery,
            group_size=2,
            execution_spec_sha256=_sha("d"),
        )

    policy_forgery = copy.deepcopy(receipt)
    policy_forgery["update"]["policy_after_sha256"] = _sha("e")
    unsigned = dict(policy_forgery)
    unsigned.pop("receipt_sha256")
    policy_forgery["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(ValueError, match="update_invalid"):
        validate_verified_transition_step_receipt(
            policy_forgery,
            group_size=2,
            execution_spec_sha256=_sha("d"),
        )


def test_verified_telemetry_round_trips_negative_rewards() -> None:
    telemetry = VerifiedTransitionTelemetry()
    telemetry.observe(
        {
            "mean_reward": -0.25,
            "degenerate": False,
        },
        optimizer_updated=True,
    )
    restored = VerifiedTransitionTelemetry.from_state(telemetry.state())

    assert restored.reward_sum == -0.25
    assert restored.verdict(GRPOConfig())["learning_signal"] is True


def test_rejected_verified_group_cannot_mutate(monkeypatch) -> None:
    lifecycle: list[str] = []

    class Ledger:
        validated = False
        terminal = None

        def validate_started_group(self, **_kwargs):
            self.validated = True

        def finish_group(self, **kwargs):
            lifecycle.append("campaign_terminal")
            self.terminal = {
                "schema": "aura.verified_transition.campaign_group_terminal.v2",
                "sequence": kwargs["sequence"],
                "group_manifest_sha256": _sha("d"),
                "status": kwargs["status"],
                "reward_receipt_sha256": kwargs["reward_receipt_sha256"],
            }
            return self.terminal

    ledger = Ledger()
    prepared = PreparedVerifiedTransitionGroup(
        campaign_sequence=0,
        transition_store=SimpleNamespace(),
        reward_receipt={},
        transition_evidence=(),
        group_manifest={"manifest_sha256": _sha("d")},
        group_manifest_attestation={},
        independent_scorer=lambda *_args: {},
        token_encoder=lambda value: value,
        token_decoder=lambda value: value,
        campaign_ledger=ledger,
        campaign_trust_policy=SimpleNamespace(policy_sha256=_sha("f")),
        campaign_manifest_sha256=_sha("e"),
        campaign_schedule_root_sha256=_sha("7"),
    )
    reward = {
        "optimizer_admitted": False,
        "optimizer_admission_reason": "right_to_wrong_present",
        "receipt_sha256": _sha("c"),
        "transitions": [{"reward_micros": -1}, {"reward_micros": -2}],
    }
    monkeypatch.setattr(
        "core.learning.verified_transition_trainer.recurrent_policy_sha256",
        lambda *_args: _sha("a"),
    )
    monkeypatch.setattr(
        "core.learning.verified_transition_trainer.validate_verified_transition_reward_batch",
        lambda *_args, **_kwargs: reward,
    )

    class Optimizer:
        def update(self, *_args):
            raise AssertionError("rejected group reached optimizer")

    class RejectionCoordinator:
        def stage_rejection(self, *, policy_sha256):
            assert policy_sha256 == _sha("a")
            lifecycle.append("rejection_intent")

        def record_campaign_terminal(self, receipt):
            assert receipt is ledger.terminal
            lifecycle.append("transaction_terminal")

    with pytest.raises(ValueError, match="rejection_transaction_required"):
        apply_prepared_verified_transition_group(
            object(),
            Optimizer(),
            (1, 2),
            (_Sample(), _Sample()),
            prepared,
            spec=SimpleNamespace(sha256=_sha("d")),
        )
    assert ledger.terminal is None
    assert lifecycle == []

    result = apply_prepared_verified_transition_group(
        object(),
        Optimizer(),
        (1, 2),
        (_Sample(), _Sample()),
        prepared,
        spec=SimpleNamespace(sha256=_sha("d")),
        rejection_transaction_coordinator=RejectionCoordinator(),
    )

    assert result.optimizer_updated is False
    assert result.policy_before_sha256 == result.policy_after_sha256
    assert ledger.validated is True
    assert ledger.terminal["status"] == "rejected"
    assert lifecycle == [
        "rejection_intent",
        "campaign_terminal",
        "transaction_terminal",
    ]


def test_recurrent_cli_refuses_before_model_load_without_verified_provider(
    tmp_path, monkeypatch
) -> None:
    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        adaptive_halting=False,
        latent_opt_mode="disabled",
        fast_weights_mode="disabled",
        decode_bridge_policy="none",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(__import__("json").dumps(spec.to_dict()), encoding="ascii")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_grpo.py",
            "--model",
            str(tmp_path / "missing-model"),
            "--out-dir",
            str(tmp_path / "out"),
            "--execution-mode",
            "recurrent",
            "--execution-spec",
            str(spec_path),
            "--temperature",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        train_grpo.main()

    assert exc.value.code == 2


def test_recurrent_main_has_no_raw_exact_adjoint_mutation_path() -> None:
    source = inspect.getsource(train_grpo.main)

    assert "exact_adjoint_sampled_group_value_and_grad" not in source
    assert "apply_prepared_verified_transition_group" in source


def test_verified_trajectory_config_loader_requires_canonical_policy(
    tmp_path,
) -> None:
    config = VerifiedTrajectoryGroupConfig(
        trajectory_config=ExactAdjointTrajectoryConfig(
            probe_steps=(1, 2),
            improvement_weight=0.5,
            displacement_weight=0.25,
            oscillation_weight=0.1,
        ),
        diversity_weight=0.2,
    )
    path = tmp_path / "trajectory.json"
    path.write_bytes(canonical_json_bytes(config.to_dict()))

    loaded = train_grpo._load_verified_trajectory_group_config(
        "recurrent",
        str(path),
    )

    assert loaded == config
    path.write_bytes(canonical_json_bytes(config.to_dict()) + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        train_grpo._load_verified_trajectory_group_config(
            "recurrent",
            str(path),
        )
    with pytest.raises(ValueError, match="only applies"):
        train_grpo._load_verified_trajectory_group_config(
            "standard",
            str(path),
        )
