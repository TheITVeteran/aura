#!/usr/bin/env python3
"""Verifier-driven RL on the resident cortex (CP233).

The training loop for CP229. Anima Rationis line 511 records the existence
proof: QwQ-32B reached DeepSeek-R1-comparable reasoning through RL over a
32B foundation with correctness verifiers for mathematics and execution
feedback for code -- the same parameter class as Aura's cortex.

Per step:

    1. sample K completions for one prompt at temperature
    2. grade each with a PROGRAM (never the model's own opinion)
    3. advantage_i = (r_i - mean r) / std r      -- the group is the baseline
    4. loss = -mean(advantage_i * logprob_i) + beta * KL(policy || reference)

The reference policy is this same model with the adapter scope disabled,
so the KL leash is measured against the true pre-RL behaviour rather than
a stale copy -- and it costs no extra memory, which matters on a host that
has already been taken down once by an unbounded run.

What this run refuses to do:

* **Report a loss curve as progress.** If every completion in a group earns
  the same grade, the advantages are all zero and the step taught nothing.
  Those groups are counted, and a run made mostly of them is declared to
  have no learning signal regardless of how tidy its loss looks.
* **Score itself on its training set.** Held-out tasks come from a
  separate seed with proven-disjoint prompts, and the verdict is the
  held-out number.
* **Claim a gain from format compliance.** Reward is correctness; format
  credit is capped, because formatting is far easier to learn than
  reasoning and a model that learns it looks like it is improving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import signal
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.learning.grpo import (  # noqa: E402
    GRPOConfig,
    GRPOTelemetry,
    group_advantages,
    grpo_loss,
    reward_from_verdict,
    sequence_token_logprobs,
    step_scores_from_ce,
    trajectory_shaped_rewards,
)
from core.learning.grpo_training_state import (  # noqa: E402
    GRPOCheckpointError,
    canonical_json_bytes,
    load_grpo_checkpoint,
    save_grpo_checkpoint,
    sha256_bytes,
)
from core.learning.recurrent_grpo_artifact_schema import (  # noqa: E402
    PROTOCOL_SCHEMA as GRPO_PROTOCOL_SCHEMA,
)
from core.learning.recurrent_grpo_artifact_schema import (  # noqa: E402  # noqa: E402
    STEP_RECEIPT_SCHEMA,
    protocol_semantic_sha256,
    recurrent_training_adequacy_report,
    validate_step_reward_channels,
)
from core.learning.recurrent_grpo_artifact_schema import (  # noqa: E402
    TRAINING_RECEIPT_SCHEMA as GRPO_TRAIN_SCHEMA,
)
from core.learning.verifiable_tasks import (  # noqa: E402
    disjoint_split,
    scaling_report,
)
from core.learning.verified_transition_measurement_chain import (  # noqa: E402
    VerifiedTransitionMeasurementChainStore,
)
from core.learning.verified_transition_rejection_transaction import (  # noqa: E402
    VerifiedTransitionRejectionTransactionCoordinator,
    VerifiedTransitionRejectionTransactionStore,
    build_rejected_transaction_trainer_step,
)
from core.learning.verified_transition_trainer import (  # noqa: E402
    VERIFIED_TRANSITION_STEP_SCHEMA,
    VerifiedTransitionCampaignClosure,
    VerifiedTransitionGroupProvider,
    VerifiedTransitionGroupProviderFactory,
    VerifiedTransitionProviderRuntime,
    VerifiedTransitionTelemetry,
    apply_prepared_verified_transition_group,
    build_verified_transition_step_receipt,
    build_verified_transition_step_static,
    validate_verified_transition_step_receipt,
)
from core.learning.verified_transition_training_evidence import (  # noqa: E402
    VerifiedTransitionReplayGroup,
)
from core.learning.verified_transition_transaction import (  # noqa: E402
    VerifiedTransitionTransactionCoordinator,
    VerifiedTransitionTransactionStore,
    build_transaction_trainer_step,
    load_trainer_checkpoint_evidence,
)
from core.learning.verified_transition_update import (  # noqa: E402
    VerifiedTransitionUpdateJournal,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402

GRPO_DATASET_SCHEMA = "aura.grpo_dataset.v1"
RNG_STRATEGY = "stateless_sha256_step_seeded_v1"
EXECUTION_MODES = ("standard", "recurrent")
TASK_SOURCES = ("verifiable", "recurrence_curriculum", "answer_channel_curriculum")
_ADAPTER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PHASE_PROGRESS_SCHEMA = "aura.grpo.phase_progress.v1"
_PHASE_PROGRESS_MAX_BYTES = 16 * 1024 * 1024


# Set by main() from --cot. Reasoning room is the fix the CP238 finding
# pointed at: the model failed program_trace at 0.05 because the terse
# FINAL_ANSWER format denied it chain-of-thought. This invites the
# token-level deliberation that actually makes models reason.
_COT_PREAMBLE = ""


def _load_phase_progress(
    path: Path,
    *,
    phase: str,
    identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Load one canonical, identity-bound pretraining phase journal."""

    if not path.exists():
        return [], False
    try:
        raw = read_stable_bytes(path, max_bytes=_PHASE_PROGRESS_MAX_BYTES)
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise GRPOCheckpointError(f"{phase} progress journal is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "phase", "identity", "records", "complete"}
        or payload.get("schema") != _PHASE_PROGRESS_SCHEMA
        or payload.get("phase") != phase
        or payload.get("identity") != dict(identity)
        or type(payload.get("complete")) is not bool
        or not isinstance(payload.get("records"), list)
        or any(not isinstance(record, dict) for record in payload["records"])
    ):
        raise GRPOCheckpointError(f"{phase} progress journal identity differs")
    if canonical_json_bytes(payload) != raw:
        raise GRPOCheckpointError(f"{phase} progress journal is not canonical")
    return [dict(record) for record in payload["records"]], bool(payload["complete"])


def _write_phase_progress(
    path: Path,
    *,
    phase: str,
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    complete: bool,
) -> None:
    payload = {
        "schema": _PHASE_PROGRESS_SCHEMA,
        "phase": phase,
        "identity": dict(identity),
        "records": [dict(record) for record in records],
        "complete": bool(complete),
    }
    atomic_write_bytes(path, canonical_json_bytes(payload), mode=0o600)


def _validate_calibration_progress_record(record: Mapping[str, Any]) -> None:
    if (
        set(record)
        != {
            "family",
            "difficulty",
            "probe_index",
            "task_id",
            "seed",
            "pass_rate",
            "answer_channel",
            "elapsed_s",
        }
        or not isinstance(record.get("family"), str)
        or type(record.get("difficulty")) is not int
        or type(record.get("probe_index")) is not int
        or not isinstance(record.get("task_id"), str)
        or type(record.get("seed")) is not int
        or isinstance(record.get("pass_rate"), bool)
        or not isinstance(record.get("pass_rate"), (int, float))
        or not 0.0 <= float(record["pass_rate"]) <= 1.0
        or not isinstance(record.get("answer_channel"), dict)
        or isinstance(record.get("elapsed_s"), bool)
        or not isinstance(record.get("elapsed_s"), (int, float))
        or not math.isfinite(float(record["elapsed_s"]))
        or float(record["elapsed_s"]) < 0.0
    ):
        raise GRPOCheckpointError("recurrent calibration progress record is invalid")


@dataclass(slots=True)
class _CalibrationProgressJournal:
    path: Path
    identity: dict[str, Any]
    records: list[dict[str, Any]]
    prior_complete: bool
    saved_count: int
    deadline: float
    replayed: int = 0

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        identity: Mapping[str, Any],
        budget_s: float,
    ) -> _CalibrationProgressJournal:
        if not math.isfinite(float(budget_s)) or float(budget_s) <= 0.0:
            raise ValueError("calibration progress budget must be finite and positive")
        records, complete = _load_phase_progress(
            path,
            phase="recurrent_calibration",
            identity=identity,
        )
        for record in records:
            _validate_calibration_progress_record(record)
        spent_s = sum(float(record["elapsed_s"]) for record in records)
        remaining_s = max(0.0, float(budget_s) - spent_s)
        return cls(
            path=path,
            identity=dict(identity),
            records=records,
            prior_complete=complete,
            saved_count=len(records),
            deadline=time.monotonic() + remaining_s,
        )

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def replay(
        self,
        *,
        family: str,
        difficulty: int,
        probe_index: int,
        task_id: str,
        seed: int,
    ) -> dict[str, Any] | None:
        if self.replayed >= self.saved_count:
            return None
        record = self.records[self.replayed]
        if (
            record["family"] != family
            or record["difficulty"] != difficulty
            or record["probe_index"] != probe_index
            or record["task_id"] != task_id
            or record["seed"] != seed
        ):
            raise GRPOCheckpointError(
                "recurrent calibration progress is not an exact probe prefix"
            )
        self.replayed += 1
        return dict(record)

    def permits_new_probe(self) -> bool:
        return not self.prior_complete and time.monotonic() < self.deadline

    def append(self, record: Mapping[str, Any]) -> None:
        if self.replayed != self.saved_count:
            raise GRPOCheckpointError(
                "recurrent calibration cannot append before replay completes"
            )
        _validate_calibration_progress_record(record)
        self.records.append(dict(record))
        _write_phase_progress(
            self.path,
            phase="recurrent_calibration",
            identity=self.identity,
            records=self.records,
            complete=False,
        )

    def finish(self) -> None:
        if self.replayed != self.saved_count:
            raise GRPOCheckpointError(
                "recurrent calibration progress exceeds the expected probe sequence"
            )
        _write_phase_progress(
            self.path,
            phase="recurrent_calibration",
            identity=self.identity,
            records=self.records,
            complete=True,
        )


def _stable_seed(base_seed: int, *parts: Any) -> int:
    """Process-independent seed for one named training decision."""
    payload = canonical_json_bytes([int(base_seed), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _repository_roots(repo_root: Path) -> tuple[Path, ...]:
    roots = [repo_root.resolve()]
    marker = repo_root / ".git"
    try:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                for parent in gitdir.resolve().parents:
                    if parent.name == ".git":
                        roots.append(parent.parent.resolve())
                        break
    except OSError:
        pass
    return tuple(dict.fromkeys(roots))


def _resolve_model_path(value: str) -> Path:
    """Resolve a model directly or from this worktree's authenticated main checkout."""

    supplied = Path(value).expanduser()
    try:
        return supplied.resolve(strict=True)
    except OSError as direct_error:
        pure = PurePosixPath(value)
        if (
            pure.is_absolute()
            or str(pure) != value
            or ".." in pure.parts
            or "\\" in value
            or "\x00" in value
        ):
            raise direct_error
        for root in _repository_roots(REPO_ROOT):
            candidate = root / pure
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            return resolved
        raise direct_error


def _source_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"training source changed while hashing: {resolved}")
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _training_source_files(
    task_source_path: Path,
    *,
    recurrent: bool,
) -> dict[str, Path]:
    source_files = {
        "trainer": Path(__file__),
        "grpo": REPO_ROOT / "core/learning/grpo.py",
        "curriculum": REPO_ROOT / "core/learning/adaptive_curriculum.py",
        "tasks": task_source_path,
        "checkpoint": REPO_ROOT / "core/learning/grpo_training_state.py",
        "artifact_schema": REPO_ROOT / "core/learning/recurrent_grpo_artifact_schema.py",
        "adapter": REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py",
    }
    if not recurrent:
        return source_files

    source_files.update(
        {
            "recurrent_grpo": REPO_ROOT / "core/learning/recurrent_grpo.py",
            "recurrent_objective": (REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"),
            "execution_spec": REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py",
            "latent_engine": REPO_ROOT / "core/brain/llm/latent_cortex/engine.py",
            "recurrence": REPO_ROOT / "core/brain/llm/latent_cortex/recurrence.py",
            "verified_trainer": REPO_ROOT / "core/learning/verified_transition_trainer.py",
            "transition_campaign": (REPO_ROOT / "core/learning/verified_transition_campaign.py"),
            "transition_episode": (REPO_ROOT / "core/learning/verified_transition_episode.py"),
            "transition_reward": REPO_ROOT / "core/learning/verified_transition_reward.py",
            "scope_reachability": REPO_ROOT / "core/learning/scope_reachability.py",
            "transition_admission": (
                REPO_ROOT / "core/learning/verified_transition_group_admission.py"
            ),
            "transition_update": REPO_ROOT / "core/learning/verified_transition_update.py",
            "transition_training_evidence": (
                REPO_ROOT / "core/learning/verified_transition_training_evidence.py"
            ),
            "campaign_trust": (REPO_ROOT / "core/brain/llm/latent_cortex/campaign_trust.py"),
            "transition_provider": (REPO_ROOT / "core/learning/verified_transition_provider.py"),
            "transition_provider_factory": (
                REPO_ROOT / "core/learning/verified_transition_production_factory.py"
            ),
            "transition_launch_bundle": (
                REPO_ROOT / "core/learning/verified_transition_launch_bundle.py"
            ),
            "transition_launch_runner": (
                REPO_ROOT / "tools/run_verified_recurrent_grpo_training.py"
            ),
            "transition_launch_materializer": (
                REPO_ROOT / "tools/materialize_verified_recurrent_grpo_launch.py"
            ),
            "transition_recurrent_evidence": (
                REPO_ROOT / "core/learning/verified_recurrent_transition_evidence.py"
            ),
            "transition_recurrent_repository": (
                REPO_ROOT / "core/learning/verified_recurrent_transition_repository.py"
            ),
            "transition_policy_probe": (
                REPO_ROOT / "core/learning/verified_transition_policy_probe.py"
            ),
            "transition_measurement_chain": (
                REPO_ROOT / "core/learning/verified_transition_measurement_chain.py"
            ),
            "transition_policy_state_replay": (
                REPO_ROOT / "core/learning/verified_transition_policy_state_replay.py"
            ),
            "transition_policy_state_replay_worker": (
                REPO_ROOT / "tools/replay_verified_recurrent_policy_states.py"
            ),
            "transition_policy_state_replay_resume": (
                REPO_ROOT / "tools/resume_durable_external_verifier_job.py"
            ),
            "durable_external_verifier_job": (
                REPO_ROOT / "core/learning/durable_external_verifier_job.py"
            ),
            "recurrent_training_prompt": (REPO_ROOT / "core/learning/recurrent_training_prompt.py"),
            "atomic_writer": REPO_ROOT / "core/runtime/atomic_writer.py",
            "file_read_gateway": REPO_ROOT / "core/runtime/file_read_gateway.py",
            "file_write_gateway": REPO_ROOT / "core/runtime/file_write_gateway.py",
            "transition_transaction": (
                REPO_ROOT / "core/learning/verified_transition_transaction.py"
            ),
            "transition_rejection_transaction": (
                REPO_ROOT / "core/learning/verified_transition_rejection_transaction.py"
            ),
            "transition_causal_campaign": (
                REPO_ROOT / "core/learning/verified_transition_causal_campaign.py"
            ),
            "verified_training_task": (REPO_ROOT / "core/learning/verified_training_task.py"),
            "verified_token_trace": REPO_ROOT / "core/learning/verified_token_trace.py",
        }
    )

    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        REQUIRED_SOURCE_ROLES,
    )

    actual_roles = set(source_files)
    if actual_roles != REQUIRED_SOURCE_ROLES:
        missing = sorted(REQUIRED_SOURCE_ROLES - actual_roles)
        extra = sorted(actual_roles - REQUIRED_SOURCE_ROLES)
        raise GRPOCheckpointError(
            "recurrent GRPO source inventory differs from its identity contract: "
            f"missing={missing}, extra={extra}"
        )
    return source_files


def _task_record(task: Any) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "domain": task.domain,
        "depth": task.depth,
        "knowledge": task.knowledge,
        "grader": task.grader,
        "expected": task.expected,
        "metadata": task.metadata,
    }


def _dataset_payload(
    train_tasks: Sequence[Any], holdout_tasks: Sequence[Any], *, seed: int
) -> dict[str, Any]:
    return {
        "schema": GRPO_DATASET_SCHEMA,
        "seed": int(seed),
        "train": [_task_record(task) for task in train_tasks],
        "holdout": [_task_record(task) for task in holdout_tasks],
    }


def _assert_exact_adapter_keys(expected: Mapping[str, Any], loaded: Mapping[str, Any]) -> None:
    expected_keys = set(expected)
    loaded_keys = set(loaded)
    if loaded_keys == expected_keys:
        return
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    raise GRPOCheckpointError(
        f"checkpoint adapter keyset differs (missing={missing[:5]}, unexpected={unexpected[:5]})"
    )


def _assert_exact_tensor_layout(
    expected: Mapping[str, Any], loaded: Mapping[str, Any], *, role: str
) -> None:
    expected_keys = set(expected)
    loaded_keys = set(loaded)
    if loaded_keys != expected_keys:
        missing = sorted(expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - expected_keys)
        raise GRPOCheckpointError(
            f"staged {role} keyset differs (missing={missing[:5]}, unexpected={unexpected[:5]})"
        )
    for key in sorted(expected_keys):
        expected_value = expected[key]
        loaded_value = loaded[key]
        expected_shape = tuple(int(size) for size in expected_value.shape)
        loaded_shape = tuple(int(size) for size in loaded_value.shape)
        if expected_shape != loaded_shape or str(expected_value.dtype) != str(loaded_value.dtype):
            raise GRPOCheckpointError(f"staged {role} tensor layout differs at {key}")


def _point_estimate_delta(
    baseline: Mapping[str, Any] | None, final: Mapping[str, Any] | None
) -> float | None:
    if baseline is None or final is None:
        return None
    return round(float(final["overall"]) - float(baseline["overall"]), 6)


def _should_halt_for_no_learning_signal(
    telemetry: GRPOTelemetry,
    config: GRPOConfig,
    *,
    min_groups: int,
    step_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the no-signal verdict once RL has enough evidence to stop."""

    if type(min_groups) is not int or min_groups < 1:
        raise ValueError("min_groups must be positive")
    if telemetry.groups < min_groups:
        return None
    verdict = telemetry.verdict(config)
    if verdict.get("learning_signal") is False:
        return _signal_admission_report(verdict, step_receipts=step_receipts)
    return None


def _answer_channel_report_from_verdicts(
    verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize whether the trainer is grading reasoning or answer-channel failure."""

    completions = len(verdicts)
    if completions == 0:
        return {
            "completions": 0,
            "parseable": 0,
            "unparseable": 0,
            "correct": 0,
            "parseable_fraction": 0.0,
            "correct_fraction": 0.0,
            "grade_reasons": {},
        }
    reasons = Counter(_grade_reason(verdict) for verdict in verdicts)
    parseable = sum(1 for verdict in verdicts if verdict.get("parsed") is not None)
    correct = sum(1 for verdict in verdicts if bool(verdict.get("correct")))
    return {
        "completions": completions,
        "parseable": parseable,
        "unparseable": completions - parseable,
        "correct": correct,
        "parseable_fraction": round(parseable / completions, 4),
        "correct_fraction": round(correct / completions, 4),
        "grade_reasons": dict(sorted(reasons.items())),
    }


def _calibration_reward_observations(
    answer_channel: Mapping[str, Any],
) -> tuple[float, ...]:
    """Reconstruct the independently graded outcomes in one calibration group."""

    completions = answer_channel.get("completions")
    correct = answer_channel.get("correct")
    if (
        type(completions) is not int
        or type(correct) is not int
        or completions < 1
        or correct < 0
        or correct > completions
    ):
        raise GRPOCheckpointError("calibration answer-channel counts are invalid")
    return (1.0,) * correct + (0.0,) * (completions - correct)


def _merge_answer_channel_reports(
    step_receipts: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    totals = Counter()
    reasons = Counter()
    trajectory_shaped_groups = 0
    degenerate_trajectory_shaped_groups = 0
    for receipt in step_receipts or ():
        channel = receipt.get("answer_channel")
        if isinstance(channel, Mapping):
            for key in ("completions", "parseable", "unparseable", "correct"):
                value = channel.get(key, 0)
                if isinstance(value, int) and value >= 0:
                    totals[key] += value
            raw_reasons = channel.get("grade_reasons", {})
            if isinstance(raw_reasons, Mapping):
                for reason, count in raw_reasons.items():
                    if isinstance(reason, str) and isinstance(count, int) and count > 0:
                        reasons[reason] += count
        advantage = receipt.get("advantage_report")
        if isinstance(advantage, Mapping) and advantage.get("trajectory_shaped"):
            trajectory_shaped_groups += 1
            if advantage.get("degenerate"):
                degenerate_trajectory_shaped_groups += 1
    completions = totals["completions"]
    return {
        "completions": completions,
        "parseable": totals["parseable"],
        "unparseable": totals["unparseable"],
        "correct": totals["correct"],
        "parseable_fraction": round(totals["parseable"] / completions, 4) if completions else 0.0,
        "correct_fraction": round(totals["correct"] / completions, 4) if completions else 0.0,
        "grade_reasons": dict(sorted(reasons.items())),
        "trajectory_shaped_groups": trajectory_shaped_groups,
        "degenerate_trajectory_shaped_groups": degenerate_trajectory_shaped_groups,
    }


def _signal_admission_report(
    verdict: Mapping[str, Any],
    *,
    step_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Receipt-level diagnosis for whether GRPO had a trainable signal.

    GRPO can fail before it ever tests reasoning: the model might not emit a
    parseable answer contract, or trajectory credit might be constant across a
    group. This keeps those separate from true reasoning difficulty.
    """

    report = dict(verdict)
    report["schema"] = "aura.grpo_signal_admission.v1"
    channel = _merge_answer_channel_reports(step_receipts)
    report["answer_channel"] = channel
    if report.get("learning_signal") is not False:
        return report
    completions = int(channel.get("completions") or 0)
    parseable_fraction = float(channel.get("parseable_fraction") or 0.0)
    correct = int(channel.get("correct") or 0)
    if completions and correct == 0 and parseable_fraction < 0.25:
        report["diagnosis"] = (
            "answer_channel_blocked: sampled completions rarely produced a "
            "parseable verifier answer, so GRPO measured contract emission "
            "failure before reasoning correctness"
        )
        report["required_next_gate"] = (
            "repair decode contract/pretraining or run a parseability scaffold "
            "until sampled groups have parseable variance"
        )
    elif channel.get("trajectory_shaped_groups") and channel.get(
        "trajectory_shaped_groups"
    ) == channel.get("degenerate_trajectory_shaped_groups"):
        report["diagnosis"] = (
            "trajectory_credit_constant: recurrent CE shaping was present but "
            "did not distinguish completions, so no preference signal reached "
            "the adapter"
        )
        report["required_next_gate"] = (
            "restore discriminative trajectory credit or fall back to a task "
            "cell with verifier reward variance before launching long training"
        )
    if not isinstance(report.get("diagnosis"), str) or not report["diagnosis"].strip():
        report["diagnosis"] = (
            "no_training_groups_observed: calibration or admission stopped the "
            "run before an optimizer group executed"
        )
        report["required_next_gate"] = (
            "inspect the calibration admission receipt and satisfy its required "
            "next gate before launching resident training"
        )
    return report


def _calibration_admission_report(
    calibration: Mapping[str, Any],
    *,
    allow_unexplored_frontier: bool,
) -> dict[str, Any]:
    """Decide whether calibration found enough signal to spend training budget."""

    learnable = list(calibration.get("learnable") or [])
    unexplored = list(calibration.get("unexplored") or [])
    probes = list(calibration.get("probes") or [])
    answer_channel = calibration.get("answer_channel")
    parseable_fraction = 0.0
    if isinstance(answer_channel, Mapping):
        parseable_fraction = float(answer_channel.get("parseable_fraction") or 0.0)
    if learnable:
        return {
            "schema": "aura.grpo_calibration_admission.v1",
            "training_admitted": True,
            "reason": "measured_learnable_cells",
            "learnable_cells": learnable,
            "allow_unexplored_frontier": bool(allow_unexplored_frontier),
        }
    if allow_unexplored_frontier and unexplored:
        return {
            "schema": "aura.grpo_calibration_admission.v1",
            "training_admitted": True,
            "reason": "unexplored_frontier_allowed",
            "unexplored_cells": unexplored,
            "allow_unexplored_frontier": True,
        }
    if probes and parseable_fraction < 0.25:
        diagnosis = "answer_channel_blocked"
        next_gate = (
            "repair recurrent decode contract or pretrain the answer channel before resident GRPO"
        )
    elif bool(calibration.get("partial")) and unexplored:
        diagnosis = "partial_calibration_without_measured_learnable_cell"
        next_gate = (
            "increase calibration coverage or reduce cell space until at least "
            "one measured cell has reward variance"
        )
    else:
        diagnosis = "no_measured_learnable_cell"
        next_gate = (
            "redesign curriculum difficulty, verifier, or trajectory credit "
            "before launching training"
        )
    return {
        "schema": "aura.grpo_calibration_admission.v1",
        "training_admitted": False,
        "reason": diagnosis,
        "required_next_gate": next_gate,
        "allow_unexplored_frontier": bool(allow_unexplored_frontier),
        "measured_probes": len(probes),
        "parseable_fraction": round(parseable_fraction, 4),
    }


def _calibration_token_budget(max_tokens: int, requested: int) -> int:
    if requested not in (0, max_tokens):
        raise ValueError(
            "calibration tokens must equal training max tokens; a shorter "
            "probe truncates reasoning and corrupts learnability"
        )
    return max_tokens


def _verified_campaign_halt_is_resumable(halt_reason: str) -> bool:
    """Keep the verified task ledger open across bounded exact resumes."""

    return halt_reason in {"wall_clock_budget", "interrupted", "operator_pause"}


def _operator_pause_request(
    out_dir: Path,
    *,
    adapter_id: str,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    path = out_dir / "pause.request.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise GRPOCheckpointError("operator pause request path is invalid")
    try:
        request = json.loads(read_stable_bytes(path, max_bytes=16 * 1024))
    except (OSError, TypeError, ValueError) as exc:
        raise GRPOCheckpointError("operator pause request is invalid") from exc
    if (
        not isinstance(request, dict)
        or set(request)
        != {
            "schema",
            "adapter_id",
            "protocol_sha256",
            "requested_at_unix_ns",
            "request_sha256",
        }
        or request.get("schema") != "aura.grpo.operator_pause_request.v1"
        or request.get("adapter_id") != adapter_id
        or request.get("protocol_sha256") != protocol_sha256
        or type(request.get("requested_at_unix_ns")) is not int
        or int(request["requested_at_unix_ns"]) <= 0
    ):
        raise GRPOCheckpointError("operator pause request is invalid")
    material = dict(request)
    claimed = material.pop("request_sha256")
    if claimed != sha256_bytes(canonical_json_bytes(material)):
        raise GRPOCheckpointError("operator pause request digest is invalid")
    return request


def _acknowledge_operator_pause(
    out_dir: Path,
    *,
    request: Mapping[str, Any],
    step: int,
    checkpoint_path: Path,
    policy_sha256: str | None,
) -> Path:
    body = {
        "schema": "aura.grpo.operator_pause_receipt.v1",
        "request_sha256": request["request_sha256"],
        "adapter_id": request["adapter_id"],
        "protocol_sha256": request["protocol_sha256"],
        "step": step,
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "policy_sha256": policy_sha256,
        "acknowledged_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "receipt_sha256": sha256_bytes(canonical_json_bytes(body))}
    destination = out_dir / "pause-receipts" / f"{request['request_sha256']}.json"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _publish_immutable_bytes(
        destination,
        canonical_json_bytes(receipt),
        role="operator pause receipt",
    )
    request_path = out_dir / "pause.request.json"
    if request_path.exists():
        request_path.unlink()
    return destination


def _task_gold_answer_text(task: Any) -> str:
    """Canonical text whose tokens represent the verifier's correct answer."""

    answer = getattr(task, "answer", None)
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    expected = getattr(task, "expected", None)
    if callable(expected):
        expected = expected()
    if isinstance(expected, (dict, list)):
        return "FINAL_ANSWER: " + json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    if expected is not None:
        return f"FINAL_ANSWER: {expected}"
    raise ValueError("task does not expose a canonical verifier answer")


def _tokenize_gold_answer(tokenizer: Any, task: Any) -> list[int]:
    answer_text = _task_gold_answer_text(task)
    tokens = list(tokenizer.encode(answer_text, add_special_tokens=False))
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("canonical verifier answer produced invalid tokens")
    return tokens


def _shape_recurrent_rewards_from_ce_trails(
    rewards: Sequence[float],
    ce_trails: Sequence[Sequence[float]],
    *,
    shaping_weight: float,
) -> dict[str, Any]:
    """Bounded recurrent credit with verifier rewards preserved separately."""

    score_trails = [step_scores_from_ce(trail) for trail in ce_trails]
    shaped = trajectory_shaped_rewards(
        rewards,
        score_trails,
        shaping_weight=shaping_weight,
    )
    return {
        **shaped,
        "ce_trails": [[round(float(value), 6) for value in trail] for trail in ce_trails],
        "score_trails": [[round(float(value), 6) for value in trail] for trail in score_trails],
    }


def _advantage_report_with_verifier_rate(
    shaped_report: dict[str, Any],
    verifier_report: dict[str, Any],
) -> dict[str, Any]:
    """Use shaped advantages while preserving verifier-rate telemetry."""

    verifier_mean = float(verifier_report["mean_reward"])
    if not 0.0 <= verifier_mean <= 1.0:
        raise ValueError("verifier mean_reward must be a rate in [0, 1]")
    report = dict(shaped_report)
    report["shaped_mean_reward"] = report["mean_reward"]
    report["shaped_reward_std"] = report.get("reward_std")
    report["shaped_degenerate"] = report.get("degenerate")
    report["shaped_all_correct"] = report.get("all_correct")
    report["shaped_all_wrong"] = report.get("all_wrong")
    report["shaped_uniform_partial"] = report.get("uniform_partial")
    report["mean_reward"] = verifier_report["mean_reward"]
    report["verifier_reward_std"] = verifier_report.get("reward_std")
    report["verifier_degenerate"] = verifier_report.get("degenerate")
    if report.get("degenerate"):
        # A degenerate shaped group has no optimizer signal. Telemetry should
        # diagnose the verifier state, not misread a constant CE-shaping offset
        # as format credit or partial correctness.
        report["all_correct"] = verifier_report.get("all_correct")
        report["all_wrong"] = verifier_report.get("all_wrong")
        report["uniform_partial"] = verifier_report.get("uniform_partial")
    report["trajectory_shaped"] = True
    return report


def _build_recurrent_step_receipt(
    *,
    step_number: int,
    task_id: str,
    sample_seed: int,
    execution_spec_sha256: str,
    samples: Sequence[Mapping[str, Any]],
    effective_rewards: Sequence[float],
    verifier_rewards: Sequence[float],
    answer_channel: Mapping[str, Any],
    verifier_advantage_report: Mapping[str, Any],
    trajectory_credit: Mapping[str, Any] | None,
    advantage_report: Mapping[str, Any],
    step_kind: str,
    update: Mapping[str, Any] | None,
    policy_after_sha256: str,
    trajectory_credit_enabled: bool,
    trajectory_shaping_weight: float,
    advantage_clip: float,
) -> dict[str, Any]:
    """Build one producer-format receipt and replay it before persistence."""

    receipt = {
        "schema": STEP_RECEIPT_SCHEMA,
        "step": step_number,
        "task_id": task_id,
        "sample_seed": sample_seed,
        "execution_spec_sha256": execution_spec_sha256,
        "samples": [dict(sample) for sample in samples],
        "rewards": [float(value) for value in effective_rewards],
        "verifier_rewards": [float(value) for value in verifier_rewards],
        "answer_channel": dict(answer_channel),
        "verifier_advantage_report": dict(verifier_advantage_report),
        "trajectory_credit": (dict(trajectory_credit) if trajectory_credit is not None else None),
        "advantage_report": dict(advantage_report),
        "step_kind": step_kind,
        "update": dict(update) if update is not None else None,
        "policy_after_sha256": policy_after_sha256,
    }
    validate_step_reward_channels(
        receipt,
        group_size=len(samples),
        trajectory_credit_enabled=trajectory_credit_enabled,
        shaping_weight=trajectory_shaping_weight,
        advantage_clip=advantage_clip,
    )
    return receipt


def _build_task_split(
    *,
    task_source: str,
    domains: list[str],
    depths: list[int],
    train_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[Any], list[Any], Path]:
    """Build one source-bound split without mixing training registries."""
    if task_source == "verifiable":
        train, holdout = disjoint_split(
            domains=domains,
            depths=depths,
            train_per_cell=train_per_cell,
            holdout_per_cell=holdout_per_cell,
            seed=seed,
        )
        source = REPO_ROOT / "core/learning/verifiable_tasks.py"
    elif task_source == "recurrence_curriculum":
        from core.learning.recurrence_curriculum import disjoint_task_split

        train, holdout = disjoint_task_split(
            families=domains,
            depths=depths,
            train_per_cell=train_per_cell,
            holdout_per_cell=holdout_per_cell,
            seed=seed,
        )
        source = REPO_ROOT / "core/learning/recurrence_curriculum.py"
    elif task_source == "answer_channel_curriculum":
        from core.learning.answer_channel_curriculum import disjoint_task_split

        train, holdout = disjoint_task_split(
            families=domains,
            depths=depths,
            train_per_cell=train_per_cell,
            holdout_per_cell=holdout_per_cell,
            seed=seed,
        )
        source = REPO_ROOT / "core/learning/answer_channel_curriculum.py"
    else:
        raise ValueError(f"unsupported task source: {task_source}")
    return list(train), list(holdout), source


def _publish_adapter_snapshot(path: Path, tensors: Mapping[str, Any]) -> None:
    import mlx.core as mx

    scratch = path.parent / f".{path.stem}.{time.time_ns()}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        atomic_write_bytes(path, scratch.read_bytes(), mode=0o600)
    finally:
        scratch.unlink(missing_ok=True)


def _publish_immutable_tensor_snapshot(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    role: str,
) -> None:
    import mlx.core as mx

    scratch = path.parent / f".{path.stem}.{time.time_ns()}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        _publish_immutable_bytes(
            path,
            scratch.read_bytes(),
            role=role,
        )
    finally:
        scratch.unlink(missing_ok=True)


def _publish_immutable_bytes(path: Path, payload: bytes, *, role: str) -> None:
    if path.is_symlink():
        raise GRPOCheckpointError(f"{role} symlink is forbidden")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise GRPOCheckpointError(f"{role} is unreadable") from exc
        if existing != payload:
            raise GRPOCheckpointError(f"{role} differs from the frozen run")
        return
    if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
        if path.is_symlink() or path.read_bytes() != payload:
            raise GRPOCheckpointError(f"{role} publication raced with different bytes")


def _publish_update_canary_step_timing(
    out_dir: Path,
    *,
    adapter_id: str,
    protocol_sha256: str,
    step_receipt: Mapping[str, Any],
    checkpoint_path: Path,
    started_at_unix_ns: int,
    started_monotonic_ns: int,
    finished_at_unix_ns: int,
    finished_monotonic_ns: int,
) -> Path:
    """Seal end-to-end canary latency after the trainer checkpoint is durable."""

    step = step_receipt.get("step")
    task_id = step_receipt.get("task_id")
    step_receipt_sha256 = step_receipt.get("receipt_sha256")
    if (
        type(step) is not int
        or step <= 0
        or not isinstance(task_id, str)
        or not task_id
        or not isinstance(step_receipt_sha256, str)
        or len(step_receipt_sha256) != 64
        or finished_monotonic_ns < started_monotonic_ns
        or finished_at_unix_ns < started_at_unix_ns
    ):
        raise GRPOCheckpointError("update canary timing inputs are invalid")
    checkpoint_complete = checkpoint_path / "complete.json"
    if checkpoint_complete.is_symlink() or not checkpoint_complete.is_file():
        raise GRPOCheckpointError("update canary checkpoint is not complete")
    checkpoint_payload = checkpoint_complete.read_bytes()
    body = {
        "schema": "aura.recurrent_grpo.update_canary_step_timing.v1",
        "adapter_id": adapter_id,
        "protocol_sha256": protocol_sha256,
        "step": step,
        "task_id": task_id,
        "step_receipt_sha256": step_receipt_sha256,
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "checkpoint_complete_sha256": sha256_bytes(checkpoint_payload),
        "started_at_unix_ns": started_at_unix_ns,
        "finished_at_unix_ns": finished_at_unix_ns,
        "elapsed_monotonic_ns": finished_monotonic_ns - started_monotonic_ns,
        "includes_durable_checkpoint_publication": True,
    }
    receipt = {**body, "receipt_sha256": sha256_bytes(canonical_json_bytes(body))}
    destination = (
        out_dir / "update-canary-step-timings" / f"step-{step:08d}.json"
    )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _publish_immutable_bytes(
        destination,
        canonical_json_bytes(receipt),
        role="update canary step timing",
    )
    return destination


def _artifact_binding(relative: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _read_recurrent_bundle_artifacts(
    out_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, bytes]:
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        declared_bindings,
    )

    root = out_dir.resolve(strict=True)
    artifacts: dict[str, bytes] = {}
    for _role, binding in declared_bindings(manifest):
        path = (root / binding["path"]).resolve(strict=True)
        if path.parent != root and root not in path.parents:
            raise GRPOCheckpointError("recurrent adapter artifact escapes run root")
        artifacts[binding["path"]] = path.read_bytes()
    artifacts["training_completion.json"] = (root / "training_completion.json").read_bytes()
    return artifacts


def _validate_published_recurrent_bundle(
    out_dir: Path,
    *,
    adapter_id: str,
    base_identity: Mapping[str, Any],
    behavior_identity: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    transition_closure: VerifiedTransitionCampaignClosure | None = None,
    transition_groups: Sequence[VerifiedTransitionReplayGroup] = (),
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        MANIFEST_FILE,
        strict_json_loads,
        validate_recurrent_grpo_adapter_identity,
        validate_recurrent_grpo_adapter_identity_with_verified_transitions,
    )

    manifest_bytes = (out_dir / MANIFEST_FILE).read_bytes()
    manifest = strict_json_loads(manifest_bytes, role="published_manifest")
    adapter_path = out_dir / manifest["adapter"]["path"]
    kwargs = {
        "adapter_id": adapter_id,
        "actual_base_checkpoint": base_identity,
        "actual_model_behavior_bundle": behavior_identity,
        "actual_personality_adapter": personality_identity,
        "actual_runtime_environment": runtime_identity,
        "artifacts": _read_recurrent_bundle_artifacts(out_dir, manifest),
        "tensor_metadata": inspect_mlx_tensor_metadata(adapter_path),
    }
    if transition_closure is None:
        if transition_groups:
            raise GRPOCheckpointError(
                "transition replay groups require a verified campaign closure"
            )
        return validate_recurrent_grpo_adapter_identity(manifest_bytes, **kwargs)
    return validate_recurrent_grpo_adapter_identity_with_verified_transitions(
        manifest_bytes,
        **kwargs,
        transition_campaign_ledger=transition_closure.campaign_ledger,
        transition_policy=transition_closure.campaign_trust_policy,
        transition_groups=transition_groups,
    )


def _publish_recurrent_adapter_bundle(
    out_dir: Path,
    *,
    adapter_id: str,
    protocol: Mapping[str, Any],
    protocol_bytes: bytes,
    dataset_bytes: bytes,
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    execution_spec: Any,
    source_roles: Mapping[str, Path],
    transition_closure: VerifiedTransitionCampaignClosure | None = None,
    transition_groups: Sequence[VerifiedTransitionReplayGroup] = (),
) -> dict[str, Any]:
    """Publish and immediately revalidate a campaign-loadable GRPO identity."""

    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        COMPLETION_SCHEMA,
        LOADER_CONFIG_SCHEMA,
        MANIFEST_FILE,
        MANIFEST_SCHEMA,
        TRAINING_METHOD,
        declared_bindings,
        required_source_roles,
        validate_recurrent_grpo_adapter_identity,
        validate_recurrent_grpo_adapter_identity_with_verified_transitions,
    )

    completion_path = out_dir / "training_completion.json"
    if completion_path.exists():
        return _validate_published_recurrent_bundle(
            out_dir,
            adapter_id=adapter_id,
            base_identity=protocol["base_checkpoint"],
            behavior_identity=protocol["model_behavior"],
            personality_identity=protocol["personality_adapter"],
            runtime_identity=protocol["runtime"],
            transition_closure=transition_closure,
            transition_groups=transition_groups,
        )
    expected_source_roles = required_source_roles(str(protocol.get("schema")))
    if set(source_roles) != expected_source_roles:
        raise GRPOCheckpointError("recurrent GRPO source inventory is incomplete")
    if receipt.get("execution_mode") != "recurrent":
        raise GRPOCheckpointError("only recurrent GRPO can publish this identity")
    termination = receipt.get("termination")
    if (
        not isinstance(termination, Mapping)
        or termination.get("reason") != "max_steps"
        or termination.get("completed_budget") is not True
        or termination.get("signal") is not None
    ):
        raise GRPOCheckpointError("recurrent GRPO training is not complete")

    campaign_dir = ensure_private_directory(out_dir / "campaign_adapter")
    source_adapter = out_dir / "grpo_adapters.safetensors"
    adapter_bytes = source_adapter.read_bytes()
    documents = {
        "campaign_adapter/adapters.safetensors": adapter_bytes,
        "campaign_adapter/adapter_final.safetensors": adapter_bytes,
        "campaign_adapter/grpo_receipt.json": receipt_bytes,
        "campaign_adapter/training_protocol.json": protocol_bytes,
        "campaign_adapter/dataset_manifest.json": dataset_bytes,
        "campaign_adapter/execution_spec.json": canonical_json_bytes(execution_spec.to_dict()),
    }
    for relative, payload in documents.items():
        _publish_immutable_bytes(
            out_dir / relative,
            payload,
            role=relative.replace("/", " "),
        )

    tensor_metadata = inspect_mlx_tensor_metadata(campaign_dir / "adapters.safetensors")
    tensor_records = [record.to_dict() for record in tensor_metadata]
    projection_paths = sorted(
        {record["key"].removesuffix(".lora_a").removesuffix(".lora_b") for record in tensor_records}
    )
    targets = [part.strip() for part in protocol["training"]["lora_targets"].split(",")]
    trainable_params = sum(math.prod(record["shape"]) for record in tensor_records)
    unique_layers = {int(path.split(".")[2]) for path in projection_paths}
    loader_config = {
        "schema": LOADER_CONFIG_SCHEMA,
        "fine_tune_type": "recurrent_grpo_scoped_lora",
        "loader": "aura_custom_loader_required",
        "model": protocol["model_path"],
        "num_layers": len(unique_layers),
        "wrapped_projection_count": len(projection_paths),
        "lora_parameters": {
            "rank": protocol["training"]["lora_rank"],
            "scale": 20.0,
            "dropout": 0.0,
            "keys": targets,
        },
        "execution_spec_sha256": execution_spec.sha256,
        "training_method": TRAINING_METHOD,
    }
    loader_bytes = canonical_json_bytes(loader_config)
    _publish_immutable_bytes(
        campaign_dir / "adapter_config.json",
        loader_bytes,
        role="campaign adapter loader config",
    )

    sources: dict[str, dict[str, Any]] = {}
    for role in sorted(expected_source_roles):
        protocol_binding = protocol["sources"][role]
        snapshot_relative = f"source_snapshots/{role}.py"
        snapshot_bytes = (out_dir / snapshot_relative).read_bytes()
        if (
            len(snapshot_bytes) != protocol_binding["size_bytes"]
            or sha256_bytes(snapshot_bytes) != protocol_binding["sha256"]
        ):
            raise GRPOCheckpointError(f"frozen recurrent source differs: {role}")
        sources[role] = {
            "origin_path": protocol_binding["path"],
            "snapshot_path": snapshot_relative,
            "sha256": protocol_binding["sha256"],
            "size_bytes": protocol_binding["size_bytes"],
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "adapter_id": adapter_id,
        "training_method": TRAINING_METHOD,
        "base_checkpoint": protocol["base_checkpoint"],
        "model_behavior_bundle": protocol["model_behavior"],
        "personality_adapter": protocol["personality_adapter"],
        "training_runtime": protocol["runtime"],
        "adapter": _artifact_binding("campaign_adapter/adapters.safetensors", adapter_bytes),
        "adapter_alias": _artifact_binding(
            "campaign_adapter/adapter_final.safetensors", adapter_bytes
        ),
        "loader_config": _artifact_binding("campaign_adapter/adapter_config.json", loader_bytes),
        "training_receipt": _artifact_binding("campaign_adapter/grpo_receipt.json", receipt_bytes),
        "training_protocol": _artifact_binding(
            "campaign_adapter/training_protocol.json", protocol_bytes
        ),
        "dataset_manifest": _artifact_binding(
            "campaign_adapter/dataset_manifest.json", dataset_bytes
        ),
        "execution_spec": _artifact_binding(
            "campaign_adapter/execution_spec.json",
            documents["campaign_adapter/execution_spec.json"],
        ),
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "dataset_sha256": sha256_bytes(dataset_bytes),
        "execution_spec_sha256": execution_spec.sha256,
        "sources": sources,
        "lora": {
            "rank": protocol["training"]["lora_rank"],
            "targets": targets,
            "wrapped_projections": len(projection_paths),
            "projection_paths": projection_paths,
            "trainable_params": trainable_params,
        },
        "tensors": tensor_records,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _publish_immutable_bytes(
        out_dir / MANIFEST_FILE,
        manifest_bytes,
        role="recurrent GRPO adapter manifest",
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "step": receipt["steps"],
        "optimizer_updates": receipt["optimizer_updates"],
        "adapter_sha256": manifest["adapter"]["sha256"],
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "execution_spec_sha256": execution_spec.sha256,
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }
    completion_bytes = canonical_json_bytes(completion)
    preflight_artifacts: dict[str, bytes] = {}
    for _role, binding in declared_bindings(manifest):
        preflight_artifacts[binding["path"]] = (out_dir / binding["path"]).read_bytes()
    preflight_artifacts["training_completion.json"] = completion_bytes
    identity_kwargs = {
        "adapter_id": adapter_id,
        "actual_base_checkpoint": protocol["base_checkpoint"],
        "actual_model_behavior_bundle": protocol["model_behavior"],
        "actual_personality_adapter": protocol["personality_adapter"],
        "actual_runtime_environment": protocol["runtime"],
        "artifacts": preflight_artifacts,
        "tensor_metadata": tensor_metadata,
    }
    if transition_closure is None:
        if transition_groups:
            raise GRPOCheckpointError(
                "transition replay groups require a verified campaign closure"
            )
        preflight_identity = validate_recurrent_grpo_adapter_identity(
            manifest_bytes, **identity_kwargs
        )
    else:
        preflight_identity = validate_recurrent_grpo_adapter_identity_with_verified_transitions(
            manifest_bytes,
            **identity_kwargs,
            transition_campaign_ledger=transition_closure.campaign_ledger,
            transition_policy=transition_closure.campaign_trust_policy,
            transition_groups=transition_groups,
        )
    _publish_immutable_bytes(
        completion_path,
        completion_bytes,
        role="recurrent GRPO training completion",
    )
    published_identity = _validate_published_recurrent_bundle(
        out_dir,
        adapter_id=adapter_id,
        base_identity=protocol["base_checkpoint"],
        behavior_identity=protocol["model_behavior"],
        personality_identity=protocol["personality_adapter"],
        runtime_identity=protocol["runtime"],
        transition_closure=transition_closure,
        transition_groups=transition_groups,
    )
    if published_identity != preflight_identity:
        raise GRPOCheckpointError("published recurrent identity differs from preflight")
    return published_identity


def _render(tokenizer, task) -> str:
    from core.learning.recurrent_training_prompt import (
        render_recurrent_training_prompt_text,
    )

    return render_recurrent_training_prompt_text(
        tokenizer,
        task,
        include_chain_of_thought=bool(_COT_PREAMBLE),
    )


def _answer_contract_instruction(task: Any) -> str:
    """Serving-side answer-channel scaffold without leaking answer values."""

    from core.learning.recurrent_training_prompt import answer_contract_instruction

    return answer_contract_instruction(task)


def _load_execution_spec(mode: str, path: str | None):
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {mode}")
    if mode == "standard":
        if path:
            raise ValueError("--execution-spec only applies to recurrent mode")
        return None
    if not path:
        raise ValueError("recurrent mode requires --execution-spec")
    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

    spec_path = Path(path).expanduser().resolve(strict=True)
    if not spec_path.is_file():
        raise ValueError("execution spec must be a regular file")
    try:
        payload = json.loads(spec_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("execution spec is not readable canonical JSON") from exc
    return RLCExecutionSpec.from_dict(payload)


def _load_verified_trajectory_group_config(
    mode: str,
    path: str | None,
):
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {mode}")
    if mode == "standard":
        if path:
            raise ValueError("--verified-trajectory-config only applies to recurrent mode")
        return None
    if not path:
        return None
    from core.learning.recurrent_grpo import VerifiedTrajectoryGroupConfig

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("verified trajectory config cannot be a symlink")
    config_path = source.resolve(strict=True)
    if not config_path.is_file():
        raise ValueError("verified trajectory config must be a regular file")
    try:
        encoded = read_stable_bytes(config_path, max_bytes=65_536)
        payload = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("verified trajectory config is not readable canonical JSON") from exc
    if canonical_json_bytes(payload) != encoded:
        raise ValueError("verified trajectory config JSON is not canonical")
    return VerifiedTrajectoryGroupConfig.from_dict(payload)


def _rendered_task_prompt(tokenizer, task) -> tuple[str, list[int]]:
    from core.learning.recurrent_training_prompt import (
        render_recurrent_training_prompt,
    )

    rendered, tokens = render_recurrent_training_prompt(
        tokenizer,
        task,
        include_chain_of_thought=bool(_COT_PREAMBLE),
    )
    return rendered, list(tokens)


def _task_prompt_tokens(tokenizer, task) -> list[int]:
    return _rendered_task_prompt(tokenizer, task)[1]


def _scheduled_verified_training_task(
    provider: VerifiedTransitionGroupProvider,
    tasks_by_id: Mapping[str, Any],
    *,
    campaign_sequence: int,
) -> tuple[Any, int]:
    """Resolve one task only from the provider's frozen signed schedule."""

    scheduled = provider.training_schedule_entry(sequence=campaign_sequence)
    if scheduled.campaign_sequence != campaign_sequence:
        raise RuntimeError("verified provider returned a different schedule sequence")
    task = tasks_by_id.get(scheduled.task_id)
    if task is None:
        raise RuntimeError("verified provider scheduled a task outside the frozen dataset")
    return task, scheduled.trainer_sample_seed


def _grade_reason(verdict: Mapping[str, Any]) -> str:
    if isinstance(verdict.get("reason"), str) and verdict["reason"]:
        return str(verdict["reason"])
    return "correct" if bool(verdict.get("correct")) else "incorrect"


def sample_recurrent_group(
    model,
    tokenizer,
    task,
    *,
    spec,
    size: int,
    max_tokens: int,
    seed: int,
    sampling_config: Any | None = None,
    verified_group_provider: Any | None = None,
    campaign_sequence: int | None = None,
    model_path: str | None = None,
    token_trace_adapter: Any | None = None,
):
    """Bounded behavior-policy completions from the fixed recurrent graph."""

    from core.learning.recurrent_grpo import (
        RecurrentSamplingAdmissionError,
        RecurrentSamplingConfig,
        recurrent_policy_sha256,
        sample_final_recurrent_transition_completion,
        sample_recurrent_completion,
        validate_recurrent_policy_sample_receipt,
    )
    from core.learning.verified_transition_group_admission import (
        sampling_config_sha256,
    )

    _prompt_text, prompt_tokens = _rendered_task_prompt(tokenizer, task)
    requested_sampling = sampling_config
    sampling = sampling_config or RecurrentSamplingConfig(max_tokens=max_tokens)
    if not isinstance(sampling, RecurrentSamplingConfig):
        raise TypeError("sampling_config must be a RecurrentSamplingConfig")
    if sampling.max_tokens != max_tokens:
        raise ValueError("sampling_config max_tokens must match max_tokens")
    if (verified_group_provider is None) is not (campaign_sequence is None):
        raise ValueError("verified_group_provider and campaign_sequence must be supplied together")
    if verified_group_provider is not None:
        if token_trace_adapter is None:
            raise RuntimeError(
                "verified recurrent sampling requires a bound tokenizer trace adapter"
            )
        policy_sha256 = recurrent_policy_sha256(model, spec)
        plan = verified_group_provider.sampling_plan(
            sequence=campaign_sequence,
            task=task,
            prompt_tokens=prompt_tokens,
            policy_sha256=policy_sha256,
        )
        if not isinstance(plan.sampling_config, Mapping) or not plan.sampling_config:
            raise RuntimeError("verified sampling plan omitted its frozen sampling configuration")
        planned_sampling = RecurrentSamplingConfig(**dict(plan.sampling_config))
        if planned_sampling.max_tokens != max_tokens:
            raise RuntimeError("verified sampling plan token budget differs from trainer request")
        if (
            requested_sampling is not None
            and requested_sampling.to_dict() != planned_sampling.to_dict()
        ):
            raise RuntimeError("caller sampling configuration differs from verified plan")
        sampling = planned_sampling
        entries = tuple(plan.entries)
        if (
            plan.campaign_sequence != campaign_sequence
            or plan.task_id != getattr(task, "task_id", None)
            or plan.policy_sha256 != policy_sha256
            or plan.execution_spec_sha256 != spec.sha256
            or len(entries) != size
        ):
            raise RuntimeError("verified sampling plan differs from trainer request")
        samples = []
        completions: list[str] = []
        for entry in entries:
            sample = sample_final_recurrent_transition_completion(
                model,
                prompt_tokens,
                spec=spec,
                branch_index=entry.producing_branch_index,
                seed=entry.sample_seed,
                episode_id=entry.episode_id,
                sampling=sampling,
                tokenizer=tokenizer,
                model_path=model_path,
            )
            if (
                sample.episode_id != entry.episode_id
                or sample.rng_root_sha256 != entry.rng_root_sha256
                or sample.branch_index != entry.producing_branch_index
                or sample.seed != entry.sample_seed
                or sampling_config_sha256(sample) != entry.sampling_config_sha256
            ):
                raise RuntimeError("causal recurrent sample differs from signed group plan")
            validate_recurrent_policy_sample_receipt(sample.receipt())
            samples.append(sample)
            completions.append(token_trace_adapter.decode_output(sample.tokens))
        return prompt_tokens, samples, completions
    samples = []
    completions: list[str] = []
    rejected_receipts: list[dict[str, Any]] = []
    # Admission is intentionally strict and can reject correlated cached
    # samples. Eight attempts per requested member keeps the sampler bounded
    # while avoiding false exhaustion on small groups whose first few
    # deterministic seeds happen to land outside the PPO drift envelope.
    max_attempts = max(size + 2, size * 8)
    for attempt in range(max_attempts):
        if len(samples) >= size:
            break
        try:
            sample = sample_recurrent_completion(
                model,
                prompt_tokens,
                spec=spec,
                seed=_stable_seed(seed, "recurrent_completion", attempt),
                sampling=sampling,
                tokenizer=tokenizer,
            )
        except RecurrentSamplingAdmissionError as exc:
            receipt = exc.sample.receipt()
            receipt["rejected_attempt"] = attempt
            rejected_receipts.append(receipt)
            continue
        samples.append(sample)
        completions.append(tokenizer.decode(list(sample.tokens)))
    if len(samples) < size:
        payload = {
            "schema": "aura.recurrent_group_sampling_exhausted.v1",
            "requested": int(size),
            "admitted": len(samples),
            "attempts": int(max_attempts),
            "rejected": rejected_receipts,
        }
        raise RuntimeError(
            "recurrent group sampling exhausted admissible cached completions: "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)[:2000]
        )
    return prompt_tokens, samples, completions


def _record_recurrent_step_failure(
    out_dir: Path,
    *,
    protocol_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    attempted_step: int,
    last_durable_step: int,
    phase: str,
    task_id: str | None,
    sample_seed: int | None,
    samples: Sequence[Any],
    error: Exception,
) -> Path:
    """Durably bind a failed recurrent attempt without mutating its checkpoint."""

    sample_receipts = [sample.receipt() for sample in samples]
    rejected_sample = getattr(error, "sample", None)
    payload = {
        "schema": "aura.grpo_recurrent_failure.v1",
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "attempted_step": int(attempted_step),
        "last_durable_step": int(last_durable_step),
        "volatile_completed_steps": max(0, int(attempted_step) - 1 - int(last_durable_step)),
        "phase": str(phase),
        "task_id": task_id,
        "sample_seed": sample_seed,
        "samples": sample_receipts,
        "rejected_sample": (
            rejected_sample.receipt()
            if rejected_sample is not None and callable(getattr(rejected_sample, "receipt", None))
            else None
        ),
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        },
        "recorded_at_ns": time.time_ns(),
    }
    encoded = canonical_json_bytes(payload)
    incident = sha256_bytes(encoded)[:16]
    failures = ensure_private_directory(out_dir / "failures")
    path = failures / f"step-{attempted_step:06d}-{incident}.json"
    if not atomic_write_bytes_if_absent(path, encoded, mode=0o600):
        if path.read_bytes() != encoded:
            raise GRPOCheckpointError("recurrent failure receipt publication raced")
    latest = canonical_json_bytes(
        {
            "schema": "aura.grpo_recurrent_failure_pointer.v1",
            "receipt": str(path.relative_to(out_dir)),
            "receipt_sha256": sha256_bytes(encoded),
        }
    )
    atomic_write_bytes(out_dir / "latest_failure.json", latest, mode=0o600)
    return path


def sample_group(model, tokenizer, task, *, size, max_tokens, temperature, seed):
    """K completions for one prompt. Diversity is the mechanism."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _render(tokenizer, task)
    completions: list[str] = []
    for index in range(size):
        mx.random.seed(seed * 1000 + index)
        pieces: list[str] = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.95),
        ):
            pieces.append(response.text)
        completions.append("".join(pieces))
    return prompt, completions


def completion_logprob(model, tokenizer, prompt, completion, *, adapters_on):
    """Log-probability of a completion, with adapters on or off.

    Adapters off gives the reference policy for the KL term at zero extra
    memory -- a second resident copy of a 32B is exactly the kind of thing
    that took this host down.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise RuntimeError("empty completion has no EOS token for policy credit")
        completion_ids = [eos_token_id]
    full = mx.array([prompt_ids + completion_ids])
    count = len(completion_ids)

    def forward():
        logits = model(full)
        start = full.shape[1] - count - 1
        return sequence_token_logprobs(
            logits[:, start : start + count, :], mx.array([completion_ids])
        )

    if adapters_on:
        with recurrence_adapter_scope(start=None, stop=None):
            return forward()
    return forward()  # no scope => ScopedLoRALinear passes through


def evaluate_heldout(
    model,
    tokenizer,
    tasks,
    *,
    max_tokens,
    envelope,
    adapters_on: bool,
    progress_label: str = "",
    progress_every: int = 4,
):
    """Greedy held-out accuracy by depth with explicit adapter exposure."""
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    results = []
    scope = recurrence_adapter_scope(start=None, stop=None) if adapters_on else nullcontext()
    total = len(tasks)
    correct_so_far = 0
    reasons: Counter[str] = Counter()
    with scope:
        for index, task in enumerate(tasks, start=1):
            pieces: list[str] = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt=_render(tokenizer, task),
                max_tokens=max_tokens,
            ):
                pieces.append(response.text)
            verdict = task.grade("".join(pieces))
            correct = bool(verdict["correct"])
            reasons[_grade_reason(verdict)] += 1
            results.append((task, correct))
            correct_so_far += int(correct)
            if envelope is not None:
                envelope.reclaim(force=True)
            if progress_label and (
                index == total or index == 1 or index % max(1, progress_every) == 0
            ):
                print(
                    f"[{progress_label}] {index}/{total} "
                    f"running={correct_so_far / max(1, index):.3f}",
                    flush=True,
                )
    report = scaling_report(results)
    report["adapters_on"] = adapters_on
    report["execution_mode"] = "standard"
    report["score_reasons"] = dict(sorted(reasons.items()))
    return report


def evaluate_recurrent_heldout(
    model,
    tokenizer,
    tasks,
    *,
    spec,
    max_tokens: int,
    envelope,
    adapters_on: bool,
    seed: int,
    progress_label: str = "",
    progress_every: int = 4,
    progress_path: Path | None = None,
    progress_identity: Mapping[str, Any] | None = None,
):
    """Greedy held-out accuracy through the exact fixed RLC graph."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_disabled,
    )
    from core.learning.recurrent_grpo import (
        RecurrentSamplingConfig,
        cortex_config_from_execution_spec,
    )

    config = cortex_config_from_execution_spec(
        spec,
        sampling=RecurrentSamplingConfig(max_tokens=max_tokens),
    )
    config.decode_temperature = 0.0
    config.decode_contract = "final_answer_v1"
    # `max_tokens` is the preregistered scientific budget, not the beginning
    # of a second hidden budget. An incomplete contract is scored as an
    # incorrect bounded policy output at that exact ceiling.
    config.decode_contract_grace_tokens = 0
    engine = LatentCortexEngine(
        model,
        tokenizer=tokenizer,
        config=config,
        schedule_library=None,
    )
    results = []
    receipts: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    contract_reasons: Counter[str] = Counter()
    total = len(tasks)
    correct_so_far = 0
    if (progress_path is None) is not (progress_identity is None):
        raise ValueError("recurrent evaluation progress path and identity must be paired")
    prior_records: list[dict[str, Any]] = []
    prior_complete = False
    if progress_path is not None:
        assert progress_identity is not None
        prior_records, prior_complete = _load_phase_progress(
            progress_path,
            phase="recurrent_baseline",
            identity=progress_identity,
        )
        if len(prior_records) > total:
            raise GRPOCheckpointError("recurrent baseline progress exceeds task count")
        expected_prefix = [task.task_id for task in tasks[: len(prior_records)]]
        observed_prefix = [record.get("task_id") for record in prior_records]
        if observed_prefix != expected_prefix or prior_complete != (len(prior_records) == total):
            raise GRPOCheckpointError("recurrent baseline progress is not an exact task prefix")
        for task, record in zip(
            tasks[: len(prior_records)],
            prior_records,
            strict=True,
        ):
            required = {
                "task_id",
                "selected_branch",
                "steps_taken",
                "decode_termination",
                "output_tokens",
                "correct",
                "score_reason",
                "episode_ok",
                "episode_reason",
                "scored_policy_failure",
                "contract",
            }
            contract = record.get("contract")
            if (
                set(record) != required
                or type(record.get("correct")) is not bool
                or not isinstance(record.get("score_reason"), str)
                or not isinstance(record.get("episode_reason"), str)
                or type(record.get("episode_ok")) is not bool
                or type(record.get("scored_policy_failure")) is not bool
                or not isinstance(contract, dict)
                or set(contract) != {"marker_count", "complete", "valid", "reason"}
                or not isinstance(contract.get("reason"), str)
            ):
                raise GRPOCheckpointError("recurrent baseline progress record is invalid")
            correct = bool(record["correct"])
            results.append((task, correct))
            receipts.append(dict(record))
            reasons[str(record["score_reason"])] += 1
            contract_reasons[str(contract["reason"])] += 1
            correct_so_far += int(correct)
        if prior_records and progress_label:
            print(
                f"[{progress_label}] resumed {len(prior_records)}/{total} "
                f"running={correct_so_far / len(prior_records):.3f}",
                flush=True,
            )
    for index, task in enumerate(tasks[len(prior_records) :], start=len(prior_records)):
        mx.random.seed(_stable_seed(seed, "recurrent_eval", index, task.task_id))
        scope = nullcontext() if adapters_on else recurrence_adapter_disabled()
        with scope:
            result = engine.reason(
                token_ids=_task_prompt_tokens(tokenizer, task),
                decode_max_tokens=max_tokens,
                decode_sentence_grace_tokens=0,
            )
        scored_policy_failure = bool(
            not result.ok
            and (
                result.reason == "answer_replacement_abstained"
                or result.reason.startswith("decode_incomplete:")
            )
        )
        if not result.ok and not scored_policy_failure:
            raise RuntimeError(f"recurrent held-out task {task.task_id} failed: {result.reason}")
        response_text = (
            ""
            if result.reason == "answer_replacement_abstained"
            else result.text
        )
        verdict = task.grade(response_text)
        correct = bool(verdict["correct"])
        reason = _grade_reason(verdict)
        reasons[reason] += 1
        from core.brain.llm.latent_cortex.answer_contract import (
            contract_answer_state,
        )

        contract_state = contract_answer_state(response_text)
        contract_reason = str(contract_state.get("reason") or "unknown")
        contract_reasons[contract_reason] += 1
        results.append((task, correct))
        correct_so_far += int(correct)
        episode_record = {
            "task_id": task.task_id,
            "selected_branch": result.receipt.selected_branch,
            "steps_taken": result.receipt.steps_taken,
            "decode_termination": result.receipt.decode_termination,
            "output_tokens": len(result.tokens),
            "correct": bool(verdict["correct"]),
            "score_reason": reason,
            "episode_ok": bool(result.ok),
            "episode_reason": str(result.reason or ""),
            "scored_policy_failure": scored_policy_failure,
            "contract": {
                "marker_count": int(contract_state.get("marker_count") or 0),
                "complete": bool(contract_state.get("complete")),
                "valid": bool(contract_state.get("valid")),
                "reason": contract_reason,
            },
        }
        receipts.append(episode_record)
        if progress_path is not None:
            assert progress_identity is not None
            _write_phase_progress(
                progress_path,
                phase="recurrent_baseline",
                identity=progress_identity,
                records=receipts,
                complete=len(receipts) == total,
            )
        if envelope is not None:
            envelope.reclaim(force=True)
        completed = index + 1
        if progress_label and (
            completed == total or completed == 1 or completed % max(1, progress_every) == 0
        ):
            print(
                f"[{progress_label}] {completed}/{total} "
                f"running={correct_so_far / max(1, completed):.3f}",
                flush=True,
            )
    report = scaling_report(results)
    report["adapters_on"] = adapters_on
    report["execution_mode"] = "recurrent"
    report["execution_spec_sha256"] = spec.sha256
    report["episode_receipts"] = receipts
    report["score_reasons"] = dict(sorted(reasons.items()))
    report["contract_reasons"] = dict(sorted(contract_reasons.items()))
    return report


def main(
    *,
    verified_group_provider_factory: VerifiedTransitionGroupProviderFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--adapter-id",
        default="recurrent-grpo",
        help="stable adapter identity recorded in recurrent campaign bundles",
    )
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="standard",
    )
    parser.add_argument(
        "--execution-spec",
        help="strict RLCExecutionSpec JSON required by recurrent mode",
    )
    parser.add_argument(
        "--verified-trajectory-config",
        help=(
            "canonical VerifiedTrajectoryGroupConfig JSON; admits replayable "
            "trajectory terms only through the verified recurrent transaction"
        ),
    )
    parser.add_argument(
        "--task-source",
        choices=TASK_SOURCES,
        default="verifiable",
        help="immutable programmatic training registry; frontier tasks stay evaluation-only",
    )
    parser.add_argument("--domains", default="arithmetic_chain,program_trace,constraint_order")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--train-per-cell", type=int, default=32)
    parser.add_argument("--holdout-per-cell", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--kl-coefficient", type=float, default=0.04)
    parser.add_argument("--format-credit", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-credit",
        action="store_true",
        help=(
            "in recurrent mode, use bounded gold-answer CE trajectory credit "
            "when final verifier rewards would otherwise be no-signal"
        ),
    )
    parser.add_argument(
        "--trajectory-shaping-weight",
        type=float,
        default=0.25,
        help="bounded trajectory credit weight; verifier reward remains dominant",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", default="o_proj,v_proj,q_proj")
    parser.add_argument("--lora-layers", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument(
        "--min-signal-groups",
        type=int,
        default=8,
        help=(
            "minimum graded groups before a degenerate-run verdict can halt "
            "training as no_learning_signal"
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="measure pass rates before training to skip dead cells",
    )
    parser.add_argument("--calibrate-samples", type=int, default=2)
    parser.add_argument(
        "--calibrate-group",
        type=int,
        default=4,
        help="completions per calibration probe (cheaper than the train group)",
    )
    parser.add_argument(
        "--calibrate-tokens",
        type=int,
        default=0,
        help="max tokens per calibration probe; 0 = match --max-tokens "
        "(reasoning tasks need room to finish, or the probe "
        "underestimates pass rate and mislabels learnable cells)",
    )
    parser.add_argument(
        "--calibrate-minutes",
        type=float,
        default=15.0,
        help="wall-clock cap on the whole calibration phase",
    )
    parser.add_argument(
        "--cot", action="store_true", help="invite step-by-step reasoning before the answer"
    )
    parser.add_argument("--max-minutes", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument(
        "--initial-policy-probe",
        action="store_true",
        help=(
            "load the recurrent checkpoint, attach deterministic LoRA sites, "
            "seal initial_policy_probe.json, and exit before evaluation or training"
        ),
    )
    parser.add_argument(
        "--read-only-answer-channel-preflight",
        action="store_true",
        help=(
            "load the recurrent checkpoint, run the held-out answer-channel "
            "diagnostic, publish its receipt, and exit before optimizer construction"
        ),
    )
    parser.add_argument(
        "--fixed-update-canary",
        action="store_true",
        help=(
            "execute the complete externally scheduled engineering canary "
            "without adaptive early termination; reward and optimizer "
            "admission remain unchanged"
        ),
    )
    args = parser.parse_args()

    for name in (
        "train_per_cell",
        "holdout_per_cell",
        "group_size",
        "max_tokens",
        "lora_rank",
        "lora_layers",
        "max_steps",
        "eval_every",
        "checkpoint_every",
        "checkpoint_keep",
        "min_signal_groups",
        "calibrate_samples",
        "calibrate_group",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_minutes <= 0.0 or args.calibrate_minutes <= 0.0:
        parser.error("time budgets must be positive")
    if not 0.0 < args.memory_fraction <= 0.9:
        parser.error("--memory-fraction must be inside (0, 0.9]")
    try:
        _calibration_token_budget(args.max_tokens, args.calibrate_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 < args.temperature <= 2.0:
        parser.error("--temperature must be inside (0, 2]")
    if not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be inside (0, 1]")
    if not 0.0 <= args.format_credit <= 0.2:
        parser.error("--format-credit must be inside [0, 0.2]")
    if not 0.0 <= args.trajectory_shaping_weight <= 0.49:
        parser.error("--trajectory-shaping-weight must be inside [0, 0.49]")
    if args.trajectory_credit and args.execution_mode != "recurrent":
        parser.error("--trajectory-credit only applies to recurrent mode")
    if _ADAPTER_ID_RE.fullmatch(args.adapter_id) is None:
        parser.error("--adapter-id must be a stable identifier")
    try:
        execution_spec = _load_execution_spec(args.execution_mode, args.execution_spec)
        trajectory_group_config = _load_verified_trajectory_group_config(
            args.execution_mode,
            args.verified_trajectory_config,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.execution_mode == "recurrent" and args.temperature != 1.0:
        parser.error("recurrent mode requires --temperature 1")
    if (
        args.execution_mode == "recurrent"
        and verified_group_provider_factory is None
        and not args.initial_policy_probe
        and not args.read_only_answer_channel_preflight
    ):
        parser.error(
            "recurrent mode requires a post-load verified transition provider "
            "factory; preconstructed providers and raw caller-authored scalar "
            "rewards are not authorized mutation paths"
        )
    if args.execution_mode == "standard" and verified_group_provider_factory is not None:
        parser.error("a verified transition provider only applies to recurrent mode")
    if trajectory_group_config is not None and verified_group_provider_factory is None:
        parser.error("--verified-trajectory-config requires a verified transition provider")
    if args.execution_mode == "standard" and (
        args.initial_policy_probe or args.read_only_answer_channel_preflight
    ):
        parser.error("recurrent probes only apply to recurrent mode")
    if args.initial_policy_probe and args.read_only_answer_channel_preflight:
        parser.error("select exactly one recurrent probe mode")
    if verified_group_provider_factory is not None and (
        args.initial_policy_probe or args.read_only_answer_channel_preflight
    ):
        parser.error("read-only recurrent probes cannot be combined with a training provider")
    if args.fixed_update_canary:
        if (
            args.execution_mode != "recurrent"
            or verified_group_provider_factory is None
            or args.task_source != "recurrence_curriculum"
            or args.calibrate
            or not args.adapter_id.endswith("-update-canary")
            or args.checkpoint_every != 1
            or args.eval_every != args.max_steps
        ):
            parser.error(
                "--fixed-update-canary requires the externally verified "
                "recurrent one-pass canary contract"
            )
    elif args.adapter_id.endswith("-update-canary"):
        parser.error("update-canary adapter identity requires --fixed-update-canary")
    provider_contract_sha256 = None
    if verified_group_provider_factory is not None:
        provider_contract_sha256 = getattr(verified_group_provider_factory, "contract_sha256", None)
        if not isinstance(provider_contract_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", provider_contract_sha256
        ):
            parser.error("the verified transition provider must expose its frozen contract digest")
    if args.execution_mode == "recurrent" and args.trajectory_credit:
        parser.error(
            "--trajectory-credit is not authorized for proof-grade recurrent mode; "
            "transition deltas and auxiliary terms require independently replayable receipts"
        )
    if args.execution_mode == "recurrent" and args.checkpoint_every != 1:
        parser.error(
            "proof-grade recurrent mode requires --checkpoint-every 1 so no "
            "committed transition is followed by another group before it is durable"
        )

    import mlx.core as mx
    import mlx.nn as nn

    global _COT_PREAMBLE
    _COT_PREAMBLE = (
        "Work through this step by step, then end with your answer on its own line."
        if args.cot
        else ""
    )

    config = GRPOConfig(group_size=args.group_size, kl_coefficient=args.kl_coefficient)
    recurrent_config = None
    if execution_spec is not None:
        from core.learning.recurrent_grpo import RecurrentGRPOConfig

        recurrent_config = RecurrentGRPOConfig(
            kl_coefficient=args.kl_coefficient,
            advantage_clip=config.advantage_clip,
        )
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    if not domains or not depths or any(depth <= 0 for depth in depths):
        parser.error("--domains and positive --depths are required")
    train_tasks, holdout, task_source_path = _build_task_split(
        task_source=args.task_source,
        domains=domains,
        depths=depths,
        train_per_cell=args.train_per_cell,
        holdout_per_cell=args.holdout_per_cell,
        seed=args.seed,
    )
    if args.fixed_update_canary and len(train_tasks) != args.max_steps:
        parser.error(
            "--fixed-update-canary requires exactly one complete immutable task pass"
        )
    if verified_group_provider_factory is not None:
        train_tasks = list(verified_group_provider_factory.bind_training_tasks(train_tasks))
    print(
        f"[tasks] {len(train_tasks)} train / {len(holdout)} held-out "
        f"from {args.task_source} (disjoint prompts and identities verified)",
        flush=True,
    )

    out_dir = ensure_private_directory(Path(args.out_dir).expanduser().resolve())
    dataset = _dataset_payload(train_tasks, holdout, seed=args.seed)
    dataset_bytes = canonical_json_bytes(dataset)
    dataset_sha256 = sha256_bytes(dataset_bytes)

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
        model_behavior_bundle_identity,
        personality_bundle_identity,
        runtime_environment_identity,
    )

    model_path = str(_resolve_model_path(args.model))
    source_files = _training_source_files(
        task_source_path,
        recurrent=execution_spec is not None,
    )
    sources = {role: _source_binding(path) for role, path in source_files.items()}
    base_identity = full_weight_checkpoint_identity(model_path)
    behavior_identity = model_behavior_bundle_identity(model_path)
    personality_identity = personality_bundle_identity(None)
    runtime_identity = runtime_environment_identity()
    protocol = {
        "schema": GRPO_PROTOCOL_SCHEMA,
        "adapter_id": args.adapter_id,
        "model_path": model_path,
        "base_checkpoint": base_identity,
        "model_behavior": behavior_identity,
        "personality_adapter": personality_identity,
        "runtime": runtime_identity,
        "dataset_sha256": dataset_sha256,
        "sources": sources,
        "training": {
            "execution_mode": args.execution_mode,
            "execution_spec": (execution_spec.to_dict() if execution_spec is not None else None),
            "execution_spec_sha256": (
                execution_spec.sha256 if execution_spec is not None else None
            ),
            "verified_transition_provider_contract_sha256": (provider_contract_sha256),
            "domains": domains,
            "depths": depths,
            "train_per_cell": args.train_per_cell,
            "holdout_per_cell": args.holdout_per_cell,
            "group_size": args.group_size,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "kl_coefficient": args.kl_coefficient,
            "advantage_clip": config.advantage_clip,
            "format_credit": args.format_credit,
            "trajectory_credit": args.trajectory_credit,
            "trajectory_shaping_weight": args.trajectory_shaping_weight,
            "verified_trajectory_config": (
                trajectory_group_config.to_dict() if trajectory_group_config is not None else None
            ),
            "verified_trajectory_config_sha256": (
                protocol_semantic_sha256(trajectory_group_config.to_dict())
                if trajectory_group_config is not None
                else None
            ),
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "lora_layers": args.lora_layers,
            "lora_initialization_seed": _stable_seed(args.seed, "lora-init", args.adapter_id),
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "min_signal_groups": args.min_signal_groups,
            "calibrate": args.calibrate,
            "calibrate_samples": args.calibrate_samples,
            "calibrate_group": args.calibrate_group,
            "calibrate_tokens": args.calibrate_tokens,
            "calibrate_minutes": args.calibrate_minutes,
            "cot": args.cot,
            "seed": args.seed,
            "memory_fraction": args.memory_fraction,
            "rng_strategy": RNG_STRATEGY,
        },
    }
    protocol_bytes = canonical_json_bytes(protocol)
    protocol_sha256 = sha256_bytes(protocol_bytes)
    with interprocess_file_lock(out_dir / ".checkpoint.lock"):
        _publish_immutable_bytes(
            out_dir / "dataset_manifest.json", dataset_bytes, role="dataset manifest"
        )
        _publish_immutable_bytes(
            out_dir / "training_protocol.json", protocol_bytes, role="training protocol"
        )
        source_snapshot_dir = ensure_private_directory(out_dir / "source_snapshots")
        for role, source_path in source_files.items():
            source_bytes = source_path.resolve(strict=True).read_bytes()
            binding = sources[role]
            if (
                len(source_bytes) != binding["size_bytes"]
                or sha256_bytes(source_bytes) != binding["sha256"]
            ):
                raise RuntimeError(f"training source changed while snapshotting: {role}")
            _publish_immutable_bytes(
                source_snapshot_dir / f"{role}.py",
                source_bytes,
                role=f"{role} source snapshot",
            )

    started_wall = time.time()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.max_minutes * 60.0

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane

    with (
        standalone_model_lane(
            owner_id=f"train-grpo:{Path(args.out_dir).name}",
            model_path=model_path,
            purpose="training",
            preemptible=False,
            metadata={"tool": "train_grpo", "operator_launched": True},
        ),
        mlx_memory_envelope(fraction=args.memory_fraction) as envelope,
    ):
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(model_path)
        model.freeze()
        verified_group_provider: VerifiedTransitionGroupProvider | None = None
        token_trace_adapter = None
        if execution_spec is not None:
            from core.learning.verified_token_trace import (
                build_resident_tokenizer_trace_adapter,
            )

            token_trace_adapter = build_resident_tokenizer_trace_adapter(
                tokenizer,
                model_path,
            )

        from core.brain.llm.latent_cortex.recurrence_adapter import (
            ScopedLoRALinear,
            recurrence_adapter_scope,
        )
        from core.learning.recurrent_grpo import (
            attach_recurrent_policy_adapters,
            build_recurrent_policy_optimizer,
            recurrent_policy_optimizer_config,
        )

        targets = tuple(t.strip() for t in args.lora_targets.split(","))
        if execution_spec is None:
            total_layers = len(model.model.layers)
            adapted_indices = range(max(0, total_layers - args.lora_layers), total_layers)
            attached = 0
            mx.random.seed(_stable_seed(args.seed, "lora-init", args.adapter_id))
            for index in adapted_indices:
                layer = model.model.layers[index]
                for parent_name in ("self_attn", "mlp"):
                    parent = getattr(layer, parent_name, None)
                    if parent is None:
                        continue
                    for target in targets:
                        projection = getattr(parent, target, None)
                        if projection is not None and not isinstance(
                            projection,
                            ScopedLoRALinear,
                        ):
                            site = f"model.layers.{index}.{parent_name}.{target}"
                            setattr(
                                parent,
                                target,
                                ScopedLoRALinear.from_base(
                                    projection,
                                    r=args.lora_rank,
                                    block_index=index,
                                    site=site,
                                ),
                            )
                            attached += 1
        else:
            attached_sites = attach_recurrent_policy_adapters(
                model,
                execution_spec,
                lora_rank=args.lora_rank,
                lora_layers=args.lora_layers,
                lora_targets=targets,
                initialization_seed=_stable_seed(
                    args.seed,
                    "lora-init",
                    args.adapter_id,
                ),
            )
            attached = len(attached_sites)
        if not attached:
            raise RuntimeError("no projections adapted; check --lora-targets")
        print(f"[wiring] {attached} projections adapted", flush=True)

        if args.initial_policy_probe:
            from mlx.utils import tree_flatten

            from core.learning.recurrent_grpo import recurrent_policy_sha256
            from core.learning.verified_transition_policy_probe import (
                build_initial_recurrent_policy_probe,
                inspect_initial_adapter_snapshot,
                inspect_initial_optimizer_snapshot,
                validate_initial_recurrent_policy_probe_identity,
            )

            assert execution_spec is not None
            assert token_trace_adapter is not None
            probe_path = out_dir / "initial_policy_probe.json"
            snapshot_path = out_dir / "initial_adapter.safetensors"
            optimizer_snapshot_path = out_dir / "initial_optimizer.safetensors"
            initial_tensors = dict(tree_flatten(model.trainable_parameters()))
            probe_optimizer = build_recurrent_policy_optimizer(args.learning_rate)
            probe_optimizer.init(model.trainable_parameters())
            initial_optimizer_tensors = dict(tree_flatten(probe_optimizer.state))
            _publish_immutable_tensor_snapshot(
                snapshot_path,
                initial_tensors,
                role="initial recurrent adapter snapshot",
            )
            _publish_immutable_tensor_snapshot(
                optimizer_snapshot_path,
                initial_optimizer_tensors,
                role="initial recurrent optimizer snapshot",
            )
            initial_adapter_artifact = inspect_initial_adapter_snapshot(
                snapshot_path,
                execution_spec_sha256=execution_spec.sha256,
            )
            initial_optimizer_artifact = inspect_initial_optimizer_snapshot(optimizer_snapshot_path)
            probe_identity = {
                "campaign_id": args.adapter_id,
                "initial_policy_sha256": recurrent_policy_sha256(model, execution_spec),
                "dataset_sha256": dataset_sha256,
                "execution_spec_sha256": execution_spec.sha256,
                "base_checkpoint": base_identity,
                "model_behavior_bundle": behavior_identity,
                "tokenizer_bundle": token_trace_adapter.bundle_identity,
                "adapter_initialization": {
                    "seed": _stable_seed(args.seed, "lora-init", args.adapter_id),
                    "rank": args.lora_rank,
                    "layers": args.lora_layers,
                    "targets": list(targets),
                },
                "source_bindings": sources,
                "initial_adapter_artifact": initial_adapter_artifact,
                "optimizer_initialization": (recurrent_policy_optimizer_config(args.learning_rate)),
                "initial_optimizer_artifact": (initial_optimizer_artifact),
            }
            with interprocess_file_lock(out_dir / ".initial-policy-probe.lock"):
                if probe_path.exists() or probe_path.is_symlink():
                    if probe_path.is_symlink():
                        raise GRPOCheckpointError(
                            "initial recurrent policy probe symlink is forbidden"
                        )
                    raw_probe = read_stable_bytes(
                        probe_path,
                        max_bytes=16 << 20,
                    )
                    try:
                        existing_probe = json.loads(raw_probe)
                    except (
                        UnicodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise GRPOCheckpointError(
                            "initial recurrent policy probe is invalid"
                        ) from exc
                    if (
                        not isinstance(existing_probe, Mapping)
                        or canonical_json_bytes(existing_probe) != raw_probe
                    ):
                        raise GRPOCheckpointError("initial recurrent policy probe is noncanonical")
                    probe = validate_initial_recurrent_policy_probe_identity(
                        existing_probe,
                        **probe_identity,
                    )
                else:
                    probe = build_initial_recurrent_policy_probe(
                        **probe_identity,
                        created_at_unix_ns=time.time_ns(),
                    )
                    _publish_immutable_bytes(
                        probe_path,
                        canonical_json_bytes(probe),
                        role="initial recurrent policy probe",
                    )
            print(
                "[policy-probe] "
                f"policy={probe['initial_policy_sha256']} "
                f"receipt={probe['receipt_sha256']} path={probe_path}",
                flush=True,
            )
            return 0

        if args.read_only_answer_channel_preflight:
            assert execution_spec is not None
            report = evaluate_recurrent_heldout(
                model,
                tokenizer,
                holdout,
                spec=execution_spec,
                max_tokens=args.max_tokens,
                adapters_on=False,
                seed=_stable_seed(args.seed, "answer-channel-preflight"),
                progress_label="answer-channel-preflight",
                envelope=envelope,
            )
            episode_receipts = report.get("episode_receipts")
            if not isinstance(episode_receipts, list) or not episode_receipts:
                raise RuntimeError("answer-channel preflight produced no episode receipts")
            valid_contracts = sum(
                1
                for receipt in episode_receipts
                if isinstance(receipt, Mapping)
                and isinstance(receipt.get("contract"), Mapping)
                and receipt["contract"].get("valid") is True
            )
            correct = sum(
                1
                for receipt in episode_receipts
                if isinstance(receipt, Mapping) and receipt.get("correct") is True
            )
            valid_fraction = valid_contracts / len(episode_receipts)
            body = {
                "schema": "aura.recurrent_answer_channel_preflight.v1",
                "campaign_id": args.adapter_id,
                "dataset_sha256": dataset_sha256,
                "execution_spec_sha256": execution_spec.sha256,
                "base_checkpoint": base_identity,
                "model_behavior_bundle": behavior_identity,
                "tokenizer_bundle": token_trace_adapter.bundle_identity,
                "source_bindings": sources,
                "task_count": len(episode_receipts),
                "valid_contract_count": valid_contracts,
                "correct_count": correct,
                "valid_contract_fraction": round(valid_fraction, 6),
                "report": report,
                "created_at_unix_ns": time.time_ns(),
                "verdict": (
                    "answer_channel_operational"
                    if valid_fraction >= 0.5
                    else "answer_channel_blocked"
                ),
            }
            receipt = {
                **body,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(body)),
            }
            receipt_path = out_dir / "answer_channel_preflight.json"
            _publish_immutable_bytes(
                receipt_path,
                canonical_json_bytes(receipt),
                role="recurrent answer-channel preflight",
            )
            print(
                "[answer-channel-preflight] "
                f"valid={valid_contracts}/{len(episode_receipts)} "
                f"correct={correct}/{len(episode_receipts)} "
                f"verdict={receipt['verdict']} path={receipt_path}",
                flush=True,
            )
            return 0 if valid_fraction >= 0.5 else 3

        from mlx.utils import tree_flatten, tree_unflatten

        from core.learning.adaptive_curriculum import (
            AdaptiveCurriculum,
            warm_start_pass_rates,
        )

        optimizer = build_recurrent_policy_optimizer(args.learning_rate)
        optimizer.init(model.trainable_parameters())
        if verified_group_provider_factory is not None:
            assert execution_spec is not None
            verified_group_provider = verified_group_provider_factory.create(
                VerifiedTransitionProviderRuntime(
                    model=model,
                    tokenizer=tokenizer,
                    tokenizer_trace_adapter=token_trace_adapter,
                    execution_spec=execution_spec,
                    training_tasks=tuple(train_tasks),
                    output_directory=out_dir,
                    transaction_root=(out_dir / "verified-transition-transactions"),
                    dataset_sha256=dataset_sha256,
                    group_size=args.group_size,
                    sampling_max_tokens=args.max_tokens,
                )
            )
        telemetry: GRPOTelemetry | VerifiedTransitionTelemetry = (
            VerifiedTransitionTelemetry() if execution_spec is not None else GRPOTelemetry()
        )
        history: list[dict[str, Any]] = []
        step_receipts: list[dict[str, Any]] = []
        transition_replay_groups: list[VerifiedTransitionReplayGroup] = []
        transition_closure: VerifiedTransitionCampaignClosure | None = None
        baseline_eval: dict[str, Any] | None = None
        calibration: dict[str, Any] | None = None
        step = 0
        optimizer_updates = 0
        last_step_kind = "initial"
        prior_elapsed_s = 0.0
        invocation_count = 1
        pre_stage_recovery_halt = None

        by_cell: dict[tuple[str, int], list[Any]] = {}
        tasks_by_id: dict[str, Any] = {}
        for task in train_tasks:
            if task.task_id in tasks_by_id:
                raise RuntimeError(f"duplicate training task id: {task.task_id}")
            tasks_by_id[task.task_id] = task
            by_cell.setdefault((task.domain, task.depth), []).append(task)
        curriculum = AdaptiveCurriculum.over(
            sorted({domain for domain, _depth in by_cell}),
            sorted({depth for _domain, depth in by_cell}),
        )

        expected_adapters = dict(tree_flatten(model.trainable_parameters()))
        if not expected_adapters or any("lora" not in key for key in expected_adapters):
            raise RuntimeError("trainable tree contains non-LoRA parameters")

        transaction_store = (
            VerifiedTransitionTransactionStore.open(out_dir / "verified-transition-transactions")
            if execution_spec is not None
            else None
        )
        measurement_chain_store = None
        measurement_update_journal = None
        if (
            execution_spec is not None
            and trajectory_group_config is not None
            and trajectory_group_config.intervention_config is not None
        ):
            if verified_group_provider_factory is None:
                raise RuntimeError("intervention training requires a verified provider factory")
            initial_state_custody = getattr(
                verified_group_provider_factory,
                "initial_policy_state_custody",
                None,
            )
            if not isinstance(initial_state_custody, Mapping):
                raise RuntimeError(
                    "intervention training requires materialized initial "
                    "adapter and optimizer custody"
                )
            assert transaction_store is not None
            assert provider_contract_sha256 is not None
            ledger_roots = getattr(
                verified_group_provider_factory,
                "ledger_roots",
                None,
            )
            if not isinstance(ledger_roots, Mapping) or not isinstance(
                ledger_roots.get("updates"), str
            ):
                raise RuntimeError("intervention training requires the frozen update journal root")
            measurement_update_journal = VerifiedTransitionUpdateJournal.open(
                ledger_roots["updates"]
            )
            measurement_chain_store = VerifiedTransitionMeasurementChainStore.open(
                out_dir / "verified-transition-transactions",
                transaction_store=transaction_store,
                initial_policy_state_custody=initial_state_custody,
                provider_contract_sha256=provider_contract_sha256,
                training_protocol_sha256=protocol_sha256,
            )
        rejection_transaction_store = (
            VerifiedTransitionRejectionTransactionStore.open(
                out_dir / "verified-transition-transactions"
            )
            if execution_spec is not None
            else None
        )
        resumed = None
        if (out_dir / "latest.json").exists():
            resumed = load_grpo_checkpoint(
                out_dir,
                expected_protocol_sha256=protocol_sha256,
                expected_dataset_sha256=dataset_sha256,
            )
            _assert_exact_adapter_keys(expected_adapters, resumed.adapter_tensors)
            model.load_weights(list(resumed.adapter_tensors.items()), strict=False)
            optimizer.state = resumed.optimizer_state
            optimizer.init(model.trainable_parameters())
            state = resumed.state
            step = int(state["step"])
            optimizer_updates = int(state["optimizer_updates"])
            last_step_kind = str(state["last_step_kind"])
            curriculum = AdaptiveCurriculum.from_state(state["curriculum"])
            telemetry = (
                VerifiedTransitionTelemetry.from_state(state["telemetry"])
                if execution_spec is not None
                else GRPOTelemetry.from_state(state["telemetry"])
            )
            history = list(state["history"])
            raw_step_receipts = state.get("step_receipts", [])
            if not isinstance(raw_step_receipts, list) or any(
                not isinstance(entry, dict) for entry in raw_step_receipts
            ):
                raise GRPOCheckpointError("checkpoint step receipts are invalid")
            step_receipts = list(raw_step_receipts)
            if state.get("execution_mode", "standard") != args.execution_mode:
                raise GRPOCheckpointError("checkpoint execution mode differs")
            if state.get("execution_spec_sha256") != (
                execution_spec.sha256 if execution_spec is not None else None
            ):
                raise GRPOCheckpointError("checkpoint execution spec differs")
            if execution_spec is not None and len(step_receipts) != step:
                raise GRPOCheckpointError(
                    "recurrent checkpoint does not receipt every committed step"
                )
            if execution_spec is None and step_receipts:
                raise GRPOCheckpointError("standard checkpoint contains recurrent step receipts")
            if execution_spec is not None:
                assert verified_group_provider is not None
                from core.learning.recurrent_grpo import recurrent_policy_sha256

                try:
                    step_receipts = [
                        validate_verified_transition_step_receipt(
                            receipt,
                            group_size=config.group_size,
                            execution_spec_sha256=execution_spec.sha256,
                        )
                        for receipt in step_receipts
                        if receipt.get("schema") == VERIFIED_TRANSITION_STEP_SCHEMA
                    ]
                except ValueError as exc:
                    raise GRPOCheckpointError(
                        "recurrent checkpoint has an invalid verified step receipt"
                    ) from exc
                if len(step_receipts) != step:
                    raise GRPOCheckpointError(
                        "proof-grade recurrent resume rejects legacy or mixed step receipts"
                    )

                if step_receipts:
                    expected_policy = step_receipts[-1].get("policy_after_sha256")
                    if (
                        not isinstance(expected_policy, str)
                        or recurrent_policy_sha256(model, execution_spec) != expected_policy
                    ):
                        raise GRPOCheckpointError(
                            "recurrent checkpoint tensors differ from the last "
                            "committed verified transition receipt"
                        )
                restored = tuple(
                    verified_group_provider.restore_groups(
                        committed_steps=step,
                        step_receipts=step_receipts,
                    )
                )
                if len(restored) != optimizer_updates:
                    raise GRPOCheckpointError(
                        "verified transition replay group count differs from "
                        "committed optimizer updates"
                    )
                updated_step_receipts = [
                    receipt
                    for receipt in step_receipts
                    if receipt.get("step_kind") == "verified_optimizer_update"
                ]
                for receipt, replay_group in zip(updated_step_receipts, restored, strict=True):
                    update = receipt.get("update")
                    if (
                        replay_group.sequence != int(receipt["step"]) - 1
                        or not isinstance(update, Mapping)
                        or dict(replay_group.update_receipt) != dict(update)
                        or replay_group.reward_receipt.get("receipt_sha256")
                        != receipt.get("reward_receipt_sha256")
                    ):
                        raise GRPOCheckpointError(
                            "restored transition source evidence differs from "
                            "the durable trainer step receipt"
                        )
                transition_replay_groups = list(restored)
            baseline_eval = state["baseline_eval"]
            calibration = state["calibration"]
            prior_elapsed_s = float(state["elapsed_training_s"])
            invocation_count = int(state["invocation_count"]) + 1
            print(
                f"[resume] exact step={step} optimizer_updates={optimizer_updates} "
                f"checkpoint={resumed.checkpoint_dir.name}",
                flush=True,
            )
        elif (out_dir / "checkpoints" / "checkpoint_manifest.json").exists():
            raise GRPOCheckpointError(
                "legacy GRPO checkpoint lacks optimizer/protocol state; use a fresh "
                "output directory instead of claiming exact resume"
            )

        def elapsed_training_s() -> float:
            return prior_elapsed_s + (time.monotonic() - started_monotonic)

        def adapter_tensors() -> dict[str, Any]:
            tensors = dict(tree_flatten(model.trainable_parameters()))
            _assert_exact_adapter_keys(expected_adapters, tensors)
            return tensors

        last_durable_step = step

        def checkpoint_now() -> Path:
            nonlocal last_durable_step
            optimizer_tensors = dict(tree_flatten(optimizer.state))
            if not optimizer_tensors:
                raise GRPOCheckpointError("optimizer state is empty")
            path = save_grpo_checkpoint(
                out_dir,
                adapter_tensors=adapter_tensors(),
                optimizer_tensors=optimizer_tensors,
                state={
                    "protocol_sha256": protocol_sha256,
                    "dataset_sha256": dataset_sha256,
                    "step": step,
                    "curriculum": curriculum.state(),
                    "telemetry": telemetry.state(),
                    "history": history,
                    "step_receipts": step_receipts,
                    "baseline_eval": baseline_eval,
                    "calibration": calibration,
                    "elapsed_training_s": elapsed_training_s(),
                    "invocation_count": invocation_count,
                    "rng_strategy": RNG_STRATEGY,
                    "optimizer_updates": optimizer_updates,
                    "last_step_kind": last_step_kind,
                    "last_step_committed": True,
                    "execution_mode": args.execution_mode,
                    "execution_spec_sha256": (
                        execution_spec.sha256 if execution_spec is not None else None
                    ),
                },
                keep=args.checkpoint_keep,
            )
            last_durable_step = step
            return path

        if execution_spec is not None:
            assert transaction_store is not None
            assert rejection_transaction_store is not None
            assert verified_group_provider is not None
            from core.learning.recurrent_grpo import recurrent_policy_sha256

            update_transactions = transaction_store.inventory(load_tensors=False)
            rejection_transactions = rejection_transaction_store.inventory()
            if measurement_chain_store is not None:
                if measurement_update_journal is None:
                    raise GRPOCheckpointError("pre-measurement recovery journal is missing")
                recovered_pre_stage = measurement_chain_store.reconcile_interrupted_admissions(
                    update_journal=measurement_update_journal,
                    live_adapter_tensors=adapter_tensors(),
                    live_optimizer_tensors=dict(tree_flatten(optimizer.state)),
                    observed_policy_sha256=recurrent_policy_sha256(
                        model,
                        execution_spec,
                    ),
                    reconciled_at_unix_ns=time.time_ns(),
                )
                if recovered_pre_stage:
                    if (
                        resumed is None
                        or len(recovered_pre_stage) != 1
                        or recovered_pre_stage[0]["sequence"] != step
                    ):
                        raise GRPOCheckpointError(
                            "pre-stage recovery does not match the exact durable trainer checkpoint"
                        )
                    recovery_body = {
                        "schema": ("aura.grpo.pre_stage_recovery_halt.v1"),
                        "protocol_sha256": protocol_sha256,
                        "dataset_sha256": dataset_sha256,
                        "execution_spec_sha256": execution_spec.sha256,
                        "last_durable_step": step,
                        "recoveries": list(recovered_pre_stage),
                        "requires_fresh_campaign": True,
                    }
                    recovery_receipt = {
                        **recovery_body,
                        "receipt_sha256": sha256_bytes(canonical_json_bytes(recovery_body)),
                    }
                    _publish_immutable_bytes(
                        out_dir / "pre_stage_recovery_halt.json",
                        canonical_json_bytes(recovery_receipt),
                        role="pre-stage recovery halt",
                    )
                    pre_stage_recovery_halt = recovery_receipt
            update_sequences = {
                int(transaction.pending_step["sequence"]) for transaction in update_transactions
            }
            rejection_sequences = {
                int(transaction.intent["sequence"]) for transaction in rejection_transactions
            }
            if update_sequences & rejection_sequences:
                raise GRPOCheckpointError(
                    "one verified sequence has both update and rejection transactions"
                )

            pending_recovery = None
            for transaction in update_transactions:
                pending = transaction.pending_step
                sequence = int(pending["sequence"])
                admission = str(pending["group_admission_sha256"])
                if sequence < step:
                    if len(transaction.events) < 3:
                        if (
                            resumed is None
                            or sequence != step - 1
                            or int(pending["trainer_step"]) != step
                        ):
                            raise GRPOCheckpointError(
                                "historical verified transaction is not fully sealed"
                            )
                        durable_step = step_receipts[sequence]
                        if durable_step.get("group_admission_sha256") != admission:
                            raise GRPOCheckpointError(
                                "historical transaction admission differs from checkpoint"
                            )
                        if len(transaction.events) == 0:
                            transaction_store.record_update_commit(
                                sequence=sequence,
                                admission_sha256=admission,
                                update_receipt=durable_step["update"],
                            )
                        current = transaction_store.load(
                            sequence=sequence,
                            admission_sha256=admission,
                            load_tensors=False,
                        )
                        assert current is not None
                        if len(current.events) == 1:
                            transaction_store.record_campaign_terminal(
                                sequence=sequence,
                                admission_sha256=admission,
                                terminal_receipt=durable_step["terminal"],
                            )
                        current = transaction_store.load(
                            sequence=sequence,
                            admission_sha256=admission,
                            load_tensors=False,
                        )
                        assert current is not None
                        if len(current.events) == 2:
                            transaction_store.record_trainer_checkpoint(
                                sequence=sequence,
                                admission_sha256=admission,
                                checkpoint=load_trainer_checkpoint_evidence(resumed.checkpoint_dir),
                            )
                    sealed = transaction_store.load(
                        sequence=sequence,
                        admission_sha256=admission,
                        load_tensors=False,
                    )
                    if sealed is None or len(sealed.events) != 3:
                        raise GRPOCheckpointError(
                            "historical verified transaction seal is incomplete"
                        )
                    continue
                if sequence > step or pending_recovery is not None:
                    raise GRPOCheckpointError(
                        "verified transaction sequence is ahead of trainer state"
                    )
                pending_recovery = (sequence, admission)

            if pending_recovery is not None:
                sequence, admission = pending_recovery
                expected_optimizer_layout = dict(tree_flatten(optimizer.state))

                def restore_and_validate_staged_state(
                    staged: Any,
                ) -> str:
                    if staged.adapter_tensors is None or staged.optimizer_tensors is None:
                        raise GRPOCheckpointError(
                            "verified transaction recovery tensors are missing"
                        )
                    _assert_exact_tensor_layout(
                        expected_adapters,
                        staged.adapter_tensors,
                        role="adapter",
                    )
                    _assert_exact_tensor_layout(
                        expected_optimizer_layout,
                        staged.optimizer_tensors,
                        role="optimizer",
                    )
                    model.load_weights(list(staged.adapter_tensors.items()), strict=False)
                    optimizer_state = tree_unflatten(staged.optimizer_tensors)
                    if not isinstance(optimizer_state, dict):
                        raise GRPOCheckpointError("verified transaction optimizer tree is invalid")
                    optimizer.state = optimizer_state
                    optimizer.init(model.trainable_parameters())
                    mx.eval(model.parameters(), optimizer.state)
                    observed_policy = recurrent_policy_sha256(model, execution_spec)
                    if observed_policy != staged.pending_step["policy_after_sha256"]:
                        raise GRPOCheckpointError(
                            "staged transaction tensors differ from policy_after"
                        )
                    return observed_policy

                recovered = verified_group_provider.recover_transaction_publications(
                    transaction_store=transaction_store,
                    sequence=sequence,
                    admission_sha256=admission,
                    validate_staged_state=restore_and_validate_staged_state,
                )
                pending = recovered.pending_step
                recovered_step = validate_verified_transition_step_receipt(
                    build_transaction_trainer_step(recovered),
                    group_size=config.group_size,
                    execution_spec_sha256=execution_spec.sha256,
                )
                transition_replay_groups = list(
                    verified_group_provider.accept_recovered_step_receipt(recovered_step)
                )
                step_receipts.append(recovered_step)
                advantage_report = recovered_step["advantage_report"]
                assert isinstance(telemetry, VerifiedTransitionTelemetry)
                telemetry.observe(advantage_report, optimizer_updated=True)
                recovered_task = tasks_by_id.get(recovered_step["task_id"])
                if recovered_task is None:
                    raise GRPOCheckpointError(
                        "recovered transaction task is outside the frozen dataset"
                    )
                answer_channel = recovered_step["answer_channel"]
                curriculum.observe(
                    recovered_task.domain,
                    recovered_task.depth,
                    float(answer_channel["correct_fraction"]),
                    degenerate=bool(advantage_report["degenerate"]),
                )
                step = int(recovered_step["step"])
                optimizer_updates += 1
                last_step_kind = "verified_optimizer_update"
                checkpoint_path = checkpoint_now()
                transaction_store.record_trainer_checkpoint(
                    sequence=sequence,
                    admission_sha256=admission,
                    checkpoint=load_trainer_checkpoint_evidence(checkpoint_path),
                )
                print(
                    f"[recovery] completed staged verified transition step={step}",
                    flush=True,
                )

            pending_rejection = None
            for transaction in rejection_transactions:
                intent = transaction.intent
                sequence = int(intent["sequence"])
                reward_sha256 = str(intent["reward_receipt_sha256"])
                if sequence < step:
                    if len(transaction.events) < 2:
                        if (
                            resumed is None
                            or sequence != step - 1
                            or int(intent["trainer_step"]) != step
                        ):
                            raise GRPOCheckpointError(
                                "historical rejected transaction is not fully sealed"
                            )
                        durable_step = step_receipts[sequence]
                        if (
                            durable_step.get("step_kind") != "verified_rejected_group"
                            or durable_step.get("reward_receipt_sha256") != reward_sha256
                        ):
                            raise GRPOCheckpointError(
                                "historical rejection differs from checkpoint"
                            )
                        if len(transaction.events) == 0:
                            rejection_transaction_store.record_campaign_terminal(
                                sequence=sequence,
                                reward_sha256=reward_sha256,
                                terminal_receipt=durable_step["terminal"],
                            )
                        current_rejection = rejection_transaction_store.load(
                            sequence=sequence,
                            reward_sha256=reward_sha256,
                        )
                        assert current_rejection is not None
                        if len(current_rejection.events) == 1:
                            rejection_transaction_store.record_trainer_checkpoint(
                                sequence=sequence,
                                reward_sha256=reward_sha256,
                                checkpoint_dir=resumed.checkpoint_dir,
                            )
                    sealed_rejection = rejection_transaction_store.load(
                        sequence=sequence,
                        reward_sha256=reward_sha256,
                    )
                    if sealed_rejection is None or len(sealed_rejection.events) != 2:
                        raise GRPOCheckpointError(
                            "historical rejected transaction seal is incomplete"
                        )
                    continue
                if sequence > step or pending_rejection is not None:
                    raise GRPOCheckpointError(
                        "rejected transaction sequence is ahead of trainer state"
                    )
                if pending_recovery is not None:
                    raise GRPOCheckpointError(
                        "update and rejection transactions both claim the next step"
                    )
                pending_rejection = (sequence, reward_sha256)

            if pending_rejection is not None:
                sequence, reward_sha256 = pending_rejection
                recovered_rejection = verified_group_provider.recover_rejection_publications(
                    rejection_store=rejection_transaction_store,
                    sequence=sequence,
                    reward_receipt_sha256=reward_sha256,
                    validate_live_policy=lambda: recurrent_policy_sha256(model, execution_spec),
                )
                recovered_step = validate_verified_transition_step_receipt(
                    build_rejected_transaction_trainer_step(recovered_rejection),
                    group_size=config.group_size,
                    execution_spec_sha256=execution_spec.sha256,
                )
                transition_replay_groups = list(
                    verified_group_provider.accept_recovered_step_receipt(recovered_step)
                )
                if len(transition_replay_groups) != optimizer_updates:
                    raise GRPOCheckpointError(
                        "rejection recovery changed the update replay group count"
                    )
                step_receipts.append(recovered_step)
                advantage_report = recovered_step["advantage_report"]
                assert isinstance(telemetry, VerifiedTransitionTelemetry)
                telemetry.observe(advantage_report, optimizer_updated=False)
                recovered_task = tasks_by_id.get(recovered_step["task_id"])
                if recovered_task is None:
                    raise GRPOCheckpointError(
                        "recovered rejection task is outside the frozen dataset"
                    )
                answer_channel = recovered_step["answer_channel"]
                curriculum.observe(
                    recovered_task.domain,
                    recovered_task.depth,
                    float(answer_channel["correct_fraction"]),
                    degenerate=bool(advantage_report["degenerate"]),
                )
                step = int(recovered_step["step"])
                last_step_kind = "verified_rejected_group"
                checkpoint_path = checkpoint_now()
                rejection_transaction_store.record_trainer_checkpoint(
                    sequence=sequence,
                    reward_sha256=reward_sha256,
                    checkpoint_dir=checkpoint_path,
                )
                print(
                    f"[recovery] completed staged verified rejection step={step}",
                    flush=True,
                )

        # Default-open: only an explicit refusal below closes it.
        training_allowed = True
        if pre_stage_recovery_halt is not None:
            training_allowed = False
        if resumed is None:
            checkpoint_path = checkpoint_now()
            print(
                f"[checkpoint 0] pretraining phases pending at {checkpoint_path.name}",
                flush=True,
            )
        if baseline_eval is None:
            if execution_spec is None:
                baseline_eval = evaluate_heldout(
                    model,
                    tokenizer,
                    holdout,
                    max_tokens=args.max_tokens,
                    envelope=envelope,
                    adapters_on=False,
                    progress_label="baseline-standard",
                )
                baseline_role = "frozen_pretraining_baseline"
            else:
                baseline_seed = _stable_seed(args.seed, "baseline")
                baseline_progress_identity = {
                    "schema": "aura.grpo.recurrent_baseline_identity.v1",
                    "protocol_sha256": protocol_sha256,
                    "dataset_sha256": dataset_sha256,
                    "execution_spec_sha256": execution_spec.sha256,
                    "base_checkpoint": base_identity,
                    "model_behavior": behavior_identity,
                    "max_tokens": args.max_tokens,
                    "seed": baseline_seed,
                    "adapters_on": False,
                    "task_ids": [task.task_id for task in holdout],
                }
                baseline_eval = evaluate_recurrent_heldout(
                    model,
                    tokenizer,
                    holdout,
                    spec=execution_spec,
                    max_tokens=args.max_tokens,
                    envelope=envelope,
                    adapters_on=False,
                    seed=baseline_seed,
                    progress_label="baseline-recurrent",
                    progress_path=out_dir / "baseline-progress.json",
                    progress_identity=baseline_progress_identity,
                )
                baseline_role = "frozen_base_recurrent_baseline"
            baseline_eval["step"] = 0
            baseline_eval["role"] = baseline_role
            print(
                f"[baseline 0] overall={baseline_eval['overall']:.3f} "
                f"by_depth={baseline_eval['accuracy_by_depth']}",
                flush=True,
            )

            # SCOPE REACHABILITY. The baseline has just told us WHERE the
            # episodes fail. If every one of those failures lives outside
            # the parameters this run may train, no amount of optimisation
            # can reduce the loss — it has no causal path to the failure.
            #
            # This is checked here, before calibration, because calibration
            # alone costs over an hour. Seven consecutive campaigns
            # (cp259/271/273/285/291/294/305) ran with adapter_scope
            # latent_slots_only against failures that were 100% decode-path
            # output-contract failures, burned ~86 minutes each, and
            # produced no gradient because every reward was zero.
            if execution_spec is not None:
                from core.learning.scope_reachability import (
                    assess as _assess_scope_reach,
                )
                from core.learning.scope_reachability import (
                    merge_reason_counts as _merge_reasons,
                )

                _reach = _assess_scope_reach(
                    _merge_reasons(
                        baseline_eval.get("score_reasons"),
                        baseline_eval.get("contract_reasons"),
                    ),
                    adapter_scope=getattr(
                        execution_spec,
                        "adapter_scope",
                        "",
                    ),
                )
                baseline_eval["scope_reachability"] = _reach.to_dict()
                if _reach.should_refuse:
                    print(
                        "[halt] trainable scope cannot reach the observed "
                        f"failures: {_reach.detail}",
                        flush=True,
                    )
                    if _reach.remedy:
                        print(f"[halt] remedy: {_reach.remedy}", flush=True)
                    training_allowed = False
                    halt_reason = "scope_unreachable"
            checkpoint_path = checkpoint_now()
            print(
                f"[checkpoint 0] baseline durable at {checkpoint_path.name}",
                flush=True,
            )
        elif isinstance(baseline_eval.get("scope_reachability"), Mapping) and (
            baseline_eval["scope_reachability"].get("verdict") == "unreachable"
        ):
            training_allowed = False
            halt_reason = "scope_unreachable"

        if args.calibrate and calibration is None and training_allowed:
            cal_group = min(config.group_size, args.calibrate_group)
            cal_tokens = _calibration_token_budget(args.max_tokens, args.calibrate_tokens)
            cells_sorted = sorted(by_cell)
            calibration_progress_path = out_dir / "calibration-progress.json"
            calibration_progress_identity = {
                "schema": "aura.grpo.recurrent_calibration_identity.v1",
                "protocol_sha256": protocol_sha256,
                "dataset_sha256": dataset_sha256,
                "execution_spec_sha256": (
                    execution_spec.sha256 if execution_spec is not None else None
                ),
                "base_checkpoint": base_identity,
                "model_behavior": behavior_identity,
                "max_tokens": cal_tokens,
                "group_size": cal_group,
                "samples_per_cell": args.calibrate_samples,
                "minutes": args.calibrate_minutes,
                "seed": args.seed,
                "cells": [
                    {
                        "family": family,
                        "difficulty": difficulty,
                        "task_ids": [task.task_id for task in by_cell[(family, difficulty)]],
                    }
                    for family, difficulty in cells_sorted
                ],
            }
            calibration_journal = _CalibrationProgressJournal.open(
                calibration_progress_path,
                identity=calibration_progress_identity,
                budget_s=args.calibrate_minutes * 60.0,
            )
            probe_counts: dict[tuple[str, int], int] = {}
            probes: list[dict[str, Any]] = []
            print(
                f"[calibrate] {len(cells_sorted)} cells x {cal_group} completions "
                f"x {cal_tokens} tokens, cap {args.calibrate_minutes}m "
                f"(resumed {calibration_journal.saved_count} probes; "
                f"{calibration_journal.remaining_s / 60.0:.1f}m remaining)",
                flush=True,
            )

            def _measure(
                family: str,
                difficulty: int,
            ) -> tuple[float, ...] | None:
                key = (family, difficulty)
                pool = by_cell.get(key)
                probe_index = probe_counts.get(key, 0)
                probe_counts[key] = probe_index + 1
                if not pool:
                    return None
                decision_seed = _stable_seed(
                    args.seed, "calibration", family, difficulty, probe_index
                )
                probe = pool[decision_seed % len(pool)]
                saved = calibration_journal.replay(
                    family=family,
                    difficulty=difficulty,
                    probe_index=probe_index,
                    task_id=probe.task_id,
                    seed=decision_seed,
                )
                if saved is not None:
                    probes.append(dict(saved))
                    return _calibration_reward_observations(saved["answer_channel"])
                if not calibration_journal.permits_new_probe():
                    return None
                probe_started = time.monotonic()
                if execution_spec is None:
                    with recurrence_adapter_scope(start=None, stop=None):
                        _, completions = sample_group(
                            model,
                            tokenizer,
                            probe,
                            size=cal_group,
                            max_tokens=cal_tokens,
                            temperature=args.temperature,
                            seed=decision_seed,
                        )
                else:
                    _, _samples, completions = sample_recurrent_group(
                        model,
                        tokenizer,
                        probe,
                        spec=execution_spec,
                        size=cal_group,
                        max_tokens=cal_tokens,
                        seed=decision_seed,
                    )
                grade_verdicts = [probe.grade(completion) for completion in completions]
                answer_channel = _answer_channel_report_from_verdicts(grade_verdicts)
                observations = tuple(
                    float(bool(verdict["correct"])) for verdict in grade_verdicts
                )
                if observations != _calibration_reward_observations(answer_channel):
                    # The helper reconstructs count-equivalent outcomes for
                    # replay. Order is irrelevant to Bernoulli cell statistics.
                    if sorted(observations) != sorted(
                        _calibration_reward_observations(answer_channel)
                    ):
                        raise RuntimeError(
                            "calibration answer-channel counts differ from verdicts"
                        )
                rate = sum(observations) / len(observations)
                probes.append(
                    {
                        "family": family,
                        "difficulty": difficulty,
                        "probe_index": probe_index,
                        "task_id": probe.task_id,
                        "seed": decision_seed,
                        "pass_rate": round(rate, 6),
                        "answer_channel": answer_channel,
                        "elapsed_s": round(time.monotonic() - probe_started, 6),
                    }
                )
                calibration_journal.append(probes[-1])
                print(
                    f"[calibrate] {family}@{difficulty} pass={rate:.2f} "
                    f"({elapsed_training_s() / 60.0:.1f}m)",
                    flush=True,
                )
                return observations

            curriculum = warm_start_pass_rates(
                sorted({domain for domain, _depth in by_cell}),
                sorted({depth for _domain, depth in by_cell}),
                _measure,
                samples_per_cell=args.calibrate_samples,
            )
            calibration_journal.finish()
            curriculum_report = curriculum.report()
            expected_probes = len(cells_sorted) * max(2, args.calibrate_samples)
            calibration = {
                **curriculum_report,
                "max_tokens": cal_tokens,
                "group_size": cal_group,
                "probes": probes,
                "expected_probes": expected_probes,
                "partial": len(probes) < expected_probes,
            }
            calibration["answer_channel"] = _merge_answer_channel_reports(
                [{"answer_channel": probe["answer_channel"]} for probe in probes]
            )
            calibration["admission"] = _calibration_admission_report(
                calibration,
                allow_unexplored_frontier=execution_spec is None,
            )
            print(f"[calibrate] {calibration}", flush=True)
            training_allowed = training_allowed and bool(
                calibration["admission"]["training_admitted"]
            )

        # Step zero is durable only after the true frozen baseline and any
        # calibration are complete. A restart cannot silently recompute them
        # under different random process state.
        checkpoint_path = checkpoint_now()

        requested_signal: int | None = None

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal requested_signal
            if requested_signal is None:
                requested_signal = int(signum)
                print(
                    f"[signal] {signal.Signals(signum).name}; stopping after "
                    "the current committed step",
                    flush=True,
                )

        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in previous_handlers:
            signal.signal(signum, request_stop)

        halt_reason = (
            "pre_stage_admission_burned"
            if pre_stage_recovery_halt is not None
            else "calibration_not_admitted"
            if calibration and not bool(calibration.get("admission", {}).get("training_admitted"))
            else "no_reachable_frontier"
            if not training_allowed
            else "max_steps"
        )
        active_recurrent_step: dict[str, Any] | None = None
        try:
            while training_allowed and step < args.max_steps:
                pause_request = _operator_pause_request(
                    out_dir,
                    adapter_id=args.adapter_id,
                    protocol_sha256=protocol_sha256,
                )
                if pause_request is not None:
                    checkpoint_path = checkpoint_now()
                    policy_sha256 = (
                        str(step_receipts[-1]["policy_after_sha256"])
                        if step_receipts
                        else None
                    )
                    receipt_path = _acknowledge_operator_pause(
                        out_dir,
                        request=pause_request,
                        step=step,
                        checkpoint_path=checkpoint_path,
                        policy_sha256=policy_sha256,
                    )
                    halt_reason = "operator_pause"
                    print(
                        f"[pause] durable step={step} receipt={receipt_path}",
                        flush=True,
                    )
                    break
                if requested_signal is not None:
                    halt_reason = "interrupted"
                    break
                if time.monotonic() >= deadline:
                    halt_reason = "wall_clock_budget"
                    break

                canary_step_started_monotonic_ns = time.monotonic_ns()
                canary_step_started_at_unix_ns = time.time_ns()
                step_number = step + 1
                active_transaction_coordinator = None
                if execution_spec is None:
                    decision_rng = random.Random(_stable_seed(args.seed, "curriculum", step_number))
                    cell = curriculum.sample(decision_rng)
                    pool = by_cell.get(cell) or train_tasks
                    task_rng = random.Random(_stable_seed(args.seed, "task", step_number))
                    task = pool[task_rng.randrange(len(pool))]
                    sample_seed = _stable_seed(args.seed, "group", step_number, task.task_id)
                else:
                    assert verified_group_provider is not None
                    task, sample_seed = _scheduled_verified_training_task(
                        verified_group_provider,
                        tasks_by_id,
                        campaign_sequence=step_number - 1,
                    )
                recurrent_samples = None
                if execution_spec is not None:
                    active_recurrent_step = {
                        "attempted_step": step_number,
                        "phase": "sampling",
                        "task_id": task.task_id,
                        "sample_seed": sample_seed,
                        "samples": (),
                    }
                if execution_spec is None:
                    with recurrence_adapter_scope(start=None, stop=None):
                        prompt, completions = sample_group(
                            model,
                            tokenizer,
                            task,
                            size=config.group_size,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            seed=sample_seed,
                        )
                else:
                    prompt, recurrent_samples, completions = sample_recurrent_group(
                        model,
                        tokenizer,
                        task,
                        spec=execution_spec,
                        size=config.group_size,
                        max_tokens=args.max_tokens,
                        seed=sample_seed,
                        verified_group_provider=verified_group_provider,
                        campaign_sequence=step_number - 1,
                        model_path=model_path,
                        token_trace_adapter=token_trace_adapter,
                    )
                    active_recurrent_step["samples"] = tuple(recurrent_samples)
                    active_recurrent_step["phase"] = "grading"
                grade_verdicts = [task.grade(text) for text in completions]
                answer_channel = _answer_channel_report_from_verdicts(grade_verdicts)
                loss_value: float | None = None
                step_kind = "degenerate_group"
                if execution_spec is None:
                    rewards = [
                        reward_from_verdict(verdict, format_credit=args.format_credit)
                        for verdict in grade_verdicts
                    ]
                    verifier_advantage_report = group_advantages(
                        rewards, clip=config.advantage_clip
                    )
                    effective_rewards = list(rewards)
                    advantage_report = verifier_advantage_report
                    if not advantage_report["degenerate"]:
                        reference = [
                            mx.stop_gradient(
                                completion_logprob(
                                    model,
                                    tokenizer,
                                    prompt,
                                    text,
                                    adapters_on=False,
                                )
                            )
                            for text in completions
                        ]

                        def loss_fn(
                            _model,
                            *,
                            _prompt=prompt,
                            _completions=tuple(completions),
                            _advantages=tuple(advantage_report["advantages"]),
                            _reference=tuple(reference),
                        ):
                            policy = [
                                completion_logprob(
                                    _model,
                                    tokenizer,
                                    _prompt,
                                    text,
                                    adapters_on=True,
                                )
                                for text in _completions
                            ]
                            loss, _report = grpo_loss(
                                policy,
                                _advantages,
                                reference_logprobs=_reference,
                                kl_coefficient=config.kl_coefficient,
                            )
                            return loss

                        loss, grads = nn.value_and_grad(model, loss_fn)(model)
                        loss_value = float(loss)
                        optimizer.update(model, grads)
                        mx.eval(model.parameters(), optimizer.state)
                        optimizer_updates += 1
                        step_kind = "optimizer_update"
                        del grads
                        envelope.reclaim(force=True)
                else:
                    if recurrent_samples is None or recurrent_config is None:
                        raise RuntimeError("recurrent training state is missing")
                    assert verified_group_provider is not None
                    active_recurrent_step["phase"] = "verified_evidence"
                    prepared = verified_group_provider.prepare_group(
                        sequence=step_number - 1,
                        task=task,
                        prompt_text=_render(tokenizer, task),
                        prompt_tokens=prompt,
                        samples=recurrent_samples,
                        completions=completions,
                    )
                    if prepared.campaign_sequence != step_number - 1:
                        raise RuntimeError(
                            "verified transition provider returned a different sequence"
                        )
                    assert transaction_store is not None
                    assert rejection_transaction_store is not None
                    trainer_step_static = build_verified_transition_step_static(
                        samples=recurrent_samples,
                        reward_receipt=prepared.reward_receipt,
                        answer_channel=answer_channel,
                    )
                    transaction_coordinator = VerifiedTransitionTransactionCoordinator(
                        store=transaction_store,
                        sequence=step_number - 1,
                        trainer_step=step_number,
                        task_id=task.task_id,
                        trainer_sample_seed=sample_seed,
                        execution_spec_sha256=execution_spec.sha256,
                        campaign_manifest_sha256=(prepared.campaign_manifest_sha256),
                        campaign_schedule_root_sha256=(prepared.campaign_schedule_root_sha256),
                        group_manifest_sha256=str(prepared.group_manifest["manifest_sha256"]),
                        reward_receipt_sha256=str(prepared.reward_receipt["receipt_sha256"]),
                        trainer_step_static=trainer_step_static,
                        adapter_tensors=adapter_tensors,
                        optimizer_tensors=lambda: dict(tree_flatten(optimizer.state)),
                        measurement_chain=measurement_chain_store,
                    )
                    rejection_transaction_coordinator = (
                        VerifiedTransitionRejectionTransactionCoordinator(
                            store=rejection_transaction_store,
                            sequence=step_number - 1,
                            trainer_step=step_number,
                            task_id=task.task_id,
                            trainer_sample_seed=sample_seed,
                            execution_spec_sha256=execution_spec.sha256,
                            campaign_manifest_sha256=(prepared.campaign_manifest_sha256),
                            campaign_schedule_root_sha256=(prepared.campaign_schedule_root_sha256),
                            group_manifest_sha256=str(prepared.group_manifest["manifest_sha256"]),
                            reward_receipt_sha256=str(prepared.reward_receipt["receipt_sha256"]),
                            trainer_step_static=trainer_step_static,
                        )
                    )
                    active_recurrent_step["phase"] = "verified_update"
                    mutation = apply_prepared_verified_transition_group(
                        model,
                        optimizer,
                        prompt,
                        recurrent_samples,
                        prepared,
                        spec=execution_spec,
                        config=recurrent_config,
                        trajectory_group_config=trajectory_group_config,
                        transaction_coordinator=transaction_coordinator,
                        rejection_transaction_coordinator=(rejection_transaction_coordinator),
                    )
                    effective_rewards = list(mutation.structured_rewards)
                    advantage_report = group_advantages(
                        effective_rewards, clip=config.advantage_clip
                    )
                    if mutation.optimizer_updated:
                        if mutation.replay_group is None:
                            raise RuntimeError("verified update omitted its source replay group")
                        transition_replay_groups.append(mutation.replay_group)
                        optimizer_updates += 1
                        step_kind = "verified_optimizer_update"
                        envelope.reclaim(force=True)
                    else:
                        if mutation.replay_group is not None:
                            raise RuntimeError("rejected transition exposed an update replay group")
                        step_kind = "verified_rejected_group"
                    verified_step_receipt = build_verified_transition_step_receipt(
                        step_number=step_number,
                        task_id=task.task_id,
                        sample_seed=sample_seed,
                        execution_spec_sha256=execution_spec.sha256,
                        samples=recurrent_samples,
                        answer_channel=answer_channel,
                        mutation=mutation,
                    )
                    verified_group_provider.accept_step_receipt(verified_step_receipt)
                    step_receipts.append(verified_step_receipt)
                    active_transaction_coordinator = (
                        transaction_coordinator
                        if mutation.optimizer_updated
                        else rejection_transaction_coordinator
                    )

                # State mutates only after a complete optimizer update or a
                # fully graded degenerate group. The durable step is therefore
                # always replay-safe.
                if isinstance(telemetry, VerifiedTransitionTelemetry):
                    telemetry.observe(
                        advantage_report,
                        optimizer_updated=step_kind == "verified_optimizer_update",
                    )
                else:
                    telemetry.observe(advantage_report)
                curriculum.observe(
                    task.domain,
                    task.depth,
                    (
                        float(answer_channel["correct_fraction"])
                        if execution_spec is not None
                        else float(advantage_report["mean_reward"])
                    ),
                    degenerate=advantage_report["degenerate"],
                )
                step = step_number
                last_step_kind = step_kind
                no_signal = _should_halt_for_no_learning_signal(
                    telemetry,
                    config,
                    min_groups=args.min_signal_groups,
                    step_receipts=step_receipts,
                )
                if active_recurrent_step is not None:
                    active_recurrent_step["phase"] = "durable_checkpoint"

                # A completed mutation is durable before any held-out work.
                # Evaluation may be expensive or externally interrupted; it
                # must never obscure a policy update that already committed.
                if active_transaction_coordinator is not None or step % args.checkpoint_every == 0:
                    checkpoint_path = checkpoint_now()
                    if execution_spec is not None and active_transaction_coordinator is not None:
                        active_transaction_coordinator.record_trainer_checkpoint(checkpoint_path)
                if args.fixed_update_canary:
                    _publish_update_canary_step_timing(
                        out_dir,
                        adapter_id=args.adapter_id,
                        protocol_sha256=protocol_sha256,
                        step_receipt=step_receipts[-1],
                        checkpoint_path=checkpoint_path,
                        started_at_unix_ns=canary_step_started_at_unix_ns,
                        started_monotonic_ns=canary_step_started_monotonic_ns,
                        finished_at_unix_ns=time.time_ns(),
                        finished_monotonic_ns=time.monotonic_ns(),
                    )
                if active_recurrent_step is not None:
                    active_recurrent_step["phase"] = "post_update_evaluation"

                if step % 10 == 0:
                    detail = f"loss={loss_value:.4f}" if loss_value is not None else "degenerate"
                    print(
                        f"[step {step}] {detail} "
                        f"mean_r={advantage_report['mean_reward']:.2f} "
                        f"({elapsed_training_s() / 60.0:.1f}m)",
                        flush=True,
                    )

                if step % args.eval_every == 0:
                    if execution_spec is None:
                        report = evaluate_heldout(
                            model,
                            tokenizer,
                            holdout,
                            max_tokens=args.max_tokens,
                            envelope=envelope,
                            adapters_on=True,
                            progress_label=f"eval-standard-{step}",
                        )
                        report_role = "adapter_standard_decode"
                    else:
                        report = evaluate_recurrent_heldout(
                            model,
                            tokenizer,
                            holdout,
                            spec=execution_spec,
                            max_tokens=args.max_tokens,
                            envelope=envelope,
                            adapters_on=True,
                            seed=_stable_seed(args.seed, "eval", step),
                            progress_label=f"eval-recurrent-{step}",
                        )
                        report_role = "adapter_recurrent_decode"
                    report["step"] = step
                    report["role"] = report_role
                    history.append(report)
                    print(
                        f"[eval {step}] overall={report['overall']:.3f} "
                        f"delta={_point_estimate_delta(baseline_eval, report)} "
                        f"by_depth={report['accuracy_by_depth']}",
                        flush=True,
                    )
                if no_signal is not None and not args.fixed_update_canary:
                    training_allowed = False
                    halt_reason = "no_learning_signal"
                    checkpoint_path = checkpoint_now()
                    print(
                        "[halt] no learning signal after "
                        f"{no_signal['groups']} groups: {no_signal['diagnosis']}",
                        flush=True,
                    )
                    break
                if (
                    not args.fixed_update_canary
                    and not curriculum.report()["has_reachable_frontier"]
                ):
                    training_allowed = False
                    halt_reason = "frontier_exhausted"
                    print(
                        "[halt] every measured curriculum cell is saturated or hopeless",
                        flush=True,
                    )
                active_recurrent_step = None

            if step >= args.max_steps:
                halt_reason = "max_steps"
            if (
                requested_signal is None
                and training_allowed
                and (not history or history[-1].get("step") != step)
            ):
                if execution_spec is None:
                    report = evaluate_heldout(
                        model,
                        tokenizer,
                        holdout,
                        max_tokens=args.max_tokens,
                        envelope=envelope,
                        adapters_on=True,
                    )
                    report_role = "adapter_standard_decode"
                else:
                    report = evaluate_recurrent_heldout(
                        model,
                        tokenizer,
                        holdout,
                        spec=execution_spec,
                        max_tokens=args.max_tokens,
                        envelope=envelope,
                        adapters_on=True,
                        seed=_stable_seed(args.seed, "eval", step),
                    )
                    report_role = "adapter_recurrent_decode"
                report["step"] = step
                report["role"] = report_role
                history.append(report)
            checkpoint_path = checkpoint_now()
        except Exception as exc:
            if execution_spec is not None:
                context = active_recurrent_step or {
                    "attempted_step": step + 1,
                    "phase": "between_steps",
                    "task_id": None,
                    "sample_seed": None,
                    "samples": (),
                }
                failure_path = _record_recurrent_step_failure(
                    out_dir,
                    protocol_sha256=protocol_sha256,
                    dataset_sha256=dataset_sha256,
                    execution_spec_sha256=execution_spec.sha256,
                    attempted_step=context["attempted_step"],
                    last_durable_step=last_durable_step,
                    phase=context["phase"],
                    task_id=context["task_id"],
                    sample_seed=context["sample_seed"],
                    samples=context["samples"],
                    error=exc,
                )
                print(f"[failure-receipt] {failure_path}", flush=True)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        training_adequacy: dict[str, Any] | None = None
        if execution_spec is not None and halt_reason == "max_steps":
            preview_learning_signal = _signal_admission_report(
                telemetry.verdict(config),
                step_receipts=step_receipts,
            )
            training_adequacy = recurrent_training_adequacy_report(
                step_receipts=step_receipts,
                scheduled_task_ids=[task.task_id for task in train_tasks],
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                evaluation_steps=[
                    int(report["step"])
                    for report in history
                    if type(report.get("step")) is int
                ],
                learning_signal=preview_learning_signal,
            )
            print(f"[training-adequacy] {training_adequacy}", flush=True)
            if not training_adequacy["admitted"]:
                halt_reason = "training_adequacy_failed"
                training_allowed = False

        if execution_spec is not None and not _verified_campaign_halt_is_resumable(
            halt_reason
        ):
            assert verified_group_provider is not None
            transition_closure = verified_group_provider.finalize(
                completed_groups=step,
                halt_reason=halt_reason,
                replay_groups=tuple(transition_replay_groups),
            )
            if not isinstance(transition_closure, VerifiedTransitionCampaignClosure):
                raise RuntimeError(
                    "verified transition provider returned an invalid campaign closure"
                )

        adapters = adapter_tensors()
        _publish_adapter_snapshot(out_dir / "grpo_adapters.safetensors", adapters)
        curriculum_report = curriculum.report()
        print(f"[curriculum] {curriculum_report}", flush=True)

    learning_signal = _signal_admission_report(
        telemetry.verdict(config),
        step_receipts=step_receipts,
    )
    final = history[-1] if history else None
    delta = _point_estimate_delta(baseline_eval, final)
    completed = halt_reason == "max_steps"
    receipt = {
        "schema": GRPO_TRAIN_SCHEMA,
        "adapter_id": args.adapter_id,
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "model": {
            "path": model_path,
            "base_checkpoint": base_identity,
            "behavior": behavior_identity,
        },
        "config": config.to_receipt(),
        "execution_mode": args.execution_mode,
        "execution_spec": (execution_spec.to_dict() if execution_spec is not None else None),
        "execution_spec_sha256": (execution_spec.sha256 if execution_spec is not None else None),
        "domains": domains,
        "depths": depths,
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout),
        "steps": step,
        "optimizer_updates": optimizer_updates,
        "invocation_count": invocation_count,
        "termination": {
            "reason": halt_reason,
            "completed_budget": completed,
            "signal": requested_signal,
        },
        "learning_signal": learning_signal,
        "training_adequacy": training_adequacy,
        "curriculum": curriculum_report,
        "calibration": calibration,
        "baseline": baseline_eval,
        "history": history,
        "step_receipts": step_receipts,
        "final": final,
        "adapter_decode_delta": delta,
        "adapter_standard_decode_delta": (delta if execution_spec is None else None),
        "adapter_recurrent_decode_delta": (delta if execution_spec is not None else None),
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "verdict": {
            "had_signal": bool(learning_signal["learning_signal"]),
            "point_estimate_improved": bool(delta is not None and delta > 0.0),
            "causal_gain_proven": False,
            "causal_gain_blocker": (
                "requires fresh powered base/adapter x standard/RLC factorial gate"
            ),
            "diagnosis": learning_signal["diagnosis"],
        },
        "elapsed_minutes": round((time.time() - started_wall) / 60.0, 2),
    }
    receipt_bytes = canonical_json_bytes(receipt)
    atomic_write_bytes(out_dir / "grpo_receipt.json", receipt_bytes, mode=0o600)
    if execution_spec is not None and halt_reason == "max_steps":
        if transition_closure is None:
            raise RuntimeError("verified transition campaign closure is missing")
        identity = _publish_recurrent_adapter_bundle(
            out_dir,
            adapter_id=args.adapter_id,
            protocol=protocol,
            protocol_bytes=protocol_bytes,
            dataset_bytes=dataset_bytes,
            receipt=receipt,
            receipt_bytes=receipt_bytes,
            execution_spec=execution_spec,
            source_roles=source_files,
            transition_closure=transition_closure,
            transition_groups=tuple(transition_replay_groups),
        )
        if args.fixed_update_canary:
            marker = {
                "schema": "aura.recurrent_grpo.non_promotable_canary.v1",
                "adapter_id": args.adapter_id,
                "adapter_sha256": identity["adapter_sha256"],
                "composite_identity_sha256": identity["composite_identity_sha256"],
                "reasoning_gain_proven": False,
                "frontier_level_proven": False,
                "runtime_promotion_allowed": False,
                "purpose": "resident_signed_update_and_latency_engineering_canary",
            }
            _publish_immutable_bytes(
                out_dir / "NON_PROMOTABLE_CANARY.json",
                canonical_json_bytes(marker),
                role="non-promotable canary marker",
            )
        print(
            "[campaign-adapter] "
            f"identity={identity['composite_identity_sha256']} "
            f"adapter={identity['adapter_sha256']}",
            flush=True,
        )
    print(f"[verdict] {receipt['verdict']}", flush=True)
    print(f"[receipt] {out_dir / 'grpo_receipt.json'}", flush=True)
    if requested_signal is not None:
        return 128 + requested_signal
    if not training_allowed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
