"""Exclusive trainer boundary for source-verified recurrent mutations."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from core.brain.llm.latent_cortex.campaign_trust import VerifiedCampaignTrustPolicy
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.grpo import GRPOConfig, group_advantages
from core.learning.grpo_training_state import canonical_json_bytes
from core.learning.recurrent_grpo import recurrent_policy_sha256
from core.learning.verified_transition_campaign import (
    VerifiedTransitionCampaignLedger,
)
from core.learning.verified_transition_episode import TransitionArtifactStore
from core.learning.verified_transition_reward import (
    MICROS,
    VerifiedTransitionEvidence,
    rewards_for_recurrent_samples,
    validate_verified_transition_reward_batch,
)
from core.learning.verified_transition_training_evidence import (
    VerifiedTransitionReplayGroup,
)
from core.learning.verified_transition_transaction import build_trainer_step_static
from core.learning.verified_transition_update import (
    VerifiedTransitionUpdateJournal,
    apply_verified_transition_group_update,
)

VERIFIED_TRANSITION_STEP_SCHEMA = "aura.verified_transition.trainer_step.v1"
VERIFIED_TRANSITION_TELEMETRY_SCHEMA = "aura.verified_transition.telemetry.v1"
_STEP_KEYS = frozenset(
    {
        "schema",
        "step",
        "campaign_sequence",
        "task_id",
        "sample_seed",
        "execution_spec_sha256",
        "samples",
        "structured_rewards",
        "reward_receipt_sha256",
        "group_manifest_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "optimizer_admission_reason",
        "answer_channel",
        "advantage_report",
        "step_kind",
        "update",
        "terminal",
        "policy_before_sha256",
        "policy_after_sha256",
        "receipt_sha256",
    }
)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(sealed)
    ).hexdigest()
    return sealed


def _sample_receipts(samples: Sequence[Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for sample in samples:
        receipt = sample.receipt()
        if not isinstance(receipt, Mapping):
            raise ValueError("verified_transition_sample_receipt_invalid")
        receipts.append(dict(receipt))
    return receipts


@dataclass(frozen=True, slots=True)
class PreparedVerifiedTransitionGroup:
    """All independently replayable inputs for one planned campaign group."""

    campaign_sequence: int
    transition_store: TransitionArtifactStore
    reward_receipt: Mapping[str, Any]
    transition_evidence: Sequence[VerifiedTransitionEvidence]
    group_manifest: Mapping[str, Any]
    group_manifest_attestation: Mapping[str, Any]
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]]
    token_encoder: Callable[[bytes], Sequence[int]]
    token_decoder: Callable[[Sequence[int]], bytes]
    campaign_ledger: VerifiedTransitionCampaignLedger
    campaign_trust_policy: VerifiedCampaignTrustPolicy
    group_admission_receipt: Mapping[str, Any] | None = None
    update_journal: VerifiedTransitionUpdateJournal | None = None
    campaign_manifest_sha256: str = ""
    campaign_schedule_root_sha256: str = ""


@dataclass(frozen=True, slots=True)
class VerifiedTransitionSamplingEntry:
    """One externally signed sample identity exposed before generation."""

    episode_id: str
    rng_root_sha256: str
    producing_branch_index: int
    sample_seed: int
    sampling_config_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedTransitionSamplingPlan:
    """Exact admitted group plan consumed by the proof-grade sampler."""

    campaign_sequence: int
    group_manifest_sha256: str
    task_id: str
    policy_sha256: str
    prompt_tokens_sha256: str
    execution_spec_sha256: str
    entries: tuple[VerifiedTransitionSamplingEntry, ...]
    sampling_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedTransitionTrainingScheduleEntry:
    """Provider-owned task and group RNG identity for one trainer step."""

    campaign_sequence: int
    task_id: str
    trainer_sample_seed: int


@dataclass(frozen=True, slots=True)
class VerifiedTransitionProviderRuntime:
    """Live objects available only after recurrent adapters are attached."""

    model: Any
    tokenizer: Any
    execution_spec: RLCExecutionSpec
    training_tasks: tuple[Any, ...]
    output_directory: Path
    transaction_root: Path
    dataset_sha256: str
    group_size: int
    sampling_max_tokens: int


@dataclass(frozen=True, slots=True)
class VerifiedTransitionMutationResult:
    """Trainer-facing result with no unverified reward channel."""

    campaign_sequence: int
    group_manifest_sha256: str
    optimizer_updated: bool
    structured_rewards: tuple[float, ...]
    optimizer_admission_reason: str
    reward_receipt_sha256: str
    group_admission_sha256: str | None
    update_receipt_sha256: str | None
    update_receipt: Mapping[str, Any] | None
    terminal_receipt: Mapping[str, Any] | None
    policy_before_sha256: str
    policy_after_sha256: str
    replay_group: VerifiedTransitionReplayGroup | None


@dataclass(frozen=True, slots=True)
class VerifiedTransitionCampaignClosure:
    """Closed campaign material required by the final adapter identity."""

    campaign_ledger: VerifiedTransitionCampaignLedger
    campaign_trust_policy: VerifiedCampaignTrustPolicy


class VerifiedTransitionGroupProvider(Protocol):
    """Trusted campaign producer interface; it never receives the optimizer."""

    @property
    def contract_sha256(self) -> str: ...

    def training_schedule_entry(
        self, *, sequence: int
    ) -> VerifiedTransitionTrainingScheduleEntry: ...

    def sampling_plan(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        policy_sha256: str,
    ) -> VerifiedTransitionSamplingPlan: ...

    def prepare_group(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        samples: Sequence[Any],
        completions: Sequence[str],
    ) -> PreparedVerifiedTransitionGroup: ...

    def restore_groups(
        self,
        *,
        committed_steps: int,
        step_receipts: Sequence[Mapping[str, Any]],
    ) -> Sequence[VerifiedTransitionReplayGroup]: ...

    def accept_step_receipt(
        self, receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def recover_transaction_publications(
        self,
        *,
        transaction_store: Any,
        sequence: int,
        admission_sha256: str,
        validate_staged_state: Callable[[Any], str],
    ) -> Any: ...

    def recover_rejection_publications(
        self,
        *,
        rejection_store: Any,
        sequence: int,
        reward_receipt_sha256: str,
        validate_live_policy: Callable[[], str],
    ) -> Any: ...

    def accept_recovered_step_receipt(
        self, receipt: Mapping[str, Any]
    ) -> Sequence[VerifiedTransitionReplayGroup]: ...

    def finalize(
        self,
        *,
        completed_groups: int,
        halt_reason: str,
        replay_groups: Sequence[VerifiedTransitionReplayGroup],
    ) -> VerifiedTransitionCampaignClosure: ...


class VerifiedTransitionGroupProviderFactory(Protocol):
    """Construct the provider only after the live recurrent policy exists."""

    @property
    def contract_sha256(self) -> str: ...

    def bind_training_tasks(self, tasks: Sequence[Any]) -> Sequence[Any]: ...

    def create(
        self, runtime: VerifiedTransitionProviderRuntime
    ) -> VerifiedTransitionGroupProvider: ...


@dataclass(slots=True)
class VerifiedTransitionTelemetry:
    """Signal telemetry for signed transition deltas, which may be negative."""

    groups: int = 0
    admitted_groups: int = 0
    rejected_groups: int = 0
    degenerate: int = 0
    reward_sum: float = 0.0

    def observe(self, report: Mapping[str, Any], *, optimizer_updated: bool) -> None:
        mean = float(report["mean_reward"])
        if not math.isfinite(mean):
            raise ValueError("verified_transition_mean_reward_invalid")
        self.groups += 1
        self.reward_sum += mean
        self.degenerate += int(bool(report["degenerate"]))
        self.admitted_groups += int(optimizer_updated)
        self.rejected_groups += int(not optimizer_updated)

    def state(self) -> dict[str, Any]:
        return {
            "schema": VERIFIED_TRANSITION_TELEMETRY_SCHEMA,
            "groups": self.groups,
            "admitted_groups": self.admitted_groups,
            "rejected_groups": self.rejected_groups,
            "degenerate": self.degenerate,
            "reward_sum": self.reward_sum,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> VerifiedTransitionTelemetry:
        required = {
            "schema",
            "groups",
            "admitted_groups",
            "rejected_groups",
            "degenerate",
            "reward_sum",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("verified_transition_telemetry_state_invalid")
        if state.get("schema") != VERIFIED_TRANSITION_TELEMETRY_SCHEMA:
            raise ValueError("verified_transition_telemetry_schema_invalid")
        integers: dict[str, int] = {}
        for field in ("groups", "admitted_groups", "rejected_groups", "degenerate"):
            value = state.get(field)
            if type(value) is not int or value < 0:
                raise ValueError("verified_transition_telemetry_state_invalid")
            integers[field] = value
        reward_sum = state.get("reward_sum")
        if (
            isinstance(reward_sum, bool)
            or not isinstance(reward_sum, (int, float))
            or not math.isfinite(float(reward_sum))
            or integers["admitted_groups"] + integers["rejected_groups"]
            != integers["groups"]
            or integers["degenerate"] > integers["groups"]
        ):
            raise ValueError("verified_transition_telemetry_state_invalid")
        return cls(**integers, reward_sum=float(reward_sum))

    def verdict(self, config: GRPOConfig) -> dict[str, Any]:
        if self.groups == 0:
            return {
                "schema": VERIFIED_TRANSITION_TELEMETRY_SCHEMA,
                "groups": 0,
                "learning_signal": False,
                "diagnosis": "no_verified_transition_groups_observed",
            }
        degenerate_fraction = self.degenerate / self.groups
        admitted_fraction = self.admitted_groups / self.groups
        learning_signal = (
            self.admitted_groups > 0
            and degenerate_fraction <= config.max_degenerate_fraction
        )
        if self.admitted_groups == 0:
            diagnosis = "all_verified_transition_groups_rejected"
        elif degenerate_fraction > config.max_degenerate_fraction:
            diagnosis = "verified_transition_reward_variance_insufficient"
        else:
            diagnosis = "healthy_verified_transition_signal"
        return {
            "schema": VERIFIED_TRANSITION_TELEMETRY_SCHEMA,
            "groups": self.groups,
            "usable_groups": self.admitted_groups,
            "rejected_groups": self.rejected_groups,
            "admitted_fraction": round(admitted_fraction, 4),
            "degenerate_fraction": round(degenerate_fraction, 4),
            "mean_reward": round(self.reward_sum / self.groups, 6),
            "learning_signal": learning_signal,
            "diagnosis": diagnosis,
        }


def _structured_rewards(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    transitions = receipt.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("verified_transition_reward_rows_missing")
    rewards: list[float] = []
    for row in transitions:
        if not isinstance(row, Mapping) or type(row.get("reward_micros")) is not int:
            raise ValueError("verified_transition_reward_row_invalid")
        rewards.append(cast(int, row["reward_micros"]) / MICROS)
    return tuple(rewards)


def validate_verified_transition_step_receipt(
    receipt: Mapping[str, Any],
    *,
    group_size: int,
    execution_spec_sha256: str,
) -> dict[str, Any]:
    """Replay the trainer-level envelope around source-verified mutation."""

    if not isinstance(receipt, Mapping) or set(receipt) != _STEP_KEYS:
        raise ValueError("verified_transition_step_receipt_schema_invalid")
    normalized = dict(receipt)
    if normalized.get("schema") != VERIFIED_TRANSITION_STEP_SCHEMA:
        raise ValueError("verified_transition_step_receipt_version_invalid")
    observed = _sha256(
        normalized.get("receipt_sha256"), role="verified_transition_step_receipt"
    )
    unsigned = dict(normalized)
    unsigned.pop("receipt_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != observed:
        raise ValueError("verified_transition_step_receipt_digest_mismatch")
    if (
        type(normalized.get("step")) is not int
        or normalized["step"] < 1
        or normalized.get("campaign_sequence") != normalized["step"] - 1
        or not isinstance(normalized.get("task_id"), str)
        or not normalized["task_id"]
        or type(normalized.get("sample_seed")) is not int
        or normalized.get("execution_spec_sha256") != execution_spec_sha256
        or not isinstance(normalized.get("samples"), list)
        or len(normalized["samples"]) != group_size
        or not isinstance(normalized.get("answer_channel"), Mapping)
        or not isinstance(normalized.get("optimizer_admission_reason"), str)
        or not normalized["optimizer_admission_reason"]
    ):
        raise ValueError("verified_transition_step_identity_invalid")
    rewards = normalized.get("structured_rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != group_size
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in rewards
        )
    ):
        raise ValueError("verified_transition_step_rewards_invalid")
    expected_advantage = group_advantages([float(value) for value in rewards])
    if normalized.get("advantage_report") != expected_advantage:
        raise ValueError("verified_transition_step_advantage_mismatch")
    _sha256(
        normalized.get("reward_receipt_sha256"),
        role="verified_transition_step_reward_receipt",
    )
    _sha256(
        normalized.get("group_manifest_sha256"),
        role="verified_transition_step_group_manifest",
    )
    before = _sha256(
        normalized.get("policy_before_sha256"),
        role="verified_transition_step_policy_before",
    )
    after = _sha256(
        normalized.get("policy_after_sha256"),
        role="verified_transition_step_policy_after",
    )
    update = normalized.get("update")
    terminal = normalized.get("terminal")
    if normalized.get("step_kind") == "verified_optimizer_update":
        admission_sha256 = _sha256(
            normalized.get("group_admission_sha256"),
            role="verified_transition_step_group_admission",
        )
        update_sha256 = _sha256(
            normalized.get("update_receipt_sha256"),
            role="verified_transition_step_update_receipt",
        )
        if (
            not isinstance(update, Mapping)
            or update.get("schema") != "aura.verified_transition.update_receipt.v1"
            or update.get("optimizer_update_count") != 1
            or update.get("policy_before_sha256") != before
            or update.get("policy_after_sha256") != after
            or update.get("group_admission_sha256") != admission_sha256
            or update.get("receipt_sha256") != update_sha256
            or before == after
            or not isinstance(terminal, Mapping)
            or terminal.get("status") != "updated"
            or terminal.get("sequence") != normalized["campaign_sequence"]
            or terminal.get("group_manifest_sha256")
            != normalized["group_manifest_sha256"]
            or terminal.get("reward_receipt_sha256")
            != normalized["reward_receipt_sha256"]
            or terminal.get("group_admission_sha256") != admission_sha256
            or terminal.get("update_receipt_sha256") != update_sha256
        ):
            raise ValueError("verified_transition_step_update_invalid")
    elif normalized.get("step_kind") == "verified_rejected_group":
        if (
            update is not None
            or normalized.get("group_admission_sha256") is not None
            or normalized.get("update_receipt_sha256") is not None
            or not isinstance(terminal, Mapping)
            or terminal.get("status") != "rejected"
            or terminal.get("sequence") != normalized["campaign_sequence"]
            or terminal.get("group_manifest_sha256")
            != normalized["group_manifest_sha256"]
            or terminal.get("reward_receipt_sha256")
            != normalized["reward_receipt_sha256"]
            or before != after
        ):
            raise ValueError("verified_transition_step_rejection_invalid")
    else:
        raise ValueError("verified_transition_step_kind_invalid")
    return normalized


def build_verified_transition_step_receipt(
    *,
    step_number: int,
    task_id: str,
    sample_seed: int,
    execution_spec_sha256: str,
    samples: Sequence[Any],
    answer_channel: Mapping[str, Any],
    mutation: VerifiedTransitionMutationResult,
) -> dict[str, Any]:
    receipt = _seal(
        {
            "schema": VERIFIED_TRANSITION_STEP_SCHEMA,
            "step": step_number,
            "campaign_sequence": mutation.campaign_sequence,
            "task_id": task_id,
            "sample_seed": sample_seed,
            "execution_spec_sha256": execution_spec_sha256,
            "samples": _sample_receipts(samples),
            "structured_rewards": list(mutation.structured_rewards),
            "reward_receipt_sha256": mutation.reward_receipt_sha256,
            "group_manifest_sha256": mutation.group_manifest_sha256,
            "group_admission_sha256": mutation.group_admission_sha256,
            "update_receipt_sha256": mutation.update_receipt_sha256,
            "optimizer_admission_reason": mutation.optimizer_admission_reason,
            "answer_channel": dict(answer_channel),
            "advantage_report": group_advantages(mutation.structured_rewards),
            "step_kind": (
                "verified_optimizer_update"
                if mutation.optimizer_updated
                else "verified_rejected_group"
            ),
            "update": (
                dict(mutation.update_receipt)
                if mutation.update_receipt is not None
                else None
            ),
            "terminal": (
                dict(mutation.terminal_receipt)
                if mutation.terminal_receipt is not None
                else None
            ),
            "policy_before_sha256": mutation.policy_before_sha256,
            "policy_after_sha256": mutation.policy_after_sha256,
        }
    )
    return validate_verified_transition_step_receipt(
        receipt,
        group_size=len(samples),
        execution_spec_sha256=execution_spec_sha256,
    )


def build_verified_transition_step_static(
    *,
    samples: Sequence[Any],
    reward_receipt: Mapping[str, Any],
    answer_channel: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze every trainer receipt field knowable before mutation."""

    rewards = _structured_rewards(reward_receipt)
    reason = reward_receipt.get("optimizer_admission_reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("verified_transition_optimizer_admission_reason_invalid")
    return build_trainer_step_static(
        samples=_sample_receipts(samples),
        structured_rewards=list(rewards),
        optimizer_admission_reason=reason,
        answer_channel=answer_channel,
        advantage_report=group_advantages(rewards),
    )


def apply_prepared_verified_transition_group(
    model: Any,
    optimizer: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[Any],
    prepared: PreparedVerifiedTransitionGroup,
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    config: Any | None = None,
    now_unix_ns: Callable[[], int] = time.time_ns,
    transaction_coordinator: Any | None = None,
    rejection_transaction_coordinator: Any | None = None,
) -> VerifiedTransitionMutationResult:
    """Replay one group and make the verified transaction the only mutator."""

    if type(prepared.campaign_sequence) is not int or prepared.campaign_sequence < 0:
        raise ValueError("verified_transition_campaign_sequence_invalid")
    _sha256(
        prepared.campaign_manifest_sha256,
        role="verified_transition_campaign_manifest",
    )
    _sha256(
        prepared.campaign_schedule_root_sha256,
        role="verified_transition_campaign_schedule_root",
    )
    policy_before = recurrent_policy_sha256(model, spec)
    reward = validate_verified_transition_reward_batch(
        prepared.transition_store,
        prepared.reward_receipt,
        prepared.transition_evidence,
        independent_scorer=prepared.independent_scorer,
        token_encoder=prepared.token_encoder,
        token_decoder=prepared.token_decoder,
    )
    rewards = _structured_rewards(reward)
    manifest_sha256 = _sha256(
        prepared.group_manifest.get("manifest_sha256"),
        role="verified_transition_group_manifest",
    )
    admitted = reward.get("optimizer_admitted") is True
    reason = str(reward.get("optimizer_admission_reason") or "invalid")

    if not admitted:
        if prepared.group_admission_receipt is not None or prepared.update_journal is not None:
            raise ValueError("rejected_verified_transition_has_update_material")
        if rejection_transaction_coordinator is None:
            raise ValueError("verified_transition_rejection_transaction_required")
        prepared.campaign_ledger.validate_started_group(
            sequence=prepared.campaign_sequence,
            group_manifest=prepared.group_manifest,
        )
        rejection_transaction_coordinator.stage_rejection(
            policy_sha256=policy_before
        )
        terminal = prepared.campaign_ledger.finish_group(
            sequence=prepared.campaign_sequence,
            status="rejected",
            reward_receipt_sha256=cast(str, reward["receipt_sha256"]),
            group_admission_sha256=None,
            update_receipt_sha256=None,
            terminal_reason=reason,
            finished_at_unix_ns=now_unix_ns(),
            policy_after_sha256=policy_before,
        )
        policy_after = recurrent_policy_sha256(model, spec)
        if policy_after != policy_before:
            raise RuntimeError("rejected_verified_transition_changed_policy")
        rejection_transaction_coordinator.record_campaign_terminal(terminal)
        return VerifiedTransitionMutationResult(
            campaign_sequence=prepared.campaign_sequence,
            group_manifest_sha256=manifest_sha256,
            optimizer_updated=False,
            structured_rewards=rewards,
            optimizer_admission_reason=reason,
            reward_receipt_sha256=cast(str, reward["receipt_sha256"]),
            group_admission_sha256=None,
            update_receipt_sha256=None,
            update_receipt=None,
            terminal_receipt=terminal,
            policy_before_sha256=policy_before,
            policy_after_sha256=policy_after,
            replay_group=None,
        )

    if prepared.group_admission_receipt is None or prepared.update_journal is None:
        raise ValueError("admitted_verified_transition_update_material_missing")
    bound_rewards = rewards_for_recurrent_samples(reward, samples, prompt_tokens)
    if bound_rewards != rewards:
        raise ValueError("verified_transition_reward_projection_mismatch")
    update_result = apply_verified_transition_group_update(
        model,
        optimizer,
        prompt_tokens,
        samples,
        prepared.group_admission_receipt,
        reward,
        prepared.transition_evidence,
        transition_store=prepared.transition_store,
        group_manifest=prepared.group_manifest,
        group_manifest_attestation=prepared.group_manifest_attestation,
        independent_scorer=prepared.independent_scorer,
        token_encoder=prepared.token_encoder,
        token_decoder=prepared.token_decoder,
        spec=spec,
        journal=prepared.update_journal,
        campaign_ledger=prepared.campaign_ledger,
        campaign_sequence=prepared.campaign_sequence,
        bridge_tokens=bridge_tokens,
        config=config,
        now_unix_ns=now_unix_ns,
        return_terminal_receipt=True,
        transaction_coordinator=transaction_coordinator,
    )
    if not isinstance(update_result, tuple):
        raise RuntimeError("verified_transition_terminal_receipt_missing")
    update, terminal = update_result
    replay_group = VerifiedTransitionReplayGroup(
        sequence=prepared.campaign_sequence,
        transition_store=prepared.transition_store,
        group_admission_receipt=prepared.group_admission_receipt,
        reward_receipt=reward,
        transition_evidence=prepared.transition_evidence,
        samples=samples,
        prompt_tokens=prompt_tokens,
        group_manifest=prepared.group_manifest,
        group_manifest_attestation=prepared.group_manifest_attestation,
        independent_scorer=prepared.independent_scorer,
        token_encoder=prepared.token_encoder,
        token_decoder=prepared.token_decoder,
        update_journal=prepared.update_journal,
        update_receipt=update,
    )
    return VerifiedTransitionMutationResult(
        campaign_sequence=prepared.campaign_sequence,
        group_manifest_sha256=manifest_sha256,
        optimizer_updated=True,
        structured_rewards=rewards,
        optimizer_admission_reason=reason,
        reward_receipt_sha256=cast(str, reward["receipt_sha256"]),
        group_admission_sha256=cast(
            str, prepared.group_admission_receipt["receipt_sha256"]
        ),
        update_receipt_sha256=cast(str, update["receipt_sha256"]),
        update_receipt=update,
        terminal_receipt=terminal,
        policy_before_sha256=cast(str, update["policy_before_sha256"]),
        policy_after_sha256=cast(str, update["policy_after_sha256"]),
        replay_group=replay_group,
    )


__all__ = [
    "PreparedVerifiedTransitionGroup",
    "VERIFIED_TRANSITION_STEP_SCHEMA",
    "VerifiedTransitionCampaignClosure",
    "VerifiedTransitionGroupProvider",
    "VerifiedTransitionMutationResult",
    "VerifiedTransitionSamplingEntry",
    "VerifiedTransitionSamplingPlan",
    "VerifiedTransitionTrainingScheduleEntry",
    "VerifiedTransitionProviderRuntime",
    "VerifiedTransitionGroupProviderFactory",
    "VerifiedTransitionTelemetry",
    "apply_prepared_verified_transition_group",
    "build_verified_transition_step_receipt",
    "build_verified_transition_step_static",
    "validate_verified_transition_step_receipt",
]
