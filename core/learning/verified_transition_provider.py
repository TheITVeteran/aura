"""Production boundary for causally precommitted verified transitions.

The immutable contract binds what is knowable before training: the initial
policy, task inputs and order, seeds, execution graph, trust roots, scorer,
codec, storage roots, and recovery behavior.  It intentionally does *not* bind
future policy hashes or future group-manifest hashes.  Those values depend on
earlier optimizer outcomes.

Before each sampling call, ``admit_group_plan`` requires two externally issued
attestations: one over the concrete group manifest and one over a lineage
envelope that binds that manifest and the current policy to the frozen campaign
schedule root.  After mutation, ``accept_step_receipt`` advances the lineage
only when the trainer receipt and durable campaign terminal agree exactly.

Integration requires a causal-ledger adapter with these methods in addition to
the existing trainer-facing ``validate_started_group`` and ``finish_group``:

* ``admit_group_plan(...)`` durably creates the start record;
* ``group_records_unclosed(sequence=...)`` returns the exact start/terminal;
* ``validate_closed(policy=...)`` returns a close payload containing
  ``campaign_schedule_root_sha256``.

The legacy static campaign manifest cannot serve as that adapter because it
requires unknowable future group-manifest hashes.  This module does not weaken
or reinterpret that schema.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, Protocol, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    verify_role_attestation,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    sampling_config_sha256,
    validate_transition_group_manifest,
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_reward import (
    validate_verified_transition_reward_batch,
)
from core.learning.verified_transition_trainer import (
    PreparedVerifiedTransitionGroup,
    VerifiedTransitionCampaignClosure,
    VerifiedTransitionSamplingEntry,
    VerifiedTransitionSamplingPlan,
    VerifiedTransitionTrainingScheduleEntry,
    validate_verified_transition_step_receipt,
)
from core.learning.verified_transition_training_evidence import (
    VerifiedTransitionReplayGroup,
)
from core.learning.verified_transition_transaction import (
    LoadedVerifiedTransitionTransaction,
    VerifiedTransitionTransactionStore,
    validate_pending_trainer_step,
)
from core.learning.verified_transition_update import (
    VerifiedTransitionUpdateJournal,
    commit_staged_verified_transition_update,
    validate_verified_transition_update_receipt,
)

PROVIDER_CONTRACT_SCHEMA = "aura.verified_transition.provider_contract.v2"
PROVIDER_IMPLEMENTATION_ID = "aura.production_verified_transition_provider.v2"
TASK_COMMITMENT_SCHEMA = "aura.verified_transition.task_commitment.v2"
CAMPAIGN_SCHEDULE_SCHEMA = "aura.verified_transition.causal_schedule.v1"
LINEAGE_PLAN_SCHEMA = "aura.verified_transition.lineage_plan.v1"
PRODUCTION_REQUEST_SCHEMA = "aura.verified_transition.production_request.v2"
RESTORE_REQUEST_SCHEMA = "aura.verified_transition.restore_request.v2"
FINALIZE_REQUEST_SCHEMA = "aura.verified_transition.finalize_request.v2"
RECOVERY_POLICY_SCHEMA = "aura.verified_transition.recovery_policy.v1"

_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "provider",
        "trust_policy_sha256",
        "trust_root_key_id",
        "campaign_id",
        "campaign_schedule_root_sha256",
        "initial_policy_sha256",
        "scorer",
        "token_codec",
        "dataset_sha256",
        "task_schedule",
        "task_schedule_sha256",
        "ledger_roots",
        "ledger_roots_sha256",
        "recovery_policy",
        "recovery_policy_sha256",
        "frozen_at_unix_ns",
        "contract_sha256",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "implementation_id",
        "implementation_source_sha256",
        "config",
        "config_sha256",
        "evidence_producer_identity",
        "evidence_producer_source_sha256",
        "durable_artifact_loader_identity",
        "durable_artifact_loader_source_sha256",
        "campaign_finalizer_identity",
        "campaign_finalizer_source_sha256",
    }
)
_TASK_KEYS = frozenset(
    {
        "schema",
        "sequence",
        "task_id",
        "trainer_sample_seed",
        "immutable_task_sha256",
        "prompt_tokens_sha256",
        "recurrent_execution_spec_sha256",
        "sample_seeds",
    }
)
_ROOT_KEYS = frozenset(
    {"campaign", "transition_artifacts", "updates", "replay_artifacts"}
)
_RECOVERY_KEYS = frozenset(
    {
        "schema",
        "mode",
        "require_durable_artifacts",
        "reject_partial_steps",
        "reject_policy_substitution",
        "reject_campaign_substitution",
    }
)
_RECOVERY_POLICY = {
    "schema": RECOVERY_POLICY_SCHEMA,
    "mode": "exact_step_replay",
    "require_durable_artifacts": True,
    "reject_partial_steps": True,
    "reject_policy_substitution": True,
    "reject_campaign_substitution": True,
}


class VerifiedTransitionProviderError(RuntimeError):
    """Stable fail-closed provider error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionProviderError(code)


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        _fail(f"{role}_not_canonical_json")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        _fail(f"{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _absolute_root(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{role}_invalid")
    path = Path(value)
    if not path.is_absolute() or str(path) != str(path.resolve(strict=False)):
        _fail(f"{role}_not_canonical_absolute_path")
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    return str(path)


def _root_of(value: Any, *, role: str) -> str:
    root = getattr(value, "root", None)
    if root is None:
        _fail(f"{role}_root_missing")
    return _absolute_root(str(root), role=role)


def provider_implementation_source_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        _fail("provider_implementation_source_unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def callable_source_sha256(value: Callable[..., Any]) -> str:
    """Bind a callable to its source file, source span, and bytecode."""

    target: Any = inspect.unwrap(value)
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        target = type(target).__call__
    try:
        source = inspect.getsource(target).encode("utf-8")
        raw_path = inspect.getsourcefile(target)
        code = target.__code__
    except (OSError, TypeError, AttributeError) as exc:
        raise VerifiedTransitionProviderError(
            "provider_callable_source_unavailable"
        ) from exc
    if not raw_path:
        _fail("provider_callable_source_unavailable")
    path = Path(raw_path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        _fail("provider_callable_source_unavailable")
    return _digest(
        {
            "module": getattr(target, "__module__", ""),
            "qualname": getattr(target, "__qualname__", ""),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        }
    )


def _validate_task_commitment(value: Any, *, sequence: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_KEYS:
        _fail("provider_task_commitment_schema_invalid")
    item = cast(dict[str, Any], _clone(value, role="task_commitment"))
    seeds = item.get("sample_seeds")
    if (
        item.get("schema") != TASK_COMMITMENT_SCHEMA
        or item.get("sequence") != sequence
        or not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed <= (1 << 32) - 1 for seed in seeds)
    ):
        _fail("provider_task_commitment_invalid")
    _identifier(item.get("task_id"), role="provider_task")
    _integer(item.get("trainer_sample_seed"), role="provider_trainer_sample_seed")
    for field in (
        "immutable_task_sha256",
        "prompt_tokens_sha256",
        "recurrent_execution_spec_sha256",
    ):
        _sha256(item.get(field), role=f"provider_{field}")
    return item


def _schedule_root(
    *,
    campaign_id: str,
    trust_policy_sha256: str,
    dataset_sha256: str,
    task_schedule_sha256: str,
) -> str:
    return _digest(
        {
            "schema": CAMPAIGN_SCHEDULE_SCHEMA,
            "campaign_id": campaign_id,
            "trust_policy_sha256": trust_policy_sha256,
            "dataset_sha256": dataset_sha256,
            "task_schedule_sha256": task_schedule_sha256,
        }
    )


def validate_verified_transition_provider_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        _fail("provider_contract_schema_invalid")
    contract = cast(dict[str, Any], _clone(value, role="provider_contract"))
    if contract.get("schema") != PROVIDER_CONTRACT_SCHEMA:
        _fail("provider_contract_version_invalid")
    observed = _sha256(contract.get("contract_sha256"), role="provider_contract")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256")
    if observed != _digest(unsigned):
        _fail("provider_contract_digest_mismatch")

    provider = contract.get("provider")
    if not isinstance(provider, Mapping) or set(provider) != _PROVIDER_KEYS:
        _fail("provider_contract_provider_invalid")
    if provider.get("implementation_id") != PROVIDER_IMPLEMENTATION_ID:
        _fail("provider_contract_implementation_invalid")
    for field in (
        "implementation_source_sha256",
        "config_sha256",
        "evidence_producer_source_sha256",
        "durable_artifact_loader_source_sha256",
        "campaign_finalizer_source_sha256",
    ):
        _sha256(provider.get(field), role=f"provider_contract_{field}")
    for field in (
        "evidence_producer_identity",
        "durable_artifact_loader_identity",
        "campaign_finalizer_identity",
    ):
        _identifier(provider.get(field), role=f"provider_contract_{field}")
    config = provider.get("config")
    if not isinstance(config, Mapping) or not config or provider["config_sha256"] != _digest(config):
        _fail("provider_contract_config_invalid")

    trust = _sha256(contract.get("trust_policy_sha256"), role="provider_trust")
    _identifier(contract.get("trust_root_key_id"), role="provider_trust_root")
    campaign_id = _identifier(contract.get("campaign_id"), role="provider_campaign")
    dataset = _sha256(contract.get("dataset_sha256"), role="provider_dataset")
    _sha256(contract.get("initial_policy_sha256"), role="provider_initial_policy")

    scorer = contract.get("scorer")
    if not isinstance(scorer, Mapping) or set(scorer) != {"identity", "source_sha256"}:
        _fail("provider_contract_scorer_invalid")
    _identifier(scorer.get("identity"), role="provider_scorer_identity")
    _sha256(scorer.get("source_sha256"), role="provider_scorer_source")
    codec = contract.get("token_codec")
    if not isinstance(codec, Mapping) or set(codec) != {
        "identity",
        "encoder_source_sha256",
        "decoder_source_sha256",
    }:
        _fail("provider_contract_token_codec_invalid")
    _identifier(codec.get("identity"), role="provider_codec_identity")
    _sha256(codec.get("encoder_source_sha256"), role="provider_encoder")
    _sha256(codec.get("decoder_source_sha256"), role="provider_decoder")

    schedule = contract.get("task_schedule")
    if not isinstance(schedule, list) or not schedule:
        _fail("provider_contract_task_schedule_invalid")
    normalized_schedule = [
        _validate_task_commitment(item, sequence=index)
        for index, item in enumerate(schedule)
    ]
    schedule_sha = _digest(normalized_schedule)
    if contract.get("task_schedule_sha256") != schedule_sha:
        _fail("provider_contract_task_schedule_digest_mismatch")
    expected_root = _schedule_root(
        campaign_id=campaign_id,
        trust_policy_sha256=trust,
        dataset_sha256=dataset,
        task_schedule_sha256=schedule_sha,
    )
    if contract.get("campaign_schedule_root_sha256") != expected_root:
        _fail("provider_contract_campaign_schedule_root_mismatch")

    roots = contract.get("ledger_roots")
    if not isinstance(roots, Mapping) or set(roots) != _ROOT_KEYS:
        _fail("provider_contract_ledger_roots_invalid")
    normalized_roots = {
        key: _absolute_root(roots[key], role=f"provider_{key}_root")
        for key in sorted(_ROOT_KEYS)
    }
    if len(set(normalized_roots.values())) != len(normalized_roots):
        _fail("provider_contract_ledger_roots_overlap")
    if contract.get("ledger_roots_sha256") != _digest(normalized_roots):
        _fail("provider_contract_ledger_roots_digest_mismatch")

    recovery = contract.get("recovery_policy")
    if (
        not isinstance(recovery, Mapping)
        or set(recovery) != _RECOVERY_KEYS
        or dict(recovery) != _RECOVERY_POLICY
        or contract.get("recovery_policy_sha256") != _digest(recovery)
    ):
        _fail("provider_contract_recovery_policy_invalid")
    _integer(contract.get("frozen_at_unix_ns"), role="provider_frozen_at", minimum=1)
    contract["task_schedule"] = normalized_schedule
    contract["ledger_roots"] = normalized_roots
    return contract


def build_verified_transition_provider_contract(
    *,
    provider_config: Mapping[str, Any],
    evidence_producer_identity: str,
    evidence_producer_source_sha256: str,
    durable_artifact_loader_identity: str,
    durable_artifact_loader_source_sha256: str,
    campaign_finalizer_identity: str,
    campaign_finalizer_source_sha256: str,
    trust_policy_sha256: str,
    trust_root_key_id: str,
    campaign_id: str,
    initial_policy_sha256: str,
    scorer_identity: str,
    scorer_source_sha256: str,
    token_codec_identity: str,
    token_encoder_source_sha256: str,
    token_decoder_source_sha256: str,
    dataset_sha256: str,
    task_schedule: Sequence[Mapping[str, Any]],
    ledger_roots: Mapping[str, str],
    frozen_at_unix_ns: int,
) -> dict[str, Any]:
    """Seal knowable campaign facts without inventing future tensor hashes."""

    config = cast(dict[str, Any], _clone(provider_config, role="provider_config"))
    schedule = [dict(item) for item in task_schedule]
    schedule_sha = _digest(schedule)
    roots = dict(ledger_roots)
    body = {
        "schema": PROVIDER_CONTRACT_SCHEMA,
        "provider": {
            "implementation_id": PROVIDER_IMPLEMENTATION_ID,
            "implementation_source_sha256": provider_implementation_source_sha256(),
            "config": config,
            "config_sha256": _digest(config),
            "evidence_producer_identity": evidence_producer_identity,
            "evidence_producer_source_sha256": evidence_producer_source_sha256,
            "durable_artifact_loader_identity": durable_artifact_loader_identity,
            "durable_artifact_loader_source_sha256": durable_artifact_loader_source_sha256,
            "campaign_finalizer_identity": campaign_finalizer_identity,
            "campaign_finalizer_source_sha256": campaign_finalizer_source_sha256,
        },
        "trust_policy_sha256": trust_policy_sha256,
        "trust_root_key_id": trust_root_key_id,
        "campaign_id": campaign_id,
        "campaign_schedule_root_sha256": _schedule_root(
            campaign_id=campaign_id,
            trust_policy_sha256=trust_policy_sha256,
            dataset_sha256=dataset_sha256,
            task_schedule_sha256=schedule_sha,
        ),
        "initial_policy_sha256": initial_policy_sha256,
        "scorer": {"identity": scorer_identity, "source_sha256": scorer_source_sha256},
        "token_codec": {
            "identity": token_codec_identity,
            "encoder_source_sha256": token_encoder_source_sha256,
            "decoder_source_sha256": token_decoder_source_sha256,
        },
        "dataset_sha256": dataset_sha256,
        "task_schedule": schedule,
        "task_schedule_sha256": schedule_sha,
        "ledger_roots": roots,
        "ledger_roots_sha256": _digest({key: roots[key] for key in sorted(roots)}),
        "recovery_policy": dict(_RECOVERY_POLICY),
        "recovery_policy_sha256": _digest(_RECOVERY_POLICY),
        "frozen_at_unix_ns": frozen_at_unix_ns,
    }
    return validate_verified_transition_provider_contract(
        {**body, "contract_sha256": _digest(body)}
    )


class CausalCampaignLedger(Protocol):
    """Required adapter interface for a causally evolving campaign."""

    root: Path

    def admit_group_plan(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def validate_started_group(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def finish_group(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def group_records_unclosed(
        self, *, sequence: int
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def group_start(self, *, sequence: int) -> Mapping[str, Any]: ...

    def group_start_if_exists(
        self, *, sequence: int
    ) -> Mapping[str, Any] | None: ...

    def group_terminal_if_exists(
        self, *, sequence: int
    ) -> Mapping[str, Any] | None: ...

    def group_records(
        self, *, sequence: int, policy: Any
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def validate_closed(self, *, policy: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class VerifiedTransitionProductionRequest:
    schema: str
    contract_sha256: str
    campaign_schedule_root_sha256: str
    sequence: int
    task: Any
    prompt_tokens: tuple[int, ...]
    samples: tuple[Any, ...]
    completions: tuple[str, ...]
    task_commitment: Mapping[str, Any]
    lineage_plan: Mapping[str, Any]
    provider_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedTransitionRestoreRequest:
    schema: str
    contract_sha256: str
    campaign_schedule_root_sha256: str
    committed_steps: int
    step_receipts: tuple[Mapping[str, Any], ...]
    replay_artifact_root: str


@dataclass(frozen=True, slots=True)
class VerifiedTransitionFinalizeRequest:
    schema: str
    contract_sha256: str
    campaign_schedule_root_sha256: str
    completed_groups: int
    halt_reason: str
    replay_groups: tuple[VerifiedTransitionReplayGroup, ...]


@dataclass(frozen=True, slots=True)
class _AdmittedPlan:
    sequence: int
    policy_before_sha256: str
    group_manifest: Mapping[str, Any]
    group_manifest_attestation: Mapping[str, Any]
    lineage_attestation: Mapping[str, Any]
    lineage_plan: Mapping[str, Any]
    start_receipt: Mapping[str, Any]


class ProductionVerifiedTransitionGroupProvider:
    """Causal provider whose future policy lineage is learned, not guessed."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        provider_config: Mapping[str, Any],
        campaign_ledger: CausalCampaignLedger,
        campaign_trust_policy: VerifiedCampaignTrustPolicy,
        evidence_producer: Callable[
            [VerifiedTransitionProductionRequest], PreparedVerifiedTransitionGroup
        ],
        evidence_producer_identity: str,
        durable_artifact_loader: Callable[
            [VerifiedTransitionRestoreRequest], Sequence[VerifiedTransitionReplayGroup]
        ],
        durable_artifact_loader_identity: str,
        campaign_finalizer: Callable[
            [VerifiedTransitionFinalizeRequest], VerifiedTransitionCampaignClosure
        ],
        campaign_finalizer_identity: str,
        independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
        scorer_identity: str,
        token_encoder: Callable[[bytes], Sequence[int]],
        token_decoder: Callable[[Sequence[int]], bytes],
        token_codec_identity: str,
    ) -> None:
        frozen = validate_verified_transition_provider_contract(contract)
        provider = cast(Mapping[str, Any], frozen["provider"])
        scorer = cast(Mapping[str, Any], frozen["scorer"])
        codec = cast(Mapping[str, Any], frozen["token_codec"])
        config = _clone(provider_config, role="provider_config")
        if (
            provider["implementation_source_sha256"]
            != provider_implementation_source_sha256()
            or provider["config"] != config
            or provider["config_sha256"] != _digest(config)
        ):
            _fail("provider_runtime_implementation_or_config_mismatch")
        for expected_identity, expected_source, identity, callable_value, role in (
            (
                provider["evidence_producer_identity"],
                provider["evidence_producer_source_sha256"],
                evidence_producer_identity,
                evidence_producer,
                "evidence_producer",
            ),
            (
                provider["durable_artifact_loader_identity"],
                provider["durable_artifact_loader_source_sha256"],
                durable_artifact_loader_identity,
                durable_artifact_loader,
                "durable_artifact_loader",
            ),
            (
                provider["campaign_finalizer_identity"],
                provider["campaign_finalizer_source_sha256"],
                campaign_finalizer_identity,
                campaign_finalizer,
                "campaign_finalizer",
            ),
        ):
            if identity != expected_identity or callable_source_sha256(callable_value) != expected_source:
                _fail(f"provider_runtime_{role}_mismatch")
        if scorer["identity"] != scorer_identity or scorer[
            "source_sha256"
        ] != callable_source_sha256(independent_scorer):
            _fail("provider_runtime_scorer_mismatch")
        if (
            codec["identity"] != token_codec_identity
            or codec["encoder_source_sha256"] != callable_source_sha256(token_encoder)
            or codec["decoder_source_sha256"] != callable_source_sha256(token_decoder)
        ):
            _fail("provider_runtime_token_codec_mismatch")
        if (
            frozen["trust_policy_sha256"] != campaign_trust_policy.policy_sha256
            or frozen["trust_root_key_id"] != campaign_trust_policy.root_key_id
            or _root_of(campaign_ledger, role="campaign_ledger")
            != frozen["ledger_roots"]["campaign"]
        ):
            _fail("provider_runtime_trust_or_campaign_mismatch")
        for method in (
            "admit_group_plan",
            "validate_started_group",
            "finish_group",
            "group_records_unclosed",
            "group_start",
            "group_terminal_if_exists",
            "group_records",
            "validate_closed",
        ):
            if not callable(getattr(campaign_ledger, method, None)):
                _fail(f"provider_causal_ledger_{method}_missing")

        self._contract = frozen
        self._provider_config = cast(dict[str, Any], _clone(config, role="provider_config"))
        self._ledger = campaign_ledger
        self._policy = campaign_trust_policy
        self._producer = evidence_producer
        self._loader = durable_artifact_loader
        self._finalizer = campaign_finalizer
        self._scorer = independent_scorer
        self._encoder = token_encoder
        self._decoder = token_decoder
        self._plans: dict[int, _AdmittedPlan] = {}
        self._accepted_steps: list[dict[str, Any]] = []
        self._pending_sequence: int | None = None
        self._restore_attempted = False
        self._finalized = False
        self._lock = threading.RLock()

    @property
    def contract_sha256(self) -> str:
        return cast(str, self._contract["contract_sha256"])

    @property
    def expected_policy_sha256(self) -> str:
        if self._accepted_steps:
            return cast(str, self._accepted_steps[-1]["policy_after_sha256"])
        return cast(str, self._contract["initial_policy_sha256"])

    @property
    def campaign_id(self) -> str:
        return cast(str, self._contract["campaign_id"])

    @property
    def campaign_schedule_root_sha256(self) -> str:
        return cast(str, self._contract["campaign_schedule_root_sha256"])

    def task_commitment(self, *, sequence: int) -> Mapping[str, Any]:
        """Return the immutable public schedule row for runtime binding."""

        with self._lock:
            return cast(
                dict[str, Any],
                _clone(self._commitment(sequence), role="task_commitment"),
            )

    def training_schedule_entry(
        self, *, sequence: int
    ) -> VerifiedTransitionTrainingScheduleEntry:
        """Expose the provider-owned task and trainer seed before selection."""

        with self._lock:
            commitment = self._commitment(sequence)
            return VerifiedTransitionTrainingScheduleEntry(
                campaign_sequence=sequence,
                task_id=cast(str, commitment["task_id"]),
                trainer_sample_seed=cast(int, commitment["trainer_sample_seed"]),
            )

    def lineage_plan_for_manifest(
        self,
        *,
        sequence: int,
        policy_before_sha256: str,
        group_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build the exact unsigned lineage payload an external issuer signs."""

        with self._lock:
            manifest = validate_transition_group_manifest(group_manifest)
            return self._lineage_plan(
                sequence=sequence,
                policy_before_sha256=_sha256(
                    policy_before_sha256, role="provider_lineage_policy"
                ),
                group_manifest=manifest,
            )

    def _commitment(self, sequence: int) -> dict[str, Any]:
        if not 0 <= sequence < len(self._contract["task_schedule"]):
            _fail("provider_sequence_outside_contract")
        return cast(dict[str, Any], self._contract["task_schedule"][sequence])

    @staticmethod
    def _task_document(task: Any) -> dict[str, Any]:
        resolver = getattr(task, "verified_transition_task_commitment", None)
        if not callable(resolver):
            _fail("provider_runtime_task_commitment_missing")
        document = resolver()
        if not isinstance(document, Mapping) or not document:
            _fail("provider_runtime_task_commitment_invalid")
        return cast(dict[str, Any], _clone(document, role="runtime_task_commitment"))

    def _lineage_plan(
        self,
        *,
        sequence: int,
        policy_before_sha256: str,
        group_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        commitment = self._commitment(sequence)
        return {
            "schema": LINEAGE_PLAN_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "campaign_id": self._contract["campaign_id"],
            "campaign_schedule_root_sha256": self._contract[
                "campaign_schedule_root_sha256"
            ],
            "sequence": sequence,
            "task_commitment_sha256": _digest(commitment),
            "policy_before_sha256": policy_before_sha256,
            "group_manifest_sha256": group_manifest["manifest_sha256"],
        }

    def admit_group_plan(
        self,
        *,
        sequence: int,
        policy_before_sha256: str,
        group_manifest: Mapping[str, Any],
        group_manifest_attestation: Mapping[str, Any],
        lineage_attestation: Mapping[str, Any],
        admitted_at_unix_ns: int,
    ) -> Mapping[str, Any]:
        """Admit one issuer-signed concrete plan before sampling that group."""

        with self._lock:
            if self._finalized or self._pending_sequence is not None:
                _fail("provider_plan_state_invalid")
            expected_sequence = len(self._accepted_steps)
            if sequence != expected_sequence:
                _fail("provider_plan_sequence_not_next")
            expected_policy = self.expected_policy_sha256
            if _sha256(policy_before_sha256, role="provider_plan_policy") != expected_policy:
                _fail("provider_plan_policy_lineage_mismatch")
            commitment = self._commitment(sequence)
            manifest = validate_transition_group_manifest(group_manifest)
            entries = manifest["entries"]
            if (
                manifest["task_id"] != commitment["task_id"]
                or manifest["group_size"] != len(commitment["sample_seeds"])
                or [entry["sample_seed"] for entry in entries]
                != commitment["sample_seeds"]
                or any(entry["policy_sha256"] != expected_policy for entry in entries)
                or any(
                    entry["recurrent_execution_spec_sha256"]
                    != commitment["recurrent_execution_spec_sha256"]
                    for entry in entries
                )
            ):
                _fail("provider_plan_manifest_schedule_mismatch")
            admitted_at = _integer(
                admitted_at_unix_ns, role="provider_plan_admitted_at", minimum=1
            )
            if manifest["planned_at_unix_ns"] >= admitted_at:
                _fail("provider_plan_not_pre_sampling")
            verify_role_attestation(
                self._policy,
                group_manifest_attestation,
                role=TASK_ISSUER,
                expected_payload=manifest,
                not_after_unix=manifest["planned_at_unix_ns"] // 1_000_000_000,
            )
            lineage_plan = self._lineage_plan(
                sequence=sequence,
                policy_before_sha256=expected_policy,
                group_manifest=manifest,
            )
            verify_role_attestation(
                self._policy,
                lineage_attestation,
                role=TASK_ISSUER,
                expected_payload=lineage_plan,
                not_after_unix=admitted_at // 1_000_000_000,
            )
            start = self._ledger.group_start_if_exists(sequence=sequence)
            if start is None:
                start = self._ledger.admit_group_plan(
                    sequence=sequence,
                    campaign_id=self._contract["campaign_id"],
                    campaign_schedule_root_sha256=self._contract[
                        "campaign_schedule_root_sha256"
                    ],
                    policy_before_sha256=expected_policy,
                    group_manifest=manifest,
                    group_manifest_attestation=dict(group_manifest_attestation),
                    lineage_plan=lineage_plan,
                    lineage_attestation=dict(lineage_attestation),
                    policy=self._policy,
                    admitted_at_unix_ns=admitted_at,
                )
            else:
                persisted_manifest = self._validate_start_record(
                    sequence=sequence,
                    expected_policy_sha256=expected_policy,
                    start=start,
                )
                if (
                    persisted_manifest != manifest
                    or start.get("group_manifest_attestation")
                    != dict(group_manifest_attestation)
                    or start.get("lineage_plan") != lineage_plan
                    or start.get("lineage_attestation")
                    != dict(lineage_attestation)
                    or start.get("admitted_at_unix_ns") != admitted_at
                    or self._ledger.group_terminal_if_exists(sequence=sequence)
                    is not None
                ):
                    _fail("provider_plan_rehydration_mismatch")
            if not isinstance(start, Mapping):
                _fail("provider_plan_start_receipt_invalid")
            _sha256(
                start.get("campaign_manifest_sha256"),
                role="provider_plan_campaign_manifest",
            )
            if (
                start.get("campaign_schedule_root_sha256")
                != self._contract["campaign_schedule_root_sha256"]
                or start.get("sequence") != sequence
                or start.get("policy_before_sha256") != expected_policy
                or start.get("group_manifest") != manifest
                or start.get("lineage_plan") != lineage_plan
            ):
                _fail("provider_plan_start_receipt_mismatch")
            plan = _AdmittedPlan(
                sequence=sequence,
                policy_before_sha256=expected_policy,
                group_manifest=manifest,
                group_manifest_attestation=cast(
                    dict[str, Any], _clone(group_manifest_attestation, role="group_attestation")
                ),
                lineage_attestation=cast(
                    dict[str, Any], _clone(lineage_attestation, role="lineage_attestation")
                ),
                lineage_plan=lineage_plan,
                start_receipt=cast(dict[str, Any], _clone(start, role="group_start")),
            )
            self._plans[sequence] = plan
            return dict(plan.start_receipt)

    def _validate_runtime_inputs(
        self,
        *,
        commitment: Mapping[str, Any],
        task: Any,
        prompt_tokens: Sequence[int],
        samples: Sequence[Any],
        completions: Sequence[str],
        policy_sha256: str,
    ) -> None:
        task_document = self._task_document(task)
        if (
            getattr(task, "task_id", None) != commitment["task_id"]
            or _digest(task_document) != commitment["immutable_task_sha256"]
            or _digest(list(prompt_tokens)) != commitment["prompt_tokens_sha256"]
            or len(samples) != len(commitment["sample_seeds"])
            or len(completions) != len(samples)
            or any(not isinstance(text, str) for text in completions)
        ):
            _fail("provider_runtime_task_or_sample_count_mismatch")
        observed_seeds: list[int] = []
        for sample in samples:
            observed_seeds.append(getattr(sample, "seed", None))
            if (
                getattr(sample, "policy_sha256", None) != policy_sha256
                or getattr(sample, "execution_spec_sha256", None)
                != commitment["recurrent_execution_spec_sha256"]
                or getattr(sample, "prompt_tokens_sha256", None)
                != commitment["prompt_tokens_sha256"]
            ):
                _fail("provider_runtime_sample_commitment_mismatch")
        if observed_seeds != commitment["sample_seeds"]:
            _fail("provider_runtime_sample_seed_mismatch")

    def sampling_plan(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        policy_sha256: str,
    ) -> VerifiedTransitionSamplingPlan:
        """Expose one already admitted signed plan before any model sampling."""

        with self._lock:
            if self._finalized or self._pending_sequence is not None:
                _fail("provider_sampling_plan_state_invalid")
            if sequence != len(self._accepted_steps) or sequence not in self._plans:
                _fail("provider_sampling_plan_missing")
            expected_policy = self.expected_policy_sha256
            if _sha256(
                policy_sha256, role="provider_sampling_plan_policy"
            ) != expected_policy:
                _fail("provider_sampling_plan_policy_mismatch")
            commitment = self._commitment(sequence)
            plan = self._plans[sequence]
            manifest = validate_transition_group_manifest(plan.group_manifest)
            if (
                getattr(task, "task_id", None) != commitment["task_id"]
                or _digest(self._task_document(task))
                != commitment["immutable_task_sha256"]
                or _digest(list(prompt_tokens))
                != commitment["prompt_tokens_sha256"]
                or manifest["task_id"] != commitment["task_id"]
                or plan.policy_before_sha256 != expected_policy
            ):
                _fail("provider_sampling_plan_runtime_binding_mismatch")
            entries = tuple(
                VerifiedTransitionSamplingEntry(
                    episode_id=cast(str, entry["episode_id"]),
                    rng_root_sha256=cast(str, entry["rng_root_sha256"]),
                    producing_branch_index=cast(
                        int, entry["producing_branch_index"]
                    ),
                    sample_seed=cast(int, entry["sample_seed"]),
                    sampling_config_sha256=cast(
                        str, entry["sampling_config_sha256"]
                    ),
                )
                for entry in manifest["entries"]
            )
            return VerifiedTransitionSamplingPlan(
                campaign_sequence=sequence,
                group_manifest_sha256=cast(str, manifest["manifest_sha256"]),
                task_id=cast(str, commitment["task_id"]),
                policy_sha256=expected_policy,
                prompt_tokens_sha256=cast(
                    str, commitment["prompt_tokens_sha256"]
                ),
                execution_spec_sha256=cast(
                    str, commitment["recurrent_execution_spec_sha256"]
                ),
                entries=entries,
                sampling_config={},
            )

    def _validate_prepared(
        self,
        prepared: PreparedVerifiedTransitionGroup,
        *,
        plan: _AdmittedPlan,
        prompt_tokens: Sequence[int],
        samples: Sequence[Any],
    ) -> PreparedVerifiedTransitionGroup:
        if not isinstance(prepared, PreparedVerifiedTransitionGroup):
            _fail("provider_evidence_result_type_invalid")
        if (
            prepared.campaign_sequence != plan.sequence
            or prepared.campaign_ledger is not self._ledger
            or prepared.campaign_trust_policy.policy_sha256 != self._policy.policy_sha256
            or prepared.campaign_trust_policy.root_key_id != self._policy.root_key_id
            or prepared.independent_scorer is not self._scorer
            or prepared.token_encoder is not self._encoder
            or prepared.token_decoder is not self._decoder
        ):
            _fail("provider_evidence_runtime_binding_substitution")
        roots = self._contract["ledger_roots"]
        if _root_of(prepared.transition_store, role="transition_store") != roots[
            "transition_artifacts"
        ]:
            _fail("provider_evidence_artifact_root_substitution")
        if prepared.update_journal is not None and _root_of(
            prepared.update_journal, role="update_journal"
        ) != roots["updates"]:
            _fail("provider_evidence_update_root_substitution")
        manifest = validate_transition_group_manifest(prepared.group_manifest)
        if (
            manifest != plan.group_manifest
            or dict(prepared.group_manifest_attestation)
            != dict(plan.group_manifest_attestation)
        ):
            _fail("provider_evidence_group_plan_substitution")
        actual = []
        for evidence, sample in zip(prepared.transition_evidence, samples, strict=True):
            episode = getattr(evidence, "episode", None)
            if not isinstance(episode, Mapping):
                _fail("provider_evidence_episode_missing")
            actual.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "task_id": episode.get("task_id"),
                    "policy_sha256": getattr(sample, "policy_sha256", None),
                    "recurrent_execution_spec_sha256": getattr(
                        sample, "execution_spec_sha256", None
                    ),
                    "producing_branch_index": getattr(sample, "branch_index", None),
                    "sample_seed": getattr(sample, "seed", None),
                    "sampling_config_sha256": sampling_config_sha256(sample),
                }
            )
        if len(actual) != len(manifest["entries"]):
            _fail("provider_evidence_episode_count_mismatch")
        for expected, observed in zip(manifest["entries"], actual, strict=True):
            if any(expected.get(field) != value for field, value in observed.items()):
                _fail("provider_evidence_sample_manifest_mismatch")
        reward = validate_verified_transition_reward_batch(
            prepared.transition_store,
            prepared.reward_receipt,
            prepared.transition_evidence,
            independent_scorer=self._scorer,
            token_encoder=self._encoder,
            token_decoder=self._decoder,
        )
        if reward.get("optimizer_admitted") is True:
            if prepared.group_admission_receipt is None or prepared.update_journal is None:
                _fail("provider_admitted_update_material_missing")
            validate_verified_transition_group_admission(
                prepared.transition_store,
                prepared.group_admission_receipt,
                reward,
                prepared.transition_evidence,
                samples,
                prompt_tokens,
                group_manifest=manifest,
                group_manifest_attestation=prepared.group_manifest_attestation,
                independent_scorer=self._scorer,
                token_encoder=self._encoder,
                token_decoder=self._decoder,
            )
        elif prepared.group_admission_receipt is not None or prepared.update_journal is not None:
            _fail("provider_rejected_group_has_update_material")
        self._ledger.validate_started_group(
            sequence=plan.sequence, group_manifest=manifest
        )
        return PreparedVerifiedTransitionGroup(
            campaign_sequence=plan.sequence,
            transition_store=prepared.transition_store,
            reward_receipt=cast(dict[str, Any], _clone(reward, role="reward_receipt")),
            transition_evidence=tuple(prepared.transition_evidence),
            group_manifest=manifest,
            group_manifest_attestation=dict(plan.group_manifest_attestation),
            independent_scorer=self._scorer,
            token_encoder=self._encoder,
            token_decoder=self._decoder,
            campaign_ledger=cast(Any, self._ledger),
            campaign_trust_policy=self._policy,
            group_admission_receipt=(
                cast(dict[str, Any], _clone(prepared.group_admission_receipt, role="admission"))
                if prepared.group_admission_receipt is not None
                else None
            ),
            update_journal=prepared.update_journal,
            campaign_manifest_sha256=cast(
                str,
                _sha256(
                    plan.start_receipt.get("campaign_manifest_sha256"),
                    role="provider_prepared_campaign_manifest",
                ),
            ),
            campaign_schedule_root_sha256=cast(
                str, plan.start_receipt["campaign_schedule_root_sha256"]
            ),
        )

    def prepare_group(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        samples: Sequence[Any],
        completions: Sequence[str],
    ) -> PreparedVerifiedTransitionGroup:
        with self._lock:
            if self._finalized or self._pending_sequence is not None:
                _fail("provider_prepare_state_invalid")
            if sequence != len(self._accepted_steps) or sequence not in self._plans:
                _fail("provider_prepare_plan_missing")
            plan = self._plans[sequence]
            commitment = self._commitment(sequence)
            self._validate_runtime_inputs(
                commitment=commitment,
                task=task,
                prompt_tokens=prompt_tokens,
                samples=samples,
                completions=completions,
                policy_sha256=plan.policy_before_sha256,
            )
            from core.learning.recurrent_grpo import (
                validate_recurrent_policy_sample_receipt,
            )

            for expected, sample in zip(
                plan.group_manifest["entries"], samples, strict=True
            ):
                sample_receipt = validate_recurrent_policy_sample_receipt(
                    sample.receipt()
                )
                if (
                    sample_receipt["episode_id"] != expected["episode_id"]
                    or sample_receipt["rng_root_sha256"]
                    != expected["rng_root_sha256"]
                    or sample_receipt["branch_index"]
                    != expected["producing_branch_index"]
                    or sample_receipt["seed"] != expected["sample_seed"]
                    or sampling_config_sha256(sample)
                    != expected["sampling_config_sha256"]
                ):
                    _fail("provider_runtime_sample_plan_mismatch")
            request = VerifiedTransitionProductionRequest(
                schema=PRODUCTION_REQUEST_SCHEMA,
                contract_sha256=self.contract_sha256,
                campaign_schedule_root_sha256=self._contract[
                    "campaign_schedule_root_sha256"
                ],
                sequence=sequence,
                task=task,
                prompt_tokens=tuple(prompt_tokens),
                samples=tuple(samples),
                completions=tuple(completions),
                task_commitment=cast(
                    dict[str, Any], _clone(commitment, role="task_commitment")
                ),
                lineage_plan=dict(plan.lineage_plan),
                provider_config=cast(
                    dict[str, Any], _clone(self._provider_config, role="provider_config")
                ),
            )
            validated = self._validate_prepared(
                self._producer(request),
                plan=plan,
                prompt_tokens=prompt_tokens,
                samples=samples,
            )
            self._pending_sequence = sequence
            return validated

    def _records(self, sequence: int) -> tuple[dict[str, Any], dict[str, Any]]:
        records = self._ledger.group_records_unclosed(sequence=sequence)
        if not isinstance(records, tuple) or len(records) != 2:
            _fail("provider_campaign_records_invalid")
        start, terminal = records
        if not isinstance(start, Mapping) or not isinstance(terminal, Mapping):
            _fail("provider_campaign_records_invalid")
        return (
            cast(dict[str, Any], _clone(start, role="campaign_start")),
            cast(dict[str, Any], _clone(terminal, role="campaign_terminal")),
        )

    def _validate_step(
        self, *, sequence: int, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        commitment = self._commitment(sequence)
        step = validate_verified_transition_step_receipt(
            receipt,
            group_size=len(commitment["sample_seeds"]),
            execution_spec_sha256=commitment["recurrent_execution_spec_sha256"],
        )
        start, terminal = self._records(sequence)
        expected_before = (
            self._contract["initial_policy_sha256"]
            if sequence == 0
            else self._accepted_steps[sequence - 1]["policy_after_sha256"]
        )
        manifest = self._validate_start_record(
            sequence=sequence,
            expected_policy_sha256=expected_before,
            start=start,
        )
        sample_policies = [row.get("policy_sha256") for row in step["samples"]]
        if (
            step["step"] != sequence + 1
            or step["campaign_sequence"] != sequence
            or step["task_id"] != commitment["task_id"]
            or step["sample_seed"] != commitment["trainer_sample_seed"]
            or step["policy_before_sha256"] != expected_before
            or sample_policies != [expected_before] * len(sample_policies)
            or step["group_manifest_sha256"] != manifest["manifest_sha256"]
            or start.get("campaign_schedule_root_sha256")
            != self._contract["campaign_schedule_root_sha256"]
            or start.get("policy_before_sha256") != expected_before
            or terminal != step["terminal"]
            or terminal.get("campaign_manifest_sha256")
            != start.get("campaign_manifest_sha256")
            or terminal.get("campaign_schedule_root_sha256")
            != self._contract["campaign_schedule_root_sha256"]
        ):
            _fail("provider_step_lineage_crosscheck_mismatch")
        return step

    def _validate_start_record(
        self,
        *,
        sequence: int,
        expected_policy_sha256: str,
        start: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reconstruct a persisted JIT plan during acceptance and resume."""

        commitment = self._commitment(sequence)
        manifest = validate_transition_group_manifest(start.get("group_manifest"))
        entries = manifest["entries"]
        expected_plan = self._lineage_plan(
            sequence=sequence,
            policy_before_sha256=expected_policy_sha256,
            group_manifest=manifest,
        )
        group_attestation = start.get("group_manifest_attestation")
        lineage_attestation = start.get("lineage_attestation")
        admitted_at = _integer(
            start.get("admitted_at_unix_ns"),
            role="provider_start_admitted_at",
            minimum=1,
        )
        if (
            start.get("campaign_schedule_root_sha256")
            != self._contract["campaign_schedule_root_sha256"]
            or start.get("sequence") != sequence
            or start.get("policy_before_sha256") != expected_policy_sha256
            or start.get("lineage_plan") != expected_plan
            or manifest["task_id"] != commitment["task_id"]
            or [entry["sample_seed"] for entry in entries]
            != commitment["sample_seeds"]
            or any(
                entry["policy_sha256"] != expected_policy_sha256 for entry in entries
            )
            or any(
                entry["recurrent_execution_spec_sha256"]
                != commitment["recurrent_execution_spec_sha256"]
                for entry in entries
            )
            or not isinstance(group_attestation, Mapping)
            or not isinstance(lineage_attestation, Mapping)
            or manifest["planned_at_unix_ns"] >= admitted_at
        ):
            _fail("provider_start_record_reconstruction_mismatch")
        verify_role_attestation(
            self._policy,
            group_attestation,
            role=TASK_ISSUER,
            expected_payload=manifest,
            not_after_unix=manifest["planned_at_unix_ns"] // 1_000_000_000,
        )
        verify_role_attestation(
            self._policy,
            lineage_attestation,
            role=TASK_ISSUER,
            expected_payload=expected_plan,
            not_after_unix=admitted_at // 1_000_000_000,
        )
        return manifest

    def accept_step_receipt(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        """Advance policy lineage from one durable, cross-checked trainer step."""

        with self._lock:
            if self._pending_sequence is None:
                _fail("provider_accept_without_pending_group")
            sequence = self._pending_sequence
            step = self._validate_step(sequence=sequence, receipt=receipt)
            self._accepted_steps.append(step)
            self._pending_sequence = None
            return dict(step)

    def recover_transaction_publications(
        self,
        *,
        transaction_store: VerifiedTransitionTransactionStore,
        sequence: int,
        admission_sha256: str,
        validate_staged_state: Callable[[LoadedVerifiedTransitionTransaction], str],
    ) -> LoadedVerifiedTransitionTransaction:
        """Validate staged tensors, then roll them through durable ledgers."""

        with self._lock:
            if (
                self._finalized
                or self._pending_sequence is not None
                or sequence != len(self._accepted_steps)
            ):
                _fail("provider_transaction_recovery_state_invalid")
            loaded = transaction_store.load(
                sequence=sequence,
                admission_sha256=admission_sha256,
                load_tensors=True,
            )
            if loaded is None:
                _fail("provider_transaction_recovery_stage_missing")
            pending = validate_pending_trainer_step(loaded.pending_step)
            commitment = self._commitment(sequence)
            expected_policy = self.expected_policy_sha256
            start = self._ledger.group_start(sequence=sequence)
            if not isinstance(start, Mapping):
                _fail("provider_transaction_recovery_start_invalid")
            manifest = self._validate_start_record(
                sequence=sequence,
                expected_policy_sha256=expected_policy,
                start=start,
            )
            if (
                pending["sequence"] != sequence
                or pending["trainer_step"] != sequence + 1
                or pending["task_id"] != commitment["task_id"]
                or pending["trainer_sample_seed"]
                != commitment["trainer_sample_seed"]
                or pending["execution_spec_sha256"]
                != commitment["recurrent_execution_spec_sha256"]
                or pending["campaign_manifest_sha256"]
                != start.get("campaign_manifest_sha256")
                or pending["campaign_schedule_root_sha256"]
                != self._contract["campaign_schedule_root_sha256"]
                or pending["group_manifest_sha256"]
                != manifest["manifest_sha256"]
                or pending["group_admission_sha256"] != admission_sha256
                or pending["policy_before_sha256"] != expected_policy
            ):
                _fail("provider_transaction_recovery_binding_mismatch")
            restored_policy = _sha256(
                validate_staged_state(loaded),
                role="provider_transaction_recovery_restored_policy",
            )
            if restored_policy != pending["policy_after_sha256"]:
                _fail("provider_transaction_recovery_restored_policy_mismatch")

            journal = VerifiedTransitionUpdateJournal.open(
                self._contract["ledger_roots"]["updates"]
            )
            if len(loaded.events) == 0:
                update = commit_staged_verified_transition_update(
                    journal,
                    admission_sha256=admission_sha256,
                    policy_before_sha256=pending["policy_before_sha256"],
                    policy_after_sha256=pending["policy_after_sha256"],
                )
                transaction_store.record_update_commit(
                    sequence=sequence,
                    admission_sha256=admission_sha256,
                    update_receipt=update,
                )
            loaded = transaction_store.load(
                sequence=sequence,
                admission_sha256=admission_sha256,
                load_tensors=False,
            )
            assert loaded is not None
            if len(loaded.events) == 1:
                update = loaded.events[0]["evidence"]
                terminal = self._ledger.group_terminal_if_exists(sequence=sequence)
                if terminal is None:
                    terminal = self._ledger.finish_group(
                        sequence=sequence,
                        status="updated",
                        reward_receipt_sha256=pending[
                            "reward_receipt_sha256"
                        ],
                        group_admission_sha256=admission_sha256,
                        update_receipt_sha256=update["receipt_sha256"],
                        terminal_reason=(
                            "optimizer_update_recovered_from_staged_transaction"
                        ),
                        finished_at_unix_ns=max(
                            int(update["committed_at_unix_ns"]), time.time_ns()
                        ),
                        policy_after_sha256=pending["policy_after_sha256"],
                    )
                transaction_store.record_campaign_terminal(
                    sequence=sequence,
                    admission_sha256=admission_sha256,
                    terminal_receipt=terminal,
                )
            recovered = transaction_store.load(
                sequence=sequence,
                admission_sha256=admission_sha256,
                load_tensors=True,
            )
            if recovered is None or len(recovered.events) < 2:
                _fail("provider_transaction_recovery_publication_incomplete")
            return recovered

    def recover_rejection_publications(
        self,
        *,
        rejection_store: Any,
        sequence: int,
        reward_receipt_sha256: str,
        validate_live_policy: Callable[[], str],
    ) -> Any:
        """Finish one staged rejection without inventing an optimizer update."""

        from core.learning.verified_transition_rejection_transaction import (
            validate_rejection_intent,
        )

        with self._lock:
            if (
                self._finalized
                or self._pending_sequence is not None
                or sequence != len(self._accepted_steps)
            ):
                _fail("provider_rejection_recovery_state_invalid")
            loaded = rejection_store.load(
                sequence=sequence,
                reward_sha256=reward_receipt_sha256,
            )
            if loaded is None:
                _fail("provider_rejection_recovery_intent_missing")
            intent = validate_rejection_intent(loaded.intent)
            commitment = self._commitment(sequence)
            expected_policy = self.expected_policy_sha256
            start = self._ledger.group_start(sequence=sequence)
            if not isinstance(start, Mapping):
                _fail("provider_rejection_recovery_start_invalid")
            manifest = self._validate_start_record(
                sequence=sequence,
                expected_policy_sha256=expected_policy,
                start=start,
            )
            if (
                intent["sequence"] != sequence
                or intent["trainer_step"] != sequence + 1
                or intent["task_id"] != commitment["task_id"]
                or intent["trainer_sample_seed"]
                != commitment["trainer_sample_seed"]
                or intent["execution_spec_sha256"]
                != commitment["recurrent_execution_spec_sha256"]
                or intent["campaign_manifest_sha256"]
                != start.get("campaign_manifest_sha256")
                or intent["campaign_schedule_root_sha256"]
                != self._contract["campaign_schedule_root_sha256"]
                or intent["group_manifest_sha256"]
                != manifest["manifest_sha256"]
                or intent["reward_receipt_sha256"]
                != reward_receipt_sha256
                or intent["policy_sha256"] != expected_policy
                or _sha256(
                    validate_live_policy(),
                    role="provider_rejection_recovery_live_policy",
                )
                != expected_policy
            ):
                _fail("provider_rejection_recovery_binding_mismatch")
            if len(loaded.events) == 0:
                terminal = self._ledger.group_terminal_if_exists(sequence=sequence)
                if terminal is None:
                    terminal = self._ledger.finish_group(
                        sequence=sequence,
                        status="rejected",
                        reward_receipt_sha256=reward_receipt_sha256,
                        group_admission_sha256=None,
                        update_receipt_sha256=None,
                        terminal_reason=intent["trainer_step_static"][
                            "optimizer_admission_reason"
                        ],
                        finished_at_unix_ns=time.time_ns(),
                        policy_after_sha256=expected_policy,
                    )
                rejection_store.record_campaign_terminal(
                    sequence=sequence,
                    reward_sha256=reward_receipt_sha256,
                    terminal_receipt=terminal,
                )
            recovered = rejection_store.load(
                sequence=sequence,
                reward_sha256=reward_receipt_sha256,
            )
            if recovered is None or len(recovered.events) < 1:
                _fail("provider_rejection_recovery_publication_incomplete")
            return recovered

    def accept_recovered_step_receipt(
        self, receipt: Mapping[str, Any]
    ) -> Sequence[VerifiedTransitionReplayGroup]:
        """Advance lineage and replay artifacts for a recovered trainer step."""

        with self._lock:
            if self._finalized or self._pending_sequence is not None:
                _fail("provider_recovered_step_state_invalid")
            sequence = len(self._accepted_steps)
            step = self._validate_step(sequence=sequence, receipt=receipt)
            self._accepted_steps.append(step)
            request = VerifiedTransitionRestoreRequest(
                schema=RESTORE_REQUEST_SCHEMA,
                contract_sha256=self.contract_sha256,
                campaign_schedule_root_sha256=self._contract[
                    "campaign_schedule_root_sha256"
                ],
                committed_steps=len(self._accepted_steps),
                step_receipts=tuple(self._accepted_steps),
                replay_artifact_root=self._contract["ledger_roots"][
                    "replay_artifacts"
                ],
            )
            loaded = tuple(self._loader(request))
            updated = [
                item
                for item in self._accepted_steps
                if item["step_kind"] == "verified_optimizer_update"
            ]
            if [group.sequence for group in loaded] != [
                item["campaign_sequence"] for item in updated
            ]:
                _fail("provider_recovered_group_set_mismatch")
            return tuple(
                self._validate_replay_group(group, step=item)
                for group, item in zip(loaded, updated, strict=True)
            )

    def _validate_replay_group(
        self, group: VerifiedTransitionReplayGroup, *, step: Mapping[str, Any]
    ) -> VerifiedTransitionReplayGroup:
        sequence = cast(int, step["campaign_sequence"])
        roots = self._contract["ledger_roots"]
        if (
            not isinstance(group, VerifiedTransitionReplayGroup)
            or group.sequence != sequence
            or _root_of(group.transition_store, role="replay_store")
            != roots["transition_artifacts"]
            or _root_of(group.update_journal, role="replay_update_journal")
            != roots["updates"]
            or group.independent_scorer is not self._scorer
            or group.token_encoder is not self._encoder
            or group.token_decoder is not self._decoder
        ):
            _fail("provider_restore_runtime_binding_mismatch")
        manifest = validate_transition_group_manifest(group.group_manifest)
        admission = validate_verified_transition_group_admission(
            group.transition_store,
            group.group_admission_receipt,
            group.reward_receipt,
            group.transition_evidence,
            group.samples,
            group.prompt_tokens,
            group_manifest=manifest,
            group_manifest_attestation=group.group_manifest_attestation,
            independent_scorer=self._scorer,
            token_encoder=self._encoder,
            token_decoder=self._decoder,
        )
        update = validate_verified_transition_update_receipt(
            group.update_journal, group.update_receipt
        )
        if (
            manifest["manifest_sha256"] != step["group_manifest_sha256"]
            or group.reward_receipt.get("receipt_sha256")
            != step["reward_receipt_sha256"]
            or admission.get("receipt_sha256") != step["group_admission_sha256"]
            or update.get("receipt_sha256") != step["update_receipt_sha256"]
            or update.get("policy_before_sha256") != step["policy_before_sha256"]
            or update.get("policy_after_sha256") != step["policy_after_sha256"]
        ):
            _fail("provider_restore_artifact_crosscheck_mismatch")
        return group

    def restore_groups(
        self,
        *,
        committed_steps: int,
        step_receipts: Sequence[Mapping[str, Any]],
    ) -> Sequence[VerifiedTransitionReplayGroup]:
        with self._lock:
            if (
                self._finalized
                or self._restore_attempted
                or self._accepted_steps
                or self._plans
                or self._pending_sequence is not None
            ):
                _fail("provider_restore_state_invalid")
            self._restore_attempted = True
            _integer(committed_steps, role="provider_committed_steps")
            if committed_steps != len(step_receipts) or committed_steps > len(
                self._contract["task_schedule"]
            ):
                _fail("provider_restore_step_count_mismatch")
            steps: list[dict[str, Any]] = []
            for sequence, receipt in enumerate(step_receipts):
                step = self._validate_step(sequence=sequence, receipt=receipt)
                self._accepted_steps.append(step)
                steps.append(step)
            request = VerifiedTransitionRestoreRequest(
                schema=RESTORE_REQUEST_SCHEMA,
                contract_sha256=self.contract_sha256,
                campaign_schedule_root_sha256=self._contract[
                    "campaign_schedule_root_sha256"
                ],
                committed_steps=committed_steps,
                step_receipts=tuple(steps),
                replay_artifact_root=self._contract["ledger_roots"]["replay_artifacts"],
            )
            loaded = tuple(self._loader(request))
            updated = [step for step in steps if step["step_kind"] == "verified_optimizer_update"]
            if [group.sequence for group in loaded] != [
                step["campaign_sequence"] for step in updated
            ]:
                _fail("provider_restore_updated_group_set_mismatch")
            return tuple(
                self._validate_replay_group(group, step=step)
                for group, step in zip(loaded, updated, strict=True)
            )

    def finalize(
        self,
        *,
        completed_groups: int,
        halt_reason: str,
        replay_groups: Sequence[VerifiedTransitionReplayGroup],
    ) -> VerifiedTransitionCampaignClosure:
        with self._lock:
            if self._finalized or self._pending_sequence is not None:
                _fail("provider_finalize_state_invalid")
            if completed_groups != len(self._accepted_steps):
                _fail("provider_finalize_group_count_mismatch")
            _identifier(halt_reason, role="provider_halt_reason")
            request = VerifiedTransitionFinalizeRequest(
                schema=FINALIZE_REQUEST_SCHEMA,
                contract_sha256=self.contract_sha256,
                campaign_schedule_root_sha256=self._contract[
                    "campaign_schedule_root_sha256"
                ],
                completed_groups=completed_groups,
                halt_reason=halt_reason,
                replay_groups=tuple(replay_groups),
            )
            closure = self._finalizer(request)
            if not isinstance(closure, VerifiedTransitionCampaignClosure):
                _fail("provider_finalize_closure_type_invalid")
            if (
                closure.campaign_ledger is not self._ledger
                or _root_of(closure.campaign_ledger, role="closure_ledger")
                != self._contract["ledger_roots"]["campaign"]
                or closure.campaign_trust_policy.policy_sha256 != self._policy.policy_sha256
                or closure.campaign_trust_policy.root_key_id != self._policy.root_key_id
            ):
                _fail("provider_finalize_trust_or_campaign_substitution")
            closed = closure.campaign_ledger.validate_closed(policy=self._policy)
            payload = closed.get("close_payload")
            if not isinstance(payload, Mapping):
                _fail("provider_finalize_close_payload_invalid")
            statuses = payload.get("group_statuses")
            if (
                payload.get("campaign_schedule_root_sha256")
                != self._contract["campaign_schedule_root_sha256"]
                or not isinstance(statuses, list)
                or len(statuses) != len(self._contract["task_schedule"])
                or any(status not in {"updated", "rejected"} for status in statuses[:completed_groups])
                or any(status not in {"aborted", "indeterminate"} for status in statuses[completed_groups:])
            ):
                _fail("provider_finalize_campaign_status_mismatch")
            expected_updated = [
                sequence
                for sequence, status in enumerate(statuses[:completed_groups])
                if status == "updated"
            ]
            if [group.sequence for group in replay_groups] != expected_updated:
                _fail("provider_finalize_replay_group_set_mismatch")
            self._finalized = True
            return closure


__all__ = [
    "CAMPAIGN_SCHEDULE_SCHEMA",
    "CausalCampaignLedger",
    "FINALIZE_REQUEST_SCHEMA",
    "LINEAGE_PLAN_SCHEMA",
    "PRODUCTION_REQUEST_SCHEMA",
    "PROVIDER_CONTRACT_SCHEMA",
    "PROVIDER_IMPLEMENTATION_ID",
    "ProductionVerifiedTransitionGroupProvider",
    "RECOVERY_POLICY_SCHEMA",
    "RESTORE_REQUEST_SCHEMA",
    "TASK_COMMITMENT_SCHEMA",
    "VerifiedTransitionFinalizeRequest",
    "VerifiedTransitionProductionRequest",
    "VerifiedTransitionProviderError",
    "VerifiedTransitionRestoreRequest",
    "build_verified_transition_provider_contract",
    "callable_source_sha256",
    "provider_implementation_source_sha256",
    "validate_verified_transition_provider_contract",
]
