"""Causal-lineage tests for the production verified-transition provider."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    TransitionGroupPlanEntry,
    build_transition_group_manifest,
    sampling_config_sha256,
)
from core.learning.verified_transition_provider import (
    TASK_COMMITMENT_SCHEMA,
    ProductionVerifiedTransitionGroupProvider,
    VerifiedTransitionProviderError,
    build_verified_transition_provider_contract,
    callable_source_sha256,
    validate_verified_transition_provider_contract,
)
from core.learning.verified_transition_trainer import (
    PreparedVerifiedTransitionGroup,
    VerifiedTransitionCampaignClosure,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _score(_task: Any, _response: Any) -> dict[str, Any]:
    return {"correct": True}


def _encode(value: bytes) -> tuple[int, ...]:
    return tuple(value)


def _decode(value: Any) -> bytes:
    return bytes(value)


class _Task:
    def __init__(self, sequence: int) -> None:
        self.task_id = f"task-{sequence}"
        self.sequence = sequence

    def verified_transition_task_commitment(self) -> dict[str, Any]:
        return {
            "schema": "external.immutable_task.v1",
            "task_id": self.task_id,
            "public_input": f"problem-{self.sequence}",
            "answer_commitment_sha256": _sha(f"answer-{self.sequence}"),
        }


class _SamplingConfig:
    def to_dict(self) -> dict[str, Any]:
        return {"max_tokens": 32, "temperature_micros": 1_000_000}


class _Sample:
    def __init__(
        self,
        *,
        seed: int,
        policy_sha256: str,
        execution_spec_sha256: str,
        prompt_tokens_sha256: str,
        branch_index: int,
    ) -> None:
        self.seed = seed
        self.policy_sha256 = policy_sha256
        self.execution_spec_sha256 = execution_spec_sha256
        self.prompt_tokens_sha256 = prompt_tokens_sha256
        self.branch_index = branch_index
        self.sampling_config = _SamplingConfig()

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "aura.recurrent_sampling_behavior.v3",
            "seed": self.seed,
            "policy_sha256": self.policy_sha256,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
        }


class _CausalLedger:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.starts: dict[int, dict[str, Any]] = {}
        self.terminals: dict[int, dict[str, Any]] = {}
        self.closed: dict[str, Any] | None = None

    def admit_group_plan(self, **kwargs: Any) -> dict[str, Any]:
        sequence = kwargs["sequence"]
        if sequence in self.starts:
            raise RuntimeError("duplicate_start")
        start = {
            "schema": "aura.verified_transition.causal_group_start.v1",
            "campaign_manifest_sha256": _sha("campaign-manifest"),
            "campaign_schedule_root_sha256": kwargs[
                "campaign_schedule_root_sha256"
            ],
            "sequence": sequence,
            "policy_before_sha256": kwargs["policy_before_sha256"],
            "group_manifest": copy.deepcopy(kwargs["group_manifest"]),
            "group_manifest_attestation": copy.deepcopy(
                kwargs["group_manifest_attestation"]
            ),
            "lineage_plan": copy.deepcopy(kwargs["lineage_plan"]),
            "lineage_attestation": copy.deepcopy(kwargs["lineage_attestation"]),
            "admitted_at_unix_ns": kwargs["admitted_at_unix_ns"],
        }
        self.starts[sequence] = start
        return copy.deepcopy(start)

    def validate_started_group(self, *, sequence: int, group_manifest: Any) -> dict:
        start = self.starts[sequence]
        if start["group_manifest"] != group_manifest or sequence in self.terminals:
            raise RuntimeError("started_group_invalid")
        return copy.deepcopy(start)

    def finish_group(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("trainer mutation is outside this provider unit test")

    def group_records_unclosed(
        self, *, sequence: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return copy.deepcopy(self.starts[sequence]), copy.deepcopy(
            self.terminals[sequence]
        )

    def group_start(self, *, sequence: int) -> dict[str, Any]:
        return copy.deepcopy(self.starts[sequence])

    def group_start_if_exists(self, *, sequence: int) -> dict[str, Any] | None:
        start = self.starts.get(sequence)
        return copy.deepcopy(start) if start is not None else None

    def group_terminal_if_exists(self, *, sequence: int) -> dict[str, Any] | None:
        terminal = self.terminals.get(sequence)
        return copy.deepcopy(terminal) if terminal is not None else None

    def group_records(
        self, *, sequence: int, policy: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert policy.policy_sha256
        return self.group_records_unclosed(sequence=sequence)

    def validate_closed(self, *, policy: Any) -> dict[str, Any]:
        assert policy.policy_sha256
        if self.closed is None:
            raise RuntimeError("not_closed")
        return copy.deepcopy(self.closed)


class _Producer:
    def __init__(self) -> None:
        self.prepared: PreparedVerifiedTransitionGroup | None = None
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> PreparedVerifiedTransitionGroup:
        self.requests.append(request)
        assert self.prepared is not None
        return self.prepared


class _Loader:
    def __init__(self) -> None:
        self.groups: tuple[Any, ...] = ()
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> tuple[Any, ...]:
        self.requests.append(request)
        return self.groups


class _Finalizer:
    def __init__(self) -> None:
        self.closure: VerifiedTransitionCampaignClosure | None = None
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> VerifiedTransitionCampaignClosure:
        self.requests.append(request)
        assert self.closure is not None
        return self.closure


@pytest.fixture
def material(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    roots = {
        name: str((tmp_path / name).resolve())
        for name in ("campaign", "transition_artifacts", "updates", "replay_artifacts")
    }
    for root in roots.values():
        Path(root).mkdir(parents=True)
    tasks = (_Task(0), _Task(1))
    prompts = ((11, 12), (21, 22))
    execution = _sha("execution")
    initial_policy = _sha("initial-policy")
    sample_seeds = ((101, 102), (201, 202))
    trainer_seeds = (77, 88)
    schedule = tuple(
        {
            "schema": TASK_COMMITMENT_SCHEMA,
            "sequence": sequence,
            "task_id": task.task_id,
            "trainer_sample_seed": trainer_seeds[sequence],
            "immutable_task_sha256": _digest(
                task.verified_transition_task_commitment()
            ),
            "prompt_tokens_sha256": _digest(list(prompts[sequence])),
            "recurrent_execution_spec_sha256": execution,
            "sample_seeds": list(sample_seeds[sequence]),
        }
        for sequence, task in enumerate(tasks)
    )
    policy = SimpleNamespace(
        policy_sha256=_sha("trust-policy"),
        root_key_id="external-root-key",
        document={"custody": "external_service"},
    )
    ledger = _CausalLedger(Path(roots["campaign"]))
    store = SimpleNamespace(root=Path(roots["transition_artifacts"]))
    journal = SimpleNamespace(root=Path(roots["updates"]))
    producer = _Producer()
    loader = _Loader()
    finalizer = _Finalizer()
    config = {"evidence_timeout_ms": 30_000, "maximum_group_size": 8}
    contract = build_verified_transition_provider_contract(
        provider_config=config,
        evidence_producer_identity="external-evidence-service:v1",
        evidence_producer_source_sha256=callable_source_sha256(producer),
        durable_artifact_loader_identity="durable-replay-loader:v1",
        durable_artifact_loader_source_sha256=callable_source_sha256(loader),
        campaign_finalizer_identity="external-campaign-finalizer:v1",
        campaign_finalizer_source_sha256=callable_source_sha256(finalizer),
        trust_policy_sha256=policy.policy_sha256,
        trust_root_key_id=policy.root_key_id,
        campaign_id="causal-campaign",
        initial_policy_sha256=initial_policy,
        scorer_identity="frontier-independent-scorer:v1",
        scorer_source_sha256=callable_source_sha256(_score),
        token_codec_identity="byte-codec:v1",
        token_encoder_source_sha256=callable_source_sha256(_encode),
        token_decoder_source_sha256=callable_source_sha256(_decode),
        dataset_sha256=_sha("dataset"),
        task_schedule=schedule,
        ledger_roots=roots,
        frozen_at_unix_ns=1_799_999_999_000_000_000,
    )
    reward = {
        "schema": "aura.verified_transition.reward_batch.v1",
        "optimizer_admitted": True,
        "optimizer_admission_reason": "admitted",
        "receipt_sha256": _sha("reward"),
    }
    admission = {
        "schema": "aura.verified_transition.group_admission.v1",
        "receipt_sha256": _sha("admission"),
    }
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.verify_role_attestation",
        lambda _policy, attestation, **kwargs: {
            "attestation": attestation,
            "payload": kwargs["expected_payload"],
        },
    )
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_reward_batch",
        lambda *_args, **_kwargs: copy.deepcopy(reward),
    )
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_group_admission",
        lambda *_args, **_kwargs: copy.deepcopy(admission),
    )

    def make_provider(**overrides: Any) -> ProductionVerifiedTransitionGroupProvider:
        arguments = {
            "contract": contract,
            "provider_config": config,
            "campaign_ledger": ledger,
            "campaign_trust_policy": policy,
            "evidence_producer": producer,
            "evidence_producer_identity": "external-evidence-service:v1",
            "durable_artifact_loader": loader,
            "durable_artifact_loader_identity": "durable-replay-loader:v1",
            "campaign_finalizer": finalizer,
            "campaign_finalizer_identity": "external-campaign-finalizer:v1",
            "independent_scorer": _score,
            "scorer_identity": "frontier-independent-scorer:v1",
            "token_encoder": _encode,
            "token_decoder": _decode,
            "token_codec_identity": "byte-codec:v1",
        }
        arguments.update(overrides)
        return ProductionVerifiedTransitionGroupProvider(**arguments)

    return {
        "roots": roots,
        "tasks": tasks,
        "prompts": prompts,
        "execution": execution,
        "initial_policy": initial_policy,
        "sample_seeds": sample_seeds,
        "trainer_seeds": trainer_seeds,
        "schedule": schedule,
        "policy": policy,
        "ledger": ledger,
        "store": store,
        "journal": journal,
        "producer": producer,
        "loader": loader,
        "finalizer": finalizer,
        "contract": contract,
        "reward": reward,
        "admission": admission,
        "make_provider": make_provider,
    }


def _group_material(
    material: dict[str, Any], *, sequence: int, policy_sha256: str
) -> dict[str, Any]:
    prompt_sha = _digest(list(material["prompts"][sequence]))
    samples = tuple(
        _Sample(
            seed=seed,
            policy_sha256=policy_sha256,
            execution_spec_sha256=material["execution"],
            prompt_tokens_sha256=prompt_sha,
            branch_index=index,
        )
        for index, seed in enumerate(material["sample_seeds"][sequence])
    )
    entries = tuple(
        TransitionGroupPlanEntry(
            episode_id=f"episode-{sequence}-{index}",
            task_id=f"task-{sequence}",
            rng_root_sha256=_sha(f"rng-{sequence}-{index}"),
            policy_sha256=policy_sha256,
            recurrent_execution_spec_sha256=material["execution"],
            producing_branch_index=index,
            sample_seed=sample.seed,
            sampling_config_sha256=sampling_config_sha256(sample),
        )
        for index, sample in enumerate(samples)
    )
    manifest = build_transition_group_manifest(
        group_id=f"group-{sequence}",
        task_id=f"task-{sequence}",
        entries=entries,
        reward_config_sha256=_sha("reward-config"),
        planned_at_unix_ns=1_800_000_000_000_000_000 + sequence,
    )
    evidence = tuple(
        SimpleNamespace(
            episode={
                "episode_id": f"episode-{sequence}-{index}",
                "task_id": f"task-{sequence}",
            }
        )
        for index in range(2)
    )
    prepared = PreparedVerifiedTransitionGroup(
        campaign_sequence=sequence,
        transition_store=material["store"],
        reward_receipt=material["reward"],
        transition_evidence=evidence,
        group_manifest=manifest,
        group_manifest_attestation={"external": f"group-{sequence}"},
        independent_scorer=_score,
        token_encoder=_encode,
        token_decoder=_decode,
        campaign_ledger=material["ledger"],
        campaign_trust_policy=material["policy"],
        group_admission_receipt=material["admission"],
        update_journal=material["journal"],
    )
    return {"manifest": manifest, "samples": samples, "prepared": prepared}


def _admit(
    provider: ProductionVerifiedTransitionGroupProvider,
    group: dict[str, Any],
    *,
    sequence: int,
    policy_sha256: str,
) -> None:
    provider.admit_group_plan(
        sequence=sequence,
        policy_before_sha256=policy_sha256,
        group_manifest=group["manifest"],
        group_manifest_attestation=group["prepared"].group_manifest_attestation,
        lineage_attestation={"external": f"lineage-{sequence}"},
        admitted_at_unix_ns=1_800_000_001_000_000_000 + sequence,
    )


def _prepare(
    material: dict[str, Any],
    provider: ProductionVerifiedTransitionGroupProvider,
    group: dict[str, Any],
    *,
    sequence: int,
) -> None:
    material["producer"].prepared = group["prepared"]
    provider.prepare_group(
        sequence=sequence,
        task=material["tasks"][sequence],
        prompt_tokens=material["prompts"][sequence],
        samples=group["samples"],
        completions=("one", "two"),
    )


def _step(
    material: dict[str, Any],
    *,
    sequence: int,
    group: dict[str, Any],
    policy_before: str,
    policy_after: str,
) -> dict[str, Any]:
    terminal = {
        "schema": "aura.verified_transition.causal_group_terminal.v1",
        "campaign_manifest_sha256": _sha("campaign-manifest"),
        "campaign_schedule_root_sha256": material["contract"][
            "campaign_schedule_root_sha256"
        ],
        "sequence": sequence,
        "status": "updated",
        "group_manifest_sha256": group["manifest"]["manifest_sha256"],
    }
    material["ledger"].terminals[sequence] = copy.deepcopy(terminal)
    return {
        "schema": "aura.verified_transition.trainer_step.v1",
        "step": sequence + 1,
        "campaign_sequence": sequence,
        "task_id": f"task-{sequence}",
        "sample_seed": material["trainer_seeds"][sequence],
        "execution_spec_sha256": material["execution"],
        "samples": [sample.receipt() for sample in group["samples"]],
        "step_kind": "verified_optimizer_update",
        "group_manifest_sha256": group["manifest"]["manifest_sha256"],
        "reward_receipt_sha256": material["reward"]["receipt_sha256"],
        "group_admission_sha256": material["admission"]["receipt_sha256"],
        "update_receipt_sha256": _sha(f"update-{sequence}"),
        "policy_before_sha256": policy_before,
        "policy_after_sha256": policy_after,
        "terminal": terminal,
    }


def test_contract_binds_only_initial_policy_and_immutable_schedule(
    material: dict[str, Any]
) -> None:
    contract = material["contract"]
    assert validate_verified_transition_provider_contract(contract) == contract
    assert contract["initial_policy_sha256"] == material["initial_policy"]
    assert all("policy_sha256" not in row for row in contract["task_schedule"])
    assert all("group_manifest_sha256" not in row for row in contract["task_schedule"])


def test_contract_tampering_fails_closed(material: dict[str, Any]) -> None:
    tampered = copy.deepcopy(material["contract"])
    tampered["task_schedule"][1]["sample_seeds"][0] += 1
    with pytest.raises(VerifiedTransitionProviderError, match="digest_mismatch"):
        validate_verified_transition_provider_contract(tampered)


def test_sampling_requires_pre_admitted_lineage_plan(material: dict[str, Any]) -> None:
    provider = material["make_provider"]()
    group = _group_material(material, sequence=0, policy_sha256=material["initial_policy"])
    material["producer"].prepared = group["prepared"]
    with pytest.raises(VerifiedTransitionProviderError, match="prepare_plan_missing"):
        _prepare(material, provider, group, sequence=0)


def test_restart_rehydrates_exact_open_plan_without_duplicate_start(
    material: dict[str, Any],
) -> None:
    group = _group_material(
        material, sequence=0, policy_sha256=material["initial_policy"]
    )
    first = material["make_provider"]()
    _admit(first, group, sequence=0, policy_sha256=material["initial_policy"])

    restarted = material["make_provider"]()
    _admit(restarted, group, sequence=0, policy_sha256=material["initial_policy"])
    _prepare(material, restarted, group, sequence=0)

    assert list(material["ledger"].starts) == [0]
    assert material["producer"].requests[-1].sequence == 0


@pytest.mark.parametrize("failure", ["restore_error", "wrong_policy"])
def test_recovery_never_publishes_before_staged_state_validation(
    material: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    mx = pytest.importorskip("mlx.core")
    from core.learning.grpo import group_advantages
    from core.learning.verified_transition_transaction import (
        VerifiedTransitionTransactionStore,
        build_pending_trainer_step,
        build_trainer_step_static,
    )

    provider = material["make_provider"]()
    group = _group_material(
        material, sequence=0, policy_sha256=material["initial_policy"]
    )
    _admit(provider, group, sequence=0, policy_sha256=material["initial_policy"])
    start = material["ledger"].starts[0]
    static = build_trainer_step_static(
        samples=[sample.receipt() for sample in group["samples"]],
        structured_rewards=[1.0, 0.0],
        optimizer_admission_reason="admitted",
        answer_channel={"correct_fraction": 0.5},
        advantage_report=group_advantages([1.0, 0.0]),
    )
    policy_after = _sha("staged-policy-after")
    admission = material["admission"]["receipt_sha256"]
    pending = build_pending_trainer_step(
        sequence=0,
        trainer_step=1,
        task_id="task-0",
        trainer_sample_seed=material["trainer_seeds"][0],
        execution_spec_sha256=material["execution"],
        campaign_manifest_sha256=start["campaign_manifest_sha256"],
        campaign_schedule_root_sha256=material["contract"][
            "campaign_schedule_root_sha256"
        ],
        group_manifest_sha256=group["manifest"]["manifest_sha256"],
        group_admission_sha256=admission,
        reward_receipt_sha256=material["reward"]["receipt_sha256"],
        policy_before_sha256=material["initial_policy"],
        policy_after_sha256=policy_after,
        trainer_step_static=static,
        created_at_unix_ns=1,
    )
    transactions = VerifiedTransitionTransactionStore.open(
        Path(material["roots"]["replay_artifacts"]) / "transactions"
    )
    transactions.stage(
        adapter_tensors={"model.lora_a": mx.array([[1.0]])},
        optimizer_tensors={"state.step": mx.array(1)},
        pending_trainer_step=pending,
    )
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.commit_staged_verified_transition_update",
        lambda *_args, **_kwargs: pytest.fail(
            "update journal published before staged state validation"
        ),
    )

    def validate_staged(_transaction: Any) -> str:
        if failure == "restore_error":
            raise RuntimeError("corrupt staged tensor layout")
        return _sha("wrong-restored-policy")

    expected_error = RuntimeError if failure == "restore_error" else VerifiedTransitionProviderError
    with pytest.raises(expected_error):
        provider.recover_transaction_publications(
            transaction_store=transactions,
            sequence=0,
            admission_sha256=admission,
            validate_staged_state=validate_staged,
        )

    loaded = transactions.load(
        sequence=0, admission_sha256=admission, load_tensors=False
    )
    assert loaded is not None and loaded.events == ()
    assert material["ledger"].terminals == {}


def test_first_plan_rejects_noninitial_policy(material: dict[str, Any]) -> None:
    provider = material["make_provider"]()
    wrong = _sha("not-initial")
    group = _group_material(material, sequence=0, policy_sha256=wrong)
    with pytest.raises(VerifiedTransitionProviderError, match="policy_lineage_mismatch"):
        _admit(provider, group, sequence=0, policy_sha256=wrong)


def test_two_step_campaign_uses_actual_prior_policy_after_without_preregistering_it(
    material: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: step 2 is causally bound without an impossible future hash."""

    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_step_receipt",
        lambda receipt, **_kwargs: copy.deepcopy(receipt),
    )
    provider = material["make_provider"]()
    group_0 = _group_material(
        material, sequence=0, policy_sha256=material["initial_policy"]
    )
    _admit(
        provider,
        group_0,
        sequence=0,
        policy_sha256=material["initial_policy"],
    )
    _prepare(material, provider, group_0, sequence=0)
    actual_policy_after = _sha("optimizer-produced-policy-after-step-0")
    step_0 = _step(
        material,
        sequence=0,
        group=group_0,
        policy_before=material["initial_policy"],
        policy_after=actual_policy_after,
    )
    provider.accept_step_receipt(step_0)

    assert actual_policy_after not in canonical_json_bytes(material["contract"]).decode(
        "ascii"
    )
    assert provider.expected_policy_sha256 == actual_policy_after

    group_1 = _group_material(
        material, sequence=1, policy_sha256=actual_policy_after
    )
    _admit(
        provider,
        group_1,
        sequence=1,
        policy_sha256=actual_policy_after,
    )
    _prepare(material, provider, group_1, sequence=1)

    request = material["producer"].requests[-1]
    assert request.sequence == 1
    assert request.lineage_plan["policy_before_sha256"] == actual_policy_after
    assert request.lineage_plan["campaign_schedule_root_sha256"] == material[
        "contract"
    ]["campaign_schedule_root_sha256"]


def test_second_plan_rejects_policy_not_produced_by_first_step(
    material: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_step_receipt",
        lambda receipt, **_kwargs: copy.deepcopy(receipt),
    )
    provider = material["make_provider"]()
    group_0 = _group_material(
        material, sequence=0, policy_sha256=material["initial_policy"]
    )
    _admit(
        provider,
        group_0,
        sequence=0,
        policy_sha256=material["initial_policy"],
    )
    _prepare(material, provider, group_0, sequence=0)
    actual_after = _sha("actual-after")
    provider.accept_step_receipt(
        _step(
            material,
            sequence=0,
            group=group_0,
            policy_before=material["initial_policy"],
            policy_after=actual_after,
        )
    )
    substituted = _sha("substituted-after")
    group_1 = _group_material(material, sequence=1, policy_sha256=substituted)
    with pytest.raises(VerifiedTransitionProviderError, match="policy_lineage_mismatch"):
        _admit(provider, group_1, sequence=1, policy_sha256=substituted)


def test_step_receipt_must_equal_durable_campaign_terminal(
    material: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_step_receipt",
        lambda receipt, **_kwargs: copy.deepcopy(receipt),
    )
    provider = material["make_provider"]()
    group = _group_material(material, sequence=0, policy_sha256=material["initial_policy"])
    _admit(provider, group, sequence=0, policy_sha256=material["initial_policy"])
    _prepare(material, provider, group, sequence=0)
    step = _step(
        material,
        sequence=0,
        group=group,
        policy_before=material["initial_policy"],
        policy_after=_sha("after"),
    )
    material["ledger"].terminals[0]["status"] = "rejected"
    with pytest.raises(VerifiedTransitionProviderError, match="lineage_crosscheck"):
        provider.accept_step_receipt(step)


def test_acceptance_reconstructs_persisted_lineage_plan(
    material: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "core.learning.verified_transition_provider.validate_verified_transition_step_receipt",
        lambda receipt, **_kwargs: copy.deepcopy(receipt),
    )
    provider = material["make_provider"]()
    group = _group_material(material, sequence=0, policy_sha256=material["initial_policy"])
    _admit(provider, group, sequence=0, policy_sha256=material["initial_policy"])
    _prepare(material, provider, group, sequence=0)
    step = _step(
        material,
        sequence=0,
        group=group,
        policy_before=material["initial_policy"],
        policy_after=_sha("after"),
    )
    material["ledger"].starts[0]["lineage_plan"]["policy_before_sha256"] = _sha(
        "forged-lineage"
    )
    with pytest.raises(
        VerifiedTransitionProviderError,
        match="start_record_reconstruction_mismatch",
    ):
        provider.accept_step_receipt(step)


def test_finalize_requires_same_causal_schedule_root(material: dict[str, Any]) -> None:
    provider = material["make_provider"]()
    material["ledger"].closed = {
        "close_payload": {
            "campaign_schedule_root_sha256": material["contract"][
                "campaign_schedule_root_sha256"
            ],
            "group_statuses": ["aborted", "aborted"],
        }
    }
    material["finalizer"].closure = VerifiedTransitionCampaignClosure(
        campaign_ledger=material["ledger"],
        campaign_trust_policy=material["policy"],
    )
    closure = provider.finalize(
        completed_groups=0, halt_reason="preflight_failed", replay_groups=()
    )
    assert closure.campaign_ledger is material["ledger"]


def test_runtime_scorer_substitution_is_rejected(material: dict[str, Any]) -> None:
    def other_scorer(_task: Any, _response: Any) -> dict[str, Any]:
        return {"correct": False}

    with pytest.raises(VerifiedTransitionProviderError, match="runtime_scorer_mismatch"):
        material["make_provider"](independent_scorer=other_scorer)
