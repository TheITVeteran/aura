"""Verifier gradients pass through the actual recurrent hidden-state path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

import core.learning.recurrent_grpo as recurrent_grpo_runtime  # noqa: E402
from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.recurrence_curriculum import khop_reachability  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    ExactAdjointInterventionConfig,
    ExactAdjointTrajectoryConfig,
    LivePathForward,
    exact_adjoint_composite_live_path_value_and_grad,
    live_path_branch_answer_ce_trail,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentGroupClipAdmissionError,
    RecurrentGRPOConfig,
    RecurrentSamplingConfig,
    VerifiedTrajectoryGroupConfig,
    attach_recurrent_policy_adapters,
    branch_token_logprobs,
    build_recurrent_policy_optimizer,
    clipped_recurrent_grpo_objective,
    cortex_config_from_execution_spec,
    exact_adjoint_sampled_group_value_and_grad,
    exact_adjoint_verifier_group_value_and_grad,
    recurrent_completion_token_logprobs,
    recurrent_policy_optimizer_config,
    recurrent_policy_sample_from_causal_pair,
    recurrent_policy_sample_from_receipt,
    recurrent_policy_sha256,
    recurrent_policy_tensor_map_sha256,
    recurrent_sampling_rng_root_sha256,
    sample_final_recurrent_transition_pair,
    sample_recurrent_completion,
    validate_causal_recurrent_transition_pair_receipt,
    validate_recurrent_policy_sample_receipt,
    validate_verified_trajectory_group_receipt,
    verifier_group_objective,
)
from core.learning.verified_recurrent_transition_evidence import (  # noqa: E402
    VerifiedRecurrentTransitionEvidenceError,
    build_verified_recurrent_transition_evidence,
    validate_verified_recurrent_transition_evidence,
)
from core.learning.verified_recurrent_transition_repository import (  # noqa: E402
    VerifiedRecurrentTransitionRepositoryError,
    load_recurrent_replay_packages,
    produce_verified_recurrent_transition_group,
    reconstruct_recurrent_package_inputs,
    score_verified_recurrent_training_task,
)
from core.learning.verified_token_trace import (  # noqa: E402
    build_tokenizer_bundle_identity,
    tokenizer_file_bindings_from_bytes,
)
from core.learning.verified_training_task import (  # noqa: E402
    build_verified_training_task,
)
from core.learning.verified_transition_episode import (  # noqa: E402
    TransitionArtifactStore,
)
from core.learning.verified_transition_group_admission import (  # noqa: E402
    TransitionGroupPlanEntry,
    build_transition_group_manifest,
    sampling_config_sha256,
)
from core.learning.verified_transition_production_factory import (  # noqa: E402
    ProviderBoundTrainingTask,
)
from core.learning.verified_transition_reward import (  # noqa: E402
    TransitionRewardConfig,
    build_verified_transition_reward_batch,
    validate_verified_transition_reward_batch,
)
from tools.recurrence_native_train_v2 import _wrap_window_layers  # noqa: E402
from tools.train_grpo import sample_recurrent_group  # noqa: E402


def _model(seed: int = 311) -> Model:
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    return model


def _prepared(seed: int = 311) -> Model:
    model = _model(seed)
    wrapped = _wrap_window_layers(
        model,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    assert wrapped
    return model


def _set_adapter_delta(model: Model, value: float) -> None:
    for layer_index in (1, 2):
        adapter = model.model.layers[layer_index].self_attn.o_proj
        adapter.lora_b = mx.ones_like(adapter.lora_b) * value
    mx.eval(model.trainable_parameters())


def _spec(
    depth: int = 2,
    branch_roles: tuple[str, ...] = ("constructive_solution",),
) -> RLCExecutionSpec:
    return RLCExecutionSpec(
        n_slots=2,
        branch_roles=branch_roles,
        exchange_interval=1,
        recurrent_steps=depth,
        alpha=0.35,
        prelude_frac=0.25,
        coda_frac=0.25,
    )


def test_proof_campaign_adapter_topology_is_exact_and_reconstructable():
    spec = _spec()
    first = _model(seed=401)
    second = _model(seed=401)

    first_sites = attach_recurrent_policy_adapters(
        first,
        spec,
        lora_rank=2,
        lora_layers=1,
        lora_targets=("q_proj", "o_proj"),
        initialization_seed=17,
    )
    second_sites = attach_recurrent_policy_adapters(
        second,
        spec,
        lora_rank=2,
        lora_layers=1,
        lora_targets=("q_proj", "o_proj"),
        initialization_seed=17,
    )

    assert first_sites == (
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.o_proj",
    )
    assert second_sites == first_sites
    assert recurrent_policy_sha256(first, spec) == recurrent_policy_sha256(
        second,
        spec,
    )


def test_proof_campaign_adapter_topology_preflight_prevents_partial_mutation():
    model = _model(seed=403)

    with pytest.raises(ValueError, match="resolve exactly once"):
        attach_recurrent_policy_adapters(
            model,
            _spec(),
            lora_rank=2,
            lora_layers=1,
            lora_targets=("q_proj", "missing_projection"),
            initialization_seed=19,
        )

    layer = model.model.layers[2]
    assert not isinstance(layer.self_attn.q_proj, ScopedLoRALinear)
    assert not isinstance(layer.self_attn.o_proj, ScopedLoRALinear)


def test_proof_campaign_optimizer_constructor_is_explicit_and_reconstructable():
    config = recurrent_policy_optimizer_config(1e-5)
    model = _model(seed=407)
    attach_recurrent_policy_adapters(
        model,
        _spec(),
        lora_rank=2,
        lora_layers=1,
        lora_targets=("q_proj", "o_proj"),
        initialization_seed=23,
    )
    first = build_recurrent_policy_optimizer(1e-5)
    second = build_recurrent_policy_optimizer(1e-5)
    first.init(model.trainable_parameters())
    second.init(model.trainable_parameters())
    first_state = dict(tree_flatten(first.state))
    second_state = dict(tree_flatten(second.state))

    assert config == {
        "class_name": "mlx.optimizers.Adam",
        "learning_rate_hex": (1e-5).hex(),
        "betas_hex": [(0.9).hex(), (0.999).hex()],
        "eps_hex": (1e-8).hex(),
        "bias_correction": False,
    }
    assert set(first_state) == set(second_state)
    comparisons = [
        mx.array_equal(first_state[key], second_state[key])
        for key in first_state
    ]
    mx.eval(*comparisons)
    assert all(bool(value) for value in comparisons)


def test_branch_logprobs_are_bound_to_exact_tokens_and_branch():
    logits = mx.zeros((1, 2, 8))
    forward = LivePathForward(
        branch_logits=(logits,),
        branch_states=(mx.zeros((1, 2, 4)),),
        exchanges=0,
        prompt_tokens=3,
        answer_tokens=2,
        bridge_tokens=0,
    )

    logprobs = branch_token_logprobs(forward, [2, 3], branch_index=0)

    assert logprobs.shape == (2,)
    assert all(float(value) == pytest.approx(-math.log(8.0)) for value in logprobs)
    with pytest.raises(ValueError, match="branch_index"):
        branch_token_logprobs(forward, [2, 3], branch_index=1)
    with pytest.raises(ValueError, match="do not align"):
        branch_token_logprobs(forward, [2], branch_index=0)


def test_clipped_objective_is_token_normalized_and_reference_anchored():
    old = [mx.array([-2.0, -2.0]), mx.array([-3.0])]
    policy = [old[0] + math.log(1.5), old[1] + math.log(0.5)]
    reference = [mx.array([-2.0, -2.0]), mx.array([-3.0])]
    objective = clipped_recurrent_grpo_objective(
        policy,
        old,
        [1.0, -1.0],
        reference_logprobs=reference,
        config=RecurrentGRPOConfig(clip_epsilon=0.2, kl_coefficient=0.1),
    )
    receipt = objective.receipt()

    assert receipt["policy_loss"] == pytest.approx(-0.2, rel=1e-5)
    assert receipt["clip_fraction"] == 1.0
    assert receipt["reference_kl"] > 0.0
    assert receipt["completion_count"] == 2
    assert receipt["token_count"] == 3


def test_adapter_disabled_reference_keeps_recurrence_but_changes_policy():
    model = _prepared()
    _set_adapter_delta(model, 0.05)

    on = recurrent_completion_token_logprobs(
        model, [5, 9, 17], [7, 11], spec=_spec(), branch_index=0
    )
    off = recurrent_completion_token_logprobs(
        model,
        [5, 9, 17],
        [7, 11],
        spec=_spec(),
        branch_index=0,
        adapters_on=False,
    )
    mx.eval(on, off)

    assert on.shape == off.shape == (2,)
    assert not bool(mx.array_equal(on, off))


def test_execution_spec_maps_to_fixed_live_training_graph():
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    config = cortex_config_from_execution_spec(
        spec,
        sampling=RecurrentSamplingConfig(max_tokens=4),
    )

    assert config.workspace.n_slots == spec.n_slots
    assert config.workspace.roles == spec.slot_roles
    assert config.recurrence.max_steps == spec.recurrent_steps
    assert config.recurrence.min_steps == spec.recurrent_steps
    assert config.recurrence.fixed_depth is True
    assert config.branches.roles == spec.branch_roles
    assert config.branches.exchange_interval == spec.exchange_interval
    assert config.latent_opt.enabled is False
    assert config.fast_weights.enabled is False
    assert config.allow_vanilla_fallback is False
    assert config.answer_replacement_enabled is False
    assert config.decode_temperature == 1.0
    assert config.decode_top_p == 1.0

    with pytest.raises(ValueError, match="temperature=1"):
        RecurrentSamplingConfig(temperature=0.8)


def test_cached_recurrent_sampler_is_admitted_by_differentiable_policy():
    model = _prepared(seed=919)
    _set_adapter_delta(model, 0.02)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    sample = sample_recurrent_completion(
        model,
        [5, 9, 17],
        spec=spec,
        seed=117,
        sampling=RecurrentSamplingConfig(max_tokens=3),
        episode_id="engine-sample-test-117",
    )
    receipt = sample.receipt()
    validated = validate_recurrent_policy_sample_receipt(receipt)

    assert len(sample.tokens) == 3
    assert len(sample.behavior_logprobs) == len(sample.differentiable_logprobs) == 3
    assert sample.branch_index in (0, 1)
    assert sample.behavior_admitted is True
    assert receipt["behavior_admitted"] is True
    assert receipt["seed"] == 117
    assert len(receipt["prompt_tokens_sha256"]) == 64
    assert len(receipt["policy_sha256"]) == 64
    assert len(receipt["behavior_logprobs_sha256"]) == 64
    assert len(receipt["differentiable_logprobs_sha256"]) == 64
    assert receipt["cached_params_unchanged"] is True
    assert receipt["cached_nonparametric_memory_status"] == ("disabled_by_policy")
    assert receipt["cached_recurrence_adapter"]["active"] is True
    assert validated["sample_kind"] == "engine_episode"
    assert validated["episode_id"] == "engine-sample-test-117"
    assert validated["episode_receipt"]["episode_id"] == sample.episode_id


def test_cached_recurrent_sampler_keeps_bounded_incomplete_policy_trace(
    monkeypatch,
):
    from core.brain.llm.latent_cortex import engine as engine_module

    original_engine = engine_module.LatentCortexEngine

    class IncompleteDecodeEngine:
        def __init__(self, *args, **kwargs):
            self._inner = original_engine(*args, **kwargs)

        def reason(self, **kwargs):
            result = self._inner.reason(**kwargs)
            assert result.tokens
            result.ok = False
            result.reason = "decode_incomplete:budget_exhausted"
            return result

    monkeypatch.setattr(engine_module, "LatentCortexEngine", IncompleteDecodeEngine)
    model = _prepared(seed=934)
    _set_adapter_delta(model, 0.02)

    sample = sample_recurrent_completion(
        model,
        [5, 9, 17],
        spec=_spec(
            depth=2,
            branch_roles=("constructive_solution", "critical_audit"),
        ),
        seed=117,
        sampling=RecurrentSamplingConfig(max_tokens=3),
        episode_id="engine-incomplete-sample-test-117",
    )

    assert len(sample.tokens) == 3
    assert len(sample.behavior_logprobs) == 3
    assert sample.behavior_admitted is True


def test_flat_transaction_tensors_reproduce_live_policy_identity() -> None:
    model = _prepared(seed=920)
    spec = _spec(depth=2)
    tensors = dict(tree_flatten(model.trainable_parameters()))

    assert recurrent_policy_tensor_map_sha256(
        tensors,
        spec.sha256,
    ) == recurrent_policy_sha256(model, spec)

    name = sorted(tensors)[0]
    changed = dict(tensors)
    changed[name] = changed[name] + mx.ones_like(changed[name])
    assert recurrent_policy_tensor_map_sha256(
        changed,
        spec.sha256,
    ) != recurrent_policy_sha256(model, spec)


def test_causal_pair_decodes_one_frozen_edge_under_matched_randomness():
    model = _prepared(seed=927)
    _set_adapter_delta(model, 0.02)
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    sampling = RecurrentSamplingConfig(
        max_tokens=3,
        max_clipped_token_fraction=1.0,
    )
    pair = sample_final_recurrent_transition_pair(
        model,
        [5, 9, 17],
        spec=spec,
        branch_index=1,
        seed=117,
        sampling=sampling,
        episode_id="causal-edge-test-117",
    )
    receipt = validate_causal_recurrent_transition_pair_receipt(pair.receipt())
    sample = recurrent_policy_sample_from_causal_pair(
        pair,
        sampling=sampling,
    )
    sample_receipt = validate_recurrent_policy_sample_receipt(sample.receipt())

    assert pair.transition.transition_index == 2
    assert pair.parent.depth == 2
    assert pair.child.depth == 3
    assert pair.parent.seed == pair.child.seed == 117
    assert len(pair.parent.tokens) == len(pair.child.tokens) == 3
    assert pair.parent.state_sha256 == pair.transition.parent_branch_sha256s[1]
    assert pair.child.state_sha256 == pair.transition.child_branch_sha256s[1]
    assert pair.child_behavior_admitted is True
    assert receipt["fixed_token_budget"] == 3
    assert receipt["episode_id"] == "causal-edge-test-117"
    assert receipt["runtime_integrity"]["verdict"]["engine_measurements_complete"] is True
    assert receipt["recurrence_adapter"]["active"] is True
    assert sample_receipt["sample_kind"] == "causal_final_transition"
    assert sample.tokens == pair.child.tokens
    assert recurrent_policy_sample_from_receipt(sample_receipt).receipt() == (sample_receipt)


def test_causal_sample_receipt_rejects_trace_substitution():
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=928),
        [5, 9, 17],
        spec=_spec(depth=2),
        branch_index=0,
        seed=43,
        sampling=RecurrentSamplingConfig(max_tokens=2),
        episode_id="causal-trace-tamper-43",
    )
    receipt = recurrent_policy_sample_from_causal_pair(pair).receipt()
    receipt["tokens"][0] = (receipt["tokens"][0] + 1) % 32

    with pytest.raises(ValueError, match="trace"):
        validate_recurrent_policy_sample_receipt(receipt)


def test_causal_pair_becomes_independently_replayable_transition_evidence(
    tmp_path: Path,
):
    prompt = [5, 9, 17]
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=928),
        prompt,
        spec=_spec(depth=2),
        branch_index=0,
        seed=43,
        sampling=RecurrentSamplingConfig(max_tokens=2),
        episode_id="causal-evidence-43",
    )
    sample = recurrent_policy_sample_from_causal_pair(pair)

    class Task:
        task_id = "causal-evidence-task"

        @staticmethod
        def verified_transition_task_commitment():
            return {
                "schema": "test.task.v1",
                "task_id": "causal-evidence-task",
                "prompt": "select the larger token",
            }

    bundle = build_tokenizer_bundle_identity(
        tokenizer_class="test.NumericTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b'{"kind":"numeric"}',
                "tokenizer_config.json": b'{"separator":" "}',
            }
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256="3" * 64,
    )

    class Adapter:
        bundle_identity = bundle

        @staticmethod
        def encode_prompt(text):
            assert text == "Prompt"
            return prompt

        @staticmethod
        def decode_output(tokens):
            return "江山 " + " ".join(str(token) for token in tokens)

        @classmethod
        def stream_decode_deltas(cls, tokens):
            rendered = [cls.decode_output(tokens[: index + 1]) for index in range(len(tokens))]
            return [
                value if index == 0 else value[len(rendered[index - 1]) :]
                for index, value in enumerate(rendered)
            ]

    adapter = Adapter()

    def score(_task, response):
        return {
            "parsed": True,
            "correct": bool(response),
            "reason": "deterministic_test_score",
            "normalized_answer_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        }

    store = TransitionArtifactStore(tmp_path / "recurrent-evidence")
    policy = SimpleNamespace(policy_sha256="1" * 64, root_key_id="2" * 64)
    evidence = build_verified_recurrent_transition_evidence(
        store,
        task=Task(),
        prompt_text="Prompt",
        prompt_tokens=prompt,
        sample=sample,
        supplied_completion=adapter.decode_output(sample.tokens),
        independent_scorer=score,
        tokenizer_trace_adapter=adapter,
        expected_tokenizer_bundle_sha256=bundle["bundle_sha256"],
        campaign_trust_policy=policy,
        created_at_unix_ns=1_800_000_000_000_000_000,
    )

    replayed = validate_verified_recurrent_transition_evidence(
        store,
        evidence.document,
        task=Task(),
        independent_scorer=score,
        tokenizer_trace_adapter=adapter,
        expected_tokenizer_bundle_sha256=bundle["bundle_sha256"],
        campaign_trust_policy=policy,
    )
    assert replayed.document["episode_id"] == "causal-evidence-43"
    assert replayed.document["child_token_trace"]["generation"]["response_text"].startswith(
        "江山 "
    )
    assert replayed.document["child_observable_completion"][
        "optimization_token_count"
    ] == len(sample.tokens)
    stored_sample = json.loads(replayed.document["sample_receipt_json"])
    assert stored_sample["tokens"] == list(pair.child.tokens)
    reward = build_verified_transition_reward_batch(
        store,
        (replayed,),
        independent_scorer=score,
        token_encoder=lambda value: tuple(value),
        token_decoder=lambda tokens: adapter.decode_output(tokens).encode(),
        created_at_unix_ns=1_800_000_000_000_000_100,
    )
    assert (
        validate_verified_transition_reward_batch(
            store,
            reward,
            (replayed,),
            independent_scorer=score,
            token_encoder=lambda value: tuple(value),
            token_decoder=lambda tokens: adapter.decode_output(tokens).encode(),
        )
        == reward
    )
    assert reward["transitions"][0]["pass_1_output_token_ids"] == list(pair.child.tokens)
    attacked = dict(replayed.document)
    attacked["child_response_sha256"] = "0" * 64
    with pytest.raises(
        VerifiedRecurrentTransitionEvidenceError,
        match="reconstruction_mismatch",
    ):
        validate_verified_recurrent_transition_evidence(
            store,
            attacked,
            task=Task(),
            independent_scorer=score,
            tokenizer_trace_adapter=adapter,
            expected_tokenizer_bundle_sha256=bundle["bundle_sha256"],
            campaign_trust_policy=policy,
        )
    boundary_attack = copy.deepcopy(replayed.document)
    boundary_attack["child_observable_completion"]["optimization_token_count"] = 1
    boundary_unsigned = dict(boundary_attack)
    boundary_unsigned.pop("receipt_sha256")
    boundary_attack["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(boundary_unsigned)
    ).hexdigest()
    with pytest.raises(
        VerifiedRecurrentTransitionEvidenceError,
        match="observable_completion_invalid",
    ):
        validate_verified_recurrent_transition_evidence(
            store,
            boundary_attack,
            task=Task(),
            independent_scorer=score,
            tokenizer_trace_adapter=adapter,
            expected_tokenizer_bundle_sha256=bundle["bundle_sha256"],
            campaign_trust_policy=policy,
        )


def test_recurrent_package_survives_object_free_restart(tmp_path: Path):
    prompt = [5, 9, 17]
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=929),
        prompt,
        spec=_spec(depth=2),
        branch_index=0,
        seed=47,
        sampling=RecurrentSamplingConfig(max_tokens=2),
        episode_id="replay-package-47",
    )
    sample = recurrent_policy_sample_from_causal_pair(pair)

    source_task = khop_reachability(1, 929)
    public_task, _sealed_task = build_verified_training_task(
        source_task,
        answer_nonce=b"replay-package-answer-nonce-32-bytes",
    )
    task = ProviderBoundTrainingTask(
        source_task,
        public_task.to_dict(),
    )

    bundle = build_tokenizer_bundle_identity(
        tokenizer_class="test.NumericTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b'{"kind":"numeric"}',
                "tokenizer_config.json": b'{"separator":" "}',
            }
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256="4" * 64,
    )

    class Adapter:
        bundle_identity = bundle

        @staticmethod
        def encode_prompt(text):
            assert text == source_task.prompt
            return prompt

        @staticmethod
        def decode_output(tokens):
            return " ".join(str(token) for token in tokens)

        @classmethod
        def stream_decode_deltas(cls, tokens):
            rendered = [cls.decode_output(tokens[: index + 1]) for index in range(len(tokens))]
            return [
                value if index == 0 else value[len(rendered[index - 1]) :]
                for index, value in enumerate(rendered)
            ]

    score = score_verified_recurrent_training_task

    roots = {
        name: str((tmp_path / name).resolve())
        for name in (
            "campaign",
            "transition_artifacts",
            "updates",
            "replay_artifacts",
        )
    }
    for root in roots.values():
        Path(root).mkdir(mode=0o700)
    planned = 1_800_000_000_000_000_000
    reward_sha256 = hashlib.sha256(
        json.dumps(
            TransitionRewardConfig().to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    manifest = build_transition_group_manifest(
        group_id="replay-package-group",
        task_id=task.task_id,
        entries=(
            TransitionGroupPlanEntry(
                episode_id=sample.episode_id,
                task_id=task.task_id,
                rng_root_sha256=sample.rng_root_sha256,
                policy_sha256=sample.policy_sha256,
                recurrent_execution_spec_sha256=(sample.execution_spec_sha256),
                producing_branch_index=sample.branch_index,
                sample_seed=sample.seed,
                sampling_config_sha256=sampling_config_sha256(sample),
            ),
        ),
        reward_config_sha256=reward_sha256,
        planned_at_unix_ns=planned,
    )
    policy = SimpleNamespace(policy_sha256="5" * 64, root_key_id="6" * 64)
    request = SimpleNamespace(
        schema="aura.verified_transition.production_request.v2",
        contract_sha256="7" * 64,
        campaign_schedule_root_sha256="8" * 64,
        sequence=0,
        task=task,
        prompt_text=source_task.prompt,
        prompt_tokens=tuple(prompt),
        samples=(sample,),
        completions=(Adapter.decode_output(sample.tokens),),
        group_manifest=manifest,
        group_manifest_attestation={"test": "attestation"},
        provider_config={},
        ledger_roots=roots,
        campaign_ledger=SimpleNamespace(
            group_start=lambda **_kwargs: {"campaign_manifest_sha256": "9" * 64}
        ),
        campaign_trust_policy=policy,
        tokenizer_bundle_sha256=bundle["bundle_sha256"],
        tokenizer_trace_adapter=Adapter(),
        independent_scorer=score,
        token_encoder=lambda value: tuple(value),
        token_decoder=lambda tokens: Adapter.decode_output(tokens).encode(),
    )
    prepared = produce_verified_recurrent_transition_group(request)
    assert prepared.reward_receipt["optimizer_admitted"] is False
    replayed_prepared = produce_verified_recurrent_transition_group(request)
    assert replayed_prepared.reward_receipt == prepared.reward_receipt
    assert (
        replayed_prepared.transition_evidence[0].document
        == prepared.transition_evidence[0].document
    )
    step = {
        "task_id": task.task_id,
        "reward_receipt_sha256": prepared.reward_receipt["receipt_sha256"],
        "group_manifest_sha256": manifest["manifest_sha256"],
        "group_admission_sha256": None,
    }
    restore_request = SimpleNamespace(
        schema="aura.verified_transition.restore_request.v2",
        contract_sha256=request.contract_sha256,
        campaign_schedule_root_sha256=(request.campaign_schedule_root_sha256),
        committed_steps=1,
        step_receipts=(step,),
        replay_artifact_root=roots["replay_artifacts"],
    )
    packages = load_recurrent_replay_packages(restore_request)
    reopened_store = TransitionArtifactStore(roots["transition_artifacts"])
    samples, evidence = reconstruct_recurrent_package_inputs(
        packages[0],
        store=reopened_store,
        task=task,
        independent_scorer=score,
        tokenizer_trace_adapter=Adapter(),
        campaign_trust_policy=policy,
    )
    assert samples[0].receipt() == sample.receipt()
    assert (
        evidence[0].document["receipt_sha256"]
        == (prepared.transition_evidence[0].document["receipt_sha256"])
    )
    package_path = Path(roots["replay_artifacts"]) / "group-00000000.prepared.json"
    package_bytes = package_path.read_bytes()
    package_path.write_bytes(package_bytes + b" ")
    with pytest.raises(
        VerifiedRecurrentTransitionRepositoryError,
        match="recurrent_replay_package_noncanonical",
    ):
        load_recurrent_replay_packages(restore_request)


def test_causal_samples_enter_the_exact_adjoint_without_identity_loss():
    model = _prepared(seed=930)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    samples = [
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                prompt,
                spec=spec,
                branch_index=branch,
                seed=seed,
                sampling=sampling,
                episode_id=f"causal-adjoint-{branch}-{seed}",
            ),
            sampling=sampling,
        )
        for branch, seed in ((0, 47), (1, 53))
    ]

    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        [1.0, 0.0],
        spec=spec,
        config=RecurrentGRPOConfig(
            kl_coefficient=0.0,
            max_initial_clip_fraction=1.0,
            max_initial_old_policy_approx_kl=1.0,
        ),
    )

    assert result.gradients is not None
    assert result.completion_count == 2
    assert result.branch_indices == (0, 1)


def test_causal_pair_rejects_branch_state_substitution():
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=929),
        [5, 9, 17],
        spec=_spec(depth=2),
        branch_index=0,
        seed=41,
        sampling=RecurrentSamplingConfig(max_tokens=2),
    )
    attacked = pair.receipt()
    attacked["child"]["state_sha256"] = attacked["parent"]["state_sha256"]

    with pytest.raises(ValueError, match="child"):
        validate_causal_recurrent_transition_pair_receipt(attacked)


def test_trainer_group_uses_tokenizer_and_distinct_bound_seeds():
    model = _prepared(seed=941)

    class Tokenizer:
        eos_token_id = None

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            assert messages[0]["content"].endswith("\n\nsolve")
            assert "FINAL_ANSWER: {JSON object}" in messages[0]["content"]
            return "rendered"

        @staticmethod
        def encode(text, **_kwargs):
            if text == "rendered":
                return [5, 9, 17]
            assert _kwargs.get("add_special_tokens") is False
            assert isinstance(text, str) and text
            return [1 + (sum(text.encode("utf-8")) % 61)]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

        decode_output = decode

    class Task:
        task_id = "trainer-group-task"
        prompt = "solve"

    prompt, samples, completions = sample_recurrent_group(
        model,
        Tokenizer(),
        Task(),
        spec=_spec(depth=2),
        size=2,
        max_tokens=2,
        seed=51,
        sampling_config=RecurrentSamplingConfig(
            max_tokens=2,
            max_abs_logprob_drift=100.0,
            max_mean_abs_logprob_drift=100.0,
            clip_epsilon=1.0,
            max_clipped_token_fraction=1.0,
            max_old_policy_approx_kl=100.0,
        ),
    )

    assert prompt == [5, 9, 17]
    assert len(samples) == len(completions) == 2
    assert samples[0].seed != samples[1].seed
    assert all(sample.behavior_admitted for sample in samples)
    assert completions == [" ".join(str(token) for token in sample.tokens) for sample in samples]


def test_trainer_executes_exact_pre_admitted_causal_group_without_retries():
    from core.learning.verified_transition_group_admission import (
        sampling_config_sha256,
    )
    from core.learning.verified_transition_trainer import (
        VerifiedTransitionSamplingEntry,
        VerifiedTransitionSamplingPlan,
    )

    model = _prepared(seed=943)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    tokenizer_bundle = build_tokenizer_bundle_identity(
        tokenizer_class="test.SignedCausalTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b'{"kind":"numeric"}',
                "tokenizer_config.json": b'{"separator":" "}',
            }
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256="9" * 64,
    )

    class Tokenizer:
        bundle_identity = tokenizer_bundle

        @staticmethod
        def apply_chat_template(_messages, **_kwargs):
            return "rendered"

        @staticmethod
        def encode(_text, **_kwargs):
            return [5, 9, 17]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

        decode_output = decode

        @classmethod
        def stream_decode_deltas(cls, tokens):
            rendered = [
                cls.decode_output(tokens[: index + 1])
                for index in range(len(tokens))
            ]
            return [
                value if index == 0 else value[len(rendered[index - 1]) :]
                for index, value in enumerate(rendered)
            ]

    class Task:
        task_id = "signed-causal-task"
        prompt = "solve"

    class Provider:
        calls = 0

        def sampling_plan(self, *, sequence, task, prompt_tokens, policy_sha256):
            self.calls += 1
            entries = []
            prompt_sha256 = hashlib.sha256(
                json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
            ).hexdigest()
            for branch, seed in ((0, 61), (1, 67)):
                episode_id = f"signed-causal-{branch}-{seed}"
                template = SimpleNamespace(sampling_config=sampling)
                entries.append(
                    VerifiedTransitionSamplingEntry(
                        episode_id=episode_id,
                        rng_root_sha256=recurrent_sampling_rng_root_sha256(
                            episode_id=episode_id,
                            prompt_tokens_sha256=prompt_sha256,
                            policy_sha256=policy_sha256,
                            execution_spec_sha256=spec.sha256,
                            branch_index=branch,
                            seed=seed,
                            sampling_config=sampling,
                        ),
                        producing_branch_index=branch,
                        sample_seed=seed,
                        sampling_config_sha256=sampling_config_sha256(template),
                    )
                )
            return VerifiedTransitionSamplingPlan(
                campaign_sequence=sequence,
                group_manifest_sha256="a" * 64,
                task_id=task.task_id,
                policy_sha256=policy_sha256,
                prompt_tokens_sha256=prompt_sha256,
                execution_spec_sha256=spec.sha256,
                entries=tuple(entries),
                sampling_config=sampling.to_dict(),
            )

    provider = Provider()
    prompt, samples, _completions = sample_recurrent_group(
        model,
        Tokenizer(),
        Task(),
        spec=spec,
        size=2,
        max_tokens=2,
        seed=999,
        sampling_config=sampling,
        verified_group_provider=provider,
        campaign_sequence=0,
        token_trace_adapter=Tokenizer(),
    )

    assert prompt == [5, 9, 17]
    assert provider.calls == 1
    assert [sample.episode_id for sample in samples] == [
        "signed-causal-0-61",
        "signed-causal-1-67",
    ]
    assert [sample.seed for sample in samples] == [61, 67]
    assert all(sample.sample_kind == "causal_final_transition" for sample in samples)


def test_live_path_branch_answer_ce_trail_scores_each_recurrent_step():
    model = _prepared(seed=949)
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    trail = live_path_branch_answer_ce_trail(
        model,
        [5, 9, 17],
        [7, 11],
        spec=spec,
        branch_index=1,
    )

    assert len(trail) == 3
    assert all(math.isfinite(value) and value >= 0.0 for value in trail)
    with pytest.raises(ValueError, match="branch_index"):
        live_path_branch_answer_ce_trail(
            model,
            [5, 9, 17],
            [7, 11],
            spec=spec,
            branch_index=2,
        )


def test_sampled_verifier_update_improves_preference_without_base_drift():
    model = _prepared(seed=977)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(depth=2)
    sampling = RecurrentSamplingConfig(max_tokens=2)
    samples = []
    for seed in range(200, 216):
        candidate = sample_recurrent_completion(
            model,
            prompt,
            spec=spec,
            seed=seed,
            sampling=sampling,
        )
        if not samples or sum(candidate.tokens) != sum(samples[0].tokens):
            samples.append(candidate)
        if len(samples) == 2:
            break
    assert len(samples) == 2

    # A deterministic external program ranks lower token sums higher. The
    # production trainer substitutes each task's exact correctness verifier;
    # this synthetic rule keeps the contract test contamination-free.
    rewards = tuple(-float(sum(sample.tokens)) for sample in samples)
    assert rewards[0] != rewards[1]
    branches = tuple(sample.branch_index for sample in samples)
    completions = tuple(sample.tokens for sample in samples)

    def policy_scores(*, adapters_on: bool) -> tuple[float, float]:
        scores = []
        for completion, branch in zip(completions, branches, strict=True):
            values = recurrent_completion_token_logprobs(
                model,
                prompt,
                completion,
                spec=spec,
                branch_index=branch,
                adapters_on=adapters_on,
            )
            mx.eval(values)
            scores.append(float(mx.sum(values)))
        return tuple(scores)

    policy_before = policy_scores(adapters_on=True)
    reference_before = policy_scores(adapters_on=False)
    standard_before = model(mx.array([prompt]))
    mx.eval(standard_before)

    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        rewards,
        spec=spec,
        config=RecurrentGRPOConfig(kl_coefficient=0.0),
    )
    assert result.gradients is not None
    optimizer = optim.SGD(learning_rate=0.02)
    optimizer.update(model, result.gradients)
    mx.eval(model.parameters(), optimizer.state)

    policy_after = policy_scores(adapters_on=True)
    reference_after = policy_scores(adapters_on=False)
    standard_after = model(mx.array([prompt]))
    mx.eval(standard_after)
    winner = rewards.index(max(rewards))
    loser = 1 - winner

    assert policy_after[winner] - policy_after[loser] > policy_before[winner] - policy_before[loser]
    assert reference_after == pytest.approx(reference_before, abs=1e-7)
    assert bool(mx.array_equal(standard_after, standard_before))


def test_verified_trajectory_composite_assigns_credit_and_structural_terms_once(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _prepared(seed=981)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_abs_logprob_drift=2.0,
        max_mean_abs_logprob_drift=2.0,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    samples = [
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                prompt,
                spec=spec,
                branch_index=branch,
                seed=seed,
                sampling=sampling,
                episode_id=f"trajectory-composite-{branch}-{seed}",
            ),
            sampling=sampling,
        )
        for branch, seed in ((0, 71), (1, 73))
    ]
    rewards = (1.0, 0.0)
    optimization_token_counts = (
        len(samples[0].tokens) - 1,
        len(samples[1].tokens) - 1,
    )
    base = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        rewards,
        spec=spec,
        config=RecurrentGRPOConfig(
            kl_coefficient=0.0,
            max_initial_clip_fraction=1.0,
            max_initial_old_policy_approx_kl=1.0,
        ),
        optimization_token_counts=optimization_token_counts,
    )
    trajectory_config = VerifiedTrajectoryGroupConfig(
        trajectory_config=ExactAdjointTrajectoryConfig(
            probe_steps=(1, 2),
            improvement_weight=0.7,
            improvement_margin=2.0,
            displacement_weight=0.4,
            displacement_floor=0.99,
            oscillation_weight=0.3,
        ),
        diversity_weight=0.2,
        diversity_target_cos=0.0,
    )
    admission_sha256 = "a" * 64
    reward_sha256 = "b" * 64
    sample_receipts = [
        hashlib.sha256(
            json.dumps(
                sample.receipt(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        for sample in samples
    ]
    monkeypatch.setattr(
        "core.learning.verified_transition_reward.rewards_for_recurrent_samples",
        lambda _receipt, _samples, _prompt: rewards,
    )
    group_admission = {
        "receipt_sha256": admission_sha256,
        "reward_receipt_sha256": reward_sha256,
        "policy_sha256": samples[0].policy_sha256,
        "recurrent_execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": samples[0].prompt_tokens_sha256,
        "group_size": 2,
        "sample_bindings": [
            {
                "schema": "aura.verified_transition.sample_binding.v1",
                "sample_sha256": digest,
            }
            for digest in sample_receipts
        ],
    }
    source_binding = recurrent_grpo_runtime.build_verified_trajectory_group_source_binding(
        group_admission,
        {"receipt_sha256": reward_sha256},
        samples,
        prompt,
        spec=spec,
        trajectory_group_config=trajectory_config,
        advantage_clip=4.0,
        optimization_token_counts=optimization_token_counts,
    )
    result = recurrent_grpo_runtime._with_verified_trajectory_group_objective(
        model,
        prompt,
        samples,
        rewards,
        base,
        group_admission_receipt=group_admission,
        reward_receipt={"receipt_sha256": reward_sha256},
        spec=spec,
        bridge_tokens=(),
        trajectory_group_config=trajectory_config,
        advantage_clip=4.0,
        optimization_token_counts=optimization_token_counts,
    )
    objective_receipt = result.receipt()
    trajectory_receipt = objective_receipt["trajectory_receipt"]
    validated = validate_verified_trajectory_group_receipt(
        trajectory_receipt,
        advantage_report=objective_receipt["advantage_report"],
        expected_source_binding=source_binding,
    )

    assert objective_receipt["mode"] == ("exact_adjoint_trajectory_composite_single_update")
    assert (
        source_binding["schema"]
        == recurrent_grpo_runtime.VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    )
    assert (
        trajectory_receipt["schema"]
        == recurrent_grpo_runtime.VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3
    )
    assert trajectory_receipt["optimization_token_counts"] == list(
        optimization_token_counts
    )
    assert objective_receipt["trajectory_objective_value"] > 0.0
    assert objective_receipt["composite_objective_at_sampling"] == pytest.approx(
        objective_receipt["objective_at_sampling"] + objective_receipt["trajectory_objective_value"]
    )
    assert validated["group_admission_sha256"] == admission_sha256
    assert validated["positive_completion_indices"] == [0]
    assert validated["positive_advantage_weights"] == [1.0]
    assert len(validated["improvement_receipts"]) == 1
    assert validated["improvement_receipts"][0]["completion_index"] == 0
    assert validated["improvement_receipts"][0]["objective_receipt"]["branch_indices"] == [0]
    assert validated["structural_receipt"]["objective_receipt"]["branch_indices"] == [0, 1]
    assert validated["structural_receipt"]["anchor_completion_index"] == 0
    base_flat = dict(tree_flatten(base.gradients))
    composite_flat = dict(tree_flatten(result.gradients))
    assert any(
        float(mx.linalg.norm(mx.reshape(composite_flat[name] - base_flat[name], (-1,)))) > 0.0
        for name in base_flat
    )

    def reseal(attacked):
        unsigned = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
        attacked["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        return attacked

    attacked = copy.deepcopy(trajectory_receipt)
    attacked["positive_advantage_weights"][0] = 0.5
    with pytest.raises(ValueError, match="advantage weights"):
        validate_verified_trajectory_group_receipt(
            reseal(attacked),
            advantage_report=objective_receipt["advantage_report"],
        )

    attacked = copy.deepcopy(trajectory_receipt)
    attacked["advantage_clip"] = 0.5
    with pytest.raises(ValueError, match="advantages do not replay"):
        validate_verified_trajectory_group_receipt(reseal(attacked))

    for field, replacement in (
        ("reward_receipt_sha256", "c" * 64),
        ("policy_sha256", "d" * 64),
        ("execution_spec_sha256", "e" * 64),
        ("prompt_tokens_sha256", "f" * 64),
        ("sample_receipt_sha256s", ["0" * 64, "1" * 64]),
        ("completion_tokens_sha256s", ["2" * 64, "3" * 64]),
        ("verified_rewards", [0.8, 0.0]),
    ):
        attacked = copy.deepcopy(trajectory_receipt)
        attacked[field] = replacement
        with pytest.raises(ValueError, match=f"{field} differs from admitted source"):
            validate_verified_trajectory_group_receipt(
                reseal(attacked),
                expected_source_binding=source_binding,
            )

    for branches in ([1, 0], [0, 0], [0]):
        attacked = copy.deepcopy(trajectory_receipt)
        attacked["sample_branch_indices"] = branches
        with pytest.raises(ValueError, match="sample bindings"):
            validate_verified_trajectory_group_receipt(reseal(attacked))

    def reseal_exact(attacked):
        input_payload = {
            key: attacked[key]
            for key in (
                "policy_sha256",
                "prompt_tokens_sha256",
                "prompt_token_count",
                "answer_tokens_sha256",
                "answer_token_count",
                "bridge_tokens_sha256",
                "bridge_token_count",
                "token_loss_weights",
                "execution_spec_sha256",
                "recurrent_depth",
                "execution_branch_count",
                "branch_indices",
                "diversity_weight",
                "diversity_target_cos",
                "trajectory_config",
            )
        }
        encoded = json.dumps(
            input_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        attacked["objective_input_sha256"] = hashlib.sha256(
            b"aura.exact_adjoint_input.v1\0" + encoded
        ).hexdigest()
        return reseal(attacked)

    child_attacks = (
        ("policy_sha256", "9" * 64),
        ("prompt_tokens_sha256", "8" * 64),
        ("answer_tokens_sha256", "7" * 64),
        (
            "token_loss_weights",
            [0.5]
            * trajectory_receipt["improvement_receipts"][0]["objective_receipt"][
                "answer_token_count"
            ],
        ),
    )
    for field, replacement in child_attacks:
        attacked = copy.deepcopy(trajectory_receipt)
        child = attacked["improvement_receipts"][0]["objective_receipt"]
        child[field] = replacement
        attacked["improvement_receipts"][0]["objective_receipt"] = reseal_exact(child)
        with pytest.raises(ValueError, match="improvement objective differs"):
            validate_verified_trajectory_group_receipt(
                reseal(attacked),
                expected_source_binding=source_binding,
            )

    attacked = copy.deepcopy(trajectory_receipt)
    child = attacked["improvement_receipts"][0]["objective_receipt"]
    child["bridge_tokens_sha256"] = hashlib.sha256(b"[99]").hexdigest()
    child["bridge_token_count"] = 1
    attacked["improvement_receipts"][0]["objective_receipt"] = reseal_exact(child)
    with pytest.raises(ValueError, match="improvement objective differs"):
        validate_verified_trajectory_group_receipt(
            reseal(attacked),
            expected_source_binding=source_binding,
        )

    attacked = copy.deepcopy(trajectory_receipt)
    structural = attacked["structural_receipt"]["objective_receipt"]
    structural["answer_tokens_sha256"] = "6" * 64
    attacked["structural_receipt"]["objective_receipt"] = reseal_exact(structural)
    with pytest.raises(ValueError, match="structural objective differs"):
        validate_verified_trajectory_group_receipt(
            reseal(attacked),
            expected_source_binding=source_binding,
        )

    with pytest.raises(ValueError, match="requires empty bridge_tokens"):
        recurrent_grpo_runtime._with_verified_trajectory_group_objective(
            model,
            prompt,
            samples,
            rewards,
            base,
            group_admission_receipt=group_admission,
            reward_receipt={"receipt_sha256": reward_sha256},
            spec=spec,
            bridge_tokens=(99,),
            trajectory_group_config=trajectory_config,
            advantage_clip=4.0,
        )


def test_verified_intervention_composite_is_quality_weighted_and_replayable(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _prepared(seed=982)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_abs_logprob_drift=2.0,
        max_mean_abs_logprob_drift=2.0,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    samples = [
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                prompt,
                spec=spec,
                branch_index=branch,
                seed=seed,
                sampling=sampling,
                episode_id=f"intervention-composite-{branch}-{seed}",
            ),
            sampling=sampling,
        )
        for branch, seed in ((0, 79), (1, 83))
    ]
    rewards = (1.0, 0.0)
    optimization_token_counts = (
        len(samples[0].tokens) - 1,
        len(samples[1].tokens) - 1,
    )
    base = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        rewards,
        spec=spec,
        config=RecurrentGRPOConfig(
            kl_coefficient=0.0,
            max_initial_clip_fraction=1.0,
            max_initial_old_policy_approx_kl=1.0,
        ),
    )
    group_config = VerifiedTrajectoryGroupConfig(
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1, 2),
            causality_weight=0.7,
            causality_margin=2.0,
            stopping_steps=(1, 2),
            stopping_weight=0.5,
            stopping_ponder_cost=0.02,
            stopping_temperature=0.2,
        ),
        diversity_weight=0.2,
    )
    admission_sha256 = "3" * 64
    reward_sha256 = "4" * 64
    sample_receipts = [
        hashlib.sha256(
            json.dumps(
                sample.receipt(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        for sample in samples
    ]
    monkeypatch.setattr(
        "core.learning.verified_transition_reward.rewards_for_recurrent_samples",
        lambda _receipt, _samples, _prompt: rewards,
    )
    group_admission = {
        "receipt_sha256": admission_sha256,
        "reward_receipt_sha256": reward_sha256,
        "policy_sha256": samples[0].policy_sha256,
        "recurrent_execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": samples[0].prompt_tokens_sha256,
        "group_size": 2,
        "sample_bindings": [
            {
                "schema": "aura.verified_transition.sample_binding.v1",
                "sample_sha256": digest,
            }
            for digest in sample_receipts
        ],
    }
    source = recurrent_grpo_runtime.build_verified_trajectory_group_source_binding(
        group_admission,
        {"receipt_sha256": reward_sha256},
        samples,
        prompt,
        spec=spec,
        trajectory_group_config=group_config,
        advantage_clip=4.0,
        optimization_token_counts=optimization_token_counts,
    )
    result = recurrent_grpo_runtime._with_verified_trajectory_group_objective(
        model,
        prompt,
        samples,
        rewards,
        base,
        group_admission_receipt=group_admission,
        reward_receipt={"receipt_sha256": reward_sha256},
        spec=spec,
        bridge_tokens=(),
        trajectory_group_config=group_config,
        advantage_clip=4.0,
        optimization_token_counts=optimization_token_counts,
    )
    receipt = result.receipt()["trajectory_receipt"]
    validated = validate_verified_trajectory_group_receipt(
        receipt,
        advantage_report=result.receipt()["advantage_report"],
        expected_source_binding=source,
    )

    assert source["schema"] == recurrent_grpo_runtime.VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3
    assert receipt["schema"] == recurrent_grpo_runtime.VERIFIED_TRAJECTORY_GROUP_SCHEMA_V3
    assert source["optimization_token_counts"] == list(optimization_token_counts)
    assert receipt["optimization_token_counts"] == list(optimization_token_counts)
    assert source["completion_tokens_sha256s"] == [
        recurrent_grpo_runtime._tokens_sha256(sample.tokens[:count])
        for sample, count in zip(
            samples,
            optimization_token_counts,
            strict=True,
        )
    ]
    assert validated["positive_completion_indices"] == [0]
    assert validated["positive_advantage_weights"] == [1.0]
    assert len(validated["quality_receipts"]) == 1
    child = validated["quality_receipts"][0]["objective_receipt"]
    assert child["branch_indices"] == [0]
    assert child["trajectory_values"]["causality"] > 0.0
    assert child["trajectory_values"]["stopping"] > 0.0
    assert child["lesion_losses"].keys() == {"1", "2"}
    assert len(child["stopping_teacher_receipts"]) == 1
    assert validated["structural_receipt"] is not None

    def reseal(document):
        payload = {key: value for key, value in document.items() if key != "receipt_sha256"}
        document["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        return document

    attacked = copy.deepcopy(receipt)
    attacked["optimization_token_counts"][0] += 1
    with pytest.raises(ValueError, match="optimization_token_counts differs"):
        validate_verified_trajectory_group_receipt(
            reseal(attacked),
            expected_source_binding=source,
        )

    attacked = copy.deepcopy(receipt)
    attacked_child = attacked["quality_receipts"][0]["objective_receipt"]
    attacked_child["lesion_losses"]["1"][0] += 1.0
    attacked["quality_receipts"][0]["objective_receipt"] = reseal(attacked_child)
    with pytest.raises(ValueError, match="causality term does not replay"):
        validate_verified_trajectory_group_receipt(reseal(attacked))

    attacked = copy.deepcopy(receipt)
    attacked_child = attacked["quality_receipts"][0]["objective_receipt"]
    teacher = attacked_child["stopping_teacher_receipts"][0]
    teacher["probabilities"] = list(reversed(teacher["probabilities"]))
    attacked_child["stopping_teacher_receipts"][0] = reseal(teacher)
    attacked["quality_receipts"][0]["objective_receipt"] = reseal(attacked_child)
    with pytest.raises(ValueError, match="stopping teacher arithmetic"):
        validate_verified_trajectory_group_receipt(reseal(attacked))

    injected = exact_adjoint_composite_live_path_value_and_grad(
        model,
        prompt,
        samples[0].tokens,
        spec=spec,
        trajectory_config=None,
        intervention_config=group_config.intervention_config,
        policy_sha256=samples[0].policy_sha256,
        branch_index=None,
        diversity_weight=group_config.diversity_weight,
        diversity_target_cos=group_config.diversity_target_cos,
        token_loss_weights=(0.0,) * len(samples[0].tokens),
    )
    attacked = copy.deepcopy(receipt)
    previous_structural = attacked["structural_receipt"]["objective_receipt"]
    attacked["structural_receipt"]["objective_receipt"] = injected.receipt()
    attacked["trajectory_objective_value"] += injected.value - previous_structural["value"]
    with pytest.raises(ValueError, match="structural objective differs"):
        validate_verified_trajectory_group_receipt(
            reseal(attacked),
            expected_source_binding=source,
        )


def test_verified_intervention_group_config_rejects_false_measurement_boundary():
    config = VerifiedTrajectoryGroupConfig(
        trajectory_config=ExactAdjointTrajectoryConfig(
            probe_steps=(1, 2),
            improvement_weight=0.2,
        ),
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.2,
            stopping_steps=(1, 2),
            stopping_weight=0.1,
        ),
    )
    canonical = config.to_dict()
    assert (
        canonical["measurement_trust_boundary"]
        == "producer_sealed_arithmetic_external_state_replay_required"
    )

    missing = copy.deepcopy(canonical)
    del missing["measurement_trust_boundary"]
    with pytest.raises(ValueError, match="fields do not match"):
        VerifiedTrajectoryGroupConfig.from_dict(missing)

    relabeled = copy.deepcopy(canonical)
    relabeled["measurement_trust_boundary"] = "independently_replayed"
    with pytest.raises(ValueError, match="policy is unsupported"):
        VerifiedTrajectoryGroupConfig.from_dict(relabeled)

    missing_intervention = copy.deepcopy(canonical)
    missing_intervention["intervention_config"] = None
    with pytest.raises(ValueError, match="policy is unsupported"):
        VerifiedTrajectoryGroupConfig.from_dict(missing_intervention)


def test_sampled_group_rejects_excess_clip_fraction_before_adjoint():
    model = _prepared(seed=983)
    prompt = [5, 9, 17]
    spec = _spec(depth=2)
    samples = [
        sample_recurrent_completion(
            model,
            prompt,
            spec=spec,
            seed=seed,
            sampling=RecurrentSamplingConfig(max_tokens=2),
        )
        for seed in (31, 37)
    ]
    shifted = []
    for sample in samples:
        behavior = list(sample.differentiable_logprobs)
        behavior[0] -= 0.19
        differences = [
            abs(left - right)
            for left, right in zip(behavior, sample.differentiable_logprobs, strict=True)
        ]
        changed = replace(
            sample,
            behavior_logprobs=tuple(behavior),
            max_abs_logprob_drift=max(differences),
            mean_abs_logprob_drift=sum(differences) / len(differences),
            max_abs_logprob_drift_token_index=differences.index(max(differences)),
            clipped_token_fraction=0.5,
            old_policy_approx_kl=sum(
                (math.exp(target - cached) - 1.0) - (target - cached)
                for target, cached in zip(
                    sample.differentiable_logprobs,
                    behavior,
                    strict=True,
                )
            )
            / len(behavior),
            behavior_admitted=True,
            sampling_config=replace(
                sample.sampling_config,
                max_mean_abs_logprob_drift=0.1,
                max_clipped_token_fraction=1.0,
            ),
        )
        shifted.append(
            replace(
                changed,
                rng_root_sha256=recurrent_sampling_rng_root_sha256(
                    episode_id=changed.episode_id,
                    prompt_tokens_sha256=changed.prompt_tokens_sha256,
                    policy_sha256=changed.policy_sha256,
                    execution_spec_sha256=changed.execution_spec_sha256,
                    branch_index=changed.branch_index,
                    seed=changed.seed,
                    sampling_config=changed.sampling_config,
                ),
            )
        )

    with pytest.raises(RecurrentGroupClipAdmissionError) as captured:
        exact_adjoint_sampled_group_value_and_grad(
            model,
            prompt,
            shifted,
            [1.0, 0.0],
            spec=spec,
            config=RecurrentGRPOConfig(max_initial_clip_fraction=0.2),
        )

    assert captured.value.clip_fraction == pytest.approx(0.5)

    with pytest.raises(RuntimeError, match="PPO KL admission"):
        exact_adjoint_sampled_group_value_and_grad(
            model,
            prompt,
            shifted,
            [1.0, 0.0],
            spec=spec,
            config=RecurrentGRPOConfig(
                max_initial_clip_fraction=1.0,
                max_initial_old_policy_approx_kl=1e-6,
            ),
        )


def test_verifier_advantage_produces_finite_nonzero_recurrent_gradients():
    model = _prepared(seed=617)
    prompt = [5, 9, 17]
    completions = ([7, 11], [8, 12])
    spec = _spec(depth=2)
    old = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(model, prompt, tokens, spec=spec, branch_index=0)
        )
        for tokens in completions
    ]
    reference = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                model,
                prompt,
                tokens,
                spec=spec,
                branch_index=0,
                adapters_on=False,
            )
        )
        for tokens in completions
    ]
    mx.eval(old, reference)

    def loss_fn(current_model):
        policy = [
            recurrent_completion_token_logprobs(
                current_model, prompt, tokens, spec=spec, branch_index=0
            )
            for tokens in completions
        ]
        objective, report = verifier_group_objective(
            policy,
            old,
            [1.0, 0.0],
            reference_logprobs=reference,
            config=RecurrentGRPOConfig(kl_coefficient=0.04),
        )
        assert report["degenerate"] is False
        return objective.loss

    value, gradients = nn.value_and_grad(model, loss_fn)(model)
    flattened = [gradient for _name, gradient in tree_flatten(gradients)]
    mx.eval(value, gradients)

    assert math.isfinite(float(value))
    assert flattened
    assert all(bool(mx.all(mx.isfinite(gradient))) for gradient in flattened)
    assert any(float(mx.max(mx.abs(gradient))) > 0.0 for gradient in flattened)


def test_exact_adjoint_group_matches_monolithic_recurrent_grpo_gradient():
    monolithic = _prepared(seed=811)
    streamed = _prepared(seed=811)
    _set_adapter_delta(monolithic, 0.02)
    _set_adapter_delta(streamed, 0.02)
    prompt = [5, 9, 17]
    completions = ([7, 11], [8, 12])
    branches = (1, 0)
    rewards = (1.0, 0.0)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    config = RecurrentGRPOConfig(kl_coefficient=0.04)

    old = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                monolithic, prompt, tokens, spec=spec, branch_index=branch
            )
        )
        for tokens, branch in zip(completions, branches, strict=True)
    ]
    behavior = [
        old[0] + mx.array([-0.25, 0.05]),
        old[1] + mx.array([0.0, 0.10]),
    ]
    reference = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                monolithic,
                prompt,
                tokens,
                spec=spec,
                branch_index=branch,
                adapters_on=False,
            )
        )
        for tokens, branch in zip(completions, branches, strict=True)
    ]
    mx.eval(old, behavior, reference)

    def loss_fn(current_model):
        policy = [
            recurrent_completion_token_logprobs(
                current_model, prompt, tokens, spec=spec, branch_index=branch
            )
            for tokens, branch in zip(completions, branches, strict=True)
        ]
        objective, _report = verifier_group_objective(
            policy,
            behavior,
            rewards,
            reference_logprobs=reference,
            config=config,
        )
        return objective.loss

    _value, monolithic_gradients = nn.value_and_grad(monolithic, loss_fn)(monolithic)
    streamed_result = exact_adjoint_verifier_group_value_and_grad(
        streamed,
        prompt,
        completions,
        branches,
        rewards,
        spec=spec,
        behavior_logprobs=[[float(value) for value in values] for values in behavior],
        config=config,
    )
    mx.eval(monolithic_gradients, streamed_result.gradients)
    monolithic_flat = dict(tree_flatten(monolithic_gradients))
    streamed_flat = dict(tree_flatten(streamed_result.gradients))

    assert set(monolithic_flat) == set(streamed_flat)
    assert streamed_result.receipt()["has_gradient"] is True
    assert streamed_result.receipt()["clip_fraction"] == pytest.approx(0.25)
    for key in monolithic_flat:
        difference = float(mx.max(mx.abs(monolithic_flat[key] - streamed_flat[key])))
        assert difference < 2e-4, key


def test_exact_adjoint_masks_post_terminal_tokens_from_policy_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _prepared(seed=983)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_abs_logprob_drift=2.0,
        max_mean_abs_logprob_drift=2.0,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    samples = [
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                prompt,
                spec=spec,
                branch_index=branch,
                seed=seed,
                sampling=sampling,
                episode_id=f"terminal-mask-{branch}-{seed}",
            ),
            sampling=sampling,
        )
        for branch, seed in ((0, 89), (1, 97))
    ]
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_objective(
        _model,
        _prompt,
        completion_tokens,
        branch_indices,
        rewards,
        **kwargs,
    ):
        captured["completion_tokens"] = completion_tokens
        captured["branch_indices"] = branch_indices
        captured["rewards"] = rewards
        captured["behavior_logprobs"] = kwargs["behavior_logprobs"]
        return sentinel

    monkeypatch.setattr(
        recurrent_grpo_runtime,
        "exact_adjoint_verifier_group_value_and_grad",
        fake_objective,
    )
    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        (1.0, 0.0),
        spec=spec,
        config=RecurrentGRPOConfig(
            max_initial_clip_fraction=1.0,
            max_initial_old_policy_approx_kl=1.0,
        ),
        optimization_token_counts=(1, 2),
    )
    assert result is sentinel
    assert captured["completion_tokens"] == [
        samples[0].tokens[:1],
        samples[1].tokens,
    ]
    assert captured["behavior_logprobs"] == [
        samples[0].behavior_logprobs[:1],
        samples[1].behavior_logprobs,
    ]
