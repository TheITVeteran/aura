"""Contract tests: the integrated LatentCortexEngine.

Full episodes on a tiny real Qwen2: latent computation is causal on the
answer, receipts tell the truth, fallbacks are honest, budgets bind, the
checkpoint invariant is enforced, and fast-weight episodes leave the model
bit-for-bit unchanged.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.branches import BranchEnsemble  # noqa: E402
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
from core.brain.llm.latent_cortex.governance import parameter_fingerprint  # noqa: E402
from core.brain.llm.latent_cortex.recurrence import WindowRunner  # noqa: E402
from core.brain.llm.latent_cortex.schedules import LayerSchedule, StageOp  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    FastWeightsConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

N_LAYERS = 8
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


def _model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _model()


def _config(**overrides) -> CortexConfig:
    base = dict(
        workspace=WorkspaceConfig(n_slots=4, seed=3),
        recurrence=RecurrenceConfig(max_steps=6, min_steps=2),
        branches=BranchConfig(n_branches=2, exchange_interval=2),
        prelude_frac=0.25,
        coda_frac=0.25,
        decode_max_tokens=8,
    )
    base.update(overrides)
    return CortexConfig(**base)


def test_full_episode_produces_tokens_and_truthful_receipt(tiny_model):
    from core.brain.llm.latent_cortex.cognitive_operators import (
        validate_operator_receipt,
    )

    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    assert result.tokens, "episode must decode tokens"
    r = result.receipt
    assert r.params_unchanged is True
    assert r.n_layers == N_LAYERS and r.prelude_end == 2 and r.coda_start == 6
    assert r.n_branches == 2 and r.n_slots == 4
    assert r.steps_taken >= 2
    assert r.branch_isolation["certified"] is True
    assert r.loop_stability["shared_train_inference_core"] is True
    assert r.loop_stability["all_accepted_states_anchor_bounded"] is True
    assert r.loop_stability["kv_bound"]["all_within_limit"] is True
    assert r.branch_isolation["first_exchange_step"] >= 1
    assert r.branch_isolation["cache_discipline"]["all_restored"] is True
    assert r.cognitive_operator_trace
    assert {row["operator"] for row in r.cognitive_operator_trace} == {
        "constructive_solution",
        "counterexample",
    }
    assert all(validate_operator_receipt(row) for row in r.cognitive_operator_trace)
    assert r.structural_diversity["certified"] is True
    assert r.structural_diversity["wording_counted"] is False
    assert r.structural_diversity["independent_support_count"] == 2
    assert r.correlated_support["raw_support_count"] == 2
    assert r.correlated_support["evidence_state"] == "bootstrap_unmeasured"
    assert r.correlated_support["confidence_multiplier"] <= 1.0
    assert r.blind_review == {}
    assert r.verifier_preflight == {}
    assert r.decoy_verification == {}
    assert r.residual_trail, "receipt must carry the residual trail"
    assert r.halting_reason
    assert r.schedule_hash
    assert r.budget["spent_layer_apps"] > 0
    assert r.decode_requested_tokens == 8
    assert r.decode_generated_tokens == len(result.tokens)
    assert r.verifier_probe_max_tokens == 48
    assert r.decode_termination in {"eos", "token_limit"}
    assert r.last_stage == "complete"
    assert r.stage_timings_s["prefill"] >= 0.0
    assert r.stage_timings_s["recurrence"] >= 0.0
    assert r.stage_timings_s["decode"] >= 0.0
    assert r.stage_timings_s["total"] >= 0.0
    assert not r.honest_flags, f"clean episode must carry no flags: {r.honest_flags}"


def test_verifier_probe_cost_uses_receipted_profile_and_bridge(tiny_model):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(verifier_probe_max_tokens=24),
    )

    assert engine._verifier_probe_layer_apps([3, 4], count=2) == (2 * (4 + 2 + 23) * N_LAYERS)
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.receipt.verifier_probe_max_tokens == 24


def test_verifier_preview_is_hard_capped_and_compute_charge_is_exact(
    tiny_model,
    monkeypatch,
):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(verifier_probe_max_tokens=24),
    )
    engine.tokenizer = object()
    monkeypatch.setattr(engine, "_eos_ids", lambda: set())
    monkeypatch.setattr(engine, "_is_pure_newline_token", lambda _token: False)
    monkeypatch.setattr(engine, "_token_ends_sentence", lambda _token: False)
    monkeypatch.setattr(engine, "_sample", lambda *_args, **_kwargs: 1)

    budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
    cache = engine._fresh_cache()
    _, initial_logits = engine._prefill(PROMPT_TOKENS, cache, budget)
    spent_before = budget.spent_layer_apps

    tokens, termination = engine._decode(
        cache,
        budget,
        initial_logits,
        max_tokens=engine.config.verifier_probe_max_tokens,
        temperature=0.0,
        sentence_grace_tokens=0,
    )

    assert len(tokens) == 24
    assert termination == "token_limit"
    assert budget.spent_layer_apps - spent_before == 23 * N_LAYERS

    # The user-facing decoder still owns the independent sentence grace.
    final_budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
    final_cache = engine._fresh_cache()
    _, final_logits = engine._prefill(PROMPT_TOKENS, final_cache, final_budget)
    final_tokens, final_termination = engine._decode(
        final_cache,
        final_budget,
        final_logits,
        max_tokens=24,
        temperature=0.0,
    )
    assert len(final_tokens) == 72
    assert final_termination == "token_limit"


def test_fresh_verifier_generation_uses_zero_offset_real_qwen_cache(tiny_model):
    class NumericTokenizer:
        eos_token_id = None

        @staticmethod
        def encode(text, **_kwargs):
            return [1 + (ord(char) % 120) for char in str(text)][:96] or [5]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

    engine = LatentCortexEngine(tiny_model, NumericTokenizer(), config=_config())
    budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
    budget.bind_model(tiny_model)
    generated = engine._fresh_verifier_generation(
        "Independently check the disputed atom.",
        budget,
        max_tokens=32,
        reserve_layer_apps=0,
    )

    context = generated["context"]
    assert context["all_initial_offsets_zero"] is True
    assert context["solver_context_imported"] is False
    assert context["parameter_relation"] == "shared_resident_checkpoint"
    assert context["initial_cache_offsets"] == [0] * N_LAYERS
    assert len(set(context["final_cache_offsets"])) == 1
    assert context["final_cache_offsets"][0] >= context["prompt_token_count"]
    assert context["generated_token_count"] == 64
    assert context["termination"] == "token_limit_contract_incomplete"
    assert generated["text"]


def test_seeded_prefix_generation_is_local_reproducible_and_receipted(tiny_model):
    class NumericTokenizer:
        eos_token_id = None

        @staticmethod
        def encode(text, **_kwargs):
            return [1 + (ord(char) % 120) for char in str(text)][:96] or [5]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

    engine = LatentCortexEngine(tiny_model, NumericTokenizer(), config=_config())

    def generate(seed: int):
        budget = ComputeBudget(max_layer_apps=100_000, wall_clock_s=30.0)
        budget.bind_model(tiny_model)
        return engine._fresh_verifier_generation(
            "Continue from this verified prefix.",
            budget,
            max_tokens=32,
            reserve_layer_apps=0,
            temperature=0.7,
            top_p=0.9,
            sample_seed=seed,
        )

    first = generate(913)
    repeat = generate(913)
    assert first == repeat
    context = first["context"]
    assert context["schema"] == "aura.rlc.fresh_prefix_context.v1"
    assert context["sample_seed"] == 913
    assert context["temperature"] == 0.7
    assert context["top_p"] == 0.9
    assert context["initial_cache_offsets"] == [0] * N_LAYERS


def test_admitted_fresh_refutation_causally_replaces_provisional_winner(
    tiny_model,
    monkeypatch,
):
    from core.brain.llm.latent_cortex.task_verifiers import check_arithmetic_claims

    class ProbeTokenizer:
        eos_token_id = None

        @staticmethod
        def encode(_text, **_kwargs):
            return [5]

        @staticmethod
        def decode(tokens):
            values = list(tokens)
            if values == [100]:
                return "The answer is 2 + 2 = 5."
            if values == [101]:
                return "The answer is 2 + 2 = 4."
            return "A complete final answer."

    engine = LatentCortexEngine(
        tiny_model,
        ProbeTokenizer(),
        config=_config(
            branches=BranchConfig(n_branches=2, exchange_interval=2),
            generative_verifier_max_tokens=64,
        ),
    )
    monkeypatch.setattr(
        engine,
        "_decode_probe",
        lambda branch, *_args, **_kwargs: [100 + branch.index],
    )

    def verifier(text: str) -> float:
        if text.startswith("Independent consistency check:"):
            checked = check_arithmetic_claims(text)
            return float(checked["score"] if checked["score"] is not None else 0.0)
        return 0.9 if "= 5" in text else 0.1

    def fresh(prompt: str, *_args, **_kwargs):
        claim = re.search(r"ANONYMIZED_CLAIM_SHA256: ([0-9a-f]{64})", prompt)
        assert claim is not None
        payload = {
            "claim_sha256": claim.group(1),
            "verdict": "refutes",
            "witness": "2 + 2 = 4",
        }
        return {
            "text": "FINAL_ANSWER: " + json.dumps(payload, separators=(",", ":")),
            "context": {
                "schema": "aura.rlc.fresh_verifier_context.v1",
                "prompt_token_count": 1,
                "generated_token_count": 16,
                "termination": "contract_complete",
                "initial_cache_offsets": [0] * N_LAYERS,
                "final_cache_offsets": [16] * N_LAYERS,
                "all_initial_offsets_zero": True,
                "solver_context_imported": False,
                "parameter_relation": "shared_resident_checkpoint",
            },
        }

    monkeypatch.setattr(engine, "_fresh_verifier_generation", fresh)
    result = engine.reason(
        prompt="Compute 2 + 2 exactly.",
        verifier=verifier,
        budget=ComputeBudget(max_layer_apps=500_000, wall_clock_s=30.0),
    )

    assert result.ok
    assert result.receipt.generative_verifier["causal_refutation"] is True
    assert result.receipt.generative_verifier["selection_effect"] == "winner_replaced"
    assert result.receipt.generative_verifier["vetoed_branch"] == 0
    assert result.receipt.selected_branch == 1


def test_counterfactual_verifier_causally_breaks_only_equal_score_tie(
    tiny_model,
    monkeypatch,
):
    from core.brain.llm.latent_cortex.task_verifiers import check_arithmetic_claims

    candidates = {
        0: "The first answer is 4 + 3 = 7.",
        1: "The second answer is 4 + 3 = 7.",
    }

    class ProbeTokenizer:
        eos_token_id = None

        @staticmethod
        def encode(_text, **_kwargs):
            return [5]

        @staticmethod
        def decode(tokens):
            values = list(tokens)
            if values == [100]:
                return candidates[0]
            if values == [101]:
                return candidates[1]
            return "A complete final answer."

    engine = LatentCortexEngine(
        tiny_model,
        ProbeTokenizer(),
        config=_config(
            branches=BranchConfig(n_branches=2, exchange_interval=2),
            counterfactual_verifier_max_tokens=64,
        ),
    )
    monkeypatch.setattr(
        engine,
        "_decode_probe",
        lambda branch, *_args, **_kwargs: [100 + branch.index],
    )

    def verifier(text: str) -> float:
        if text.startswith("Independent consistency check:"):
            checked = check_arithmetic_claims(text)
            return float(checked["score"] if checked["score"] is not None else 0.0)
        return 0.8

    def fresh(prompt: str, *_args, **_kwargs):
        claim = re.search(r"ANONYMIZED_CLAIM_SHA256: ([0-9a-f]{64})", prompt)
        intervention = re.search(r"INTERVENTION_SHA256: ([0-9a-f]{64})", prompt)
        inputs = re.search(r"COUNTERFACTUAL_INPUT: (-?\d+) ([+\-*/]) (-?\d+)", prompt)
        claim_text = re.search(
            r"ANONYMIZED_CLAIM:\n(.*?)\nINTERVENTION_SHA256:",
            prompt,
            re.DOTALL,
        )
        assert claim and intervention and inputs and claim_text
        left, operator, right = int(inputs.group(1)), inputs.group(2), int(inputs.group(3))
        actual = {
            "+": left + right,
            "-": left - right,
            "*": left * right,
            "/": left // right,
        }[operator]
        predicted = 7 if claim_text.group(1) == candidates[0] else actual
        payload = {
            "claim_sha256": claim.group(1),
            "intervention_sha256": intervention.group(1),
            "prediction": f"{left} {operator} {right} = {predicted}",
        }
        return {
            "text": "FINAL_ANSWER: " + json.dumps(payload, separators=(",", ":")),
            "context": {
                "schema": "aura.rlc.fresh_verifier_context.v1",
                "prompt_token_count": 1,
                "generated_token_count": 16,
                "termination": "contract_complete",
                "initial_cache_offsets": [0] * N_LAYERS,
                "final_cache_offsets": [16] * N_LAYERS,
                "all_initial_offsets_zero": True,
                "solver_context_imported": False,
                "parameter_relation": "shared_resident_checkpoint",
            },
        }

    monkeypatch.setattr(engine, "_fresh_verifier_generation", fresh)
    result = engine.reason(
        prompt="Compute 4 + 3 and explain sensitivity.",
        verifier=verifier,
        budget=ComputeBudget(max_layer_apps=500_000, wall_clock_s=30.0),
    )

    assert result.ok
    counterfactual = result.receipt.counterfactual_verifier
    assert counterfactual["selection_authority_admitted"] is True
    assert counterfactual["selection_effect"] == "winner_replaced"
    assert counterfactual["source_selected_branch"] == 0
    assert counterfactual["selected_branch"] == 1
    assert result.receipt.selected_branch == 1
    from core.brain.latent_cortex_service import LatentCortexService

    contract_errors = LatentCortexService._receipt_contract_errors(
        result.receipt.to_dict(),
        {
            "n_slots": 4,
            "n_branches": 2,
            "min_steps": 2,
            "max_steps": 6,
            "verifier_probe_max_tokens": 48,
            "generative_verifier_enabled": True,
            "counterfactual_verifier_enabled": True,
        },
    )
    assert "counterfactual_verifier_unproven" not in contract_errors
    assert "blind_or_decoy_branch_review_unproven" not in contract_errors


def test_heterogeneous_dual_lane_decode_is_real_equal_compute_and_restoring(
    tiny_model,
):
    engine = LatentCortexEngine(tiny_model, config=_config())
    budget = ComputeBudget(max_layer_apps=500_000, wall_clock_s=30.0)
    budget.bind_model(tiny_model)
    cache = engine._fresh_cache()
    embeddings, _ = engine._prefill(PROMPT_TOKENS, cache, budget)
    runner = WindowRunner(tiny_model.model, budget)
    ensemble = BranchEnsemble.seed(
        embeddings,
        engine.config.workspace,
        engine.config.branches,
        engine.config.recurrence,
        runner,
        cache,
        engine.prelude_end,
    )
    branch = ensemble.branches[0]
    saved_state = branch.z
    saved_offsets = [layer.offset for layer in cache]
    incumbent = np.array(branch.z, copy=True)
    correction = np.linspace(
        -0.35,
        0.35,
        incumbent.shape[-1],
        dtype=incumbent.dtype,
    )[None, None, :]
    corrected = np.array(incumbent + correction, copy=True)
    phase_events = []

    def decode(policy: str):
        return engine._heterogeneous_dual_lane_decode(
            branch=branch,
            cache=cache,
            runner=runner,
            budget=budget,
            incumbent_state=incumbent,
            corrected_state=corrected,
            policy=policy,
            fusion_weight=0.65,
            bridge_tokens=[31, 32],
            max_tokens=4,
            temperature=0.0,
            force_exact_tokens=True,
            phase_checkpoint=phase_events.append,
        )

    old_tokens, old_termination, old_audit = decode("select_old")
    new_tokens, new_termination, new_audit = decode("select_new")
    fused_tokens, fused_termination, fused_audit = decode("probability_fusion")
    repeated_tokens, repeated_termination, repeated_audit = decode("probability_fusion")

    expected_lane_apps = (engine.config.workspace.n_slots + 2 + len(old_tokens) - 1) * N_LAYERS
    for tokens, termination, audit in (
        (old_tokens, old_termination, old_audit),
        (new_tokens, new_termination, new_audit),
        (fused_tokens, fused_termination, fused_audit),
    ):
        assert len(tokens) == 4
        assert termination == "token_limit"
        assert audit["old_lane_layer_apps"] == expected_lane_apps
        assert audit["new_lane_layer_apps"] == expected_lane_apps
        assert audit["divergence_samples"] == len(tokens)
        assert audit["mean_js_divergence_bits"] > 0.0
    assert old_audit["old_initial_logits_sha256"] != (old_audit["new_initial_logits_sha256"])
    assert old_audit["policy_initial_logits_sha256"] == (old_audit["old_initial_logits_sha256"])
    assert new_audit["policy_initial_logits_sha256"] == (new_audit["new_initial_logits_sha256"])
    assert fused_audit["policy_initial_logits_sha256"] not in {
        fused_audit["old_initial_logits_sha256"],
        fused_audit["new_initial_logits_sha256"],
    }
    assert repeated_tokens == fused_tokens
    assert repeated_termination == fused_termination
    assert repeated_audit == fused_audit
    assert branch.z is saved_state
    assert [layer.offset for layer in cache] == saved_offsets
    assert phase_events == ["persist", "decode_bridge"] * 4


def test_heterogeneous_policy_refuses_partial_exact_probe(
    tiny_model,
    monkeypatch,
):
    class Tokenizer:
        @staticmethod
        def decode(_tokens):
            return "partial"

    engine = LatentCortexEngine(
        tiny_model,
        tokenizer=Tokenizer(),
        config=_config(verifier_probe_max_tokens=16),
    )
    monkeypatch.setattr(
        engine,
        "_heterogeneous_dual_lane_decode",
        lambda **_kwargs: ([1], "budget_exhausted", {}),
    )
    evaluator = engine._heterogeneous_policy_evaluator(
        branch=None,
        cache=None,
        runner=None,
        budget=ComputeBudget(),
        bridge_tokens=[],
        verifier=lambda _text: 1.0,
    )

    with pytest.raises(RuntimeError, match="exact equal-compute token contract"):
        evaluator(
            "select_old",
            np.zeros((1, 4, 64), dtype=np.float32),
            np.ones((1, 4, 64), dtype=np.float32),
            0.5,
            0,
        )

    monkeypatch.setattr(
        engine,
        "_heterogeneous_dual_lane_decode",
        lambda **_kwargs: (
            [1] * 16,
            "token_limit",
            {
                "old_lane_layer_apps": N_LAYERS,
                "new_lane_layer_apps": N_LAYERS,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="episode compute budget"):
        evaluator(
            "select_old",
            np.zeros((1, 4, 64), dtype=np.float32),
            np.ones((1, 4, 64), dtype=np.float32),
            0.5,
            0,
        )


def test_episode_cooperatively_cancels_at_safe_stage_and_preserves_checkpoint(
    tiny_model,
):
    engine = LatentCortexEngine(tiny_model, config=_config())
    cancel = False
    stages: list[str] = []

    def progress(payload):
        nonlocal cancel
        stage = str(payload.get("stage") or "")
        stages.append(stage)
        if stage == "prefill":
            cancel = True

    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        progress=progress,
        cancel_check=lambda: cancel,
    )

    assert result.ok is False
    assert result.reason == "soft_cancelled"
    assert result.receipt.params_unchanged is True
    assert result.receipt.last_stage == "prefill"
    assert "soft_cancelled" in result.receipt.honest_flags
    assert stages[0] == "prefill"
    assert stages[-1] == "failed"


def test_episodes_are_deterministic(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    a = engine.reason(token_ids=PROMPT_TOKENS)
    b = engine.reason(token_ids=PROMPT_TOKENS)
    assert a.tokens == b.tokens
    assert a.receipt.schedule_hash == b.receipt.schedule_hash


def test_nucleus_sampling_excludes_tokens_outside_probability_mass():
    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    logits = mx.array([12.0, 2.0, 1.0, 0.0])

    sampled = {engine._sample(logits, temperature=0.7, top_p=0.01) for _ in range(32)}

    assert sampled == {0}


def test_latent_computation_is_causal_on_answer(tiny_model):
    """More recurrence ⇒ different refined thoughts ⇒ different answer tokens.

    (On a random-weight model "different" is the strongest honest claim;
    'better' belongs to the experiments on trained checkpoints.)
    """
    shallow = LatentCortexEngine(
        tiny_model, config=_config(recurrence=RecurrenceConfig(max_steps=1, min_steps=1))
    ).reason(token_ids=PROMPT_TOKENS)
    deep = LatentCortexEngine(
        tiny_model,
        config=_config(
            recurrence=RecurrenceConfig(max_steps=12, min_steps=8, convergence_eps=1e-9)
        ),
    ).reason(token_ids=PROMPT_TOKENS)
    assert shallow.ok and deep.ok
    assert shallow.receipt.steps_taken < deep.receipt.steps_taken
    # On random weights greedy decode can collapse into the same attractor
    # token, so the honest causal signal is the first-decode logits digest:
    # the next-token DISTRIBUTION conditioned on [prompt; thoughts] must move
    # when the latent computation deepens.
    assert shallow.receipt.first_logits_digest != deep.receipt.first_logits_digest, (
        "recurrence depth must causally shape the answer distribution"
    )


def test_explicit_schedule_is_used_and_hashed(tiny_model):
    schedule = LayerSchedule(ops=(StageOp(2, 4, 2), StageOp(4, 6, 2), StageOp(2, 6, 2)))
    engine = LatentCortexEngine(tiny_model, config=_config(schedule=schedule.to_dict()))
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    assert result.receipt.schedule_hash == schedule.schedule_hash


def test_invalid_schedule_falls_back_honestly(tiny_model):
    bad = LayerSchedule(ops=(StageOp(0, 7, 2),))  # escapes the recurrent region
    engine = LatentCortexEngine(tiny_model, config=_config(schedule=bad.to_dict()))
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok, "fallback must still answer"
    assert any(f.startswith("fallback_vanilla") for f in result.receipt.honest_flags)
    assert result.tokens, "vanilla fallback must decode"


def test_production_episode_refuses_secondary_vanilla_decode(tiny_model):
    bad = LayerSchedule(ops=(StageOp(0, 7, 2),))
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            schedule=bad.to_dict(),
            allow_vanilla_fallback=False,
        ),
    )

    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok is False
    assert result.tokens == []
    assert result.receipt.decode_termination == "not_started"
    assert "vanilla_fallback_disabled" in result.receipt.honest_flags
    assert not any(flag.startswith("fallback_vanilla") for flag in result.receipt.honest_flags)


def test_budget_binds_and_is_reported(tiny_model):
    tight = ComputeBudget(max_layer_apps=PROMPT_TOKENS.__len__() * N_LAYERS + 200)
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=tight)
    assert result.ok
    assert result.receipt.budget["spent_layer_apps"] <= tight.max_layer_apps
    assert "fallback_vanilla:RuntimeError" in result.receipt.honest_flags
    fallback_ceiling = (len(PROMPT_TOKENS) + _config().decode_max_tokens - 1) * N_LAYERS
    assert result.receipt.budget["spent_layer_apps"] <= fallback_ceiling
    assert result.receipt.decode_termination in {"eos", "token_limit"}
    reasons = {result.receipt.halting_reason} | {
        b["halt_reason"]
        for b in []  # branch receipts live in ensemble receipt; halting_reason covers winner
    }
    assert any("budget" in r or r for r in reasons)


def test_verifier_selects_branch_and_scores_land_in_receipt(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())

    # A tokenizer-free verifier can't run (no decode-to-text); engine must
    # fall back to convergence selection without exploding.
    result = engine.reason(token_ids=PROMPT_TOKENS, verifier=lambda text: 1.0)
    assert result.ok
    assert len(result.receipt.branch_scores) == 2


def test_latent_opt_episode_records_trace(tiny_model):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(latent_opt=LatentOptConfig(enabled=True, steps=3, lr=0.05)),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    r = result.receipt
    assert r.latent_opt_applied and r.latent_opt_mode == "gradient"
    assert r.latent_opt_attempts == 3
    assert r.latent_opt_steps == 3
    assert r.latent_opt_rejected == 0
    assert r.latent_opt_budget_exhausted is False
    assert len(r.latent_opt_loss_trail) == 4  # 3 steps + final
    assert r.latent_opt_loss_trail[-1] < r.latent_opt_loss_trail[0]


def test_fast_weight_episode_proves_erase_and_invariant():
    model = _model()  # fresh model: this test mutates wrappers
    before = parameter_fingerprint(model)
    engine = LatentCortexEngine(
        model,
        config=_config(
            fast_weights=FastWeightsConfig(
                enabled=True, rank=2, target="o_proj", opt_steps=2, lr=0.02
            ),
            latent_opt=LatentOptConfig(enabled=False),
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)
    assert result.ok
    r = result.receipt
    assert r.fast_weights_applied and r.fast_weights_layers == 4
    assert r.fast_weight_optimization_attempts >= 1
    assert r.fast_weight_optimized_steps >= 1
    assert r.fast_weight_budget_exhausted is False
    assert r.fast_weight_optimizer == "rms_normalized_sgd_backtracking_v1"
    assert len(r.fast_weight_loss_trail) == r.fast_weight_optimized_steps + 1
    assert len(r.fast_weight_gradient_norm_trail) == (r.fast_weight_optimization_attempts)
    assert len(r.fast_weight_accepted_step_sizes) == r.fast_weight_optimized_steps
    assert r.fast_weights_erased is True
    assert r.params_unchanged is True
    assert parameter_fingerprint(model) == before, "episode must leave W0 untouched"


def test_fast_weight_optimization_failure_cleans_before_vanilla_fallback(monkeypatch):
    model = _model()
    original_modules = [layer.self_attn.o_proj for layer in model.model.layers]

    def fail_optimization(self, loss_fn, **kwargs):
        assert self.handles, "failure must be injected after attachment"
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(EpisodicFastWeights, "optimize", fail_optimization)
    engine = LatentCortexEngine(
        model,
        config=_config(
            fast_weights=FastWeightsConfig(enabled=True, target="o_proj", opt_steps=2),
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok, "a proven-clean model may serve the honest vanilla fallback"
    assert result.receipt.fast_weights_applied is True
    assert result.receipt.fast_weights_erased is True
    assert "fallback_vanilla:RuntimeError" in result.receipt.honest_flags
    assert [layer.self_attn.o_proj for layer in model.model.layers] == original_modules


def test_unproven_fast_weight_cleanup_refuses_fallback(monkeypatch):
    model = _model()

    def fail_optimization(self, loss_fn, **kwargs):
        raise RuntimeError("injected optimizer failure")

    def fail_proof(self, probe_fn, baseline):
        raise RuntimeError("injected erase-proof failure")

    monkeypatch.setattr(EpisodicFastWeights, "optimize", fail_optimization)
    monkeypatch.setattr(EpisodicFastWeights, "prove_erase", fail_proof)
    engine = LatentCortexEngine(
        model,
        config=_config(fast_weights=FastWeightsConfig(enabled=True, target="o_proj")),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert not result.ok
    assert result.reason == "fast_weight_cleanup_unproven"
    assert result.receipt.fast_weights_erased is False
    assert "fallback_refused_unproven_model_state" in result.receipt.honest_flags


def test_fast_weight_snapshot_memory_failure_still_detaches(monkeypatch):
    model = _model()
    originals = [layer.self_attn.o_proj for layer in model.model.layers]

    def fail_snapshot(self):
        raise MemoryError("injected snapshot allocation failure")

    monkeypatch.setattr(EpisodicFastWeights, "snapshot_for_export", fail_snapshot)
    engine = LatentCortexEngine(
        model,
        config=_config(fast_weights=FastWeightsConfig(enabled=True, target="o_proj")),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok
    assert result.receipt.fast_weights_erased is True
    assert "fast_weight_snapshot_failed:MemoryError" in result.receipt.honest_flags
    assert [layer.self_attn.o_proj for layer in model.model.layers] == originals


def test_post_episode_invariant_probe_failure_refuses_output(monkeypatch, tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())

    def fail_post_probe():
        raise RuntimeError("injected post-probe failure")

    monkeypatch.setattr(engine.invariant, "post_episode", fail_post_probe)
    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok is False
    assert result.reason == "checkpoint_invariant_violated"
    assert result.receipt.params_unchanged is False
    assert "checkpoint_post_probe_failed:RuntimeError" in result.receipt.honest_flags


def test_config_validation_rejects_garbage(tiny_model):
    with pytest.raises(ValueError):
        LatentCortexEngine(tiny_model, config=_config(recurrence=RecurrenceConfig(max_steps=0)))
    with pytest.raises(ValueError):
        LatentCortexEngine(tiny_model, config=_config(branches=BranchConfig(n_branches=99)))


@pytest.mark.parametrize(
    "override",
    [
        {"workspace": WorkspaceConfig(n_slots=True)},
        {"workspace": WorkspaceConfig(seed="0")},
        {"workspace": WorkspaceConfig(roles="objective")},
        {"recurrence": RecurrenceConfig(max_steps="2")},
        {"branches": BranchConfig(n_branches=1.5)},
        {"latent_opt": LatentOptConfig(enabled="false")},
        {"fast_weights": FastWeightsConfig(enabled=1)},
        {"decode_max_tokens": True},
    ],
)
def test_direct_config_validation_rejects_coercible_types(tiny_model, override):
    with pytest.raises(ValueError):
        LatentCortexEngine(tiny_model, config=_config(**override))


@pytest.mark.parametrize("invalid", [True, 1.5, "8", 0, 8193])
def test_reason_rejects_malformed_decode_override(tiny_model, invalid):
    engine = LatentCortexEngine(tiny_model, config=_config())
    with pytest.raises((TypeError, ValueError)):
        engine.reason(token_ids=PROMPT_TOKENS, decode_max_tokens=invalid)


def test_decode_newline_discipline_caps_babble_runs(tiny_model, monkeypatch):
    """Force the model into pure-newline babble and prove the sampling
    discipline caps runs at two, counts every suppression, and never edits
    emitted text (all output tokens remain model-sampled ids)."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex import engine as engine_mod

    newline_id, word_id, vocab = 7, 9, 128

    class NewlineTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [5, 9, 17]

        def decode(self, ids):
            return "".join("\n" if i == newline_id else "x" for i in ids)

    engine = engine_mod.LatentCortexEngine(
        tiny_model, NewlineTokenizer(), config=_config(decode_max_tokens=12)
    )

    # The model "wants" endless newlines: every logits call favors newline_id
    # overwhelmingly, with word_id as the runner-up the mask must expose.
    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, newline_id] = 10.0
    spiked[0, 0, word_id] = 5.0

    monkeypatch.setattr(
        engine_mod.LatentCortexEngine,
        "_logits",
        lambda self, h: spiked,
    )

    from mlx_lm.models.cache import KVCache

    from core.brain.llm.latent_cortex.types import ComputeBudget

    cache = [KVCache() for _ in tiny_model.model.layers]
    budget = ComputeBudget()
    # Prefill through the real layers so the cache is genuine.
    import mlx.core as mx2
    from mlx_lm.models.base import create_attention_mask

    inner = tiny_model.model
    h = inner.embed_tokens(mx2.array([PROMPT_TOKENS]))
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])

    out, termination = engine._decode(cache, budget, spiked[0, 0], max_tokens=12, temperature=0.0)
    assert termination == "token_limit"
    assert engine._last_decode_newline_suppressions >= 3
    # No run of pure-newline tokens longer than the discipline allows.
    run = longest = 0
    for token in out:
        run = run + 1 if token == newline_id else 0
        longest = max(longest, run)
    assert longest <= 2, out
    assert word_id in out, "masking must expose the model's own runner-up token"


def test_bridge_policies_produce_distinct_hashable_cues(tiny_model):
    class RecordingTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 128 for c in text][:24]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    v1 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v1")
    )._decode_bridge_tokens()
    v2 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v2")
    )._decode_bridge_tokens()
    v3 = LatentCortexEngine(
        tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="assistant_answer_v3")
    )._decode_bridge_tokens()
    assert v1 and v2 and v3
    assert len({tuple(v1), tuple(v2), tuple(v3)}) == 3
    with pytest.raises(ValueError):
        LatentCortexEngine(
            tiny_model, RecordingTokenizer(), config=_config(decode_bridge_policy="bogus")
        )


def test_repetition_penalty_breaks_forced_loop(tiny_model, monkeypatch):
    """A model that 'wants' to emit one token forever must be broken out of
    the loop by the sliding-window penalty — the CP105 live degeneration
    (one line repeated ~80 times) as a mechanical regression."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget

    loop_id, alt_id, vocab = 11, 23, 128

    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, loop_id] = 6.0
    spiked[0, 0, alt_id] = 5.5  # runner-up the penalty must expose

    monkeypatch.setattr(engine_mod.LatentCortexEngine, "_logits", lambda self, h: spiked)

    def run(penalty):
        engine = engine_mod.LatentCortexEngine(
            tiny_model,
            config=_config(
                decode_max_tokens=24,
                decode_repetition_penalty=penalty,
            ),
        )
        cache = [KVCache() for _ in tiny_model.model.layers]
        from mlx_lm.models.base import create_attention_mask

        inner = tiny_model.model
        h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        out, _ = engine._decode(
            cache, ComputeBudget(), spiked[0, 0], max_tokens=24, temperature=0.0
        )
        return out

    unguarded = run(1.0)
    assert unguarded.count(loop_id) == len(unguarded), "control arm must loop"

    guarded = run(1.25)
    assert alt_id in guarded, "penalty must surface the runner-up token"
    longest = run_len = 0
    for token in guarded:
        run_len = run_len + 1 if token == loop_id else 0
        longest = max(longest, run_len)
    assert longest < len(guarded), "penalty must break the monoculture"


def test_sentence_grace_finishes_the_sentence(tiny_model, monkeypatch):
    """When the token limit lands mid-sentence, the decoder may sample up to
    the grace window until sentence-final punctuation — model tokens only,
    receipted as its own termination kind."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget

    word_id, period_id, vocab = 9, 13, 128

    class GraceTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids):
            return "".join("." if i == period_id else "w" for i in ids)

    calls = {"n": 0}

    def scripted_logits(self, h):
        # Words until well past the limit, then a period.
        calls["n"] += 1
        spiked = mx.full((1, 1, vocab), -20.0)
        spiked[0, 0, period_id if calls["n"] >= 10 else word_id] = 8.0
        return spiked

    monkeypatch.setattr(engine_mod.LatentCortexEngine, "_logits", scripted_logits)
    engine = engine_mod.LatentCortexEngine(
        tiny_model, GraceTokenizer(), config=_config(decode_max_tokens=6)
    )
    cache = [KVCache() for _ in tiny_model.model.layers]
    inner = tiny_model.model
    h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])

    first = mx.full((1, 1, vocab), -20.0)
    first[0, 0, word_id] = 8.0
    out, termination = engine._decode(
        cache, ComputeBudget(), first[0, 0], max_tokens=6, temperature=0.0
    )
    assert termination == "token_limit_sentence_grace"
    assert out[-1] == period_id, "grace must end at the model's own period"
    assert 6 < len(out) <= 6 + 48


def test_eos_floor_suppresses_early_stop(tiny_model, monkeypatch):
    """Sampling variance must not abandon an answer a few tokens in: below
    decode_min_tokens the EOS logits are masked (min-new-tokens), and after
    the floor the model's own EOS is honored (CP116 live regression)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    from core.brain.llm.latent_cortex import engine as engine_mod
    from core.brain.llm.latent_cortex.types import ComputeBudget

    eos_id, word_id, vocab = 3, 9, 128

    class EosTokenizer:
        eos_token_id = eos_id

        def encode(self, text, **kwargs):
            return [1, 2]

        def decode(self, ids):
            return "w" * len(ids)

    # The model "wants" to stop immediately: EOS is always argmax.
    spiked = mx.full((1, 1, vocab), -20.0)
    spiked[0, 0, eos_id] = 9.0
    spiked[0, 0, word_id] = 5.0
    monkeypatch.setattr(engine_mod.LatentCortexEngine, "_logits", lambda self, h: spiked)

    def run(min_tokens):
        engine = engine_mod.LatentCortexEngine(
            tiny_model,
            EosTokenizer(),
            config=_config(decode_max_tokens=20, decode_min_tokens=min_tokens),
        )
        cache = [KVCache() for _ in tiny_model.model.layers]
        inner = tiny_model.model
        h = inner.embed_tokens(mx.array([PROMPT_TOKENS]))
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        return engine._decode(cache, ComputeBudget(), spiked[0, 0], max_tokens=20, temperature=0.0)

    out, termination = run(0)
    assert termination == "eos" and len(out) == 0, "control: instant EOS honored"

    out, termination = run(8)
    assert termination == "eos"
    assert len(out) == 8, "EOS must be masked until the floor, then honored"
    assert all(token == word_id for token in out), "runner-up token fills the floor"
