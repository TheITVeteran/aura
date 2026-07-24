"""Configured role lesions must run as latent episodes, not refusals.

SPARK-070's dry run found CP328's sanctioned duplicate-role lesion arm and
CP331's correlation-evidence validation contradicting each other: the
uniform-role arm degraded to a vanilla-decode fallback, silently voiding
the lesion comparison. These tests pin the repaired contract."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.correlated_support import (
    ROLE_LESION_BUCKET,
    build_correlation_evidence,
    initial_exchange_weights,
    validate_correlation_evidence,
)


class TestLesionEvidence:
    def test_duplicate_roles_build_lesion_bucket_over_distinct_roles(self) -> None:
        roles = ["direct_derivation", "direct_derivation"]
        evidence = validate_correlation_evidence(None, roles=roles)
        assert evidence["bucket"] == ROLE_LESION_BUCKET
        assert evidence["roles"] == ["direct_derivation"]
        assert evidence["evidence_state"] == "bootstrap_unmeasured"

    def test_lesion_evidence_round_trips_through_revalidation(self) -> None:
        roles = ["direct_derivation", "direct_derivation", "constructive_solution"]
        evidence = validate_correlation_evidence(None, roles=roles)
        assert validate_correlation_evidence(evidence, roles=roles) == evidence

    def test_unique_roles_keep_the_strict_contract(self) -> None:
        roles = ["direct_derivation", "constructive_solution"]
        evidence = validate_correlation_evidence(None, roles=roles)
        assert evidence["bucket"] == "runtime|unmeasured"
        assert evidence["roles"] == roles
        assert validate_correlation_evidence(evidence, roles=roles) == evidence

    def test_preregistered_evidence_cannot_claim_a_lesion_run(self) -> None:
        unique_roles = ["direct_derivation", "constructive_solution"]
        preregistered = build_correlation_evidence(
            bucket="campaign|preregistered",
            roles=unique_roles,
            checked_outcomes=[],
        )
        lesioned_runtime = ["direct_derivation", "direct_derivation"]
        with pytest.raises(ValueError):
            validate_correlation_evidence(preregistered, roles=lesioned_runtime)

    def test_duplicate_programs_still_collapse_in_initial_weights(self) -> None:
        roles = ["direct_derivation", "direct_derivation"]
        weights = initial_exchange_weights(
            roles=roles,
            correlation_evidence=validate_correlation_evidence(None, roles=roles),
        )
        assert weights == {0: 0.5, 1: 0.5}
        distinct = ["direct_derivation", "constructive_solution"]
        distinct_weights = initial_exchange_weights(
            roles=distinct,
            correlation_evidence=validate_correlation_evidence(None, roles=distinct),
        )
        assert distinct_weights == {0: 1.0, 1: 1.0}


class TestLesionEpisodeRunsLatent:
    def test_uniform_role_episode_completes_without_fallback(self) -> None:
        mx = pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")
        from mlx_lm.models.qwen2 import Model, ModelArgs

        from core.brain.llm.latent_cortex.engine import LatentCortexEngine
        from core.brain.llm.latent_cortex.types import (
            BranchConfig,
            ComputeBudget,
            CortexConfig,
            LatentOptConfig,
            RecurrenceConfig,
            WorkspaceConfig,
        )

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=8,
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

        class StubTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [ord(character) % 128 for character in text][:32]

            def decode(self, ids):
                return " ".join(str(token) for token in ids)

        engine = LatentCortexEngine(
            model,
            StubTokenizer(),
            config=CortexConfig(
                workspace=WorkspaceConfig(n_slots=4, seed=11),
                recurrence=RecurrenceConfig(
                    min_steps=2, max_steps=2, convergence_eps=1e-9
                ),
                branches=BranchConfig(
                    n_branches=2,
                    roles=("direct_derivation", "direct_derivation"),
                ),
                latent_opt=LatentOptConfig(enabled=False),
                decode_max_tokens=4,
            ),
        )
        result = engine.reason(
            "lesion arm episode",
            budget=ComputeBudget(wall_clock_s=60.0),
            decode_max_tokens=4,
        )
        assert result.ok is True, result.reason
        receipt = result.receipt.to_dict()
        assert receipt["branch_isolation"]["configured_role_lesion"] is True
        support = receipt["correlated_support"]
        assert support["raw_support_count"] == 2
        assert support["effective_support_count"] < 2.0
