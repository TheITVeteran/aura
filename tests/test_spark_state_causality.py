"""SPARK-013 state causality: projection, arms, grading, replay, engine."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256
from core.brain.llm.latent_cortex.state_causality import (
    ARM_ANNOTATION,
    ARM_INERT,
    ARM_INTACT,
    ARM_LESIONED,
    ARM_SHAM,
    STATE_CAUSALITY_ARMS,
    StateCausalityError,
    answer_matches,
    build_state_binding_tasks,
    build_task_arm_states,
    engine_episode_runner,
    project_state_evidence_context,
    replay_state_causality_receipt,
    run_state_causality_experiment,
)


def _task():
    return build_state_binding_tasks(count=3, seed=11)[0]


def _claims_by_name(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {claim["experiment"]: claim for claim in receipt["claims"]}


class TestProjection:
    def test_projection_selects_content_addressed_items(self) -> None:
        arms = build_task_arm_states(_task())
        bundle = arms[ARM_INTACT]
        items, receipt = project_state_evidence_context(
            bundle.state, bundle.contents
        )
        assert [item["evidence_id"] for item in items] == [
            "evidence-decoy-fact",
            "evidence-required-fact",
        ]
        for item in items:
            assert item["context_role"] == "evidence_observation"
            assert item["instruction_authority"] is False
            assert (
                hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                == item["content_sha256"]
            )
        assert receipt["state_sha256"] == bundle.state.state_sha256
        assert {entry["evidence_id"] for entry in receipt["excluded"]} == {
            "evidence-recalled-note"
        }
        assert receipt["excluded"][0]["reason"] == "memory_wire_only"
        body = {
            key: value
            for key, value in receipt.items()
            if key != "projection_sha256"
        }
        assert receipt["projection_sha256"] == canonical_sha256(body)

    def test_content_binding_mismatch_refused(self) -> None:
        arms = build_task_arm_states(_task())
        bundle = arms[ARM_INTACT]
        contents = dict(bundle.contents)
        contents["evidence-required-fact"] = "A forged fact the state never bound."
        with pytest.raises(StateCausalityError) as error:
            project_state_evidence_context(bundle.state, contents)
        assert error.value.code == "state_causality_content_binding_mismatch"

    def test_unknown_evidence_refused(self) -> None:
        arms = build_task_arm_states(_task())
        bundle = arms[ARM_INTACT]
        contents = dict(bundle.contents)
        contents["evidence-smuggled"] = "Content with no typed state ancestor."
        with pytest.raises(StateCausalityError) as error:
            project_state_evidence_context(bundle.state, contents)
        assert error.value.code == "state_causality_unknown_evidence"

    def test_prose_summary_never_reaches_the_wire(self) -> None:
        arms = build_task_arm_states(_task())
        intact_items, _ = project_state_evidence_context(
            arms[ARM_INTACT].state, arms[ARM_INTACT].contents
        )
        annotated_items, _ = project_state_evidence_context(
            arms[ARM_ANNOTATION].state, arms[ARM_ANNOTATION].contents
        )
        assert intact_items == annotated_items
        assert (
            arms[ARM_INTACT].state.state_sha256
            != arms[ARM_ANNOTATION].state.state_sha256
        )

    def test_arm_states_differ_exactly_where_designed(self) -> None:
        arms = build_task_arm_states(_task())
        assert set(arms) == set(STATE_CAUSALITY_ARMS)
        intact_ids = {
            record.evidence_id for record in arms[ARM_INTACT].state.evidence
        }
        assert intact_ids == {
            "evidence-required-fact",
            "evidence-decoy-fact",
            "evidence-recalled-note",
        }
        # The lesion removes the information, not the slot: the required
        # record survives with information-free filler content so the arm
        # keeps exact width and layer-application parity.
        lesioned = {
            record.evidence_id: record
            for record in arms[ARM_LESIONED].state.evidence
        }
        assert set(lesioned) == intact_ids
        intact_required = next(
            record
            for record in arms[ARM_INTACT].state.evidence
            if record.evidence_id == "evidence-required-fact"
        )
        assert (
            lesioned["evidence-required-fact"].content_sha256
            != intact_required.content_sha256
        )
        sham_ids = {
            record.evidence_id for record in arms[ARM_SHAM].state.evidence
        }
        assert sham_ids == intact_ids - {"evidence-decoy-fact"}
        inert_ids = {
            record.evidence_id for record in arms[ARM_INERT].state.evidence
        }
        assert inert_ids == intact_ids - {"evidence-recalled-note"}


def _synthetic_runner(prompt: str, cognitive_context: list[dict]) -> dict:
    """Deterministic pseudo-model: output depends only on prompt + items."""

    joined = prompt + "||" + "||".join(item["text"] for item in cognitive_context)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return {
        "ok": True,
        "text": " ".join(item["text"] for item in cognitive_context) or "empty",
        "tokens": [int(byte) for byte in digest[:8]],
        "final_states_sha256": hashlib.sha256(
            b"final:" + joined.encode("utf-8")
        ).hexdigest(),
        "steps_taken": 4,
        "layer_apps": 4096,
    }


class TestExperimentGrading:
    def test_structural_claims_supported_on_synthetic_runner(self) -> None:
        tasks = build_state_binding_tasks(count=4, seed=3)
        receipt = run_state_causality_experiment(
            _synthetic_runner, tasks, minimum_tasks=4
        )
        claims = _claims_by_name(receipt)
        assert (
            claims["expS_required_lesion_changes_computation"]["tier"]
            == "SUPPORTED"
        )
        assert claims["expS_nonprojected_state_inert"]["tier"] == "SUPPORTED"
        assert claims["expS_restoration_recovers"]["tier"] == "SUPPORTED"
        assert claims["expS_prose_shadow_prohibited"]["tier"] == "SUPPORTED"
        assert (
            claims["expS_content_substitution_tracks_state"]["tier"]
            == "SUPPORTED"
        )
        loss = claims["expS_task_appropriate_loss"]
        assert loss["tier"] in {"PROVEN", "SUPPORTED", "CONJECTURE", "REFUTED"}
        assert 0.0 <= loss["evidence"]["intact_rate"] <= 1.0
        assert receipt["n_tasks"] == 4
        assert len(receipt["rows"]) == 4 * len(STATE_CAUSALITY_ARMS)

    def test_small_n_caps_at_conjecture(self) -> None:
        tasks = build_state_binding_tasks(count=2, seed=5)
        receipt = run_state_causality_experiment(
            _synthetic_runner, tasks, minimum_tasks=20
        )
        claims = _claims_by_name(receipt)
        assert (
            claims["expS_required_lesion_changes_computation"]["tier"]
            == "CONJECTURE"
        )

    def test_replay_regrades_identically(self) -> None:
        tasks = build_state_binding_tasks(count=3, seed=7)
        receipt = run_state_causality_experiment(
            _synthetic_runner, tasks, minimum_tasks=3
        )
        replay = replay_state_causality_receipt(receipt)
        assert replay["replayed"] is True
        assert replay["claims"] == receipt["claims"]

    def test_replay_rejects_tampered_rows(self) -> None:
        tasks = build_state_binding_tasks(count=3, seed=7)
        receipt = run_state_causality_experiment(
            _synthetic_runner, tasks, minimum_tasks=3
        )
        tampered_rows = [dict(row) for row in receipt["rows"]]
        tampered_rows[0]["tokens_sha256"] = "0" * 64
        tampered = {**receipt, "rows": tampered_rows}
        with pytest.raises(StateCausalityError) as error:
            replay_state_causality_receipt(tampered)
        assert error.value.code == "state_causality_receipt_digest_mismatch"

    def test_replay_rejects_regraded_claim_tampering(self) -> None:
        tasks = build_state_binding_tasks(count=3, seed=7)
        receipt = run_state_causality_experiment(
            _synthetic_runner, tasks, minimum_tasks=3
        )
        upgraded = [dict(claim) for claim in receipt["claims"]]
        for claim in upgraded:
            claim["tier"] = "PROVEN"
        body = {
            **{k: v for k, v in receipt.items() if k != "receipt_sha256"},
            "claims": upgraded,
        }
        forged = {**body, "receipt_sha256": canonical_sha256(body)}
        with pytest.raises(StateCausalityError) as error:
            replay_state_causality_receipt(forged)
        assert error.value.code == "state_causality_replay_mismatch"

    def test_lying_runner_refutes_identity_claims(self) -> None:
        calls = {"count": 0}

        def _noise_runner(prompt: str, cognitive_context: list[dict]) -> dict:
            calls["count"] += 1
            return {
                "ok": True,
                "text": f"call-{calls['count']}",
                "tokens": [calls["count"]],
                "final_states_sha256": hashlib.sha256(
                    str(calls["count"]).encode("ascii")
                ).hexdigest(),
                "steps_taken": 4,
                "layer_apps": 4096,
            }

        tasks = build_state_binding_tasks(count=3, seed=9)
        receipt = run_state_causality_experiment(
            _noise_runner, tasks, minimum_tasks=3
        )
        claims = _claims_by_name(receipt)
        assert claims["expS_restoration_recovers"]["tier"] == "REFUTED"
        assert claims["expS_prose_shadow_prohibited"]["tier"] == "REFUTED"


class TestAnswerMatching:
    def test_final_answer_extraction(self) -> None:
        assert answer_matches("thinking...\nFINAL_ANSWER: 42", "42")
        assert answer_matches("The code word is meridian.", "meridian")
        assert not answer_matches("FINAL_ANSWER: 43", "42")
        assert not answer_matches("", "42")


class TestEngineSeam:
    def test_engine_arms_prove_state_causality_on_tiny_model(self) -> None:
        mx = pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")
        from mlx_lm.models.qwen2 import Model, ModelArgs

        from core.brain.llm.latent_cortex.engine import LatentCortexEngine
        from core.brain.llm.latent_cortex.types import (
            BranchConfig,
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
            # Word-level, sha-derived (not Python's randomized hash), so the
            # decisive evidence words survive the engine's 64-token context
            # cap behind the in-band authority prefix.
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [
                    1
                    + int.from_bytes(
                        hashlib.sha256(word.encode("utf-8")).digest()[:2], "big"
                    )
                    % 127
                    for word in text.split()
                ][:64]

            def decode(self, ids):
                return " ".join(str(token) for token in ids)

        tokenizer = StubTokenizer()

        def build_engine():
            return LatentCortexEngine(
                model,
                tokenizer,
                config=CortexConfig(
                    workspace=WorkspaceConfig(n_slots=6, seed=23),
                    recurrence=RecurrenceConfig(
                        min_steps=3,
                        max_steps=3,
                        convergence_eps=1e-12,
                    ),
                    branches=BranchConfig(n_branches=1),
                    latent_opt=LatentOptConfig(enabled=False),
                    decode_max_tokens=8,
                ),
            )

        runner = engine_episode_runner(
            build_engine, decode_max_tokens=8, wall_clock_s=60.0
        )
        tasks = build_state_binding_tasks(count=2, seed=17)
        receipt = run_state_causality_experiment(
            runner, tasks, minimum_tasks=2, runner_identity={"model": "tiny-qwen2"}
        )
        claims = _claims_by_name(receipt)
        for name in (
            "expS_required_lesion_changes_computation",
            "expS_nonprojected_state_inert",
            "expS_restoration_recovers",
            "expS_prose_shadow_prohibited",
            "expS_content_substitution_tracks_state",
        ):
            assert claims[name]["tier"] == "SUPPORTED", (name, claims[name])
        assert replay_state_causality_receipt(receipt)["claims"] == receipt[
            "claims"
        ]
