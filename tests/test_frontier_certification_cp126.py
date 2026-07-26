"""CP126 frontier_certification — treatment-side integrity and arm parity.

A published capability claim must not rest on an unmeasured boolean, and an
outcome gap must not be attributable to decoding rather than the treatment.
"""
from __future__ import annotations

import copy

from core.brain.llm.latent_cortex.frontier_certification import (
    _ARM_GENERATION_PARITY_FIELDS,
    _receipt_integrity_verdict,
    _validate_arm_generation_parity,
    _validate_treatment_receipt,
)
from core.brain.llm.latent_cortex.runtime_integrity import (
    build_fast_weight_cleanup_proof,
    canonical_sha256,
)
from tests.fixtures.rlc_runtime_integrity import (
    accepted_fast_weight_learning,
    bound_runtime_integrity,
    complete_worker_identity,
)


class TestTreatmentIntegrityDigest:
    """6090a5ae + 01e8b3c1: the treatment arm's integrity claims are measured."""

    def _receipt(self, **overrides):
        episode_id = "ep-1"
        input_sha256 = "9" * 64
        worker_identity = complete_worker_identity(boot_id="1" * 32)
        learning = accepted_fast_weight_learning(
            episode_id=episode_id,
            input_tokens_sha256=input_sha256,
        )
        receipt = {
            "params_unchanged": True,
            "latent_opt_applied": True,
            "fast_weights_applied": True,
            "fast_weights_erased": True,
            "checkpoint_fingerprint": "f" * 64,
            "checkpoint_fingerprint_method": "sha256",
            "checkpoint_file_count": 8,
            "worker_boot_id": worker_identity["worker_boot_id"],
            "worker_identity": worker_identity,
            "installed_app_build_sha256": "b" * 64,
            "episode_id": episode_id,
            "input_tokens_sha256": input_sha256,
            "schedule_hash": "c" * 64,
            "latent_opt_mode": "gradient",
            "runtime_integrity": bound_runtime_integrity(
                episode_id=episode_id,
                input_tokens_sha256=input_sha256,
                fast_weights_applied=True,
                fast_weight_learning=learning,
                worker_identity=worker_identity,
            ),
        }
        receipt.update(overrides)
        return receipt

    def _run(self, receipt):
        reasons: list[str] = []
        gaps: list[str] = []
        _validate_treatment_receipt(
            {"trial_id": "t1", "treatment_receipt": receipt},
            "f" * 64,
            "1" * 32,
            "b" * 64,
            reasons,
            gaps,
        )
        return reasons, gaps

    def test_refuted_digest_rejects_the_trial(self):
        receipt = self._receipt()
        proof = copy.deepcopy(receipt["runtime_integrity"])
        proof["parameters"]["after"]["sha256"] = "8" * 64
        proof["parameters"]["unchanged"] = False
        proof["verdict"]["engine_measurements_complete"] = False
        proof["verdict"]["safe_to_continue"] = False
        proof["verdict"]["reasons"] = ["parameter_canary_changed"]
        proof["receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in proof.items()
                if key != "receipt_sha256"
            }
        )
        receipt["runtime_integrity"] = proof
        reasons, _ = self._run(receipt)
        assert any("params_unchanged_refuted_by_digest" in r for r in reasons)

    def test_refuted_erasure_rejects_the_trial(self):
        receipt = self._receipt()
        learning = accepted_fast_weight_learning(
            episode_id=receipt["episode_id"],
            input_tokens_sha256=receipt["input_tokens_sha256"],
        )
        failed_cleanup = build_fast_weight_cleanup_proof(
            episode_id=receipt["episode_id"],
            input_tokens_sha256=receipt["input_tokens_sha256"],
            detached=True,
            erase_proven=False,
            lease_released=True,
            conflicts=0,
            pre_probe_sha256="7" * 64,
            post_probe_sha256="8" * 64,
            layer_ids=["layers.1.o_proj", "layers.2.o_proj"],
        )
        receipt["runtime_integrity"] = bound_runtime_integrity(
            episode_id=receipt["episode_id"],
            input_tokens_sha256=receipt["input_tokens_sha256"],
            fast_weights_applied=True,
            fast_weight_learning=learning,
            fast_weight_cleanup=failed_cleanup,
            worker_identity=receipt["worker_identity"],
        )
        reasons, _ = self._run(receipt)
        assert any("fast_weights_erased_refuted_by_digest" in r for r in reasons)

    def test_absent_digest_disqualifies_and_is_distinguishable(self):
        receipt = self._receipt()
        receipt.pop("runtime_integrity")
        reasons, gaps = self._run(receipt)
        # A bare boolean is not measured evidence, and the producer
        # (EpisodeReceipt.to_dict) DOES emit the digests — so an absent
        # verdict on a published claim disqualifies the trial...
        assert set(gaps) == {"treatment_params_unchanged", "treatment_fast_weights_erased"}
        assert any("unproven_no_digest" in r for r in reasons)
        # ...while staying distinguishable from "measured and false".
        assert not any("refuted_by_digest" in r for r in reasons)

    def test_proven_digest_leaves_no_gap(self):
        receipt = self._receipt()
        reasons, gaps = self._run(receipt)
        assert gaps == []
        assert not any("refuted_by_digest" in r for r in reasons)

    def test_verdict_helper_defaults_to_unproven(self):
        assert _receipt_integrity_verdict(None, "params_unchanged") == "unproven"
        assert _receipt_integrity_verdict({}, "params_unchanged") == "unproven"


class TestArmGenerationParity:
    """8a56c486: seeds and decoding must be paired across arms."""

    def test_declared_mismatch_rejects_the_trial(self):
        reasons: list[str] = []
        gaps = _validate_arm_generation_parity(
            "t1",
            {"decode_temperature": 0.7, "decode_seed": 11},
            {"decode_temperature": 0.0, "decode_seed": 11},
            reasons,
        )
        assert any("generation_parity_mismatch:decode_temperature" in r for r in reasons)
        assert "decode_temperature" not in gaps

    def test_matching_values_pass(self):
        reasons: list[str] = []
        _validate_arm_generation_parity(
            "t1",
            {"decode_seed": 7, "decode_top_p": 0.9},
            {"decode_seed": 7, "decode_top_p": 0.9},
            reasons,
        )
        assert not any("generation_parity_mismatch" in r for r in reasons)

    def test_undeclared_on_both_sides_is_a_gap_not_a_rejection(self):
        reasons: list[str] = []
        gaps = _validate_arm_generation_parity("t1", {}, {}, reasons)
        assert set(gaps) == set(_ARM_GENERATION_PARITY_FIELDS)
        assert reasons == []

    def test_one_sided_declaration_is_a_gap(self):
        reasons: list[str] = []
        gaps = _validate_arm_generation_parity(
            "t1", {"decode_seed": 3}, {}, reasons
        )
        assert "decode_seed" in gaps
        assert reasons == []

    def test_missing_receipts_are_reported(self):
        reasons: list[str] = []
        _validate_arm_generation_parity("t1", None, {}, reasons)
        assert reasons == ["t1:generation_parity_receipts_missing"]

    def test_parity_covers_seed_and_template_identity(self):
        for field in ("decode_seed", "tokenizer_sha256", "prompt_template_sha256"):
            assert field in _ARM_GENERATION_PARITY_FIELDS
