"""CP126 frontier_certification — treatment-side integrity and arm parity.

A published capability claim must not rest on an unmeasured boolean, and an
outcome gap must not be attributable to decoding rather than the treatment.
"""
from __future__ import annotations

from core.brain.llm.latent_cortex.frontier_certification import (
    _ARM_GENERATION_PARITY_FIELDS,
    _receipt_integrity_verdict,
    _validate_arm_generation_parity,
    _validate_treatment_receipt,
)


class TestTreatmentIntegrityDigest:
    """6090a5ae + 01e8b3c1: the treatment arm's integrity claims are measured."""

    def _receipt(self, **overrides):
        receipt = {
            "params_unchanged": True,
            "latent_opt_applied": True,
            "fast_weights_applied": True,
            "fast_weights_erased": True,
            "checkpoint_fingerprint": "a" * 64,
            "checkpoint_fingerprint_method": "sha256",
            "checkpoint_file_count": 3,
            "worker_boot_id": "boot",
            "installed_app_build_sha256": "b" * 64,
            "episode_id": "ep-1",
            "schedule_hash": "c" * 64,
            "latent_opt_mode": "gradient",
        }
        receipt.update(overrides)
        return receipt

    def _run(self, receipt):
        reasons: list[str] = []
        gaps: list[str] = []
        _validate_treatment_receipt(
            {"trial_id": "t1", "treatment_receipt": receipt},
            "a" * 64,
            "boot",
            "b" * 64,
            reasons,
            gaps,
        )
        return reasons, gaps

    def test_refuted_digest_rejects_the_trial(self):
        receipt = self._receipt(
            integrity_verdicts={
                "params_unchanged": {"verdict": "refuted"},
                "fast_weights_erased": {"verdict": "proven"},
            }
        )
        reasons, _ = self._run(receipt)
        assert any("params_unchanged_refuted_by_digest" in r for r in reasons)

    def test_refuted_erasure_rejects_the_trial(self):
        receipt = self._receipt(
            integrity_verdicts={
                "params_unchanged": {"verdict": "proven"},
                "fast_weights_erased": {"verdict": "refuted"},
            }
        )
        reasons, _ = self._run(receipt)
        assert any("fast_weights_erased_refuted_by_digest" in r for r in reasons)

    def test_absent_digest_disqualifies_and_is_distinguishable(self):
        reasons, gaps = self._run(self._receipt())
        # A bare boolean is not measured evidence, and the producer
        # (EpisodeReceipt.to_dict) DOES emit the digests — so an absent
        # verdict on a published claim disqualifies the trial...
        assert set(gaps) == {"treatment_params_unchanged", "treatment_fast_weights_erased"}
        assert any("unproven_no_digest" in r for r in reasons)
        # ...while staying distinguishable from "measured and false".
        assert not any("refuted_by_digest" in r for r in reasons)

    def test_proven_digest_leaves_no_gap(self):
        receipt = self._receipt(
            integrity_verdicts={
                "params_unchanged": {"verdict": "proven"},
                "fast_weights_erased": {"verdict": "proven"},
            }
        )
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
