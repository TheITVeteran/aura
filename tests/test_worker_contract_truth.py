"""CP126 batch-3 truth contracts for the MLX worker.

Pins the fabrication/salvage fixes: sanitizer mode asymmetry, backslash
literal preservation, syntactic proof-fragment validation, the capability
inventory evidence boundary, and the operator-evidence model-merit gate.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _OPERATOR_EVIDENCE_PREFIX,
    _capability_inventory_minimum_grounding,
    _collapse_escape_noise,
    _normalize_strict_answer_response,
    _normalize_strict_value_response,
    _operator_evidence_model_contribution_insufficient,
    _proof_evaluation_fragment_incomplete,
    _sanitize_telemetry_leakage,
)


class TestSanitizerModeAsymmetry:
    def test_proof_mode_keeps_large_integers(self):
        text = "The factorial is 2432902008176640000000 exactly."
        assert _sanitize_telemetry_leakage(text, is_proof=True) == text

    def test_conversational_mode_rejects_digit_walls(self):
        assert (
            _sanitize_telemetry_leakage(
                "id 123456789012345678901234", is_proof=False
            )
            is None
        )

    def test_corrupted_language_rejected_in_every_mode(self):
        corrupted = "The xublcate value converges to 4 after evaluation runs."
        assert _sanitize_telemetry_leakage(corrupted, is_proof=False) is None
        assert _sanitize_telemetry_leakage(corrupted, is_proof=True) is None

    def test_backend_markers_stay_conversational_only(self):
        # "proceeding" is a common English word; proof/eval content keeps it.
        text = "Proceeding with the derivation, the limit equals 3 as required."
        assert _sanitize_telemetry_leakage(text, is_proof=True) == text
        assert _sanitize_telemetry_leakage(text, is_proof=False) is None


class TestEscapeNoisePreservation:
    def test_backslash_literals_survive(self):
        path_answer = r"C:\Users\bryan\tmp\report.txt"
        assert _collapse_escape_noise(path_answer) == path_answer

    def test_regex_answers_survive(self):
        regex_answer = r"^\d{4}-\d{2}-\d{2}$"
        assert _collapse_escape_noise(regex_answer) == regex_answer

    def test_pure_formatting_noise_collapses(self):
        assert _collapse_escape_noise("42\\n\\nDone") == "42  Done"

    def test_strict_value_keeps_windows_path(self):
        value = r"C:\temp\out.csv"
        assert _normalize_strict_value_response(value) == value

    def test_strict_answer_envelope_content_verbatim(self):
        raw = r"<answer>\d+\.\d+</answer>"
        assert (
            _normalize_strict_answer_response(raw, envelope_prefixed=False) == raw
        )


class TestProofFragmentValidation:
    def test_python_fence_must_parse(self):
        broken = "```python\ndef f(:\n    return 1\n```"
        assert _proof_evaluation_fragment_incomplete(broken) is True

    def test_valid_python_fence_is_complete(self):
        ok = "```python\ndef f(x):\n    return x * 2\n```"
        assert _proof_evaluation_fragment_incomplete(ok) is False

    def test_json_fence_must_parse(self):
        broken = '```json\n{"a": 1,\n```'
        assert _proof_evaluation_fragment_incomplete(broken) is True

    def test_bare_json_still_validated(self):
        assert _proof_evaluation_fragment_incomplete('{"a": [1, 2]}') is False
        assert _proof_evaluation_fragment_incomplete('{"a": [1, 2') is True

    def test_csv_requires_uniform_columns(self):
        uniform = "name,count\nalpha,3\nbeta,5"
        ragged = "name,count\nalpha,3\nthen the rest, of this is, prose noise"
        assert _proof_evaluation_fragment_incomplete(uniform) is False
        # Ragged rows no longer certify completeness via the CSV rule; the
        # short prose tail then fails the fragment length checks.
        assert _proof_evaluation_fragment_incomplete(ragged) is True


class TestCapabilityInventoryGrounding:
    _GROUNDED = (
        "I can use browser/web research and file/desktop actions, all "
        "governed by Will/authority checks, and I am not executing any of "
        "them in this turn."
    )
    _NO_BOUNDARY = (
        "I can use browser/web research and file/desktop actions, all "
        "governed by Will/authority checks."
    )

    def test_boundary_is_required(self):
        grounded, evidence = _capability_inventory_minimum_grounding(self._GROUNDED)
        assert grounded is True
        assert evidence["boundary"] is True

    def test_missing_boundary_and_effect_evidence_fails(self):
        # The old `or not effect-evidence` escape admitted exactly this case.
        grounded, evidence = _capability_inventory_minimum_grounding(
            self._NO_BOUNDARY
        )
        assert grounded is False
        assert evidence["boundary"] is False
        assert evidence["effect_evidence"] is False


class TestOperatorEvidenceModelMerit:
    def test_prefix_alone_cannot_carry_the_contract(self):
        # The fixed prefix contains every required evidence term; a fragment
        # of model fluff must still fail on its own merit.
        assert _operator_evidence_model_contribution_insufficient("Sure.") is True

    def test_substantive_continuation_passes(self):
        continuation = (
            "For this task I would define the check, run the governed probe, "
            "capture its receipt and trace, and stop if the result is unsafe "
            "or ambiguous."
        )
        assert (
            _operator_evidence_model_contribution_insufficient(continuation)
            is False
        )

    def test_prefix_contains_all_required_terms(self):
        # Documents WHY the merit gate exists: scaffolding satisfies the
        # combined-text term check by construction.
        body = _OPERATOR_EVIDENCE_PREFIX.lower()
        for term in ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood"):
            assert term in body
