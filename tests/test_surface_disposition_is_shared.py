"""Three gates, one policy.

The MLX worker, the inference gate and the response-generation phase can each
end a turn on their own. Each had grown its own list of what counts as
"unsafe", so teaching two of them that a shortfall is repairable still lost the
turn at the third — measured live 2026-07-26, in that exact order:

    Cortex produced repairable user-facing draft (final_answer_missing,
    len=735). Passing it to downstream chat repair…
    ResponseGeneration rejected unsafe user-facing draft (final_answer_missing,
    len=735).
"""

import pytest

from core.conversation.surface_disposition import (
    SHORTFALL_REASONS,
    SurfaceDisposition,
    disposition_for,
    draft_is_servable,
    integrity_failures,
)

pytestmark = pytest.mark.unit


class TestTheTwoKindsOfFailure:
    def test_nothing_wrong_is_served(self):
        assert disposition_for(()) is SurfaceDisposition.SERVE

    def test_a_shortfall_is_repaired_not_discarded(self):
        for reason in ("final_answer_missing", "truncated_tail", "incomplete_code_response"):
            assert disposition_for((reason,)) is SurfaceDisposition.REPAIR, reason
            assert draft_is_servable((reason,)), reason

    def test_text_that_must_not_be_spoken_is_discarded(self):
        for reason in (
            "raw_lane_telemetry",
            "prompt_artifact",
            "corrupted_language",
            "function_word_starvation",
            "cognitive_engine_failure_envelope",
        ):
            assert disposition_for((reason,)) is SurfaceDisposition.DISCARD, reason
            assert not draft_is_servable((reason,)), reason

    def test_one_integrity_failure_outweighs_any_number_of_shortfalls(self):
        mixed = ("final_answer_missing", "truncated_tail", "raw_lane_telemetry")
        assert disposition_for(mixed) is SurfaceDisposition.DISCARD
        assert integrity_failures(mixed) == ("raw_lane_telemetry",)

    def test_thinness_is_not_a_shortfall(self):
        """A truncated derivation has content the person can use; a thin reply
        has none, and downstream repair cannot invent the missing answer.
        Those need another generation — the existing documented decision."""
        for reason in (
            "too_thin_for_user_turn",
            "too_thin_for_operational_status_turn",
            "reliability_diagnostic_too_thin",
        ):
            assert disposition_for((reason,)) is SurfaceDisposition.DISCARD, reason

    def test_an_unclassified_reason_fails_safe(self):
        """A reason nobody has triaged is not assumed safe to speak."""
        assert disposition_for(("a_reason_added_next_week",)) is SurfaceDisposition.DISCARD


class TestEveryGateConsultsIt:
    """A new shortfall reason must not need three separate edits."""

    def test_the_worker_vetoes_on_integrity_only(self):
        import inspect

        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker._surface_quality_failure_reasons)
        assert "integrity_failures" in source

    def test_the_inference_gate_passes_shortfalls_downstream(self):
        import inspect

        from core.brain import inference_gate

        source = inspect.getsource(
            inference_gate._should_pass_user_facing_draft_downstream
        )
        assert "draft_is_servable" in source

    def test_response_generation_keeps_a_servable_draft(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "core"
            / "phases"
            / "response_generation.py"
        ).read_text(encoding="utf-8")
        rejection = source.split("rejected unsafe user-facing draft", 1)[0][-2000:]
        assert "draft_is_servable" in rejection


class TestTheReasonsThatMatterAreClassified:
    def test_the_live_shortfalls_are_all_repairable(self):
        """Every reason a correct answer was actually killed for."""
        for reason in (
            "final_answer_missing",
            "truncated_tail",
            "incomplete_code_response",
        ):
            assert reason in SHORTFALL_REASONS, reason
