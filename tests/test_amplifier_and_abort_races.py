"""Two defects the live capability run surfaced within minutes of starting.

**The critic was judged as if it were the reply.** An internal critique is prose
ABOUT an answer, not an answer. The 2026-07-25 run rejected
``The proposed answer is 100% correct.`` for
``off_topic_self_reflection_reply,missing_requested_objective_facets`` against
``validation_source=inference_gate.visible_user_message`` — the critic did its
job perfectly and the reply gate threw the whole reasoning pass away for not
sounding like chat. That is an amplifier disabled by a category error.

**A lost abort race was recorded as damage.** ``inference_gate_generation_timeout``
fired, the generation finished first, and the abort then recorded a degradation
AND a MARGINAL fault, and killed a healthy idle worker — buying a cold reload
for a turn that had already been answered.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytestmark = pytest.mark.unit


class TestTheCriticIsInternal:
    def test_critique_generation_drops_the_user_surface_contract(self):
        from core.brain import reasoning_strategies

        src = inspect.getsource(reasoning_strategies)
        block = src[src.index("critique_kwargs = dict(kwargs)") :][:1600]
        assert 'critique_kwargs["clean_user_surface_contract"] = False' in block, (
            "an internal critique judged as a chat reply kills the amplifier"
        )
        assert 'critique_kwargs["user_surface_validation_prompt"] = ""' in block

    def test_the_critique_is_still_marked_as_an_internal_stage(self):
        from core.brain import reasoning_strategies

        src = inspect.getsource(reasoning_strategies)
        assert 'critique_kwargs["internal_reasoning_stage"] = "critique"' in src

    def test_recursion_is_still_prevented(self):
        """The existing guard must survive the change."""
        from core.brain import reasoning_strategies

        src = inspect.getsource(reasoning_strategies)
        assert 'critique_kwargs["bypass_critique"] = True' in src

    def test_a_confirming_critique_still_returns_the_original(self):
        """The live text — the critic confirming — must keep the draft."""
        from core.brain import reasoning_strategies

        src = inspect.getsource(reasoning_strategies)
        assert '"100% correct"' in src and '"is correct"' in src, (
            "confirmation phrases are how a clean critique returns the draft"
        )


class TestALostAbortRaceIsNotDamage:
    @pytest.fixture()
    def marker_re(self):
        from core.brain.llm.mlx_client import _ABORT_RACE_MARKERS_RE

        return _ABORT_RACE_MARKERS_RE

    @pytest.mark.parametrize(
        "reason",
        [
            "inference_gate_generation_timeout:Reflex:14.4s",
            "first_token_timeout",
            "kernel_soft_deadline",
            "deadline_missed",
        ],
    )
    def test_timeout_shaped_reasons_are_races(self, marker_re, reason):
        assert marker_re.search(reason)

    @pytest.mark.parametrize(
        "reason",
        [
            "memory_pressure_critical",
            "operator_requested_kill",
            "crash_loop_backoff",
            "model_swap",
        ],
    )
    def test_arbitrary_kills_are_not_races(self, marker_re, reason):
        assert not marker_re.search(reason)

    def test_the_race_path_leaves_the_worker_up(self):
        from core.brain.llm import mlx_client

        src = inspect.getsource(mlx_client)
        block = src[src.index("if not had_active_request:") :][:1800]
        race = block[block.index("_ABORT_RACE_MARKERS_RE") :][:400]
        assert "return False" in race, (
            "a worker with no work is not a worker to kill"
        )
        assert "_record_mlx_degradation" not in race, (
            "losing a race the timeout will always sometimes lose is not a fault"
        )

    def test_an_arbitrary_idle_kill_is_still_recorded(self):
        from core.brain.llm import mlx_client

        src = inspect.getsource(mlx_client)
        block = src[src.index("if not had_active_request:") :][:2200]
        assert "force_abort_without_active_request" in block, (
            "an emergency kill of an idle worker must stay visible"
        )
