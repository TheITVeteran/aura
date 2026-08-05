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
            "endpoint_timeout:Cortex:150.0s",
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

    def _idle_but_alive_client(self):
        """A healthy resident worker with nothing running on it."""
        from types import SimpleNamespace

        from core.brain.llm.mlx_client import MLXLocalClient

        client = MLXLocalClient(model_path="/models/Qwen2.5-32B-Instruct-4bit")
        client._record_degraded_event = lambda *a, **k: None
        client._replace_ipc_queues = lambda *a, **k: None
        self.killed = False

        def _kill():
            self.killed = True

        client._process = SimpleNamespace(
            is_alive=lambda: not self.killed,
            pid=99,
            kill=_kill,
            join=lambda timeout=None: None,
        )
        return client

    def test_the_race_path_leaves_the_worker_up(self):
        """Behaviour, not source layout: a lost race must not cost a reload."""
        client = self._idle_but_alive_client()
        assert client.force_abort_active_generation("first_token_timeout") is False
        assert self.killed is False, "a worker with no work is not a worker to kill"

    def test_an_arbitrary_idle_kill_no_longer_happens_and_is_still_recorded(self):
        """CP126 ccb125e0: the reason's WORDING used to decide this.

        A reason matching the race markers spared the worker; anything else
        killed a healthy resident model. Whether a 20GB worker survived
        therefore depended on how a caller phrased a string, which is not a
        property of the worker or of the work. Now nothing is killed when
        nothing is running — and an abort on a stale premise stays visible.
        """
        from core.brain.llm import mlx_client

        recorded = []
        original = mlx_client._record_mlx_degradation
        mlx_client._record_mlx_degradation = lambda exc, **kw: recorded.append(str(exc))
        try:
            client = self._idle_but_alive_client()
            assert client.force_abort_active_generation("operator_requested_kill") is False
        finally:
            mlx_client._record_mlx_degradation = original

        assert self.killed is False
        assert any("force_abort_without_active_request" in msg for msg in recorded), (
            "an abort on a stale premise must stay visible"
        )
