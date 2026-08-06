"""Ask before killing; and don't take a step that delivered nothing.

Two applications of the same optimizer discipline.

**The preemption ladder.** ``soft_cancel_active_generation`` documents itself
as "the cheap first rung on the preemption ladder before
``force_abort_active_generation``", and roughly ten call sites use it that way.
The foreground first-token watchdog did not: it went straight to the kill,
which costs a ~60-90s reload of a 20GB resident model. A first-token overrun
has two causes that look identical from outside — a worker wedged in prefill,
which cannot poll the cancel word and must be killed, and a worker one decode
step from its first token, which observes the cancel immediately. Only the
first needs a reload, and telling them apart costs one bounded wait.

It also aborted without ``expected_request_id`` despite having just checked
that id, so a new foreground request starting in the window between the check
and the kill was the one that died.

**Trust-region acceptance.** Gauss-Newton, Levenberg-Marquardt and Dogleg all
compute the reduction a step PREDICTS, measure the reduction it DELIVERS, and
reject the step when the ratio is poor — they do not move merely because a step
was computed. ``repair_is_an_improvement`` checked "introduces no new
objection" and "is not much shorter" but never that the objection the repair
was invoked for was actually gone.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from core.conversation.surface_disposition import repair_is_an_improvement

pytestmark = pytest.mark.unit


# ── The preemption ladder ────────────────────────────────────────────────


class TestTheWatchdogAsksBeforeKilling:
    @staticmethod
    def _client(monkeypatch, *, acks: bool, alive: bool = True):
        from core.brain.llm.mlx_client import MLXLocalClient

        client = MLXLocalClient(model_path="/models/Qwen2.5-7B-Instruct-4bit")
        client._current_request_id = "req-1"
        client._current_request_seq = 7
        client._current_request_started_at = time.time() - 60.0
        # Enough of a process for teardown to close this client cleanly — the
        # autouse fixture that closes MLX clients inside their own test calls
        # kill/join, and a bare namespace turns that into a CRITICAL
        # "worker_survived_kill" degradation from a worker that never existed.
        client._process = SimpleNamespace(
            is_alive=lambda: alive,
            kill=lambda: None,
            terminate=lambda: None,
            join=lambda *a, **k: None,
            close=lambda: None,
            pid=0,
            exitcode=0,
        )

        requested: list[str] = []

        def _soft_cancel(reason: str = ""):
            requested.append(reason)
            client._soft_cancel_target = {
                "req_id": "req-1",
                "seq": 7,
                "requested_monotonic": time.monotonic(),
                "reason": reason,
            }
            if acks:
                client._soft_cancel_ack = {
                    "req_id": "req-1",
                    "observed_monotonic": time.monotonic() + 0.001,
                }
            return {"requested": True, "reason": reason, "active_seq": 7}

        monkeypatch.setattr(client, "soft_cancel_active_generation", _soft_cancel)
        return client, requested

    def test_a_cooperative_cancel_is_tried_first(self, monkeypatch):
        client, requested = self._client(monkeypatch, acks=True)

        assert client._first_token_watchdog_soft_cancel("req-1") is True
        assert requested == ["first_token_wall_clock_watchdog"]

    def test_an_unacknowledged_cancel_escalates(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        client, requested = self._client(monkeypatch, acks=False)
        started = time.monotonic()

        assert client._first_token_watchdog_soft_cancel("req-1") is False

        waited = time.monotonic() - started
        assert requested, "it must still have ASKED before giving up"
        assert waited >= mod._WATCHDOG_ENFORCEMENT_SLACK_S * 0.8
        # …and the wait is bounded, not a hope.
        assert waited < mod._WATCHDOG_ENFORCEMENT_SLACK_S + 1.0

    def test_a_dead_worker_escalates_immediately(self, monkeypatch):
        client, _requested = self._client(monkeypatch, acks=False, alive=False)
        started = time.monotonic()

        assert client._first_token_watchdog_soft_cancel("req-1") is False
        assert time.monotonic() - started < 0.5

    def test_a_job_that_ended_on_its_own_is_not_killed(self, monkeypatch):
        """Nothing to abort, and aborting now takes whatever started after it."""
        client, _requested = self._client(monkeypatch, acks=False)

        def _finish_soon() -> None:
            time.sleep(0.1)
            client._current_request_id = "req-2"

        threading.Thread(target=_finish_soon, daemon=True).start()
        assert client._first_token_watchdog_soft_cancel("req-1") is True

    def test_a_cancel_that_was_never_requested_does_not_claim_success(
        self, monkeypatch
    ):
        from core.brain.llm.mlx_client import MLXLocalClient

        client = MLXLocalClient(model_path="/models/Qwen2.5-7B-Instruct-4bit")
        monkeypatch.setattr(
            client,
            "soft_cancel_active_generation",
            lambda reason="": {"requested": False, "detail": "no_active_generation"},
        )
        assert client._first_token_watchdog_soft_cancel("req-1") is False

    def test_the_watchdog_end_to_end_asks_then_aborts_the_right_request(
        self, monkeypatch
    ):
        """Let the real timer fire and watch what it does.

        Both halves in one run: the cooperative rung is tried first, and the
        escalation names the request the watchdog checked rather than whatever
        happens to be running when it gets there.
        """
        client, requested = self._client(monkeypatch, acks=False)
        aborted: list[dict] = []

        monkeypatch.setattr(
            client,
            "force_abort_active_generation",
            lambda reason="", **kw: aborted.append({"reason": reason, **kw}) or True,
        )
        monkeypatch.setattr(client, "_is_primary_or_deep_lane", lambda: True)
        client._current_first_token_at = 0.0

        timer = client._start_foreground_first_token_watchdog(
            "req-1", foreground_request=True, hard_ceiling_s=10.0
        )
        assert timer is not None
        try:
            # fire_after is max(10, ceiling + slack); drive it directly rather
            # than waiting out the real interval.
            timer.cancel()
            timer.function()
        finally:
            timer.cancel()

        assert requested == ["first_token_wall_clock_watchdog"], (
            "the kill must be preceded by an ask"
        )
        assert len(aborted) == 1
        assert aborted[0]["expected_request_id"] == "req-1", (
            "an unbound abort kills whatever started after the id check"
        )

    def test_a_worker_that_answers_the_cancel_is_never_aborted(self, monkeypatch):
        client, _requested = self._client(monkeypatch, acks=True)
        aborted: list[dict] = []

        monkeypatch.setattr(
            client,
            "force_abort_active_generation",
            lambda reason="", **kw: aborted.append({"reason": reason, **kw}) or True,
        )
        monkeypatch.setattr(client, "_is_primary_or_deep_lane", lambda: True)
        client._current_first_token_at = 0.0

        timer = client._start_foreground_first_token_watchdog(
            "req-1", foreground_request=True, hard_ceiling_s=10.0
        )
        try:
            timer.cancel()
            timer.function()
        finally:
            timer.cancel()

        assert aborted == [], "a cooperative stop must not also pay for a reload"

    def test_the_slack_is_one_number_used_twice(self):
        """A second, differently-chosen constant is how timing contracts drift.

        Measured rather than read: the timer's own interval must be the hard
        ceiling plus exactly the slack that the ack-wait also uses.
        """
        import core.brain.llm.mlx_client as mod

        from core.brain.llm.mlx_client import MLXLocalClient

        client = MLXLocalClient(model_path="/models/Qwen2.5-7B-Instruct-4bit")
        object.__setattr__(client, "_is_primary_or_deep_lane", lambda: True)

        timer = client._start_foreground_first_token_watchdog(
            "req-x", foreground_request=True, hard_ceiling_s=30.0
        )
        assert timer is not None
        try:
            assert timer.interval == pytest.approx(
                30.0 + mod._WATCHDOG_ENFORCEMENT_SLACK_S
            )
        finally:
            timer.cancel()
        assert mod._WATCHDOG_ENFORCEMENT_SLACK_S > 0.0


# ── Trust-region repair acceptance ───────────────────────────────────────


class TestARepairMustDeliverWhatItPredicted:
    QUESTION = "Answer in exactly three sentences: what is a transformer?"
    #: One sentence where three were asked for —
    #: `missing_requested_sentence_count`, a checkable shortfall.
    ORIGINAL = (
        "A transformer is a neural network architecture that relies on "
        "self-attention to relate every position in a sequence to every "
        "other one."
    )
    #: Reworded, still one sentence. The repair changed something and fixed
    #: nothing — which is precisely the step a trust region rejects.
    REWORDED = (
        "A transformer is a neural-network architecture built around "
        "self-attention, which relates each position in a sequence to all "
        "the others."
    )

    def _reasons(self, text: str) -> set[str]:
        from core.conversation.response_reliability import assess_user_facing_reply

        return set(assess_user_facing_reply(self.QUESTION, text).reasons)

    def test_the_setup_is_what_it_claims(self):
        """Both drafts carry the same objection, so only `targeted` separates them."""
        original = self._reasons(self.ORIGINAL)
        reworded = self._reasons(self.REWORDED)
        assert "missing_requested_sentence_count" in original
        assert "missing_requested_sentence_count" in reworded
        assert not (reworded - original), "the reword must introduce nothing new"

    def test_without_a_target_the_reword_is_accepted(self):
        """The old contract: no new objections, not much shorter — so it wins."""
        assert repair_is_an_improvement(self.ORIGINAL, self.REWORDED, self.QUESTION)

    def test_naming_the_target_rejects_it(self):
        assert not repair_is_an_improvement(
            self.ORIGINAL,
            self.REWORDED,
            self.QUESTION,
            targeted=("missing_requested_sentence_count",),
        )

    def test_a_repair_that_actually_fixes_it_is_still_accepted(self):
        fixed = (
            "A transformer is a neural network architecture. It relies on "
            "self-attention to relate every position in a sequence to every "
            "other one. That is what replaced recurrence for sequence "
            "modelling."
        )
        assert "missing_requested_sentence_count" not in self._reasons(fixed)
        assert repair_is_an_improvement(
            self.ORIGINAL,
            fixed,
            self.QUESTION,
            targeted=("missing_requested_sentence_count",),
        )

    def test_an_untargeted_objection_surviving_does_not_reject(self):
        """Only what the repair was CALLED for is held against it."""
        assert repair_is_an_improvement(
            self.ORIGINAL,
            self.REWORDED,
            self.QUESTION,
            targeted=("some_unrelated_reason",),
        )

    def test_a_target_may_be_given_as_a_bare_string(self):
        assert not repair_is_an_improvement(
            self.ORIGINAL,
            self.REWORDED,
            self.QUESTION,
            targeted="missing_requested_sentence_count",
        )

    def test_the_chat_route_names_its_target(self):
        """The retry knows what it is for; it must say so."""
        from pathlib import Path

        route = (
            Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat.py"
        )
        body = route.read_text(encoding="utf-8")
        assert "targeted=(failure_reason,)" in body
