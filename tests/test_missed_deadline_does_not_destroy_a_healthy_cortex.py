"""A turn running out of time is not a wedged worker.

LIVE DEFECT, 2026-07-25. Bryan asked a follow-up question and got nothing
back at all. The trace:

    First-token HARD CEILING exceeded (livelocked: heartbeats but zero
      tokens) for Aura-32B-... (82.5s elapsed, sla=240.0s, hard=82.0s).
    Cortex still sending heartbeats (1.8s ago). Recycling after this
      abandoned foreground request so late text cannot bleed into the
      next turn.
    Cortex generation exceeded inference-gate timeout 86.0s
    [MLX] Abort ... arrived after the generation finished ...; nothing to
      abort, leaving the worker up.
    Endpoint Cortex failed validation: client_returned_no_text
    Circuit OPEN for Cortex

Three things in that trace are worth separating.

The 82.0s "hard ceiling" was NOT the livelock formula — that computes
around 450s for a foreground request. It was the caller's remaining
wall-clock minus a reserve, derived from an 86s inference-gate budget. The
label said livelock; the number came from a deadline.

The generation FINISHED, a few seconds after we stopped waiting. The abort
arrived to find nothing to abort. The worker was never wedged — the turn
was slower than its budget under 80% RAM.

And then a healthy, heartbeating, 20GB warm model was marked for recycle,
which made the NEXT turn slower, which made the next deadline likelier to
expire. The recycle was the part of that cascade we chose.

Orphaned output is already fenced three ways: the pending generation is
dropped, the request id no longer matches, and the worker is soft-cancelled
between tokens. Destroying the warm lane was never what kept late text out
of the next turn.
"""
from __future__ import annotations

import inspect
import re

from core.brain.llm import mlx_client


def _abandonment_source() -> str:
    """The first-token abandonment branch, read from the module."""
    source = inspect.getsource(mlx_client)
    start = source.index("livelock_ceiling = self._first_token_hard_ceiling")
    end = source.index("soft_cancel_active_generation(\"abandoned_first_token_sla\")", start)
    return source[start:end]


class TestTheTwoCeilingsAreDistinguished:
    def test_the_livelock_ceiling_is_computed_separately(self):
        """Previously the formula ceiling was immediately overwritten by
        min(formula, deadline), so by the time the branch ran there was no
        way to ask which one had fired."""
        assert "livelock_ceiling = self._first_token_hard_ceiling" in _abandonment_source()

    def test_the_livelock_verdict_tests_the_formula_ceiling(self):
        assert re.search(
            r"livelocked\s*=\s*elapsed_without_token\s*>\s*livelock_ceiling",
            _abandonment_source(),
        )

    def test_the_effective_ceiling_is_still_the_stricter_of_the_two(self):
        """Bounding by the caller's deadline must not regress — waiting past
        the caller is its own defect."""
        source = _abandonment_source()
        assert "min(" in source
        assert "request_hard_ceiling," in source


class TestRecycleIsReservedForRealWedges:
    def test_a_healthy_worker_past_its_deadline_is_not_recycled(self):
        """The branch Bryan's turn took. It must not set a reboot reason."""
        source = _abandonment_source()
        healthy_branch = source[source.index("else:", source.index("elif livelocked")):]
        assert "_deferred_reboot_reason" not in healthy_branch
        assert "KEEPING the warm lane" in healthy_branch

    def test_a_genuine_livelock_still_recycles(self):
        """The protection this ceiling exists for is intact."""
        source = _abandonment_source()
        livelock_branch = source[
            source.index("elif livelocked"): source.index("else:", source.index("elif livelocked"))
        ]
        assert "_deferred_reboot_reason" in livelock_branch

    def test_a_silent_worker_still_recycles(self):
        """No heartbeat for 30s is a wedge regardless of which ceiling
        fired, and stays the hardest verdict."""
        source = _abandonment_source()
        assert "heartbeat_age > 30.0" in source
        silent_branch = source[
            source.index("if heartbeat_age > 30.0"): source.index("elif livelocked")
        ]
        assert 'self._deferred_reboot_reason = "first_token_sla_exceeded"' in silent_branch

    def test_the_missed_deadline_is_still_recorded(self):
        """Keeping the worker must not make the miss invisible — a turn that
        ran out of budget is real degradation even with a healthy lane."""
        source = _abandonment_source()
        assert "first_token_deadline_exceeded_worker_healthy" in source


class TestOrphanedOutputIsStillFenced:
    def test_the_pending_generation_is_dropped(self):
        assert "self._pending_generations.pop(req_id, None)" in _abandonment_source()

    def test_the_worker_is_soft_cancelled_between_tokens(self):
        source = inspect.getsource(mlx_client)
        assert 'soft_cancel_active_generation("abandoned_first_token_sla")' in source

    def test_the_request_id_guards_the_branch(self):
        """Fencing by identity is what actually stops bleed into the next
        turn; the recycle was only ever belt-and-braces on top of it."""
        assert "req_id == self._current_request_id" in _abandonment_source()
