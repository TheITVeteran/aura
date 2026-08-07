"""Distil a session once it ends, not every five minutes forever.

The synthesis loop called ``_run_synthesis`` on every tick regardless of
whether anything had happened or whether the conversation was still going.
A forty-turn session was re-distilled eight times over, each pass re-reading
the same growing set; a session that ended thirty seconds after a tick
waited a full cycle anyway.

``core/voice/duplex/session.py`` has published ``session_ended`` on the bus
since it was written, and nothing had ever subscribed to it.
"""
from __future__ import annotations

import time

import pytest

from core.memory.memory_synthesizer import MemorySynthesizer


@pytest.fixture
def synth(tmp_path):
    return MemorySynthesizer(snapshot_path=tmp_path / "worldview.json")


class TestBoundaryDetection:
    def test_an_idle_runtime_is_not_an_ended_session(self, synth):
        # Nothing has ever arrived. _last_memory_at is 0.0, and reading that
        # as "the last message was in 1970" would distil an empty runtime on
        # its very first tick.
        assert synth.session_is_over() is False

    def test_a_live_conversation_is_not_over(self, synth):
        synth.notify_new_memory()

        assert synth.session_is_over() is False

    def test_a_quiet_gap_ends_the_session(self, synth):
        synth.notify_new_memory()
        synth._last_memory_at = time.time() - synth.SESSION_IDLE_SECONDS - 1

        assert synth.session_is_over() is True

    def test_a_pause_shorter_than_a_cycle_is_not_an_ending(self, synth):
        synth.notify_new_memory()
        synth._last_memory_at = time.time() - (synth.SESSION_IDLE_SECONDS / 2)

        assert synth.session_is_over() is False

    def test_nothing_new_means_nothing_to_distil(self, synth):
        # Long quiet with no unsynthesised material must not re-distil the
        # same worldview every cycle forever.
        synth._last_memory_at = time.time() - synth.SESSION_IDLE_SECONDS - 1000
        synth._new_since_synthesis = 0

        assert synth.session_is_over() is False


class TestExplicitSignal:
    def test_a_surface_that_knows_ends_the_session_immediately(self, synth):
        synth.notify_new_memory()
        assert synth.session_is_over() is False

        synth.notify_session_ended({"surface": "voice_duplex"})

        # No waiting for the idle gap.
        assert synth.session_is_over() is True

    def test_a_signal_alone_does_not_distil_an_empty_session(self, synth):
        synth.notify_session_ended({})

        assert synth.session_is_over() is False

    def test_new_material_means_the_session_resumed(self, synth):
        synth.notify_new_memory()
        synth.notify_session_ended({})
        assert synth.session_is_over() is True

        # Without clearing the flag, one session_ended would make every later
        # tick distil — exactly the every-few-minutes behaviour removed here.
        synth.notify_new_memory()

        assert synth.session_is_over() is False

    def test_the_volume_trigger_still_covers_a_session_that_never_ends(self, synth):
        # A conversation that never goes quiet must not go un-synthesised.
        for _ in range(synth.SYNTHESIS_TRIGGER_COUNT):
            synth.notify_new_memory()

        assert synth._new_since_synthesis >= synth.SYNTHESIS_TRIGGER_COUNT
        assert synth.session_is_over() is False, "still talking"
