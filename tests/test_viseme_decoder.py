"""Bounded-vocabulary lip reading: viseme classification, sequence
matching, honest refusals, and end-to-end decode through the pipeline.
"""
from __future__ import annotations

import numpy as np

from core.senses.viseme_decoder import (
    DEFAULT_VOCABULARY,
    VisemeDecoder,
    classify_viseme,
    collapse_runs,
)
from core.senses.visual_speech import VisualSpeechPipeline


# ── frame classification ─────────────────────────────────────────

def test_viseme_classes_from_geometry():
    assert classify_viseme(0.005, 0.40) == "C"   # closed / bilabial
    assert classify_viseme(0.08, 0.40) == "O"    # open vowel
    assert classify_viseme(0.03, 0.30) == "R"    # rounded
    assert classify_viseme(0.025, 0.50) == "W"   # wide


def test_run_collapse():
    assert collapse_runs("CCCOOORRR") == "COR"
    assert collapse_runs("") == ""
    assert collapse_runs("COCO") == "COCO"


# ── vocabulary matching ──────────────────────────────────────────

def _speak(decoder: VisemeDecoder, sequence: str, dwell: int = 3):
    geometry = {
        "C": (0.005, 0.40), "O": (0.08, 0.40),
        "R": (0.03, 0.30), "W": (0.025, 0.50),
    }
    for viseme in sequence:
        aperture, width = geometry[viseme]
        for _ in range(dwell):
            decoder.feed(aperture, width, speaking=True)


def test_every_vocabulary_word_decodes_from_its_own_template():
    for word, template in DEFAULT_VOCABULARY.items():
        decoder = VisemeDecoder()
        _speak(decoder, template)
        result = decoder.decode()
        assert result.word == word, (word, result.to_dict())
        assert result.confidence == 1.0


def test_one_viseme_slip_still_matches_when_unambiguous():
    decoder = VisemeDecoder()
    _speak(decoder, "WCRO")  # 'stop' (WCRC) with the final closure misread
    result = decoder.decode()
    assert result.word == "stop"
    assert 0.0 < result.confidence < 1.0


def test_confusable_slip_is_refused_as_ambiguous():
    decoder = VisemeDecoder()
    _speak(decoder, "WCOC")  # distance 1 from yes, stop AND open
    result = decoder.decode()
    assert result.word is None
    assert result.reason == "ambiguous_between_candidates"


def test_gibberish_is_refused_not_guessed():
    decoder = VisemeDecoder()
    _speak(decoder, "OWOWOWOW")
    result = decoder.decode()
    assert result.word is None
    assert result.reason in {"no_vocabulary_match", "ambiguous_between_candidates"}


def test_short_utterances_are_refused():
    decoder = VisemeDecoder()
    decoder.feed(0.08, 0.40, speaking=True)
    result = decoder.decode()
    assert result.word is None
    assert result.reason == "insufficient_articulation"


def test_silence_accumulates_nothing():
    decoder = VisemeDecoder()
    decoder.feed(0.08, 0.40, speaking=False)
    assert decoder.frame_count == 0


# ── end to end through the pipeline ──────────────────────────────

class _ScriptedTracker:
    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def lip_metrics(self, frame):
        if self._index >= len(self._frames):
            return {"aperture": 0.005, "width": 0.40}
        aperture, width = self._frames[self._index]
        self._index += 1
        return {"aperture": aperture, "width": width}


def test_pipeline_reads_a_word_end_to_end():
    """Articulated 'no' (C→R with articulation motion), then stillness:
    the utterance boundary triggers the decoder and the observation
    carries a real transcript."""
    frames = []
    # 'aura' = OCO. Wiggle WITHIN each viseme class (real articulators
    # never freeze) so speech activity stays hot while the viseme
    # sequence stays clean.
    for _ in range(8):
        frames.append((0.055, 0.40))   # O
        frames.append((0.095, 0.40))   # O
    for _ in range(5):
        frames.append((0.003, 0.40))   # C
        frames.append((0.013, 0.40))   # C
    for _ in range(8):
        frames.append((0.055, 0.40))   # O
        frames.append((0.095, 0.40))   # O
    frames.extend([(0.075, 0.40)] * 45)  # stillness → utterance ends

    pipeline = VisualSpeechPipeline(
        fps=15.0, face_detector=lambda g: None,
        lip_tracker=_ScriptedTracker(frames))
    transcripts = []
    for i in range(len(frames)):
        obs = pipeline.process_frame(np.zeros((8, 8), dtype=np.uint8), at=i / 15.0)
        if obs.transcript is not None:
            transcripts.append((obs.transcript, obs.transcript_source))
    assert transcripts, "no transcript was produced"
    word, source = transcripts[0]
    assert word == "aura"
    assert source == "viseme_command_decoder"
    assert pipeline.last_lip_read is not None
    assert pipeline.last_lip_read.to_dict()["honest_scope"] == (
        "bounded_command_vocabulary_not_open_speech")
