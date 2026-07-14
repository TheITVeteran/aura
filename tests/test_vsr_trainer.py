"""Trainable VSR (the license-clean free path): the whole learnable
pipeline must close — synthetic labeled clips train to near-zero CTC
loss with exact greedy decode, and the trained net exports to the ONNX
contract the runtime backend already loads.

The full capability suite requires the ML dependency profile and runs on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.vsr_ctc import Vocabulary
from core.perception.vsr_trainer import (
    LabeledClip,
    synthesize_clip,
)

VOCAB = Vocabulary("abcdefgh ")


def _corpus(words, seed=0):
    return [synthesize_clip(word, VOCAB, seed=seed + i) for i, word in enumerate(words)]


def test_synthetic_clip_shape_and_determinism():
    a = synthesize_clip("bead", VOCAB, seed=3)
    b = synthesize_clip("bead", VOCAB, seed=3)
    assert a.mouth_crops.shape == (4 * 4, 88, 88)  # 4 chars × 4 frames/char
    np.testing.assert_array_equal(a.mouth_crops, b.mouth_crops)
    assert a.transcript == "bead"


def test_training_drives_ctc_loss_down_and_learns_transcripts():
    from core.perception.vsr_trainer import train_vsr

    clips = _corpus(["bead", "cab", "faded", "gee"])
    result = train_vsr(clips, VOCAB, epochs=250, learning_rate=4e-3, seed=1)

    # The learnable pipeline closes: loss collapses and every training
    # clip decodes back to its exact transcript.
    assert result.final_loss < result.initial_loss * 0.2
    assert result.final_loss < 0.5
    assert result.train_accuracy == 1.0


def test_trained_model_predicts_via_greedy_decode():
    from core.perception.vsr_trainer import predict_transcript, train_vsr

    clips = _corpus(["cab", "beef", "dead"])
    result = train_vsr(clips, VOCAB, epochs=250, learning_rate=4e-3, seed=2)
    for clip in clips:
        assert predict_transcript(result.model, clip.mouth_crops, VOCAB) == clip.transcript


def test_export_to_onnx_matches_runtime_backend(tmp_path):
    import onnxruntime

    from core.perception.vsr_trainer import export_onnx, train_vsr

    clips = _corpus(["cab", "fee"])
    result = train_vsr(clips, VOCAB, epochs=60, learning_rate=4e-3, seed=3)
    model_path = tmp_path / "self_trained.onnx"
    provenance = export_onnx(result.model, model_path, num_frames=12)

    assert model_path.exists()
    assert provenance["license"].startswith("owner-trained")
    assert provenance["acknowledged"] is True
    # Sidecar written next to the model.
    assert model_path.with_suffix(".provenance.json").exists()

    # The exported graph loads and runs on the same runtime as the
    # downloaded models would — same input name, same (1, T, V) logits.
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"])
    from core.perception.vsr_onnx_backend import preprocess_mouth_crops

    tensor = preprocess_mouth_crops(clips[0].mouth_crops)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0]
    assert logits.shape[0] == 1
    assert logits.shape[2] == VOCAB.size


def test_empty_corpus_rejected():
    from core.perception.vsr_trainer import train_vsr

    with pytest.raises(ValueError, match="at least one"):
        train_vsr([], VOCAB)


def test_corpus_roundtrip_and_trains(tmp_path):
    from core.perception.vsr_trainer import VSRCorpus, train_vsr

    corpus = VSRCorpus(tmp_path / "vsr_corpus")
    words = ["cab", "bead", "fee"]
    for i, word in enumerate(words):
        corpus.add(synthesize_clip(word, VOCAB, seed=i), clip_id=f"clip_{i}")

    loaded = corpus.load()
    assert sorted(c.transcript for c in loaded) == sorted(words)
    assert (tmp_path / "vsr_corpus" / "manifest.jsonl").exists()

    # The persisted corpus trains directly — the full free loop from
    # stored data to a learned model.
    result = train_vsr(loaded, VOCAB, epochs=200, learning_rate=4e-3, seed=5)
    assert result.train_accuracy == 1.0


def test_corpus_rejects_bad_clip_id(tmp_path):
    from core.perception.vsr_trainer import VSRCorpus

    corpus = VSRCorpus(tmp_path / "c")
    with pytest.raises(ValueError, match="alphanumeric"):
        corpus.add(synthesize_clip("ab", VOCAB), clip_id="!!!")
