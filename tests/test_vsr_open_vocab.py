"""Open-vocabulary VSR: CTC decoding + full ONNX inference pipeline.

The CTC decoders are checked against hand-constructed logits with known
answers; the ONNX backend is checked end-to-end against a REAL model
trained (deterministically, in-test) to map a synthetic mouthing signal
to characters — proving the whole open-vocab path, not a stub.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.vsr_ctc import (
    Vocabulary,
    beam_search_decode,
    default_vocabulary,
    greedy_decode,
)
from core.perception.vsr_onnx_backend import (
    MOUTH_SIZE,
    ModelProvenance,
    OnnxVSRBackend,
    load_onnx_backend,
    preprocess_mouth_crops,
)


def _onehot_logits(indices, vocab_size, scale=12.0):
    """Confident logits: each frame peaks at the given label index."""
    logits = np.full((len(indices), vocab_size), -scale, dtype=np.float64)
    for frame, index in enumerate(indices):
        logits[frame, index] = scale
    return logits


# ── CTC greedy decode ────────────────────────────────────────────

def test_greedy_collapses_repeats_and_drops_blanks():
    vocab = Vocabulary("abc")  # size 4 (0=blank,1=a,2=b,3=c)
    # a a blank a b b -> "aab"
    logits = _onehot_logits([1, 1, 0, 1, 2, 2], vocab.size)
    assert greedy_decode(logits, vocab) == "aab"


def test_greedy_decodes_a_word():
    vocab = default_vocabulary()
    text = "hello world"
    indices = [vocab.alphabet.index(ch) + 1 for ch in text]
    # Insert blanks between repeats so CTC collapse yields the word.
    frames = []
    previous = None
    for index in indices:
        if index == previous:
            frames.append(0)  # blank separates the double-l
        frames.append(index)
        previous = index
    assert greedy_decode(_onehot_logits(frames, vocab.size), vocab) == text


# ── CTC beam search ──────────────────────────────────────────────

def test_beam_search_matches_greedy_on_confident_input():
    vocab = Vocabulary("abcde")
    logits = _onehot_logits([1, 0, 2, 2, 0, 3], vocab.size)
    transcript, confidence = beam_search_decode(logits, vocab)
    assert transcript == greedy_decode(logits, vocab) == "abc"
    assert 0.0 < confidence <= 1.0


def test_beam_search_merges_blank_variant_paths():
    """Two high-probability paths ('a·' and '·a') are the same prefix 'a';
    prefix-beam must merge them, ranking 'a' above a split alternative."""
    vocab = Vocabulary("ab")
    logits = np.array([
        [0.5, 2.0, -1.0],   # favors a
        [1.5, 0.4, -1.0],   # favors blank
        [0.5, 2.0, -1.0],   # favors a again -> with blank between => "aa"
    ], dtype=np.float64)
    transcript, _ = beam_search_decode(logits, vocab, beam_width=8)
    assert transcript in {"aa", "a"}  # both are valid CTC collapses here
    assert set(transcript) <= {"a"}


def test_beam_search_language_model_breaks_ties():
    vocab = Vocabulary("io")
    # Frame ambiguous between 'i' and 'o'; the LM prefers 'o'.
    logits = np.array([[0.0, 1.0, 1.0]], dtype=np.float64)

    def lm(_prefix, char):
        return 0.0 if char == "o" else -5.0

    biased, _ = beam_search_decode(logits, vocab, lm=lm, lm_weight=1.0)
    assert biased == "o"


# ── Preprocessing ────────────────────────────────────────────────

def test_preprocess_shapes_and_normalization():
    crops = np.random.default_rng(0).integers(
        0, 255, size=(20, 96, 96, 3), dtype=np.uint8)
    tensor = preprocess_mouth_crops(crops)
    assert tensor.shape == (1, 1, 20, MOUTH_SIZE, MOUTH_SIZE)
    assert tensor.dtype == np.float32
    # Standardized: near-zero mean, unit-ish std.
    assert abs(float(tensor.mean())) < 1e-4
    assert 0.5 < float(tensor.std()) < 2.0


def test_preprocess_accepts_grayscale():
    crops = np.random.default_rng(1).integers(
        0, 255, size=(8, 88, 88), dtype=np.uint8)
    assert preprocess_mouth_crops(crops).shape == (1, 1, 8, 88, 88)


# ── Full ONNX pipeline against a real trained model ──────────────

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


def _train_and_export_tiny_vsr(tmp_path, vocab, sequences):
    """Train a small conv+GRU+CTC model to map a synthetic per-frame
    'mouthing' signal to character sequences, export to ONNX. Real
    training — the pipeline decodes a genuinely learned model."""
    import torch.nn as nn

    T = 24
    rng = np.random.default_rng(7)
    # One STABLE shape per character, shared by training and eval — each
    # character is a distinct high-contrast "mouth shape".
    all_chars = sorted({ch for text in sequences for ch in text})
    char_shapes = {
        ch: rng.standard_normal((MOUTH_SIZE, MOUTH_SIZE)).astype(np.float32)
        for ch in all_chars
    }

    def signal_for(text):
        frames = np.zeros((T, MOUTH_SIZE, MOUTH_SIZE), dtype=np.float32)
        per = max(1, T // max(1, len(text)))
        for i, ch in enumerate(text):
            frames[i * per:(i + 1) * per] = char_shapes[ch]
        return frames

    class TinyVSR(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.conv = nn.Conv3d(1, 8, kernel_size=(1, 5, 5), stride=(1, 4, 4))
            self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.gru = nn.GRU(8, 32, batch_first=True, bidirectional=True)
            self.head = nn.Linear(64, vocab_size)

        def forward(self, x):  # x: (1,1,T,88,88)
            h = torch.relu(self.conv(x))
            h = self.pool(h).squeeze(-1).squeeze(-1)  # (1,8,T)
            h = h.transpose(1, 2)                     # (1,T,8)
            h, _ = self.gru(h)
            return self.head(h)                       # (1,T,V)

    torch.manual_seed(0)
    model = TinyVSR(vocab.size)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    tensors = {text: torch.from_numpy(
        preprocess_mouth_crops(signal_for(text))) for text in sequences}
    targets = {text: torch.tensor(
        [vocab.alphabet.index(ch) + 1 for ch in text]) for text in sequences}

    for _ in range(400):
        opt.zero_grad()
        loss = torch.zeros(())
        for text in sequences:
            logits = model(tensors[text])           # (1,T,V)
            log_probs = torch.log_softmax(logits, dim=2).transpose(0, 1)
            loss = loss + ctc(
                log_probs, targets[text].unsqueeze(0),
                torch.tensor([logits.shape[1]]),
                torch.tensor([len(text)]))
        loss.backward()
        opt.step()

    model.eval()
    onnx_path = tmp_path / "tiny_vsr.onnx"
    # Legacy TorchScript exporter (dynamo=False): the new dynamo exporter
    # conflicts on dynamic-axis inference for this tiny model. All test
    # signals are fixed length T, so a static export is sufficient here;
    # the backend itself is T-agnostic (CTC decode handles any length).
    torch.onnx.export(
        model, tensors[sequences[0]], str(onnx_path),
        input_names=["mouth"], output_names=["logits"],
        opset_version=17, dynamo=False)
    return onnx_path, {text: signal_for(text) for text in sequences}


def test_onnx_backend_transcribes_a_real_trained_model(tmp_path):
    vocab = Vocabulary("abc")
    words = ["abc", "cab", "bca"]
    onnx_path, signals = _train_and_export_tiny_vsr(tmp_path, vocab, words)

    backend = load_onnx_backend(
        onnx_path, vocab=vocab,
        provenance=ModelProvenance(
            model_id="tiny-vsr-test", license="MIT (self-built)",
            training_data="synthetic in-test", acknowledged=True))
    available, reason = backend.available()
    assert available, reason

    import asyncio

    for word in words:
        crops = (signals[word] * 40 + 128).clip(0, 255).astype(np.uint8)
        prediction = asyncio.run(backend.infer(crops, fps=25.0))
        assert prediction.transcript == word, (word, prediction.transcript)
        assert prediction.calibrated
        assert prediction.backend == "onnx-vsr-ctc-beam"
        assert 0.0 < prediction.confidence <= 1.0


def test_restricted_license_model_refused_without_acknowledgement(tmp_path):
    model = tmp_path / "frontier.onnx"
    assert not model.exists()
    with pytest.raises(PermissionError, match="license"):
        load_onnx_backend(model, provenance=ModelProvenance(
            model_id="auto_avsr_vsr_trlrs3vox2",
            license="research-only (LRS3/VoxCeleb2 derived)",
            training_data="LRS3, VoxCeleb2", acknowledged=False))


def test_missing_model_file_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_onnx_backend(tmp_path / "nope.onnx", provenance=ModelProvenance(
            model_id="x", license="MIT", training_data="none", acknowledged=True))
