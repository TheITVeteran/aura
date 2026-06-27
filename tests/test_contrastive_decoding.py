"""Tests for the contrastive-decoding + reasoning-steering decision math (numpy core)."""
from __future__ import annotations

import numpy as np

from core.brain.llm.contrastive_decoding import (
    ReasoningSteeringProcessor,
    build_reasoning_bias,
    build_reasoning_logits_processors,
    contrastive_combine_np,
    plausible_mask_np,
    steering_combine_np,
)


def test_plausible_mask_keeps_top_tokens():
    logits = np.array([10.0, 9.0, 1.0, -5.0])
    mask = plausible_mask_np(logits, beta=0.1)
    assert mask[0] and mask[1]      # top two are plausible
    assert not mask[3]              # far tail is masked out


def test_contrastive_masks_implausible_tokens():
    smart = np.array([5.0, 4.0, -10.0])
    amateur = np.array([0.0, 0.0, 0.0])
    out = contrastive_combine_np(smart, amateur, alpha=0.5, beta=0.1)
    assert np.isneginf(out[2])      # implausible token can never be sampled
    assert np.isfinite(out[0]) and np.isfinite(out[1])


def test_contrastive_suppresses_amateur_preference():
    # Both tokens equally plausible to the strong model, but the amateur strongly
    # prefers token 0 → contrastive decoding should favour token 1.
    smart = np.array([2.0, 2.0])
    amateur = np.array([5.0, 0.0])
    out = contrastive_combine_np(smart, amateur, alpha=1.0, beta=0.5)
    assert out[1] > out[0]


def test_contrastive_fail_open_on_shape_mismatch():
    smart = np.array([1.0, 2.0, 3.0])
    amateur = np.array([1.0, 2.0])
    out = contrastive_combine_np(smart, amateur)
    assert np.array_equal(out, smart)


def test_contrastive_never_all_neginf():
    smart = np.array([0.0, 0.0, 0.0])
    amateur = np.array([0.0, 0.0, 0.0])
    out = contrastive_combine_np(smart, amateur, alpha=2.0, beta=0.5)
    assert np.any(np.isfinite(out))


def test_steering_only_affects_plausible_tokens():
    logits = np.array([10.0, 9.5, -20.0])
    # Bias tries to boost an implausible token (idx 2) and suppress a plausible one.
    out = steering_combine_np(logits, {2: 100.0, 0: -1.0}, beta=0.1)
    assert out[2] == logits[2]          # implausible token untouched
    assert out[0] == logits[0] - 1.0    # plausible token adjusted


def test_build_reasoning_bias_encodes_filler():
    class _Tok:
        def encode(self, word, add_special_tokens=False):
            return [abs(hash(word)) % 1000]  # one id per word

    bias = build_reasoning_bias(_Tok(), suppress=-2.0)
    assert bias
    assert all(v == -2.0 for v in bias.values())


def test_build_reasoning_bias_handles_no_tokenizer():
    assert build_reasoning_bias(object()) == {}


def test_factory_assembles_processors():
    class _Tok:
        def encode(self, word, add_special_tokens=False):
            return [abs(hash(word)) % 1000]

    # No amateur, steering on → exactly the steering processor.
    procs = build_reasoning_logits_processors(_Tok(), enable_steering=True)
    assert len(procs) == 1
    assert isinstance(procs[0], ReasoningSteeringProcessor)

    # Amateur + steering → both.
    procs2 = build_reasoning_logits_processors(
        _Tok(), enable_steering=True, amateur_logits_fn=lambda toks: None
    )
    assert len(procs2) == 2

    # Nothing enabled → empty.
    assert build_reasoning_logits_processors(_Tok()) == []


# --- amateur logits source (real model wiring) -------------------------------
import os
from pathlib import Path

import pytest

_AMATEUR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct-4bit"


def test_amateur_loader_graceful_on_bad_path():
    from core.brain.llm.contrastive_decoding import get_amateur_logits_fn

    assert get_amateur_logits_fn("/nonexistent/model/path") is None


def test_factory_skips_contrastive_when_amateur_unavailable():
    from core.brain.llm.contrastive_decoding import build_reasoning_logits_processors

    class _Tok:
        def encode(self, w, add_special_tokens=False):
            return [abs(hash(w)) % 1000]

    procs = build_reasoning_logits_processors(
        _Tok(), enable_steering=True, amateur_model_path="/nonexistent/model"
    )
    # Amateur failed to load → only the steering processor remains.
    assert len(procs) == 1


@pytest.mark.skipif(not _AMATEUR.exists(), reason="local 1.5B amateur model not present")
def test_real_amateur_forward_pass_shape():
    import mlx.core as mx

    from core.brain.llm.contrastive_decoding import ContrastiveLogitsProcessor, get_amateur_logits_fn

    fn = get_amateur_logits_fn(str(_AMATEUR))
    assert fn is not None
    logits = fn(mx.array([9707, 1879, 374]))
    assert logits is not None
    assert logits.shape[0] == 1 and logits.shape[-1] > 100000  # Qwen2.5 vocab
    # Real contrastive combine produces a finite distribution.
    out = ContrastiveLogitsProcessor(fn, alpha=0.5, beta=0.1)(mx.array([9707, 1879, 374]), logits)
    assert bool(mx.any(mx.isfinite(out)).item())
