"""Tests for CAA extraction hooks."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from training.extract_steering_vectors import _extract_hidden_states


class _FakeMx:
    float32 = np.float32

    @staticmethod
    def array(value):
        return np.array(value)

    @staticmethod
    def eval(_value=None):
        return None


class _Tokenizer:
    @staticmethod
    def encode(_text: str):
        return [1, 2, 3]


class _Layer:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, x):
        return x + self.offset


class _InnerModel:
    def __init__(self) -> None:
        self.layers = [_Layer(1.0), _Layer(2.0)]


class _Model:
    def __init__(self) -> None:
        self.model = _InnerModel()

    def __call__(self, _tokens):
        h = np.zeros((1, 3, 4), dtype=np.float32)
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_extract_hidden_states_uses_class_level_call_hook():
    captured = _extract_hidden_states(
        _Model(),
        _Tokenizer(),
        "hello",
        [0, 1],
        _FakeMx,
    )

    assert sorted(captured) == [0, 1]
    assert captured[0].shape == (4,)
    assert captured[1].shape == (4,)
    assert np.allclose(captured[0], np.ones(4, dtype=np.float32))
    assert np.allclose(captured[1], np.full(4, 3.0, dtype=np.float32))


def test_caa_extractor_exposes_bounded_slice_mode():
    source = (Path(__file__).resolve().parents[1] / "training" / "extract_steering_vectors.py").read_text(
        encoding="utf-8"
    )

    assert "--dimensions" in source
    assert "--max-prompts-per-polarity" in source
    assert "Unknown CAA dimension" in source
