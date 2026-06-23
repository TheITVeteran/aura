"""Tests for the persistent voltage-STDP engram association field."""
from __future__ import annotations

import numpy as np

from core.memory.engram_association import EngramAssociationField


def _field(tmp_path, n=64):
    return EngramAssociationField(n_slots=n, path=str(tmp_path / "assoc.npy"))


def test_slot_mapping_is_stable_across_instances(tmp_path):
    a = _field(tmp_path)
    b = _field(tmp_path)
    assert a._slot("apple") == b._slot("apple")          # hashlib, not salted hash()
    assert isinstance(a._slot("apple"), int)


def test_co_recall_potentiates_association_over_unrelated(tmp_path):
    f = _field(tmp_path)
    for _ in range(40):
        f.learn([["apple", "red"], ["fruit", "sweet"]])    # apple & fruit co-recalled
    co = f.association("apple", "fruit")
    unrelated = f.association("apple", "spaceship")          # never co-recalled
    assert co > unrelated
    assert co > 0.0


def test_learn_requires_two_distinct_items(tmp_path):
    f = _field(tmp_path)
    assert f.learn([["solo"]]) is False                     # nothing to associate
    assert f.learn([["a"], ["b"]]) is True


def test_association_boost_reflects_learned_links(tmp_path):
    f = _field(tmp_path)
    for _ in range(40):
        f.learn([["paris", "city"], ["france", "country"]])
    boost = f.association_boost(["paris"], ["france"])
    none = f.association_boost(["paris"], ["banana"])
    assert boost > none
    assert boost >= 0.0


def test_weights_stay_bounded_after_heavy_learning(tmp_path):
    f = _field(tmp_path)
    rng = np.random.default_rng(0)
    vocab = [f"c{i}" for i in range(30)]
    for _ in range(200):
        g1 = list(rng.choice(vocab, 2))
        g2 = list(rng.choice(vocab, 2))
        f.learn([g1, g2])
    assert np.isfinite(f.engine.W).all()
    assert np.abs(f.engine.W).max() <= f.engine.cfg.weight_clip + 1e-6
    assert f.engine.is_stable()


def test_persistence_round_trip(tmp_path):
    f = _field(tmp_path)
    for _ in range(25):                                      # crosses the autosave threshold
        f.learn([["x", "y"], ["z", "w"]])
    f.flush()
    reloaded = _field(tmp_path)
    assert np.allclose(reloaded.engine.W, f.engine.W)
