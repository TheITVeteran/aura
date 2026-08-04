"""Train the readout, and prove the trainer can be believed in both directions.

Gating the untrained random projection away from users was the safety fix. It
was not the capability fix, and saying "unmeasured" is not the same as making
the thing work. This is the capability: a readout actually fitted to real pairs,
scored on held-out data against the random projection it replaces.

A trainer is only worth its report if it can do two things:

  1. FIND a relation that is really there;
  2. DECLINE to report one that is not.

Both are tested. The second matters more — a fitter that always announces an
improvement is how a random projection became "a learned-readout style head" in
the first place.

The bound is stated rather than discovered later: 64 dimensions of affect and
cognitive summary cannot encode which content words a sentence needs. What they
can carry is the part of token choice that depends on state — register,
hedging, directness. So the target is a state-conditioned token prior, and the
claim it supports is "the substrate measurably shapes word choice", not "the
substrate speaks".
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.substrate_readout_training import (
    MIN_TRAINING_PAIRS,
    ReadoutPair,
    pairs_from_history,
    save_fit,
    train_readout,
)

STATE_DIM = 64
VOCAB = 200


def _state_dependent_pairs(n: int = 4000, seed: int = 0) -> list[ReadoutPair]:
    """Tokens genuinely drawn from a state-conditioned distribution."""
    rng = np.random.default_rng(seed)
    true_w = rng.standard_normal((VOCAB, STATE_DIM)) * 0.6
    pairs = []
    for _ in range(n):
        state = rng.standard_normal(STATE_DIM) * 0.4
        logits = true_w @ state
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        pairs.append(ReadoutPair(tuple(state), int(rng.choice(VOCAB, p=probabilities))))
    return pairs


def _independent_pairs(n: int = 4000, seed: int = 1) -> list[ReadoutPair]:
    """Tokens drawn with no reference to state. There is nothing to learn."""
    rng = np.random.default_rng(seed)
    return [
        ReadoutPair(tuple(rng.standard_normal(STATE_DIM) * 0.4), int(rng.integers(0, VOCAB)))
        for _ in range(n)
    ]


class TestItFindsWhatIsThere:
    def test_a_real_state_token_relation_is_recovered(self):
        fit = train_readout(_state_dependent_pairs(), vocab_size=VOCAB, epochs=120)
        assert fit is not None
        assert fit.beats_random is True
        assert fit.improvement_nats > 0.05

    def test_the_gain_shows_up_where_it_matters(self):
        """Top-k hit rate is the part a caller would actually feel."""
        fit = train_readout(_state_dependent_pairs(), vocab_size=VOCAB, epochs=120)
        assert fit.trained_top_k > fit.random_top_k * 2.0
        assert fit.trained_top_k > fit.top_k / VOCAB  # better than chance

    def test_the_report_says_what_it_is_and_is_not(self):
        fit = train_readout(_state_dependent_pairs(), vocab_size=VOCAB, epochs=60)
        report = fit.as_report()
        assert "does not generate language on its own" in report["interpretation"]
        assert report["n_holdout"] > 0
        assert report["n_train"] > report["n_holdout"]


class TestItDeclinesWhatIsNot:
    def test_no_relation_is_not_reported_as_one(self):
        """The control. A fitter that always finds something finds nothing."""
        fit = train_readout(_independent_pairs(), vocab_size=VOCAB, epochs=120)
        assert fit is not None
        assert fit.beats_random is False, fit.as_report()

    def test_top_k_stays_at_chance_when_there_is_no_signal(self):
        fit = train_readout(_independent_pairs(), vocab_size=VOCAB, epochs=120)
        chance = fit.top_k / VOCAB
        assert fit.trained_top_k == pytest.approx(chance, abs=0.05)

    def test_the_verdict_needs_BOTH_likelihood_and_ranking(self):
        """Held-out likelihood alone drifts up from fitting noise; the
        conjunction with top-k is what keeps the verdict honest."""
        fit = train_readout(_independent_pairs(), vocab_size=VOCAB, epochs=120)
        assert fit.trained_top_k <= fit.random_top_k + 1e-9
        assert fit.beats_random is False


class TestItRefusesToFitNothing:
    def test_too_few_pairs_produces_no_head_at_all(self):
        pairs = _state_dependent_pairs(n=MIN_TRAINING_PAIRS - 1)
        assert train_readout(pairs, vocab_size=VOCAB) is None

    def test_a_token_outside_the_vocabulary_is_an_error(self):
        pairs = _state_dependent_pairs(n=MIN_TRAINING_PAIRS + 50)
        pairs[0] = ReadoutPair(pairs[0].state, VOCAB + 5)
        with pytest.raises(ValueError, match="outside the declared vocabulary"):
            train_readout(pairs, vocab_size=VOCAB)


def test_history_records_flatten_into_pairs():
    records = [((0.1, 0.2), [3, 4, 5]), ((0.3, 0.4), [6])]
    pairs = pairs_from_history(records)
    assert len(pairs) == 4
    assert pairs[0].state == (0.1, 0.2)
    assert [p.token_id for p in pairs[:3]] == [3, 4, 5]


def test_a_saved_fit_carries_its_report(tmp_path):
    fit = train_readout(_state_dependent_pairs(n=600), vocab_size=VOCAB, epochs=40)
    target = save_fit(fit, tmp_path / "readout")
    assert target.with_suffix(".npy").exists()
    assert target.with_suffix(".json").exists()
    assert "improvement_nats" in target.with_suffix(".json").read_text()
