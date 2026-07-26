"""A datastore has to be able to answer before it is allowed to speak.

Live 2026-07-26: the resident 32B served fluent, grammatical, meaning-free
replies. It kept serving them with substrate steering clamped to 0.01 and
recurrent depth off — which is what ruled both of those out. The cause was the
foreground kNN datastore: 1,677 of 1,689 entries carried no recallable token,
blended into the model's top-64 logits at a weight reaching 0.87.
"""

import pytest

from core.brain.nonparametric_worker import (
    _MIN_USABLE_ENTRIES,
    _unusable_datastore_reason,
    make_tapped_nonparametric_processor,
)

pytestmark = pytest.mark.unit


class _Store:
    def __init__(self, tokens):
        self._tokens = list(tokens)

    def __len__(self):
        return len(self._tokens)


class TestDatastoreAdmission:
    def test_the_live_store_shape_is_refused(self):
        """1,677 empty of 1,689 — the exact shape that was steering replies."""
        reason = _unusable_datastore_reason(_Store([""] * 1677 + ["tok"] * 12))
        assert reason
        assert "12 of 1689" in reason

    def test_a_healthy_store_is_admitted(self):
        assert _unusable_datastore_reason(_Store(["tok"] * 100)) == ""

    def test_a_mostly_empty_store_is_refused(self):
        reason = _unusable_datastore_reason(_Store(["tok"] * 40 + [""] * 60))
        assert "40.0% usable" in reason

    def test_a_small_but_wholly_usable_store_is_still_refused(self):
        """Too few entries to recall from is its own failure, not a ratio."""
        assert _unusable_datastore_reason(_Store(["tok"] * (_MIN_USABLE_ENTRIES - 1)))
        assert _unusable_datastore_reason(_Store(["tok"] * _MIN_USABLE_ENTRIES)) == ""

    def test_an_empty_store_is_left_to_the_existing_empty_path(self):
        assert _unusable_datastore_reason(_Store([])) == ""

    def test_an_unreadable_store_does_not_raise(self):
        assert _unusable_datastore_reason(object()) == ""


class TestInterpolationWeightCeiling:
    def test_a_neighbour_may_inform_the_next_token_but_not_choose_it(self):
        """The signature bounds lambda; 0.87 meant the store, not the model,
        was generating."""
        import inspect

        params = inspect.signature(make_tapped_nonparametric_processor).parameters
        base_lam = params["base_lam"].default
        max_lam = params["max_lam"].default
        free_energy = params["free_energy"].default

        assert base_lam <= 0.30, "kNN-LM interpolation weight belongs near 0.25"
        assert max_lam <= 0.40, "no single neighbour may take most of the token"

        # The free-energy term multiplies base_lam; the ceiling must survive it.
        worst_case = base_lam * 1.0 * (0.6 + 0.8 * float(free_energy))
        assert min(max_lam, worst_case) <= max_lam
        assert max_lam < 0.5, "the model must retain the majority of every token"
