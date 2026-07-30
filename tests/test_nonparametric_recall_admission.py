"""A datastore has to be able to answer before it is allowed to speak.

Live 2026-07-26: the resident 32B served fluent, grammatical, meaning-free
replies. It kept serving them with substrate steering clamped to 0.01 and
recurrent depth off — which is what ruled both of those out. The cause was the
foreground kNN datastore: 1,677 of 1,689 entries carried no recallable token,
blended into the model's top-64 logits at a weight reaching 0.87.
"""

import pytest

from core.brain.nonparametric_worker import (
    _MIN_ENTRIES_TO_STEER_GENERATION,
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
        assert _unusable_datastore_reason(_Store([""] * 1677 + ["tok"] * 12))

    def test_the_same_store_is_still_refused_once_it_decodes(self):
        """DECODABLE IS NOT APPLICABLE, and this is the measured proof.

        Decoding that store's token ids from the resident tokenizer took it
        from 12 usable entries to 1,488 — genuinely correct per-token text —
        and the ratio guard passed. The store then steered live generation and
        two conversation turns degraded immediately:

            "dopamine, serotonin and norepagephrine like apes"
            "the inner core could prefer crystallishing iron alloys"

        Those 1,689 keys came from a CODING corpus. At that count over a
        5120-wide space the nearest neighbour of "octopus cognition" is
        whichever coding token is least far away, which is noise wearing the
        shape of grammar. Density is a separate requirement from decodability
        and it has to be checked first.
        """
        reason = _unusable_datastore_reason(_Store(["tok"] * 1488 + [""] * 201))
        assert reason
        assert "too sparse" in reason

    def test_a_dense_healthy_store_is_admitted(self):
        assert _unusable_datastore_reason(
            _Store(["tok"] * _MIN_ENTRIES_TO_STEER_GENERATION)
        ) == ""

    def test_a_mostly_empty_store_is_refused_even_when_dense(self):
        """The ratio check still matters once a store is big enough to reach
        it — a million empty slots are not a million neighbours."""
        usable = int(_MIN_ENTRIES_TO_STEER_GENERATION * 0.4)
        empty = _MIN_ENTRIES_TO_STEER_GENERATION - usable
        reason = _unusable_datastore_reason(_Store(["tok"] * usable + [""] * empty))
        assert "40.0% usable" in reason

    def test_a_sparse_store_is_declined_but_never_quarantined(self):
        """A small store has not grown yet; it is not broken. Quarantining it
        would throw away the beginning of a good one."""
        reason = _unusable_datastore_reason(_Store(["tok"] * _MIN_USABLE_ENTRIES))
        assert "too sparse" in reason
        import inspect

        import core.brain.nonparametric_worker as worker

        source = inspect.getsource(worker._unusable_datastore_reason)
        sparse_return = source.index("too sparse")
        quarantine = source.index("_quarantine_unusable_datastore")
        assert sparse_return < quarantine, (
            "the sparse path must return before anything can quarantine"
        )

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
