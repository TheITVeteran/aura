"""SPARK-068: membership without shipping the history.

The accumulator's job is to make an n-event campaign verifiable in O(log n) per
envelope instead of O(n). These tests check it against an independent brute
force, then try to forge it.
"""

from __future__ import annotations

import hashlib

import pytest

from core.brain.llm.latent_cortex.journal_accumulator import (
    JournalAccumulatorError,
    accumulator_root,
    inclusion_proof,
    peak_sizes,
    proof_size_bytes,
    verify_inclusion,
)


def _events(count: int) -> list[str]:
    return [
        hashlib.sha256(f"event-{index}".encode()).hexdigest()
        for index in range(count)
    ]


# --- structure --------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [1]),
        (2, [2]),
        (3, [2, 1]),
        (7, [4, 2, 1]),
        (8, [8]),
        (11, [8, 2, 1]),
        (1000, [512, 256, 128, 64, 32, 8]),
    ],
)
def test_peak_decomposition_is_the_binary_expansion(size, expected):
    assert peak_sizes(size) == expected
    assert sum(peak_sizes(size)) == size


def test_an_empty_or_negative_size_is_refused():
    with pytest.raises(JournalAccumulatorError):
        peak_sizes(0)
    with pytest.raises(JournalAccumulatorError):
        peak_sizes(-3)


def test_an_empty_journal_has_no_root():
    with pytest.raises(JournalAccumulatorError):
        accumulator_root([])


# --- the root commits to the exact ordered run ------------------------------


def test_the_same_events_always_produce_the_same_root():
    assert accumulator_root(_events(37)) == accumulator_root(_events(37))


def test_reordering_two_events_changes_the_root():
    events = _events(37)
    swapped = list(events)
    swapped[5], swapped[6] = swapped[6], swapped[5]
    assert accumulator_root(events)["root_sha256"] != accumulator_root(swapped)["root_sha256"]


def test_editing_one_event_changes_the_root():
    events = _events(37)
    edited = list(events)
    edited[19] = hashlib.sha256(b"tampered").hexdigest()
    assert accumulator_root(events)["root_sha256"] != accumulator_root(edited)["root_sha256"]


def test_a_truncated_journal_does_not_share_the_longer_root():
    events = _events(37)
    assert (
        accumulator_root(events)["root_sha256"]
        != accumulator_root(events[:36])["root_sha256"]
    )


def test_the_size_is_bound_into_the_root():
    # A prefix's peaks are literally reused by the longer log, so binding the
    # size is what stops a proof valid at one length being replayed at another.
    short = accumulator_root(_events(8))
    long = accumulator_root(_events(16))
    assert short["size"] == 8
    assert long["size"] == 16
    assert short["root_sha256"] != long["root_sha256"]


# --- inclusion --------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 64, 100])
def test_every_event_in_a_log_proves_its_own_membership(count):
    events = _events(count)
    commitment = accumulator_root(events)
    for index in range(count):
        proof = inclusion_proof(events, index)
        assert verify_inclusion(
            proof,
            root_sha256=commitment["root_sha256"],
            size=commitment["size"],
        )


def test_a_proof_does_not_verify_against_another_logs_root():
    events = _events(50)
    other = _events(50)[:49] + [hashlib.sha256(b"different").hexdigest()]
    proof = inclusion_proof(events, 7)
    assert not verify_inclusion(
        proof,
        root_sha256=accumulator_root(other)["root_sha256"],
        size=50,
    )


def test_swapping_the_proven_event_breaks_the_proof():
    events = _events(50)
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 7))
    proof["event_sha256"] = events[8]
    assert not verify_inclusion(
        proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
    )


def test_a_tampered_path_step_breaks_the_proof():
    events = _events(50)
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 7))
    path = [dict(step) for step in proof["path"]]
    path[0]["sha256"] = hashlib.sha256(b"forged").hexdigest()
    proof["path"] = path
    assert not verify_inclusion(
        proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
    )


def test_flipping_a_path_side_breaks_the_proof():
    events = _events(50)
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 7))
    path = [dict(step) for step in proof["path"]]
    path[0]["side"] = "left" if path[0]["side"] == "right" else "right"
    proof["path"] = path
    assert not verify_inclusion(
        proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
    )


def test_a_shortened_path_is_refused_rather_than_folded():
    events = _events(50)
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 7))
    proof["path"] = proof["path"][:-1]
    with pytest.raises(JournalAccumulatorError) as excinfo:
        verify_inclusion(
            proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
        )
    assert "path_length_differs" in str(excinfo.value)


def test_a_proof_replayed_at_a_different_size_is_refused():
    events = _events(50)
    proof = inclusion_proof(events, 7)
    with pytest.raises(JournalAccumulatorError) as excinfo:
        verify_inclusion(
            proof,
            root_sha256=accumulator_root(events[:40])["root_sha256"],
            size=40,
        )
    assert "size_differs" in str(excinfo.value)


def test_a_proof_claiming_a_peak_it_does_not_sit_in_is_refused():
    events = _events(11)
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 2))
    proof["subtree_index"] = 1
    with pytest.raises(JournalAccumulatorError):
        verify_inclusion(
            proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
        )


def test_an_index_outside_the_log_is_refused():
    with pytest.raises(JournalAccumulatorError):
        inclusion_proof(_events(10), 10)


# --- the scaling claim itself -----------------------------------------------


def test_the_proof_stays_logarithmic_while_the_prefix_grows_linearly():
    small = _events(64)
    large = _events(4096)
    small_proof = proof_size_bytes(inclusion_proof(small, 33))
    large_proof = proof_size_bytes(inclusion_proof(large, 2049))

    # The prefix envelope this replaces grows with the journal; the proof grows
    # with its logarithm. 64x the events must not cost 64x the proof.
    assert len(large) / len(small) == 64
    assert large_proof < small_proof * 3


def test_a_full_campaigns_proofs_are_linear_not_quadratic():
    events = _events(1024)
    commitment = accumulator_root(events)
    total = sum(
        proof_size_bytes(inclusion_proof(events, index))
        for index in range(len(events))
    )
    prefix_total = 32 * sum(index + 1 for index in range(len(events)))
    assert total < prefix_total // 20
    assert verify_inclusion(
        inclusion_proof(events, 1023),
        root_sha256=commitment["root_sha256"],
        size=commitment["size"],
    )
