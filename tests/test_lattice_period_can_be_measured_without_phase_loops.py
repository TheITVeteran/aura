"""Grouping lattice phases keeps the pixel scores and bounds the number of median calls."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.perception import a_board_seen as board_seen
from core.perception.a_board_seen import _lattice_phase_invariant, _period, the_lattice

pytestmark = pytest.mark.unit


def _legacy_period(profile: np.ndarray, least: int, most: int) -> tuple[int, int, float]:
    """The spacing search before grouping phases, retained as an independent reference."""
    overall = float(profile.mean()) or 1.0
    found: list[tuple[int, int, float]] = []
    for spacing in range(max(4, least), max(5, most) + 1):
        best = (spacing, 0, 0.0)
        for phase in range(spacing):
            teeth = profile[phase::spacing]
            if teeth.size < 3:
                continue
            near = np.maximum(teeth, np.concatenate([profile[phase + 1::spacing], np.zeros(1)])[:teeth.size])
            score = float(np.median(near)) / overall
            if score > best[2]:
                best = (spacing, phase, score)
        found.append(best)
    if not found:
        return (0, 0, 0.0)
    top = max(score for _s, _p, score in found)
    return next(one for one in found if one[2] >= 0.85 * top)


def _assert_same(profile: np.ndarray, least: int, most: int) -> None:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _period(profile, least, most) == _legacy_period(profile, least, most)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.uint8, np.bool_])
def test_random_and_noncontiguous_profiles_keep_the_same_spacing_phase_and_score(dtype):
    rng = np.random.default_rng(10132026)
    for size in [0, 1, 7, 8, 11, 12, 13, 17, 31, 59, 65, 101, 191]:
        raw = rng.integers(0, 40, size * 2).astype(dtype)
        for profile in (raw[:size], raw[::2]):
            for least, most in [(4, 14), (11, 7), (size // 19, size // 3)]:
                _assert_same(profile, least, most)


@pytest.mark.parametrize("size", [9, 13, 17, 29, 63, 99, 192])
def test_signed_profiles_and_exact_mean_cancellation_keep_the_same_result(size):
    rng = np.random.default_rng(size)
    for profile in (rng.uniform(-70, 120, size).astype(np.float32),
                    rng.integers(-9, 10, size).astype(np.float64),
                    np.tile([-1., 1.], size)):
        _assert_same(profile, 4, min(size + 5, 40))


@pytest.mark.parametrize("profile", [
    np.array([np.nan, 1., 2., 0.] * 11),
    np.array([np.inf, 1., -2., 0.] * 11),
    np.array([-np.inf, 1., -2., 0.] * 11),
    np.full(33, np.nan),
])
def test_nonfinite_scores_keep_the_original_strict_positive_selection(profile):
    _assert_same(profile, 4, 14)


def test_the_first_phase_wins_a_tie_and_the_short_phases_gain_no_teeth():
    assert _period(np.ones(31, np.float32), 4, 12) == (4, 0, 1.0)
    assert _period(np.ones(7, np.float32), 4, 12) == (4, 0, 0.0)
    assert _period(np.arange(12, dtype=np.float32), 12, 7) == (0, 0, 0.0)


def test_the_final_pixel_keeps_its_zero_neighbor():
    profile = np.array([-1., -1., -5., -5., -2., -2., -5., -5., -3.], np.float32)
    assert _period(profile, 4, 4) == (4, 0, -1. / float(profile.mean()))


def test_the_smallest_nearly_best_spacing_wins():
    profile = np.array([5., 0., 0., 0.] * 12)
    assert _period(profile, 4, 16) == (4, 0, 4.0)


def test_each_spacing_uses_at_most_two_grouped_medians(monkeypatch):
    profile = np.random.default_rng(9).uniform(0, 100, 959).astype(np.float32)
    expected = _legacy_period(profile, 50, 320)
    calls = []
    original = np.median

    def counted(values, *args, **kwargs):
        calls.append(values.shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(board_seen.np, "median", counted)
    assert _period(profile, 50, 320) == expected
    wanted = sum(int(profile.size // spacing >= 3)
                 + int(bool(profile.size % spacing) and profile.size // spacing + 1 >= 3)
                 for spacing in range(50, 321))
    assert len(calls) == wanted
    assert all(len(shape) == 2 for shape in calls)


def _invented_board(rows: int, columns: int, square: int, ruled: bool) -> np.ndarray:
    inset = 11
    picture = np.full((rows * square + 2 * inset, columns * square + 2 * inset, 3), 67, np.uint8)
    for row in range(rows):
        for column in range(columns):
            colour = 33 if (row + column) % 2 else 201
            y0, x0 = inset + row * square, inset + column * square
            picture[y0:y0 + square, x0:x0 + square] = 151 if ruled else colour
    if ruled:
        for row in range(rows + 1):
            at = inset + row * square
            picture[at:at + 2, inset:inset + columns * square] = 0
        for column in range(columns + 1):
            at = inset + column * square
            picture[inset:inset + rows * square, at:at + 2] = 0
    return picture


@pytest.mark.parametrize("rows,columns,square", [(3, 3, 13), (8, 8, 30), (12, 17, 6), (19, 19, 13)])
@pytest.mark.parametrize("ruled", [False, True])
def test_invented_boards_retain_the_original_lattice(rows, columns, square, ruled, monkeypatch):
    picture = _invented_board(rows, columns, square, ruled)
    actual = the_lattice(picture)
    monkeypatch.setattr(board_seen, "_period", _legacy_period)
    assert actual == the_lattice(picture)


@pytest.mark.parametrize("ruled", [False, True])
def test_an_invented_eight_by_eight_board_is_still_detected(ruled):
    assert the_lattice(_invented_board(8, 8, 30, ruled)) == (10., 10., 30., 30., 8, 8)


def test_random_and_blank_pictures_keep_the_original_result(monkeypatch):
    rng = np.random.default_rng(71)
    pictures = [rng.integers(0, 256, (144, 240, 3), dtype=np.uint8),
                np.full((144, 240, 3), 77, np.uint8)]
    actual = [the_lattice(picture) for picture in pictures]
    monkeypatch.setattr(board_seen, "_period", _legacy_period)
    assert actual == [the_lattice(picture) for picture in pictures]


def test_the_registered_lattice_phase_invariant_checks_its_canaries():
    assert _lattice_phase_invariant() == ()
