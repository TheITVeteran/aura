"""An early visual miss cannot hide changed geometry behind unchanged words."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.visual_probe_cache import (
    VisualProbeMiss,
    VisualProbeWitness,
    _visual_probe_miss_invariant,
)

pytestmark = pytest.mark.unit


def test_rejection_requires_all_pixels_and_the_same_interpretation_and_surface():
    picture = np.zeros((20, 20, 3), dtype=np.uint8)
    def witness(image=picture, context=("document", "viewport", 1), inputs="find a table"):
        return VisualProbeWitness.measure(image, context=context, inputs=inputs)
    original = witness()
    assert original is not None
    miss = VisualProbeMiss(original, 100.0)
    assert miss.applies(witness(picture.copy()), now=101.0)
    picture[19, 19, 2] = 1  # a pixel outside the centre and common sampling grids
    assert not miss.applies(witness(), now=101.0)
    picture[19, 19, 2] = 0
    assert not miss.applies(witness(context=("other document", "viewport", 1)), now=101.0)
    assert not miss.applies(witness(context=("document", "viewport", 2)), now=101.0)
    assert not miss.applies(witness(inputs="find a ruled table"), now=101.0)
    for now in (99.0, 112.0, float("nan"), float("inf")):
        assert not miss.applies(original, now=now)
    assert not miss.applies(None, now=101.0)
    assert _visual_probe_miss_invariant() == ()


@pytest.mark.parametrize("picture", [None, [], np.zeros((0, 5, 3), dtype=np.uint8),
                                    np.full((5, 5, 3), float("nan")), np.zeros((5, 5, 2), dtype=np.uint8)])
def test_unmeasured_pixels_cannot_create_a_rejection(picture):
    assert VisualProbeWitness.measure(picture, context=("surface",), inputs="") is None


@pytest.mark.asyncio
async def test_real_board_detector_retries_changed_pixels_with_unchanged_words(monkeypatch):
    from core.agency import playing_a_board
    from core.skills import screen_pursuit_as_it_happens as pursuit

    picture = np.full((300, 300, 3), 235, np.uint8)
    played = []

    async def look():
        return picture, 100.0

    async def play(*args, **kwargs):
        played.append(True)
        return {"moves": 0, "ended": "out of time"}

    monkeypatch.setattr(playing_a_board, "play_a_board", play)
    reflexes = SimpleNamespace(keep={}, look=look, words=["YOUR TURN"], said_now="YOUR TURN", stretches=[],
                               ends_at=1e9, over_because="", page=object(), band=(0, 0, 1, 1))
    assert not await pursuit._played_a_board(reflexes, 100.0)
    assert isinstance(reflexes.keep["no_board_on"], VisualProbeMiss)
    for k in range(4):
        picture[:, min(299, k * 100):min(300, k * 100 + 3)] = 90
        picture[min(299, k * 100):min(300, k * 100 + 3), :] = 90
    assert await pursuit._played_a_board(reflexes, 104.0)
    assert played and "no_board_on" not in reflexes.keep


@pytest.mark.asyncio
async def test_well_miss_uses_the_last_watched_pixels_then_retries_new_pixels(monkeypatch):
    from core.agency import playing_a_well
    from core.skills import screen_pursuit_as_it_happens as pursuit

    picture = np.zeros((10, 10, 3), dtype=np.uint8)
    watched = []

    async def look():
        return picture.copy(), 100.0

    async def no_well(look, known=None, seen=None):
        watched.append(True)
        await look()
        picture[3, 3] += 1  # it changes while the temporal probe watches
        await look()
        return None

    monkeypatch.setattr(playing_a_well, "a_well_with_a_piece_coming_down", no_well)
    reflexes = SimpleNamespace(no_well_here=False, keep={}, look=look, words=["SCORE 0"], said_now="SCORE 0",
                               waited_at_a_well=0.0)
    assert not await pursuit._played_a_well(reflexes, 100.0)
    assert not await pursuit._played_a_well(reflexes, 104.0)
    assert len(watched) == 1
    picture[2, 2] = 100
    assert not await pursuit._played_a_well(reflexes, 108.0)
    assert len(watched) == 2


@pytest.mark.asyncio
async def test_absent_capture_does_not_keep_a_board_rejection(monkeypatch):
    from core.skills import screen_pursuit_as_it_happens as pursuit

    async def look():
        return None

    reflexes = SimpleNamespace(keep={}, look=look, words=[], said_now="", ends_at=1e9)
    assert not await pursuit._played_a_board(reflexes, 100.0)
    assert reflexes.keep["no_board_on"] is None
