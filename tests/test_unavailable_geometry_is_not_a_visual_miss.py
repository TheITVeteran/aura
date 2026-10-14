"""The actual board and well routes cannot cache unavailable worker results."""
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import a_picture_examined_without_holding_the_loop as probing
from core.skills import screen_pursuit_as_it_happens as pursuit


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['board', 'well'])
async def test_the_route_keeps_unmeasured_geometry_available_and_capture_on_its_loop(monkeypatch, kind):
    from core.agency import playing_a_well
    from core.perception import a_board_seen

    release = threading.Event()
    owners = []
    thread = threading.get_ident()
    original = probing.PictureProbe

    def owner():
        made = original(wait_s=.01)
        owners.append(made)
        return made

    async def look():
        assert threading.get_ident() == thread
        return np.zeros((10, 10, 3), np.uint8), 100.

    def slow(*_args):
        assert release.wait(2.)
        return None

    monkeypatch.setattr(probing, 'PictureProbe', owner)
    reflexes = SimpleNamespace(keep={}, look=look, words=[], said_now='', ends_at=1e9,
                               no_well_here=False, waited_at_a_well=0.)
    if kind == 'board':
        monkeypatch.setattr(a_board_seen, 'a_board_in', slow)
        route = pursuit._played_a_board
    else:
        monkeypatch.setattr(playing_a_well, '_what_a_well_picture_shows', slow)
        route = pursuit._played_a_well
    try:
        assert not await route(reflexes, 100.)
        assert f'no_{kind}_on' not in reflexes.keep
        assert reflexes.keep[f'{kind}_looked_for_at'] == -1e9
    finally:
        release.set()
        for made in owners:
            await made.close()
            if made._operation is not None:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(made._operation.raw)), 2.)


@pytest.mark.asyncio
async def test_closing_the_actual_hand_over_retires_its_geometry_worker():
    reflexes = pursuit.PlayingAsItHappens(object(), (0., 0., 1., 1.), '', 100.)
    owner = reflexes._board_probe = probing.PictureProbe()
    await reflexes.close()
    assert owner.status == 'closed' and reflexes._board_probe is None
    assert await owner.examine(lambda: 'late board') is probing.UNMEASURED
