"""A word read earlier may return only with its own fresh, unchanged drawing."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.observed_transfer import ScreenReading, freeze_screen, make_snapshot
from core.perception.word_labels_seen_again import (
    WordLabelsSeenAgain,
    _visible_word_labels_invariant,
)
from core.skills.visible_words_from_their_pixels import current_visible_words

pytestmark = pytest.mark.unit


def _reading(at=1., text='Continue', picture=None, **kwargs):
    if picture is None:
        picture = np.random.default_rng(73).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    return ScreenReading({'layout': [{'text': text, 'x': .25, 'y': .25, 'width': .5, 'height': .25}],
                          'surface_id': 'view', 'bounds': [0, 0, 96, 64], 'capture_at': at,
                          'settled': True, **kwargs}, picture_snapshot=make_snapshot(picture))


def test_word_pixel_invariant():
    assert _visible_word_labels_invariant() == ()


@pytest.mark.parametrize('change', ['surface', 'bounds', 'work_epoch', 'occluded', 'stale', 'future', 'pixels'])
def test_changed_or_unmeasured_view_cannot_restore_a_label(change):
    screen = freeze_screen(_reading())
    memory = WordLabelsSeenAgain()
    memory.remember(screen, now=1.1)
    changed = replace(screen, regions=(), capture_at=2., epoch='2')
    if change == 'surface':
        changed = replace(changed, surface=(('surface_id', 'other'),))
    elif change == 'bounds':
        changed = replace(changed, bounds=(0, 0, 95, 64))
    elif change == 'work_epoch':
        changed = replace(changed, work_epoch='other')
    elif change == 'occluded':
        changed = replace(changed, occluded=True)
    elif change == 'stale':
        changed = replace(changed, capture_at=-10.)
    elif change == 'future':
        changed = replace(changed, capture_at=3.)
    else:
        changed = replace(changed, picture=make_snapshot(np.zeros((64, 96, 3), np.uint8)))
    assert memory.recovered(changed, now=2.1) == []


def test_an_unsettled_capture_needs_a_second_distinct_fresh_picture():
    screen = replace(freeze_screen(_reading()), settled=False)
    memory = WordLabelsSeenAgain()
    memory.remember(screen, now=1.1)
    assert not memory.anchors
    memory.remember(screen, now=1.1, again=screen)
    assert not memory.anchors
    memory.remember(screen, now=1.1, again=replace(screen, capture_at=1.05, epoch='second'))
    assert len(memory.anchors) == 1


def test_a_damaged_reading_is_recovered_before_menu_interpretation(monkeypatch):
    from core.skills import visible_words_from_their_pixels as labels

    monkeypatch.setattr(labels.time, 'time', lambda: 100.)
    monkeypatch.setattr(labels.time, 'monotonic', lambda: 10.)
    reflex = SimpleNamespace(page=SimpleNamespace(url='local', viewport_size={'width': 96, 'height': 64}),
                             band=(0., 0., 1., 1.), keep={})
    first = _reading(100., on_her_page=True, read_within='the page')
    assert current_visible_words(reflex, first) is first
    monkeypatch.setattr(labels.time, 'time', lambda: 101.)
    monkeypatch.setattr(labels.time, 'monotonic', lambda: 11.)
    broken = _reading(101., text='C0nt1nue', on_her_page=True, read_within='the page')
    fixed = current_visible_words(reflex, broken)
    assert fixed['layout'][0]['text'] == 'continue'
    assert broken['layout'][0]['text'] == 'C0nt1nue'
    assert len(reflex.keep['visible_word_labels'].anchors) == 1


def test_recovery_never_refreshes_a_labels_expiry():
    memory = WordLabelsSeenAgain()
    memory.remember(freeze_screen(_reading()), now=1.1)
    for at in (2., 50., 110.):
        assert memory.recovered(freeze_screen(_reading(at, text='C0nt1nue')), now=at + .1)
    assert not memory.recovered(freeze_screen(_reading(122., text='C0nt1nue')), now=122.1)
