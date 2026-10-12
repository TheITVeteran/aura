"""Actual exterior pixels distinguish visible translation from moving windows."""
import numpy as np
import pytest

from core.perception.boundary_support_for_translation import boundaries_agree, visible_boundary

pytestmark = pytest.mark.unit


def _picture(x=20, y=18, width=12, height=14, *, edge_mix=1.):
    floor = np.array([30, 70, 100], dtype=np.float32)
    colour = np.array([210, 180, 70], dtype=np.float32)
    picture = np.broadcast_to(floor, (64, 80, 3)).copy()
    picture[y:y + height, x:x + width] = colour
    if edge_mix < 1.:
        mixed = floor + edge_mix * (colour - floor)
        picture[y:y + height, x] = mixed
        picture[y:y + height, x + width - 1] = mixed
        picture[y, x:x + width] = mixed
        picture[y + height - 1, x:x + width] = mixed
    support = np.ones((height, width), bool)
    return picture.astype(np.uint8), (x, y, x + width, y + height), support


def _boundary(painted):
    picture, box, support = painted
    return visible_boundary(picture, *box, support)


@pytest.mark.parametrize('dx,dy', [(7, 0), (-7, 0), (0, 6), (0, -6), (5, 4)])
def test_plain_body_translation_has_matching_actual_exterior_boundaries(dx, dy):
    assert boundaries_agree(_boundary(_picture()), _boundary(_picture(20 + dx, 18 + dy)))


def test_one_pixel_raster_extent_and_antialiasing_do_not_erase_visible_translation():
    before = _boundary(_picture(edge_mix=.65))
    after = _boundary(_picture(25, 21, width=13, height=13, edge_mix=.8))
    assert boundaries_agree(before, after)


def test_a_window_sliding_on_a_uniform_wall_has_no_boundary_certificate():
    picture = np.full((64, 80, 3), 180, np.uint8)
    support = np.ones((14, 12), bool)
    for x in (12, 20, 28):
        assert visible_boundary(picture, x, 18, x + 12, 32, support) is None
    assert not boundaries_agree(None, None)


def test_three_real_edges_cannot_certify_a_window_that_cuts_the_fourth():
    picture, box, support = _picture(width=30)
    left, top, _right, bottom = box
    assert visible_boundary(picture, left, top, left + 12, bottom, support[:, :12]) is None


def test_clipping_touching_and_occlusion_can_abstain():
    assert _boundary(_picture(x=1)) is None
    picture, box, support = _picture()
    left, top, right, bottom = box
    picture[top:bottom, right:right + 5] = picture[top, left]
    assert visible_boundary(picture, *box, support) is None
    picture, box, support = _picture()
    picture[top:bottom, right - 3:right] = picture[0, 0]
    assert visible_boundary(picture, *box, support) is None


def test_different_surroundings_do_not_require_a_fixed_background_colour():
    before = _boundary(_picture())
    picture, box, support = _picture(26, 24)
    body = picture[box[1]:box[3], box[0]:box[2]].copy()
    picture[:] = (80, 20, 40)
    picture[box[1]:box[3], box[0]:box[2]] = body
    assert boundaries_agree(before, visible_boundary(picture, *box, support))


def test_distinct_objects_and_changed_outlines_do_not_supply_correspondence():
    before = _boundary(_picture())
    picture, box, support = _picture(26, 24)
    picture[box[1]:box[3], box[0]:box[2]] = (40, 220, 220)
    assert not boundaries_agree(before, visible_boundary(picture, *box, support))
    assert not boundaries_agree(before, _boundary(_picture(width=20)))


@pytest.mark.parametrize('bad', ['missing', 'shape', 'nan', 'time_like_bounds'])
def test_invalid_observations_remain_unavailable(bad):
    picture, box, support = _picture()
    if bad == 'missing':
        support = None
    elif bad == 'shape':
        support = np.ones((2, 2), bool)
    elif bad == 'nan':
        picture = picture.astype(float)
        picture[box[1]:box[3], box[0]:box[2]] = np.nan
    else:
        box = tuple(float(value) for value in box)
    assert visible_boundary(picture, *box, support) is None


def test_registered_visible_boundary_invariant():
    from core.perception.boundary_support_for_translation import _visible_boundary_invariant

    assert _visible_boundary_invariant() == ()


def test_mask_holes_do_not_warn_or_manufacture_missing_edges():
    picture, box, support = _picture()
    support[:, 2] = False
    picture[box[1]:box[3], box[0] + 2] = picture[0, 0]
    measured = visible_boundary(picture, *box, support)
    assert measured is not None and boundaries_agree(measured, measured)
    support[:] = False
    assert visible_boundary(picture, *box, support) is None
