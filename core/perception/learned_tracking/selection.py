"""Explicit tracker selection reaches the actual motion observer's worker."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime.flags import FlagKind, declare

_BACKEND = declare('AURA_PLAY_POINT_BACKEND', kind=FlagKind.STRING, default='classical',
    description='Select classical points, online_tapir or cotracker3 for real-time motion eyes',
    owner='core/perception/learned_tracking/selection.py')
_WEIGHTS = declare('AURA_PLAY_POINT_WEIGHTS', kind=FlagKind.STRING, default='',
    description='Local verified checkpoint for the explicitly selected point tracker',
    owner='core/perception/learned_tracking/selection.py')
_DEVICE = declare('AURA_PLAY_POINT_DEVICE', kind=FlagKind.STRING, default='cpu',
    description='Explicit compute device for the selected numerical point tracker',
    owner='core/perception/learned_tracking/selection.py')
_MODE = declare('AURA_PLAY_POINT_MODE', kind=FlagKind.STRING, default='correcting',
    description='Use learned background corrections or direct learned point observation',
    owner='core/perception/learned_tracking/selection.py')
_PIXELS = declare('AURA_PLAY_POINT_PIXEL_BUDGET', kind=FlagKind.INT, default=256_000,
    description='Bound source pixels retained for point correspondence before reduction',
    owner='core/perception/learned_tracking/selection.py')


def selected_motion_eyes(kinds: list[Any] | None = None) -> Any:
    """Construct on the actual observation worker; public mirrors retain no model."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    name = _BACKEND.value()
    if name == 'classical':
        return WhatMoves(kinds=kinds)
    if name not in ('online_tapir', 'cotracker3') or not _WEIGHTS.value():
        raise ValueError('motion eyes need an explicit supported backend and local checkpoint')
    from core.perception.learned_tracking.models import load_backend
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes

    weights, device = Path(_WEIGHTS.value()), _DEVICE.value()
    pixels = _PIXELS.value()
    if type(pixels) is not int or not 256 <= pixels <= 1_000_000:
        raise ValueError('point tracking needs a bounded source pixel budget')

    def teacher() -> Any:
        return PointsForMotionEyes(load_backend(name, weights, device=device), pixel_budget=pixels)

    if _MODE.value() == 'direct':
        return WhatMoves(kinds=kinds, points=teacher())
    if _MODE.value() != 'correcting':
        raise ValueError('point observation needs a supported scheduling mode')
    from core.perception.learned_tracking.corrections_while_points_keep_moving import (
        CorrectionsWhilePointsMove,
    )

    return WhatMoves(kinds=kinds, points=CorrectionsWhilePointsMove(teacher, pixel_budget=pixels,
                                                                sequence_required=name == 'cotracker3'))


def learned_points_requested() -> bool:
    return _BACKEND.value() != 'classical'
