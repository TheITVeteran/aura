"""Explicit tracker selection reaches the actual motion observer's worker."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime.flags import FlagKind, declare

_BACKEND = declare('AURA_PLAY_POINT_BACKEND', kind=FlagKind.STRING, default='classical',
    description='Select classical points, original continuity points or online_tapir for motion eyes',
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
    if name not in ('online_tapir', 'continuity_points') or name == 'online_tapir' and not _WEIGHTS.value():
        raise ValueError('motion eyes need an explicit supported backend and local checkpoint')
    from core.perception.learned_tracking.models import load_backend
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes

    weights, device = Path(_WEIGHTS.value()), _DEVICE.value()
    pixels = _PIXELS.value()
    if type(pixels) is not int or not 256 <= pixels <= 1_000_000:
        raise ValueError('point tracking needs a bounded source pixel budget')

    def teacher(*, fallback: bool = False) -> Any:
        if name == 'continuity_points':
            from core.perception.learned_tracking.continuity_backend import ContinuityBackend

            backend = ContinuityBackend()
        else:
            backend = load_backend(name, weights, device=device)
        return PointsForMotionEyes(backend, pixel_budget=pixels, fallback_when_busy=fallback)

    mode = _MODE.value()
    if mode not in ('direct', 'correcting'):
        raise ValueError('point observation needs a supported scheduling mode')
    from core.perception.learned_tracking.corrections_while_points_keep_moving import (
        CorrectionsWhilePointsMove,
    )

    if mode == 'direct':
        from core.runtime.optional_compute import OptionalComputeBusy

        try:
            eyes = WhatMoves(kinds=kinds, points=teacher(fallback=name == 'online_tapir'))
        except OptionalComputeBusy as why:
            points = CorrectionsWhilePointsMove(teacher, pixel_budget=pixels)
            points.correction_status = f'correction deferred: {why}'
            eyes = WhatMoves(kinds=kinds, points=points)
        eyes.point_tracking_requested = name
        return eyes
    eyes = WhatMoves(kinds=kinds, points=CorrectionsWhilePointsMove(teacher, pixel_budget=pixels))
    eyes.point_tracking_requested = name
    return eyes


def learned_points_requested() -> bool:
    return _BACKEND.value() != 'classical'
