"""Explicit tracker selection reaches the actual motion observer's worker."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime.flags import FlagKind, declare

_BACKEND = declare('AURA_PLAY_POINT_BACKEND', kind=FlagKind.STRING, default='classical',
    description='Select classical points or the measured online_tapir adapter for real-time motion eyes',
    owner='core/perception/learned_tracking/selection.py')
_WEIGHTS = declare('AURA_PLAY_POINT_WEIGHTS', kind=FlagKind.STRING, default='',
    description='Local verified checkpoint for the explicitly selected point tracker',
    owner='core/perception/learned_tracking/selection.py')
_DEVICE = declare('AURA_PLAY_POINT_DEVICE', kind=FlagKind.STRING, default='cpu',
    description='Explicit compute device for the selected numerical point tracker',
    owner='core/perception/learned_tracking/selection.py')


def selected_motion_eyes(kinds: list[Any] | None = None) -> Any:
    """Construct on the actual observation worker; public mirrors retain no model."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    name = _BACKEND.value()
    if name == 'classical':
        return WhatMoves(kinds=kinds)
    if name != 'online_tapir' or not _WEIGHTS.value():
        raise ValueError('motion eyes need an explicit supported backend and local checkpoint')
    from core.perception.learned_tracking.models import load_backend
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes

    backend = load_backend(name, Path(_WEIGHTS.value()), device=_DEVICE.value())
    return WhatMoves(kinds=kinds, points=PointsForMotionEyes(backend))


def learned_points_requested() -> bool:
    return _BACKEND.value() != 'classical'
