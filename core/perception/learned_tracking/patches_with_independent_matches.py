"""Measure bounded RGB patch matches without learned weights or external tracker code."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class PatchMatch:
    x: float
    y: float
    score: float


def patch(pixels: np.ndarray, x: float, y: float, radius: int) -> np.ndarray | None:
    x, y = int(round(x)), int(round(y))
    if x - radius < 0 or y - radius < 0 or x + radius >= pixels.shape[1] or y + radius >= pixels.shape[0]:
        return None
    return pixels[y-radius:y+radius+1, x-radius:x+radius+1].astype(np.float32).copy()


def _scores(region: np.ndarray, template: np.ndarray) -> np.ndarray:
    size = template.shape[0]
    centered = template - template.mean(axis=(0, 1), keepdims=True)
    norm = float(np.sum(centered * centered))
    if norm < size * size * 3 * 25:
        return np.full((region.shape[0]-size+1, region.shape[1]-size+1), -1., np.float32)
    integral = np.pad(region.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    squared = np.pad((region * region).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    squares = squared[size:, size:] - squared[:-size, size:] - squared[size:, :-size] + squared[:-size, :-size]
    variance = np.maximum(0., squares - total * total / (size * size)).sum(axis=2)
    numerator = 0.
    for channel in range(3):
        values = region[..., channel]
        numerator = numerator + signal.correlate(values, centered[..., channel], mode='valid', method='fft')
    return np.clip(numerator / np.sqrt(np.maximum(1., variance * norm)), -1., 1.)


def locate(pixels: np.ndarray, templates: tuple[np.ndarray, ...], predicted: tuple[float, float],
           search: int, *, minimum: float = .86, angles: tuple[float, ...] = (0.,),
           radius: int | None = None) -> PatchMatch | None:
    """Repeated appearances abstain unless a unique current pixel match exists."""
    if not templates:
        return None
    radius = templates[0].shape[0] // 2 if radius is None else radius
    x, y = int(round(predicted[0])), int(round(predicted[1]))
    left, top = max(radius, x-search), max(radius, y-search)
    right, bottom = min(pixels.shape[1]-radius-1, x+search), min(pixels.shape[0]-radius-1, y+search)
    if left > right or top > bottom:
        return None
    region = pixels[top-radius:bottom+radius+1, left-radius:right+radius+1].astype(np.float32)
    # Turning a raster twice adds interpolation blur. Compare at the same
    # low-pass scale when checking rotation in either direction.
    rotating = any(angle != 0. for angle in angles)
    if rotating:
        region = ndimage.gaussian_filter(region, (.65, .65, 0.))
    best = np.full((bottom-top+1, right-left+1), -1., np.float32)
    for template in templates:
        if rotating:
            template = ndimage.gaussian_filter(template, (.65, .65, 0.))
        for angle in angles:
            turned = template if angle == 0. else ndimage.rotate(template, angle, axes=(1, 0), reshape=False,
                                                                order=1, mode='nearest')
            middle = turned.shape[0] // 2
            turned = turned[middle-radius:middle+radius+1, middle-radius:middle+radius+1]
            best = np.maximum(best, _scores(region, turned))
    row, col = np.unravel_index(int(np.argmax(best)), best.shape)
    score = float(best[row, col])
    if score < minimum:
        return None
    alternatives = best.copy()
    alternatives[max(0, row-2):row+3, max(0, col-2):col+3] = -1.
    if score - float(alternatives.max()) < .05:
        return None
    measured = patch(pixels, left+col, top+row, radius)
    # Correlation alone ignores constant channel offsets and can match a
    # differently coloured copy. Retained RGB means constrain that case.
    if measured is None or min(float(np.linalg.norm(measured.mean((0, 1))-t.mean((0, 1))))
                               for t in templates) > 65.:
        return None
    return PatchMatch(float(left+col), float(top+row), score)
