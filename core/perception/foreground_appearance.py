"""Exclude a measured flat exterior from an object's bounded appearance witness."""
from __future__ import annotations

import numpy as np

from core.verify.invariants import invariant


def valid_mask(mask: bytes, edge: int) -> bool:
    """A mask retains substantial foreground support and has exactly one bit per pixel."""
    return (type(edge) is int and 4 <= edge <= 64 and isinstance(mask, bytes) and len(mask) == edge * edge and set(mask) <= {0, 1}
            and .20 <= sum(mask) / len(mask) <= .85)


def foreground_mask(appearance: bytes, edge: int) -> bytes:
    """Infer only a consistent edge-connected flat background; otherwise retain the whole patch."""
    if type(edge) is not int or not 4 <= edge <= 64 or not isinstance(appearance, bytes) or len(appearance) != edge * edge * 3:
        return b""
    pixels = np.frombuffer(appearance, np.uint8).reshape(edge, edge, 3).astype(np.int16)
    boundary = np.concatenate((pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1]))
    colours, counts = np.unique(boundary // 8, axis=0, return_counts=True)
    dominant = colours[int(counts.argmax())]
    group = np.all(boundary // 8 == dominant, axis=1)
    if float(group.mean()) < .80:
        return b""
    background = np.median(boundary[group], axis=0)
    alike = np.max(np.abs(pixels - background), axis=2) <= 12
    exterior = np.zeros((edge, edge), dtype=bool)
    todo = [(r, c) for r in range(edge) for c in range(edge)
            if (r in (0, edge - 1) or c in (0, edge - 1)) and alike[r, c]]
    while todo:
        r, c = todo.pop()
        if exterior[r, c]:
            continue
        exterior[r, c] = True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < edge and 0 <= nc < edge and alike[nr, nc] and not exterior[nr, nc]:
                todo.append((nr, nc))
    mask = (~exterior).astype(np.uint8).tobytes()
    return mask if valid_mask(mask, edge) else b""


@invariant("perception.foreground_masks_require_measured_support", scope="perception",
           owner="core/perception/foreground_appearance.py", observational=False)
def _foreground_mask_invariant() -> tuple:
    picture = np.full((16, 16, 3), 240, dtype=np.uint8)
    picture[3:13, 3:13] = (30, 100, 180)
    picture[7:9, 7:9] = 240  # enclosed highlights retain their evidence
    mask = foreground_mask(picture.tobytes(), 16)
    assert valid_mask(mask, 16)
    assert mask[0] == 0 and mask[7 * 16 + 7] == 1
    assert not valid_mask(bytes(256), 16)
    assert not valid_mask(bytes([1] * 256), 16)
    assert not foreground_mask(b"", 16)
    return ()
