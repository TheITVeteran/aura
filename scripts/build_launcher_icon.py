#!/usr/bin/env python3
"""Generate Aura's launcher icon assets."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "interface" / "static"


def _vertical_gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    mask = Image.linear_gradient("L").resize((size, size))
    return Image.composite(
        Image.new("RGBA", (size, size), bottom + (255,)),
        Image.new("RGBA", (size, size), top + (255,)),
        mask,
    )


def _radial_mask(size: int, blur: int = 0) -> Image.Image:
    mask = Image.radial_gradient("L").resize((size, size))
    mask = ImageChops.invert(mask)
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def _ellipse_bounds(cx: int, cy: int, radius: int) -> tuple[int, int, int, int]:
    return (cx - radius, cy - radius, cx + radius, cy + radius)


# ── the neuron mark ────────────────────────────────────────────────────────
#
# The mark is authored once, in the same 200x200 space as the web splash
# (interface/static/index.html) and the native launcher (scripts/AuraLauncher.swift),
# so the three surfaces stay the same drawing rather than three drifting
# lookalikes. `_Mark` maps that space onto whatever pixel size is requested.


class _Mark:
    """Maps the shared 200x200 mark space onto the icon canvas."""

    def __init__(self, size: int, cx: int, cy: int, extent: float) -> None:
        self.scale = (size * extent) / 200.0
        self.cx = cx
        self.cy = cy

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return (self.cx + (x - 100.0) * self.scale, self.cy + (y - 100.0) * self.scale)

    def px(self, v: float) -> int:
        """A mark-space length in canvas pixels, never rounded away to nothing."""
        return max(1, int(round(v * self.scale)))


def _pixel(draw: ImageDraw.ImageDraw, mark: _Mark, x: float, y: float, side: float, color) -> None:
    """A square sprite node — the arcade vocabulary, never a soft dot."""
    px, py = mark.pt(x, y)
    half = (side * mark.scale) / 2.0
    draw.rectangle((px - half, py - half, px + half, py + half), fill=color)


def _fibre(draw: ImageDraw.ImageDraw, mark: _Mark, points, width: float, color) -> None:
    draw.line([mark.pt(*p) for p in points], fill=color, width=mark.px(width), joint="curve")


def _myelinated(draw: ImageDraw.ImageDraw, mark: _Mark, points, width: float, color,
                dash: float = 9.0, gap: float = 4.5) -> None:
    """A dashed polyline: the segmented armour that reads as myelin sheath.

    PIL has no dash support, so walk the polyline by arc length and stamp the
    lit stretches. Walking the real geometry (rather than dashing each straight
    segment independently) keeps the rhythm continuous across the axon's knee.
    """
    span = dash + gap
    carry = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if seg <= 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        pos = 0.0
        while pos < seg:
            phase = (carry + pos) % span
            if phase < dash:
                lit = min(dash - phase, seg - pos)
                a = mark.pt(x0 + ux * pos, y0 + uy * pos)
                b = mark.pt(x0 + ux * (pos + lit), y0 + uy * (pos + lit))
                draw.line([a, b], fill=color, width=mark.px(width))
                pos += lit
            else:
                pos += span - phase
        carry = (carry + seg) % span


def _draw_neuron(canvas: Image.Image, size: int, cx: int, cy: int) -> Image.Image:
    """Soma, dendrites, myelinated axon, and travelling spikes — in that order."""
    mark = _Mark(size, cx, cy, extent=0.76)

    cyan = (0, 255, 225, 255)
    cyan_dim = (0, 255, 225, 190)
    node = (143, 255, 240, 255)
    violet = (181, 92, 255, 255)
    bouton = (217, 179, 255, 255)
    spike = (255, 255, 255, 255)

    # Ambient bloom under the mark, so the fibres sit in light rather than on
    # flat black. Blurred on its own layer to keep the sprite edges hard.
    bloom = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bloom_draw = ImageDraw.Draw(bloom)
    for pts, width, color in (
        (((44, 52), (68, 72), (86, 88)), 5.0, (0, 255, 225, 120)),
        (((30, 92), (60, 96), (84, 98)), 5.0, (0, 255, 225, 110)),
        (((74, 26), (82, 58), (92, 84)), 5.0, (0, 255, 225, 115)),
        (((52, 142), (72, 124), (88, 110)), 5.0, (0, 255, 225, 105)),
        (((122, 118), (146, 150), (168, 162)), 7.0, (181, 92, 255, 150)),
    ):
        _fibre(bloom_draw, mark, pts, width, color)
    bloom_draw.ellipse(
        _ellipse_bounds(cx, cy, int(size * 0.20)), fill=(133, 72, 255, 96)
    )
    bloom = bloom.filter(ImageFilter.GaussianBlur(int(size * 0.028)))
    canvas = Image.alpha_composite(canvas, bloom)

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Dendrites, then their forks.
    _fibre(draw, mark, ((44, 52), (68, 72), (86, 88)), 3.2, cyan)
    _fibre(draw, mark, ((30, 92), (60, 96), (84, 98)), 3.0, cyan_dim)
    _fibre(draw, mark, ((74, 26), (82, 58), (92, 84)), 3.1, cyan)
    _fibre(draw, mark, ((52, 142), (72, 124), (88, 110)), 3.0, cyan_dim)
    for fork in (((68, 72), (58, 44)), ((60, 96), (34, 112)), ((82, 58), (104, 38))):
        _fibre(draw, mark, fork, 2.4, (0, 255, 225, 150))

    # Myelinated axon out to the terminal, then the two boutons.
    _myelinated(draw, mark, ((122, 118), (146, 150), (168, 162)), 5.0, violet)
    _fibre(draw, mark, ((168, 162), (180, 172)), 2.6, (181, 92, 255, 200))
    _fibre(draw, mark, ((168, 162), (182, 154)), 2.6, (181, 92, 255, 200))

    # Square terminal nodes.
    for x, y, side in ((43.5, 51.5, 11), (29.5, 91.5, 11), (73.5, 25.5, 11), (51.5, 141.5, 11)):
        _pixel(draw, mark, x, y, side, node)
    for x, y, side in ((56, 42, 8), (32, 110, 8), (102, 36, 8)):
        _pixel(draw, mark, x, y, side, (143, 255, 240, 225))
    for x, y, side in ((181, 173, 10), (183, 155, 10), (168, 162, 12)):
        _pixel(draw, mark, x, y, side, bouton)

    # Spikes caught mid-flight on their fibres.
    _pixel(draw, mark, 68, 72, 9, spike)
    _pixel(draw, mark, 146, 150, 10, spike)

    # Soma: a chunky hexagon, not a sphere.
    hexagon = [
        mark.pt(*p)
        for p in ((100, 68), (126, 83), (126, 113), (100, 128), (74, 113), (74, 83))
    ]
    draw.polygon(hexagon, fill=(138, 62, 220, 255), outline=cyan, width=mark.px(3.4))
    _pixel(draw, mark, 100, 98, 18, (255, 255, 255, 255))

    return Image.alpha_composite(canvas, layer)


def _scanlines(canvas: Image.Image, size: int, inset: int, corner: int) -> Image.Image:
    """CRT banding, clipped to the plate so it never reads as a floating box."""
    lines = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    lines_draw = ImageDraw.Draw(lines)
    pitch = max(3, int(size * 0.012))
    thickness = max(1, pitch // 3)
    for y in range(0, size, pitch):
        lines_draw.rectangle((0, y, size, y + thickness), fill=(0, 0, 0, 62))
    plate = Image.new("L", (size, size), 0)
    ImageDraw.Draw(plate).rounded_rectangle(
        (inset, inset, size - inset, size - inset), radius=corner, fill=255
    )
    lines.putalpha(ImageChops.multiply(lines.getchannel("A"), plate))
    return Image.alpha_composite(canvas, lines)


def build_icon(size: int = 1024) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inset = int(size * 0.07)
    corner = int(size * 0.23)

    # Background plate
    bg = _vertical_gradient(size, (5, 3, 14), (18, 10, 30))
    bg_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(bg_mask).rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=corner,
        fill=255,
    )
    canvas = Image.composite(bg, canvas, bg_mask)

    # Vignette
    vignette = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vignette_mask = _radial_mask(size, blur=int(size * 0.015))
    vignette.putalpha(Image.eval(vignette_mask, lambda px: int(px * 0.42)))
    canvas = Image.alpha_composite(canvas, vignette)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=corner,
        outline=(105, 84, 164, 118),
        width=max(4, size // 170),
    )

    cx = size // 2
    cy = int(size * 0.515)

    canvas = _draw_neuron(canvas, size, cx, cy)
    canvas = _scanlines(canvas, size, inset, corner)

    return canvas


def save_png_variants(image: Image.Image) -> None:
    image.save(ROOT / "aura_icon.png", format="PNG")
    image.save(STATIC_DIR / "icon.png", format="PNG")
    image.resize((512, 512), Image.LANCZOS).save(STATIC_DIR / "icon-512.png", format="PNG")
    image.resize((192, 192), Image.LANCZOS).save(STATIC_DIR / "icon-192.png", format="PNG")


def save_icns(image: Image.Image) -> None:
    image.save(ROOT / "aura_icon.icns")


def main() -> None:
    image = build_icon()
    save_png_variants(image)
    save_icns(image)
    print(f"Updated {ROOT / 'aura_icon.png'}")
    print(f"Updated {ROOT / 'aura_icon.icns'}")
    print(f"Updated {STATIC_DIR / 'icon.png'}")
    print(f"Updated {STATIC_DIR / 'icon-512.png'}")
    print(f"Updated {STATIC_DIR / 'icon-192.png'}")


if __name__ == "__main__":
    main()
