"""Content must never depend on a CSS animation having run to be visible.

The failure this locks out, in Bryan's words: "another claude session broke
Aura's UI. Nothing is showing." Then, minutes later, "the neural steam appeared
for a moment and then went away again", "3 dots arent appearing either", and
finally "now it's back".

The mechanism was one CSS idiom, repeated:

    @keyframes staggerIn { from { opacity: 0 } to { opacity: 1 } }
    .sidebar { animation: staggerIn 0.6s ease-out both; }

`animation-fill-mode: both` fills BACKWARDS as well as forwards, so before the
animation runs the element holds the 0% keyframe — opacity 0. An animation is
not a guarantee that it runs: a hidden or occluded window, a suspended
compositor, `animation-play-state: paused`, or a `display: none` ancestor at
creation time all leave the element pinned on that first keyframe. Measured
live on the running instance: with `document.hidden === true`, 81 of 81
`.msg`/`.thought-card` elements computed to `opacity: 0` while the DOM was
fully populated. Her answers existed and were not on screen.

That idiom sat on `header`, `.chat-panel` and `.sidebar` — the entire product,
chat plus all six panels — plus every message, every neural-feed card, the
splash button and the thinking dots. It is also why a frame-rate governor that
merely paused animations (d71dda1cd, reverted in e513112c6) could blank the
whole UI: the governor tripped the trap, it did not create it.

So: an entry animation may MOVE something in. It may not be the only reason
that something is visible.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "interface" / "static"

# `animation:` shorthand carrying a backwards-filling fill-mode.
_ANIMATION_SHORTHAND = re.compile(r"animation\s*:\s*([^;{}]+)", re.IGNORECASE)
_BACKWARDS_FILL = re.compile(r"(?<![\w-])(both|backwards)(?![\w-])", re.IGNORECASE)
_KEYFRAMES = re.compile(r"@keyframes\s+([\w-]+)\s*\{", re.IGNORECASE)

# Keywords that appear in the `animation` shorthand and are never a name.
_SHORTHAND_KEYWORDS = {
    "normal", "reverse", "alternate", "alternate-reverse",
    "none", "forwards", "backwards", "both",
    "running", "paused",
    "infinite",
    "linear", "ease", "ease-in", "ease-out", "ease-in-out", "step-start",
    "step-end",
}


def _iter_css() -> list[Path]:
    files = sorted(STATIC_DIR.glob("*.css"))
    assert files, f"no stylesheets found under {STATIC_DIR}"
    return files


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _balanced_block(css: str, open_brace_index: int) -> str:
    """Return the text inside the block whose opening brace is at the index."""
    depth = 0
    for pos in range(open_brace_index, len(css)):
        char = css[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace_index + 1:pos]
    return ""


def _invisible_start_keyframes(css: str) -> set[str]:
    """Names of @keyframes whose first frame makes the element invisible."""
    invisible: set[str] = set()
    for match in _KEYFRAMES.finditer(css):
        name = match.group(1)
        body = _balanced_block(css, match.end() - 1)
        # Each frame is `<selector> { ... }`; we only care about 0%/from.
        for frame in re.finditer(r"([^{}]+)\{([^{}]*)\}", body):
            selectors = {s.strip().lower() for s in frame.group(1).split(",")}
            if not ({"0%", "from"} & selectors):
                continue
            declarations = frame.group(2)
            for opacity in re.finditer(
                r"opacity\s*:\s*([0-9.]+)", declarations, re.IGNORECASE
            ):
                if float(opacity.group(1)) == 0.0:
                    invisible.add(name)
    return invisible


def _animation_names(shorthand: str) -> set[str]:
    names = set()
    for token in re.split(r"[\s,]+", shorthand.strip()):
        token = token.strip()
        if not token or token.lower() in _SHORTHAND_KEYWORDS:
            continue
        # Skip times (0.6s, 240ms), functions (cubic-bezier(...)), var(), counts.
        if re.fullmatch(r"[\d.]+m?s", token, re.IGNORECASE):
            continue
        if "(" in token or ")" in token:
            continue
        if re.fullmatch(r"[\d.]+", token):
            continue
        names.add(token)
    return names


def _offenders(css: str) -> list[tuple[str, str]]:
    """(animation-name, shorthand) pairs that are invisible until they run."""
    clean = _strip_comments(css)
    invisible = _invisible_start_keyframes(clean)
    found: list[tuple[str, str]] = []
    for match in _ANIMATION_SHORTHAND.finditer(clean):
        shorthand = match.group(1)
        if not _BACKWARDS_FILL.search(shorthand):
            continue
        for name in _animation_names(shorthand) & invisible:
            found.append((name, " ".join(shorthand.split())))
    return found


def test_detector_catches_the_real_pattern():
    """The gate must not be able to pass by failing to look.

    This is the exact shape that blanked the UI; if the detector cannot see
    it, the assertion below is worthless.
    """
    bad = """
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sidebar { animation: staggerIn 0.6s ease-out 0.2s both; }
    """
    assert _offenders(bad) == [("staggerIn", "staggerIn 0.6s ease-out 0.2s both")]

    good = """
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sidebar { animation: staggerIn 0.6s ease-out; }
    """
    assert _offenders(good) == []

    # A backwards fill over a keyframe that starts visible is fine.
    harmless = """
    @keyframes slide { from { transform: translateY(6px); } to { transform: none; } }
    .card { animation: slide 0.2s ease-out both; }
    """
    assert _offenders(harmless) == []


def test_no_stylesheet_makes_content_invisible_until_an_animation_runs():
    failures: list[str] = []
    for path in _iter_css():
        for name, shorthand in _offenders(path.read_text(encoding="utf-8")):
            failures.append(
                f"{path.relative_to(PROJECT_ROOT)}: `animation: {shorthand}` fills "
                f"backwards from @keyframes {name}, whose first frame is "
                f"opacity 0 — this element is invisible until the animation "
                f"runs, and an animation is not a guarantee"
            )

    assert not failures, (
        "content may not depend on a CSS animation to be visible:\n  "
        + "\n  ".join(failures)
    )


def _keyframe_bodies(css: str) -> dict[str, str]:
    return {
        match.group(1): _balanced_block(css, match.end() - 1)
        for match in _KEYFRAMES.finditer(css)
    }


# The elements that ARE the product. If one of these is invisible, Aura looks
# broken or absent, so their entrances may move them but may not fade them.
LOAD_BEARING = [
    ("aura.css", r"header,\s*\.chat-panel,\s*\.sidebar\s*\{", "the shell: chat and all six panels"),
    ("aura.css", r"\.splash-start-btn\s*\{", "the control that gets past the splash"),
    ("presence_design.css", r"\.msg\s*\{", "every message, including Aura's answers"),
    ("presence_design.css", r"\.thought-card,\s*\.neural-line\s*\{", "the live neural feed"),
]


def test_motion_not_visibility_check_would_catch_a_regression():
    """Prove the gate below can fail, using the pre-fix rule verbatim."""
    regressed = _strip_comments("""
    header,
    .chat-panel,
    .sidebar { animation: staggerIn 0.6s ease-out; }
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(12px); }
        to { transform: translateY(0); }
    }
    """)
    rule = re.search(r"header,\s*\.chat-panel,\s*\.sidebar\s*\{([^}]*)\}", regressed)
    assert rule
    names = _animation_names(_ANIMATION_SHORTHAND.search(rule.group(1)).group(1))
    bodies = _keyframe_bodies(regressed)
    assert names == {"staggerIn"}
    assert re.search(r"opacity\s*:", bodies["staggerIn"], re.IGNORECASE), (
        "the gate's opacity probe no longer sees an opacity ramp it must catch"
    )

    # ...and passes on the shipped, motion-only form.
    fixed = _keyframe_bodies(_strip_comments("""
    @keyframes staggerIn {
        from { transform: translateY(12px); }
        to { transform: translateY(0); }
    }
    """))
    assert not re.search(r"opacity\s*:", fixed["staggerIn"], re.IGNORECASE)


def test_load_bearing_ui_animates_motion_not_visibility():
    """Removing the backwards fill was not enough — the ramp itself is the trap.

    Measured on the live instance with the window not visible:
    `animation-fill-mode: none`, animation *running*, and the shell computing
    to `opacity: 0`. An animation that ramps opacity from 0 hides its element
    for as long as its clock is stopped, whichever fill-mode is set. So for
    these selectors the entrance must not touch opacity at all.
    """
    sources = {
        name: _strip_comments((STATIC_DIR / name).read_text(encoding="utf-8"))
        for name in {entry[0] for entry in LOAD_BEARING}
    }
    keyframes = {
        name: _keyframe_bodies(css) for name, css in sources.items()
    }

    failures: list[str] = []
    for filename, selector, why in LOAD_BEARING:
        css = sources[filename]
        rule = re.search(selector + r"([^}]*)\}", css)
        assert rule, f"{filename}: no rule matching {selector} — re-check this contract"

        shorthand = _ANIMATION_SHORTHAND.search(rule.group(1))
        if not shorthand:
            continue

        for animation in _animation_names(shorthand.group(1)):
            body = keyframes[filename].get(animation)
            if body is None:
                continue
            if re.search(r"opacity\s*:", body, re.IGNORECASE):
                failures.append(
                    f"{filename}: {selector} animates @keyframes {animation}, "
                    f"which sets opacity — {why} would be invisible whenever "
                    f"that animation's clock is stopped"
                )

    assert not failures, "\n  ".join(["load-bearing UI fades:"] + failures)


def test_thinking_dots_are_visible_when_not_blinking():
    """`blink` rests at opacity 0 for 80% of its cycle; the dots must not."""
    css = _strip_comments((STATIC_DIR / "aura.css").read_text(encoding="utf-8"))

    rule = re.search(r"\.typing-dots\s+span\s*\{([^}]*)\}", css)
    assert rule, "the .typing-dots span rule is gone — re-check this contract"
    assert not _BACKWARDS_FILL.search(rule.group(1)), (
        "the thinking dots must not fill backwards from blink's 0% frame"
    )

    frames = re.search(r"@keyframes\s+blink\s*\{", css)
    assert frames, "@keyframes blink is gone — re-check this contract"
    body = _balanced_block(css, frames.end() - 1)
    resting = re.search(r"0%\s*,\s*80%\s*,\s*100%\s*\{([^}]*)\}", body)
    assert resting, "blink no longer has a 0%/80%/100% resting frame"
    opacity = re.search(r"opacity\s*:\s*([0-9.]+)", resting.group(1))
    assert opacity and float(opacity.group(1)) > 0, (
        "blink's resting frame is opacity 0, so a dot paused there disappears"
    )
