"""What window is in front, cheaply, without reading it.

This exists so the privacy rule can run BEFORE a capture. Every path that
reads the screen needs to know whether the foreground is private, and it
needs to know without looking at the screen — a check that has to capture in
order to decide whether capturing was allowed has already done the thing it
was deciding about.

macOS gives the frontmost application name and its window title through the
Accessibility API without any screen content, so the question "may I look?"
is answerable for the price of one attribute read.

Deliberately tiny and dependency-light: it is called on the hot path of every
screen read, including the ambient loop's tick, and a heavyweight import here
would make the cheap gate expensive enough to skip.
"""

from __future__ import annotations

import subprocess

from core.runtime.errors import record_degradation

#: Bounded hard. This runs on every screen read, and a hung osascript would
#: stall the caller — including the response lane answering "what's on my
#: screen".
_TIMEOUT_S = 1.5

_SCRIPT = """
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    set windowTitle to ""
    try
        tell process frontApp
            set windowTitle to name of front window
        end tell
    end try
    return frontApp & "|" & windowTitle
end tell
"""


def frontmost_window_hint() -> tuple[str, str]:
    """(app, title) for the frontmost window, or ("", "") when unknown.

    An unknown foreground returns EMPTY rather than raising, and callers must
    treat empty as "cannot tell". Whether that means allow or refuse is the
    caller's decision: the ambient loop refuses, an explicit user request
    proceeds, and both are right for their context. This function does not
    make that policy — it only reports.
    """
    try:
        completed = subprocess.run(
            ["osascript", "-e", _SCRIPT],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record_degradation(
            "screen_context",
            exc,
            severity="debug",
            action="frontmost window unknown; the privacy gate had nothing to check",
        )
        return "", ""
    if completed.returncode != 0:
        # Accessibility permission not granted is the common case here, and
        # it is not an error worth escalating: the caller finds out it cannot
        # read the screen either, one step later.
        return "", ""
    raw = str(completed.stdout or "").strip()
    app, _, title = raw.partition("|")
    return app.strip(), title.strip()


__all__ = ["frontmost_window_hint"]
