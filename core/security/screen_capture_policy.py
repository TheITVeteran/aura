"""One privacy boundary for every read of the user's desktop.

Screen pixels and accessibility text are equivalent from a privacy
perspective: either can expose an incognito page, password manager, or other
foreground content.  Callers therefore do not decide this independently.
They ask this module immediately before a read, and an unknown foreground is
not treated as permission.

The admission object intentionally never contains the application or window
title.  A denial receipt that names a private window has leaked the metadata
the denial exists to protect.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import StrEnum

from core.runtime.permission_gates import screen_allowed

_PRIVATE_WINDOW_MARKERS: tuple[str, ...] = (
    "incognito",
    "private browsing",
    "private window",
    "inprivate",
    "guest",
    "1password",
    "bitwarden",
    "keychain access",
    "keeper",
    "lastpass",
    "dashlane",
    "authenticator",
    "banking",
    "password",
)

_PRIVATE_APPS: frozenset[str] = frozenset(
    {
        "1password",
        "1password 7",
        "bitwarden",
        "keychain access",
        "keeper password manager",
        "lastpass",
        "dashlane",
        "gpg keychain",
        "secretive",
    }
)

# A browser name without a readable title cannot prove that the active window
# is not private.  Other apps can legitimately have no titled window (Finder
# desktop, menu-bar utilities), so the stricter title requirement is scoped to
# applications that implement private browsing.
_PRIVATE_BROWSING_APPS: frozenset[str] = frozenset(
    {
        "arc",
        "brave browser",
        "firefox",
        "google chrome",
        "microsoft edge",
        "opera",
        "safari",
        "vivaldi",
    }
)

_PRIVATE_RE = re.compile(
    "|".join(re.escape(marker) for marker in _PRIVATE_WINDOW_MARKERS),
    re.IGNORECASE,
)


class ScreenCaptureDenial(StrEnum):
    """Stable, non-disclosing reasons a desktop read did not happen."""

    NONE = "none"
    RUNTIME_SETTING_DISABLED = "runtime_setting_disabled"
    PRIVATE_FOREGROUND = "private_foreground"
    FOREGROUND_UNKNOWN = "foreground_unknown"
    BROWSER_TITLE_UNKNOWN = "browser_title_unknown"


@dataclass(frozen=True, slots=True)
class ScreenCaptureAdmission:
    """Privacy-safe result of checking whether Aura may read the desktop."""

    allowed: bool
    reason: ScreenCaptureDenial = ScreenCaptureDenial.NONE
    context_known: bool = False

    @property
    def public_error(self) -> str:
        if self.allowed:
            return ""
        if self.reason is ScreenCaptureDenial.RUNTIME_SETTING_DISABLED:
            return "screen capture is disabled by permissions.screen"
        if self.reason is ScreenCaptureDenial.PRIVATE_FOREGROUND:
            return "screen capture refused because the foreground is private"
        return "screen capture refused because foreground privacy could not be verified"

    def to_receipt(self) -> dict[str, str | bool]:
        return {
            "schema": "aura.security.screen_capture_admission.v1",
            "allowed": self.allowed,
            "reason": self.reason.value,
            "context_known": self.context_known,
        }


class ScreenCaptureDeniedError(PermissionError):
    """Raised before a backend is touched when desktop reading is not admitted."""

    def __init__(self, admission: ScreenCaptureAdmission) -> None:
        self.admission = admission
        super().__init__(admission.public_error)


def is_private_screen_context(app: str, title: str) -> bool:
    """Return whether foreground metadata identifies a private context."""

    normalized_app = str(app or "").strip().lower()
    if normalized_app in _PRIVATE_APPS:
        return True
    return bool(_PRIVATE_RE.search(f"{app or ''} {title or ''}"))


def evaluate_screen_capture_admission(
    *,
    context: tuple[str, str] | None = None,
) -> ScreenCaptureAdmission:
    """Evaluate the universal desktop-read policy without acquiring pixels.

    ``context`` is injectable for deterministic tests.  Production callers
    omit it and use the bounded foreground metadata probe.
    """

    if not screen_allowed():
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.RUNTIME_SETTING_DISABLED,
        )

    if context is None:
        try:
            from core.senses.screen_context import frontmost_window_hint

            context = frontmost_window_hint()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            context = ("", "")

    try:
        app, title = context
    except (TypeError, ValueError):
        app, title = "", ""
    app = str(app or "").strip()
    title = str(title or "").strip()

    if not app and not title:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
        )
    if is_private_screen_context(app, title):
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.PRIVATE_FOREGROUND,
            context_known=True,
        )
    if app.lower() in _PRIVATE_BROWSING_APPS and not title:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN,
            context_known=False,
        )
    return ScreenCaptureAdmission(allowed=True, context_known=True)


def require_screen_capture_admission(
    *,
    context: tuple[str, str] | None = None,
) -> ScreenCaptureAdmission:
    admission = evaluate_screen_capture_admission(context=context)
    if not admission.allowed:
        raise ScreenCaptureDeniedError(admission)
    return admission


async def evaluate_screen_capture_admission_async() -> ScreenCaptureAdmission:
    """Run the bounded metadata probe off the event loop."""

    return await asyncio.to_thread(evaluate_screen_capture_admission)


async def require_screen_capture_admission_async() -> ScreenCaptureAdmission:
    admission = await evaluate_screen_capture_admission_async()
    if not admission.allowed:
        raise ScreenCaptureDeniedError(admission)
    return admission


__all__ = [
    "ScreenCaptureAdmission",
    "ScreenCaptureDeniedError",
    "ScreenCaptureDenial",
    "evaluate_screen_capture_admission",
    "evaluate_screen_capture_admission_async",
    "is_private_screen_context",
    "require_screen_capture_admission",
    "require_screen_capture_admission_async",
]
