"""Canonical macOS app-bundle manifest.

Single source of truth for the TCC *usage description* strings that must appear
in the packaged app's ``Info.plist`` and the hardened-runtime *entitlements* the
bundle declares. Both the build script (scripts/build_app.py) and the tests
import from here so the shipped bundle and our expectations never drift.

Why each key matters for Aura specifically:

  - Apple Events (``NSAppleEventsUsageDescription`` + the automation entitlement)
    is the big one — without it, every osascript / ScriptingBridge call to
    Notes, Mail, Finder, System Events or the browser is killed by macOS with
    error -1743 ("Not authorized to send Apple events"). This is the permission
    most commonly missing from hand-rolled Python .app bundles.
  - Microphone / Speech feed the voice wake-word loop.
  - Camera / Screen Recording feed visual perception.
  - The folder usage strings let Aura read/write the user's working files.

A usage-description string is REQUIRED for the corresponding TCC prompt to even
appear; an app that touches the mic with no ``NSMicrophoneUsageDescription`` is
terminated by the OS instead of being prompted. These are needed even for an
unsigned / "unknown developer" build.
"""
from __future__ import annotations

import plistlib
from pathlib import Path

# Info.plist privacy usage descriptions (TCC). Key → human-readable reason.
TCC_USAGE_DESCRIPTIONS: dict[str, str] = {
    "NSMicrophoneUsageDescription": (
        "Aura needs microphone access for voice interaction and the wake-word loop."
    ),
    "NSCameraUsageDescription": (
        "Aura needs camera access for visual processing and spatial awareness."
    ),
    "NSSpeechRecognitionUsageDescription": (
        "Aura needs speech recognition to convert your spoken audio to text."
    ),
    "NSAppleEventsUsageDescription": (
        "Aura uses Apple Events to control apps like Notes, Mail, Finder and your "
        "browser so it can act on your behalf on the desktop."
    ),
    "NSDesktopFolderUsageDescription": (
        "Aura needs Desktop access to read and organize the files you ask it to work with."
    ),
    "NSDocumentsFolderUsageDescription": (
        "Aura needs Documents access to read and organize the files you ask it to work with."
    ),
    "NSDownloadsFolderUsageDescription": (
        "Aura needs Downloads access to read and organize the files you ask it to work with."
    ),
    "NSSystemAdministrationUsageDescription": (
        "Aura needs system administration access to manage its own runtime and local resources."
    ),
}

# Hardened-runtime entitlements. Required if/when the bundle is codesigned;
# harmless on an unsigned build. The apple-events + library-validation pair is
# what lets a Python/PyInstaller bundle both load its native deps and drive
# other apps under the hardened runtime.
HARDENED_RUNTIME_ENTITLEMENTS: dict[str, bool] = {
    "com.apple.security.automation.apple-events": True,
    "com.apple.security.device.audio-input": True,
    "com.apple.security.device.camera": True,
    # Python ships unsigned .so/.dylib files and JITs Metal kernels; without
    # these a hardened-runtime bundle crashes on import.
    "com.apple.security.cs.allow-jit": True,
    "com.apple.security.cs.allow-unsigned-executable-memory": True,
    "com.apple.security.cs.disable-library-validation": True,
}

# Apple Events permission failures surface as this OSStatus.
APPLE_EVENTS_NOT_AUTHORIZED_ERR = -1743


def info_plist_overrides() -> dict[str, str]:
    """The Info.plist keys the build must ensure are present."""
    return dict(TCC_USAGE_DESCRIPTIONS)


def write_entitlements_plist(path: str | Path) -> Path:
    """Write the hardened-runtime entitlements to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        plistlib.dump(dict(HARDENED_RUNTIME_ENTITLEMENTS), handle)
    return target
