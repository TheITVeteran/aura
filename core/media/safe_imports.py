"""core/media/safe_imports.py — macOS Compatibility Layer
Ensures media stack doesn't crash on macOS when multiple backends collide.
"""
import builtins
import sys
import os
import logging

logger = logging.getLogger("Aura.MediaSafe")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_CV2_IMPORT_GUARD_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__


def _env_true(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in _TRUE_VALUES


def media_sidecar_process_allowed() -> bool:
    """True when media libraries may load inside this process.

    The primary Aura brain process must not load OpenCV on macOS because the
    desktop STT path uses PyAV. The sensory worker is the production process
    boundary for camera/screen media libraries, so it marks itself explicitly.
    """

    return _env_true("AURA_MEDIA_SIDECAR_PROCESS") or _env_true(
        "AURA_ALLOW_INPROCESS_CV2_WITH_STT"
    )


def cv2_main_process_blocked() -> bool:
    """True when the macOS main process should not import OpenCV.

    ``faster-whisper`` depends on PyAV. Loading OpenCV's AVFoundation stack in
    the same process can register duplicate Objective-C classes. The sidecar
    sensory worker remains the production camera path; this only prevents
    unsafe in-process imports.
    """
    if sys.platform != "darwin":
        return False
    if media_sidecar_process_allowed():
        return False
    return True


def install_main_process_cv2_guard() -> None:
    """Reject accidental OpenCV imports in Aura's primary macOS process.

    The guard is intentionally narrow: it only blocks top-level ``cv2`` imports
    in the primary process. The sensory sidecar and an explicit developer
    override may still import cv2. This turns a crash-prone Objective-C class
    collision into a deterministic, recoverable ImportError at the unsafe call
    site.
    """

    global _CV2_IMPORT_GUARD_INSTALLED
    if _CV2_IMPORT_GUARD_INSTALLED:
        return

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if level == 0 and (name == "cv2" or name.startswith("cv2.")) and cv2_main_process_blocked():
            raise ImportError(
                "OpenCV import is blocked in Aura's primary macOS process; "
                "use the sensory sidecar or set AURA_ALLOW_INPROCESS_CV2_WITH_STT=1 for diagnostics."
            )
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    _CV2_IMPORT_GUARD_INSTALLED = True

def prevent_collisions():
    """Apply environment and import shims for macOS media stability."""
    if sys.platform != "darwin":
        return

    # Patch 24: Fix media stack collisions on macOS
    # Prevents "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized."
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Prevents CV2/AVFoundation deadlocks
    os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1"
    
    logger.info("🍎 macOS Media Collision Guards applied.")

# Auto-apply on import
prevent_collisions()
