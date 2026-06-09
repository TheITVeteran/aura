"""PyWebView environment diagnostic.

Desktop-only check that the webview dependency stack is importable and
can construct a window object. Skips cleanly when pywebview is not
installed — this module previously ran at import time and killed the
entire pytest collection with sys.exit(1) on machines without the
desktop stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

webview = pytest.importorskip(
    "webview", reason="pywebview not installed; desktop diagnostic only"
)

pytestmark = pytest.mark.desktop


def test_webview_window_can_be_constructed():
    """Window construction proves the native dependencies are present.

    We deliberately do not call webview.start(): the run loop would hang
    a headless environment. Construction success is the signal that the
    desktop stack (pywebview + platform backend) is wired.
    """
    window = webview.create_window("Aura Diagnostic", "https://example.com")
    assert window is not None
    assert getattr(window, "title", "") == "Aura Diagnostic"
