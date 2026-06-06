"""environments/desktop_world/desktop_simulator.py
Simulates active window layouts, front applications, and visual displays.
"""
from typing import Dict, List, Any


class VirtualDesktopWorld:
    """Simulates macOS active window hierarchies and coordinates."""

    def __init__(self):
        self._windows = [
            {"title": "Terminal", "bounds": {"x": 100, "y": 100, "w": 800, "h": 600}, "focused": True},
            {"title": "Google Chrome", "bounds": {"x": 200, "y": 200, "w": 1024, "h": 768}, "focused": False}
        ]

    def get_window_hierarchy(self) -> List[Dict[str, Any]]:
        return self._windows

    def focus_window(self, title: str) -> bool:
        found = False
        for w in self._windows:
            if w["title"] == title:
                w["focused"] = True
                found = True
            else:
                w["focused"] = False
        return found
