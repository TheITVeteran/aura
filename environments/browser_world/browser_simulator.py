"""environments/browser_world/browser_simulator.py
Simulates a browser session with DOM parsing and load transitions.
"""
from typing import Dict, Any, List


class VirtualBrowserWorld:
    """Simulates active URLs, page titles, and DOM elements."""

    def __init__(self):
        self._current_url = "about:blank"
        self._dom_tree = {"tag": "body", "children": [{"tag": "h1", "text": "Blank Page"}]}

    def load_url(self, url: str) -> None:
        self._current_url = url
        if "google.com" in url:
            self._dom_tree = {"tag": "body", "children": [{"tag": "input", "name": "q"}]}
        else:
            self._dom_tree = {"tag": "body", "children": [{"tag": "p", "text": "Page loaded."}]}

    def get_active_tab(self) -> Dict[str, Any]:
        return {
            "url": self._current_url,
            "dom": self._dom_tree
        }
