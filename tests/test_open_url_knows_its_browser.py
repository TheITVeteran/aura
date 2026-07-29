"""Nobody ever asked the browser to come forward.

Live 2026-07-29, every research run died at its first step:

    open_url failed: URL dispatch succeeded, but the target browser/tab could
    not be semantically confirmed (frontmost=unavailable, active_url=unavailable)

Read as written, the browser refused. What actually happened is that when no
browser is named, the code sampled the frontmost app the instant after
`open` — a race it always loses, because the browser has not come forward
yet and the app in front is still Aura. That is not in the allowed browser
set, so `expected_browser` was "", so the wait-for-frontmost step was skipped
entirely, so nothing was ever verified and nothing ever brought the browser
up.

LaunchServices knows which browser handles http(s) before the race starts.
"""

from __future__ import annotations

from core.skills.computer_use import _ALLOWED_URL_BROWSERS, ComputerUseSkill


def test_the_default_browser_is_read_from_the_system():
    name = ComputerUseSkill()._default_browser_name()
    assert name, "no default browser resolved"
    assert name in _ALLOWED_URL_BROWSERS


def test_an_unreadable_handler_list_still_answers(monkeypatch, tmp_path):
    """A missing preference file means Safari, which is the macOS default
    and not a guess."""
    import core.skills.computer_use as module

    class _Home:
        @staticmethod
        def home():
            return tmp_path

        def __truediv__(self, other):  # pragma: no cover - unused
            return tmp_path / other

    monkeypatch.setattr(module, "Path", _Home)
    assert ComputerUseSkill()._default_browser_name() == "Safari"


def test_bundle_ids_map_only_to_allowed_browsers():
    for name in ComputerUseSkill._BROWSER_BUNDLE_NAMES.values():
        assert name in _ALLOWED_URL_BROWSERS, name


def test_open_url_resolves_the_browser_before_reading_the_screen():
    """Order is the fix. Reading the screen first is the bug."""
    import inspect

    source = inspect.getsource(ComputerUseSkill._execute_action)
    resolve = source.index("_default_browser_name")
    poll = source.index("observed_browser = await asyncio.to_thread(")
    assert resolve < poll, (
        "the registered default must be consulted before falling back to "
        "whatever happens to be on screen"
    )
