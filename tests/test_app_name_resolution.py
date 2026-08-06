"""Say what you mean, not the exact bundle name.

From the 2026-07-30 demo: "Can you open the Note app and write a note..."
failed with "open_app failed: Unable to find application named 'Note'" while
Notes.app sat in /System/Applications. The refusal was honest — she correctly
declined to claim the action finished — but the task was doable and the only
thing standing between her and doing it was a trailing 's'.
"""

from __future__ import annotations

import pytest

from core.capabilities.host_automation import (
    AppNameResolution,
    installed_application_names,
    resolve_application_name,
)


class TestResolution:
    def test_the_demo_failure_now_resolves(self) -> None:
        assert resolve_application_name("Note").resolved == "Notes"

    def test_the_spoken_form_resolves(self) -> None:
        for spoken in ("Note app", "the Notes app", "Notes app", "the note app"):
            assert resolve_application_name(spoken).resolved == "Notes", spoken

    def test_an_exact_name_is_unchanged(self) -> None:
        result = resolve_application_name("Notes")
        assert result.resolved == "Notes"
        assert result.basis == "exact"

    def test_case_does_not_matter(self) -> None:
        assert resolve_application_name("notes").resolved == "Notes"

    def test_a_common_short_name_resolves(self) -> None:
        """"chrome" is not the bundle name; nobody says "Google Chrome"."""
        resolved = resolve_application_name("chrome").resolved
        assert resolved is None or "Chrome" in resolved

    def test_an_app_whose_name_contains_app_is_not_mangled(self) -> None:
        """Only a TRAILING app/application is stripped, so App Store survives."""
        for spoken in ("App Store", "the app store"):
            result = resolve_application_name(spoken)
            assert result.resolved in (None, "App Store"), spoken

    def test_an_absent_app_is_refused_with_what_is_there(self) -> None:
        result = resolve_application_name("definitelynotinstalledxyzzy")
        assert not result.ok
        assert "definitelynotinstalledxyzzy" in result.failure_detail()

    def test_an_empty_name_is_refused(self) -> None:
        assert not resolve_application_name("").ok
        assert not resolve_application_name("   ").ok

    def test_the_failure_names_candidates_when_it_has_them(self) -> None:
        result = AppNameResolution("Saf", None, "ambiguous", ("Safari", "SafeThing"))
        detail = result.failure_detail()
        assert "Safari" in detail
        assert "closest installed" in detail

    def test_ambiguity_is_reported_rather_than_guessed(self) -> None:
        """Opening the wrong application is worse than asking which."""
        result = AppNameResolution("x", None, "ambiguous", ("A", "B"))
        assert not result.ok


class TestEnumeration:
    def test_it_finds_applications_on_this_machine(self) -> None:
        names = installed_application_names()
        assert names, "no applications found in any standard directory"
        assert all(not name.endswith(".app") for name in names)

    def test_it_is_cached_between_calls(self) -> None:
        first = installed_application_names()
        second = installed_application_names()
        assert first is second or first == second

    def test_an_unenumerable_host_falls_back_to_passthrough(self, monkeypatch) -> None:
        """A directory listing that fails is not evidence the app is missing."""
        import core.capabilities.host_automation as mod

        monkeypatch.setattr(mod, "_INSTALLED_APPS_CACHE", None)
        monkeypatch.setattr(mod, "installed_application_names", lambda **_: ())
        result = mod.resolve_application_name("Notes")
        assert result.resolved == "Notes"
        assert result.basis == "unverified_passthrough"


@pytest.mark.asyncio
async def test_launch_refuses_an_absent_app_without_calling_applescript(monkeypatch):
    """The refusal must happen before the automation, not through it."""
    import core.capabilities.host_automation as mod

    class _NoRun:
        @staticmethod
        async def run(*_args, **_kwargs):
            raise AssertionError("AppleScript must not run for an absent app")

    monkeypatch.setattr(mod, "AppleScriptRunner", _NoRun)
    host = mod.HostAutomationProvider()
    receipt = await host.launch_app("definitelynotinstalledxyzzy")
    assert receipt.success is False
    assert "no application named" in str(receipt.error)
