from __future__ import annotations

from core.runtime.app_target_resolution import (
    InstalledApp,
    resolve_installed_app_target,
)


def test_alias_is_grounded_in_the_installed_app_inventory() -> None:
    result = resolve_installed_app_target(
        "my Note app",
        installed_apps=(InstalledApp("Notes", "/System/Applications/Notes.app"),),
    )

    assert result.resolved == "Notes"
    assert result.app_path == "/System/Applications/Notes.app"
    assert result.method == "installed_exact"
    assert result.corrected is True


def test_unique_inflection_is_corrected_without_guessing() -> None:
    result = resolve_installed_app_target(
        "Reminder",
        installed_apps=(
            InstalledApp("Reminders", "/System/Applications/Reminders.app"),
            InstalledApp("Notes", "/System/Applications/Notes.app"),
        ),
    )

    assert result.resolved == "Reminders"
    assert result.method == "installed_inflection"


def test_ambiguous_or_weak_similarity_remains_launchservices_unverified() -> None:
    result = resolve_installed_app_target(
        "Studio",
        installed_apps=(
            InstalledApp("Audio Studio", "/Applications/Audio Studio.app"),
            InstalledApp("Video Studio", "/Applications/Video Studio.app"),
        ),
    )

    assert result.resolved == "Studio"
    assert result.app_path == ""
    assert result.method == "launchservices_unverified"
    assert result.alternatives


def test_generic_app_word_is_not_launchable() -> None:
    result = resolve_installed_app_target(
        "the app",
        installed_apps=(InstalledApp("Notes", "/System/Applications/Notes.app"),),
    )

    assert result.launchable is False
    assert result.method == "missing_target"
