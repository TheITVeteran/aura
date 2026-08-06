"""Three places that reported doing something they had not done.

All three were sitting behind the enterprise gate's file-name allowlist for
stub/placeholder keywords, which is why nobody had looked at them: the gate
knew the word was there and had been told to ignore that file.

  * StartupValidator marked a check with no handler ``passed = True`` and
    wrote "Check not implemented (ignored)". "Dangerous Files Purged" is a
    critical check; renaming its handler would have turned it green.
  * DummyTTS logged the text and returned, exactly like an engine that had
    just made a sound.
  * Ears.mock_hear returned None down all four paths, so a delivered
    transcript and a dead event loop looked identical to the caller.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest


class TestAnUnrunCheckIsNotAPassedCheck:
    @staticmethod
    def _validator():
        from core.resilience.startup_validator import StartupValidator

        return StartupValidator(orchestrator=SimpleNamespace())

    def test_every_declared_check_has_a_handler_today(self) -> None:
        """The guard below only matters if this ever stops being true — so
        the pair is: nothing is unimplemented now, and if something becomes
        unimplemented it fails rather than passes."""
        validator = self._validator()
        missing = [
            check.id
            for check in validator.checks
            if getattr(validator, f"_check_{check.id}", None) is None
        ]
        assert missing == []

    def test_a_check_with_no_handler_fails(self, monkeypatch) -> None:
        from core.resilience.startup_validator import StartupValidator, ValidationCheck

        validator = self._validator()
        orphan = ValidationCheck(
            "orphan_01", "Orphan", "declared with no handler", critical=False
        )
        validator.checks = [orphan]

        recorded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "core.resilience.startup_validator.record_degradation",
            lambda subsystem, exc, **kw: recorded.append((subsystem, str(exc))),
        )

        asyncio.run(validator.validate_all())

        assert orphan.passed is False
        assert "NOT VERIFIED" in orphan.message
        assert recorded and recorded[0][0] == "startup_validator"
        assert "orphan_01" in recorded[0][1]

    def test_an_unimplemented_critical_check_blocks_startup(self, monkeypatch) -> None:
        """The whole point. Silence about a critical precondition is a
        refusal to start, not permission to continue."""
        from core.resilience.startup_validator import StartupValidator, ValidationCheck

        validator = self._validator()
        validator.checks = [
            ValidationCheck("orphan_02", "Orphan", "no handler", critical=True)
        ]
        monkeypatch.setattr(
            "core.resilience.startup_validator.record_degradation",
            lambda *a, **k: None,
        )

        assert asyncio.run(validator.validate_all()) is False

    def test_a_handler_that_passes_still_passes(self) -> None:
        """Fail-closed must not mean fail-always."""
        from core.resilience.startup_validator import StartupValidator, ValidationCheck

        validator = self._validator()
        check = ValidationCheck("live_01", "Live", "has a handler", critical=True)
        validator.checks = [check]

        async def _handler(c: ValidationCheck) -> None:
            c.passed = True
            c.message = "verified"

        validator._check_live_01 = _handler  # type: ignore[attr-defined]

        assert asyncio.run(validator.validate_all()) is True
        assert check.passed is True


class TestAVoiceThatCannotSpeakSaysSo:
    def test_the_dummy_engine_records_a_degradation(self, monkeypatch) -> None:
        from core.embodiment import voice_presence

        recorded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            voice_presence,
            "record_degradation",
            lambda subsystem, exc, **kw: recorded.append((subsystem, str(exc))),
        )

        engine = voice_presence.DummyTTS()
        asyncio.run(engine.speak("hello"))

        assert recorded and recorded[0][0] == "voice_presence"
        assert sys.platform in recorded[0][1]

    def test_it_reports_once_not_once_per_utterance(self, monkeypatch) -> None:
        """A degradation per sentence buries the condition it is reporting."""
        from core.embodiment import voice_presence

        recorded: list[str] = []
        monkeypatch.setattr(
            voice_presence,
            "record_degradation",
            lambda subsystem, exc, **kw: recorded.append(subsystem),
        )

        engine = voice_presence.DummyTTS()
        for _ in range(5):
            asyncio.run(engine.speak("hello"))

        assert len(recorded) == 1

    @pytest.mark.parametrize(
        "platform,expected", [("darwin", True), ("linux", True), ("aix", False)]
    )
    def test_availability_is_readable_before_asking_for_speech(
        self, monkeypatch, platform: str, expected: bool
    ) -> None:
        from core.embodiment import voice_presence

        monkeypatch.setattr(voice_presence.sys, "platform", platform)
        presence = voice_presence.VoicePresence(orchestrator=SimpleNamespace())

        assert presence.voice_output_available is expected


class TestTheHearingSeamSaysWhetherItDelivered:
    @staticmethod
    def _ears():
        from core.senses.ears import SovereignEars

        return SovereignEars.__new__(SovereignEars)

    def test_no_engine_is_reported_as_not_delivered(self) -> None:
        ears = self._ears()
        ears._engine = None
        assert ears.inject_transcript_for_test("hello") is False

    def test_a_live_transcript_handler_is_reported_as_delivered(self) -> None:
        heard: list[str] = []

        async def _handler(text: str) -> None:
            heard.append(text)

        ears = self._ears()
        ears._engine = SimpleNamespace(_on_transcript=_handler)

        async def _drive() -> bool:
            delivered = ears.inject_transcript_for_test("hello")
            await asyncio.sleep(0)
            return delivered

        assert asyncio.run(_drive()) is True
        assert heard == ["hello"]

    def test_the_old_name_is_gone(self) -> None:
        """``mock_hear`` had no callers anywhere in the repository. A test
        seam that reads like production code is how it stayed invisible."""
        from core.senses.ears import SovereignEars

        assert not hasattr(SovereignEars, "mock_hear")
