from __future__ import annotations

import logging

from core.security.user_recognizer import UserRecognizer


def _recognizer_without_disk_state() -> UserRecognizer:
    recognizer = UserRecognizer.__new__(UserRecognizer)
    recognizer._session_verified = False
    recognizer._session_override_reason = ""
    return recognizer


def test_owner_session_override_announces_only_the_state_transition(caplog):
    recognizer = _recognizer_without_disk_state()

    with caplog.at_level(logging.INFO, logger="Aura.UserRecognizer"):
        assert recognizer.override_session_owner("owner_session_cookie") is True
        assert recognizer.override_session_owner("owner_session_cookie") is False
        assert recognizer.override_session_owner("owner_session_cookie") is False

    announcements = [
        record
        for record in caplog.records
        if "session owner override applied" in record.getMessage()
    ]
    assert len(announcements) == 1
    assert recognizer.is_session_verified() is True


def test_session_reset_allows_next_owner_transition_to_be_announced(caplog):
    recognizer = _recognizer_without_disk_state()
    recognizer.override_session_owner("owner_session_cookie")
    recognizer.reset_session()

    with caplog.at_level(logging.INFO, logger="Aura.UserRecognizer"):
        assert recognizer.override_session_owner("owner_session_cookie") is True

    assert any(
        "session owner override applied" in record.getMessage()
        for record in caplog.records
    )
