from __future__ import annotations

from core.consciousness.adaptive_mood import SQLITE_BUSY_TIMEOUT_MS, AdaptiveMoodCoefficients


def test_adaptive_mood_configures_wal_and_busy_timeout(tmp_path) -> None:
    mood = AdaptiveMoodCoefficients(db_path=tmp_path / "adaptive_mood.sqlite3")

    with mood._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) >= SQLITE_BUSY_TIMEOUT_MS


def test_adaptive_mood_persists_under_repeated_updates(tmp_path) -> None:
    db_path = tmp_path / "adaptive_mood.sqlite3"
    mood = AdaptiveMoodCoefficients(db_path=db_path)
    chemicals = {name: 0.1 for name in mood.chemicals}

    for _ in range(8):
        mood.update_from_outcome(chemicals, {"valence": 0.3, "stress": 0.1})

    reloaded = AdaptiveMoodCoefficients(db_path=db_path)
    assert reloaded.total_updates() >= 8
