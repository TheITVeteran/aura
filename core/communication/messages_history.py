"""Strict read-only access to Apple's local Messages history database."""

from __future__ import annotations

import asyncio
import sqlite3
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = Path.home() / "Library" / "Messages" / "chat.db"
MAX_MESSAGE_CHARS = 8_000
MAX_MESSAGE_BYTES = 24_000
MAX_BATCH = 8


class MessagesTransportError(RuntimeError):
    """Base transport failure with no contact or message data in its text."""


class MessagesHistoryUnavailableError(MessagesTransportError):
    """Messages history could not be read under the runtime's current TCC identity."""


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    row_id: int
    guid: str
    text: str


class MessagesHistoryReader:
    """Strict, read-only adapter over Apple's local Messages history schema."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_HISTORY_PATH).expanduser()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path.is_symlink() or not self.db_path.is_file():
            raise MessagesHistoryUnavailableError("messages_history_unavailable")
        quoted = urllib.parse.quote(str(self.db_path), safe="/")
        try:
            connection = sqlite3.connect(
                f"file:{quoted}?mode=ro",
                uri=True,
                timeout=1.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA busy_timeout=1000")
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise MessagesHistoryUnavailableError("messages_history_unavailable") from exc

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        try:
            message_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(message)")
            }
            handle_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(handle)")
            }
        except sqlite3.Error as exc:
            raise MessagesHistoryUnavailableError(
                "messages_history_schema_unavailable"
            ) from exc
        if not {"guid", "text", "handle_id", "is_from_me"}.issubset(
            message_columns
        ) or "id" not in handle_columns:
            raise MessagesHistoryUnavailableError("messages_history_schema_unsupported")

    async def probe(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._probe_sync)

    def _probe_sync(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            self._verify_schema(connection)
            connection.execute("SELECT 1 FROM message LIMIT 1").fetchone()
            return {"readable": True, "schema_supported": True}
        except sqlite3.Error as exc:
            raise MessagesHistoryUnavailableError("messages_history_read_denied") from exc
        finally:
            connection.close()

    async def latest_row_id(self, destination: str, *, from_me: bool) -> int:
        return await asyncio.to_thread(
            self._latest_row_id_sync,
            destination,
            from_me,
        )

    def _latest_row_id_sync(self, destination: str, from_me: bool) -> int:
        connection = self._connect()
        try:
            self._verify_schema(connection)
            row = connection.execute(
                "SELECT COALESCE(MAX(m.ROWID), 0) "
                "FROM message AS m JOIN handle AS h ON h.ROWID=m.handle_id "
                "WHERE h.id=? COLLATE NOCASE AND m.is_from_me=?",
                (destination, 1 if from_me else 0),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except sqlite3.Error as exc:
            raise MessagesHistoryUnavailableError("messages_history_query_failed") from exc
        finally:
            connection.close()

    async def messages_after(
        self,
        destination: str,
        *,
        from_me: bool,
        after_row_id: int,
        limit: int = MAX_BATCH,
    ) -> list[HistoryMessage]:
        return await asyncio.to_thread(
            self._messages_after_sync,
            destination,
            from_me,
            after_row_id,
            limit,
        )

    def _messages_after_sync(
        self,
        destination: str,
        from_me: bool,
        after_row_id: int,
        limit: int,
    ) -> list[HistoryMessage]:
        bounded_limit = max(1, min(int(limit), MAX_BATCH))
        connection = self._connect()
        try:
            self._verify_schema(connection)
            rows = connection.execute(
                "SELECT m.ROWID AS row_id, COALESCE(m.guid, '') AS guid, m.text AS text "
                "FROM message AS m JOIN handle AS h ON h.ROWID=m.handle_id "
                "WHERE h.id=? COLLATE NOCASE AND m.is_from_me=? AND m.ROWID>? "
                "ORDER BY m.ROWID ASC LIMIT ?",
                (destination, 1 if from_me else 0, max(0, int(after_row_id)), bounded_limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise MessagesHistoryUnavailableError("messages_history_query_failed") from exc
        finally:
            connection.close()
        messages: list[HistoryMessage] = []
        for row in rows:
            raw_text = row["text"]
            # Rich attributed bodies are archived object graphs and remain
            # untrusted. Only bounded standard text enters Aura's chat lane.
            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text:
                continue
            encoded = text.encode("utf-8", errors="strict")
            if len(text) > MAX_MESSAGE_CHARS or len(encoded) > MAX_MESSAGE_BYTES:
                continue
            messages.append(
                HistoryMessage(
                    row_id=int(row["row_id"]),
                    guid=str(row["guid"] or ""),
                    text=text,
                )
            )
        return messages
