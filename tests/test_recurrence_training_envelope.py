from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_recurrence_training_envelope as envelope


class _FakeMLX:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def set_memory_limit(self, value: int) -> int:
        self.calls.append(("memory", value))
        return 0

    def set_cache_limit(self, value: int) -> int:
        self.calls.append(("cache", value))
        return 0

    def clear_cache(self) -> None:
        self.calls.append(("clear", None))


def test_configure_mlx_applies_bounded_limits_before_cache_clear() -> None:
    fake = _FakeMLX()
    receipt = envelope._configure_mlx(fake, memory_gb=40.0, cache_gb=2.0)

    assert fake.calls == [
        ("memory", 40 * envelope.GIB),
        ("cache", 2 * envelope.GIB),
        ("clear", None),
    ]
    assert receipt == {
        "memory_limit_bytes": 40 * envelope.GIB,
        "cache_limit_bytes": 2 * envelope.GIB,
        "cache_cleared_before_model_load": True,
    }


def test_configure_mlx_rejects_cache_at_or_above_memory() -> None:
    with pytest.raises(ValueError, match="cache limit"):
        envelope._configure_mlx(_FakeMLX(), memory_gb=2.0, cache_gb=2.0)


def test_envelope_artifact_is_idempotent_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "envelope.json"
    payload = {"schema": envelope.SCHEMA, "memory_limit_bytes": 123}
    envelope._write_envelope(path, payload)
    envelope._write_envelope(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    with pytest.raises(RuntimeError, match="differs"):
        envelope._write_envelope(path, {**payload, "memory_limit_bytes": 456})
