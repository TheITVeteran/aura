from __future__ import annotations

from core.consciousness import unified_self
from core.consciousness.unified_self import UnifiedSelf


def test_unified_self_corrupt_persistence_degrades_to_default_state(monkeypatch, tmp_path):
    path = tmp_path / "unified_self.json"
    path.write_text("{not-json", encoding="utf-8")
    recorded = []
    monkeypatch.setattr(
        unified_self,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    state = UnifiedSelf(str(path)).get_state()

    assert state.name == "Aura"
    assert recorded == [("unified_self", "JSONDecodeError")]


def test_unified_self_uses_narrow_recoverable_exceptions():
    with open(unified_self.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "except Exception" not in text
    assert "except BaseException" not in text
