from pathlib import Path

import pytest


def test_semantic_fact_extractor_records_bad_pattern_without_crashing():
    from core.memory.semantic_fact_extractor import FactType, SemanticFactExtractor

    extractor = SemanticFactExtractor()
    extractor._all_patterns = [(FactType.USER_PREFERENCE, "(", "user", "prefers")]

    assert extractor.extract_facts("I prefer concise answers", "Noted.", session_id="test") == []


@pytest.mark.asyncio
async def test_profile_manager_reports_partial_fact_write_failures():
    from core.memory.profile_manager import ProfileManager
    from core.memory.semantic_fact_extractor import FactType, SemanticFact

    class Extractor:
        def extract_facts(self, **_kwargs):
            return [
                SemanticFact(
                    fact_type=FactType.USER_PREFERENCE,
                    subject="user",
                    predicate="prefers",
                    object="concise status updates",
                    source_text="I prefer concise status updates",
                ),
                SemanticFact(
                    fact_type=FactType.RELATIONSHIP_FACT,
                    subject="relationship",
                    predicate="shared_goal",
                    object="ship Aura cleanly",
                    source_text="we should ship Aura cleanly together",
                ),
            ]

    class FailingUserProfile:
        def __init__(self):
            self.calls = 0

        def add_or_update_fact(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("user profile write failed")

    class AuraProfile:
        def __init__(self):
            self.facts = []

        def add_or_reinforce_fact(self, **kwargs):
            self.facts.append(dict(kwargs))

    manager = ProfileManager()
    aura_profile = AuraProfile()
    manager._fact_extractor = Extractor()
    user_profile = FailingUserProfile()
    manager._user_profile = user_profile
    manager._aura_profile = aura_profile

    user_count, aura_count = await manager.learn_from_turn(
        "I prefer concise status updates",
        "We should ship Aura cleanly together.",
        session_id="test",
    )

    status = manager.get_status()
    assert (user_count, aura_count) == (0, 1)
    assert user_profile.calls == 1
    assert len(aura_profile.facts) == 1
    assert status["initialized"] is True
    assert status["learning_attempts"] == 1
    assert status["learning_failures"] == 1
    assert status["fact_processing_failures"] == 1
    assert "user profile write failed" in status["last_failure_reason"]


def test_profile_memory_runtime_uses_narrow_recoverable_exceptions():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "core/memory/aura_self_profile.py",
        "core/memory/profile_manager.py",
        "core/memory/semantic_fact_extractor.py",
        "core/memory/user_profile.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "except Exception" not in source
        assert "except BaseException" not in source
