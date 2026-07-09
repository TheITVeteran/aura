"""Contracts for the learning self-report — grounded answers, gated injection.

Aura's claims about her own learning must come from receipts (ledger,
flywheel stats, scheduler state), fire only on learning-shaped questions,
and state refusals/empty logs as plainly as promotions.
"""
from __future__ import annotations

import json

import pytest

import core.learning.learning_selfreport as selfreport_mod
from core.learning.learning_selfreport import (
    LearningSelfReport,
    _asks_about_learning,
    get_learning_selfreport,
    reset_learning_selfreport_for_test,
)

pytestmark = pytest.mark.unit


class TestQuestionGate:
    @pytest.mark.parametrize("question", [
        "what have you learned lately?",
        "What did you learn today",
        "have you been practicing anything?",
        "are you improving yourself?",
        "tell me about your training",
        "how is your self-improvement going",
        "did the weight update run last night?",
        "have you gotten better at math?",
        "are you teaching yourself things while I'm gone?",
    ])
    def test_learning_questions_fire(self, question):
        assert _asks_about_learning(question)

    @pytest.mark.parametrize("text", [
        "what's the weather like today",
        "I learned a lot at work this week",          # user's learning, not hers
        "my sister is training for a marathon",
        "let's practice our presentation tomorrow",
        "can you look up the train schedule",
        "",
    ])
    def test_ordinary_turns_pay_nothing(self, text):
        assert not _asks_about_learning(text)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Fake scheduler in the container + real flywheel state file on disk."""
    reset_learning_selfreport_for_test()

    class FakeScheduler:
        def get_status(self):
            return {
                "last_status": "promoted",
                "last_generation_id": "g0001-123",
                "lineage": {
                    "generations": 3,
                    "promoted": 2,
                    "refused": 1,
                    "verdict": "BOUNDED_SELF_OPTIMIZATION",
                },
            }

    import core.container as container_mod

    monkeypatch.setattr(
        container_mod.ServiceContainer,
        "get",
        classmethod(
            lambda cls, name, default=None: FakeScheduler()
            if name == "weight_compounding" else default
        ),
    )
    learning_dir = tmp_path / "learning"
    learning_dir.mkdir(parents=True)
    (learning_dir / "selfplay_flywheel.json").write_text(json.dumps({
        "bursts": 5, "total_attempts": 40, "total_correct": 26,
        "total_pairs": 12, "correct_rate_ema": 0.7,
    }), encoding="utf-8")

    import core.config as config_mod

    class FakePaths:
        data_dir = tmp_path

    class FakeConfig:
        paths = FakePaths()

    monkeypatch.setattr(config_mod, "get_config", lambda: FakeConfig())
    yield
    reset_learning_selfreport_for_test()


class TestBlockContent:
    def test_grounded_numbers_reach_the_block(self, wired):
        block = get_learning_selfreport().get_context_injection("what have you learned lately?")
        assert "LEARNING SELF-KNOWLEDGE" in block
        assert "3 recorded generation(s)" in block
        assert "2 promoted, 1 refused" in block
        assert "BOUNDED_SELF_OPTIMIZATION" in block
        assert "40 verified" in block
        assert "12 win/loss" in block
        assert "~65%" in block                       # 26/40 stated, not invented
        assert "do not invent" in block

    def test_never_trained_is_stated_plainly(self, wired, monkeypatch):
        import core.container as container_mod

        monkeypatch.setattr(
            container_mod.ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: default),
        )
        reset_learning_selfreport_for_test()
        block = get_learning_selfreport().get_context_injection("what did you learn today")
        assert "no training generation has completed yet" in block
        assert "nothing to claim yet" in block

    def test_non_learning_turn_returns_empty(self, wired):
        assert get_learning_selfreport().get_context_injection("how are you?") == ""

    def test_block_is_cached_within_ttl(self, wired):
        report = get_learning_selfreport()
        first = report.get_context_injection("what have you learned?")
        assert report.get_context_injection("your training status?") is first

    def test_collection_failure_returns_empty_never_raises(self, wired, monkeypatch):
        reset_learning_selfreport_for_test()
        monkeypatch.setattr(
            LearningSelfReport,
            "_build_block",
            lambda self: (_ for _ in ()).throw(RuntimeError("ledger unreadable")),
        )
        assert get_learning_selfreport().get_context_injection("what have you learned?") == ""


class TestConversationWiring:
    def test_conversation_support_injects_learning_block(self):
        from pathlib import Path

        source = Path("core/runtime/conversation_support.py").read_text(encoding="utf-8")
        assert "get_learning_selfreport" in source
        assert "learning self-knowledge block" in source
