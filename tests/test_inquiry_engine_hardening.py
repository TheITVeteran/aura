from __future__ import annotations

import asyncio
import json
import time


def _blank_engine(tmp_path):
    from core.epistemics.inquiry_engine import InquiryEngine

    engine = InquiryEngine.__new__(InquiryEngine)
    engine._questions = []
    engine._settled = []
    engine._db_path = tmp_path / "inquiry_journal.json"
    engine._api_adapter = None
    engine._epistemic = None
    engine._insight_journal = None
    engine._belief_engine = None
    engine._research_task = None
    engine.running = False
    return engine


def test_inquiry_engine_quarantines_corrupt_journal(monkeypatch, tmp_path):
    from core.epistemics import inquiry_engine as module

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    engine = _blank_engine(tmp_path)
    engine._db_path.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    engine._load()

    assert engine._questions == []
    assert not engine._db_path.exists()
    assert list(tmp_path.glob("inquiry_journal.corrupt-*.json"))
    assert calls
    assert calls[-1][0] == "inquiry_engine"
    assert "quarantined corrupt store" in calls[-1][2]["action"]


def test_inquiry_engine_load_sanitizes_persisted_questions(monkeypatch, tmp_path):
    from core.epistemics import inquiry_engine as module

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    now = time.time()
    engine = _blank_engine(tmp_path)
    engine._db_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "What is robust autonomy?",
                        "domain": "agency",
                        "urgency": 7,
                        "opened_at": now - 10,
                        "last_active": now + 1000,
                        "evidence": [
                            {"content": "A", "source": "test", "weight": 5, "confidence": -3},
                            "bad evidence",
                        ],
                    }
                ],
                "settled": [{"question": "", "domain": "bad"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    engine._load()

    assert len(engine._questions) == 1
    question = engine._questions[0]
    assert question.urgency == 1.0
    assert question.last_active <= time.time()
    assert len(question.evidence) == 1
    assert question.evidence[0].weight == 1.0
    assert question.evidence[0].confidence == 0.0
    assert any("skipped invalid" in call[2]["action"] for call in calls)


def test_inquiry_engine_malformed_research_result_keeps_question_open(monkeypatch, tmp_path):
    from core.epistemics import inquiry_engine as module
    from core.epistemics.inquiry_engine import OpenQuestion

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    engine = _blank_engine(tmp_path)
    question = OpenQuestion(
        id="q1",
        question="What should Aura learn next?",
        domain="learning",
        urgency=0.5,
        opened_at=time.time(),
        last_active=time.time(),
    )

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    asyncio.run(engine._process_research_result(question, "```json\n{bad\n```"))

    assert question.research_attempts == 1
    assert tmp_path.joinpath("inquiry_journal.json").exists()
    assert calls
    assert "malformed research result" in calls[-1][2]["action"]
