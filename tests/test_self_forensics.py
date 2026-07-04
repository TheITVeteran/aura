"""Self-forensics grounding — she answers about her own failures from
her black boxes, never from invention.

Live regression (July 4): asked about her overnight death she
confabulated electromagnetic interference three drafts in a row while
the true cause sat in her own records. The honesty gate rejected every
draft; nothing supplied the evidence a truthful draft needed.
"""
from __future__ import annotations

import inspect
import json
import time

from core.introspection.self_forensics import (
    build_self_forensics_context,
    is_self_forensics_question,
)


class TestDetector:
    def test_matches_the_live_session_questions(self):
        for question in (
            "did you crash last night",
            "What was the cause",
            "what was the root cause of the crash?",
            "why did you shut down",
            "what happened last night",
            "why did you disappear",
            "you restarted overnight?",
        ):
            assert is_self_forensics_question(question), question

    def test_ordinary_questions_stay_out(self):
        for question in (
            "what is the weather",
            "tell me about the moon landing",
            "how does your memory work",
            "can you hear me?",
        ):
            assert not is_self_forensics_question(question), question


class TestEvidenceBlock:
    def test_reads_real_grace_flag(self, tmp_path, monkeypatch):
        run_dir = tmp_path / ".aura" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "grace_exit.flag").write_text(json.dumps({
            "schema": "aura.shutdown_grace.v1",
            "pid": 1234,
            "reason": "coordinator",
            "created_at_unix": time.time() - 1800,
        }))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Path.home() honors HOME on posix.
        block = build_self_forensics_context()
        assert "GROUNDED SELF-FORENSICS" in block
        assert "reason='coordinator'" in block
        assert "0.5h ago" in block

    def test_no_evidence_yields_honest_unavailability(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)  # no data/error_logs here
        block = build_self_forensics_context()
        assert "do " in block and "not invent a cause" in block

    def test_instruction_forbids_invention(self):
        block = build_self_forensics_context()
        lowered = block.lower()
        assert "never invent" in lowered or "not invent" in lowered


class TestEngineWiring:
    def test_grounding_block_is_wired_into_think(self):
        from core.brain import cognitive_engine

        src = inspect.getsource(cognitive_engine)
        assert "SELF-FORENSICS EVIDENCE" in src
        assert "is_self_forensics_question" in src
