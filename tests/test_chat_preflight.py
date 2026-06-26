"""tests/test_chat_preflight.py
─────────────────────────────────
Unit tests for the chat preflight helpers (file-reference detection,
file loading with sandboxing, pending-chat queue).

Run:
    .venv/bin/python -m unittest tests.test_chat_preflight -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conversation import chat_preflight as cp  # noqa: E402
from core.conversation.chat_preflight import (  # noqa: E402
    PendingChat,
    answer_pending,
    build_file_context_block,
    clamp_composed_chat_context,
    compose_chat_directive_prefix,
    consume_for_session,
    enqueue,
    extract_file_references,
    format_resume_prefix,
    has_unanswered_for_session,
    load_referenced_files,
    schedule_background_retry,
)


def _temp_path(suffix: str = ".jsonl") -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    p = Path(path)
    if p.exists():
        p.unlink()
    return p


class TestFileReferenceDetection(unittest.TestCase):
    def test_look_at_pattern(self):
        refs = extract_file_references("Look at the file aura/knowledge/bryan-curated-media.md")
        self.assertIn("aura/knowledge/bryan-curated-media.md", refs)

    def test_at_pattern(self):
        refs = extract_file_references(
            "I dropped a curated media list at aura/knowledge/bryan-curated-media.md"
        )
        self.assertIn("aura/knowledge/bryan-curated-media.md", refs)

    def test_read_pattern(self):
        refs = extract_file_references(
            "Read scoping/fuse-comparison-9870-vs-7500.md and tell me what you think"
        )
        self.assertIn("scoping/fuse-comparison-9870-vs-7500.md", refs)

    def test_no_false_positives(self):
        self.assertEqual(extract_file_references("Just a normal chat with no files"), [])
        self.assertEqual(extract_file_references(""), [])
        self.assertEqual(extract_file_references("How's the weather today.com"), [])

    def test_caps_at_max(self):
        msg = " ".join([f"look at file{i}.md" for i in range(20)])
        self.assertLessEqual(len(extract_file_references(msg)), 3)

    def test_dedup(self):
        refs = extract_file_references("Look at X.md. Read X.md. Open X.md.")
        # Should appear once, not three times
        self.assertEqual(refs.count("X.md"), 1)


class TestFileLoading(unittest.TestCase):
    def test_loads_existing_real_file(self):
        # The curated-media doc was shipped in the previous commit
        files = load_referenced_files(["aura/knowledge/bryan-curated-media.md"])
        self.assertEqual(len(files), 1)
        display, content = files[0]
        self.assertTrue(display.endswith("bryan-curated-media.md"))
        self.assertGreater(len(content), 100)
        self.assertIn("curated", content.lower())

    def test_rejects_traversal(self):
        # Must not be able to escape PROJECT_ROOT
        files = load_referenced_files(["../../../../etc/passwd"])
        self.assertEqual(files, [])

    def test_rejects_unsupported_extension(self):
        # .safetensors files exist but should be filtered
        files = load_referenced_files(["training/adapters/aura-personality/adapters.safetensors"])
        self.assertEqual(files, [])

    def test_missing_file_returns_empty(self):
        files = load_referenced_files(["this/path/does/not/exist.md"])
        self.assertEqual(files, [])

    def test_build_context_block_format(self):
        block = build_file_context_block(["aura/knowledge/bryan-curated-media.md"])
        self.assertIn("=== FILE:", block)
        self.assertIn("=== END", block)
        self.assertIn("references files", block)

    def test_load_referenced_files_reads_only_budgeted_prefix(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            big_file = root / "large.md"
            big_file.write_text("x" * 10_000, encoding="utf-8")

            previous_root = cp.PROJECT_ROOT
            cp.PROJECT_ROOT = root
            try:
                files = load_referenced_files(["large.md"], remaining_budget=1200)
            finally:
                cp.PROJECT_ROOT = previous_root

        self.assertEqual(len(files), 1)
        self.assertIn("truncated", files[0][1])
        self.assertLess(len(files[0][1]), 1400)


class TestPendingQueue(unittest.TestCase):
    def setUp(self):
        self.path = _temp_path()

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_enqueue_and_unanswered_check(self):
        enqueue("session-1", "What is the meaning of life?", reason="timeout", path=self.path)
        self.assertTrue(has_unanswered_for_session("session-1", path=self.path))
        self.assertFalse(has_unanswered_for_session("other-session", path=self.path))

    def test_answer_marks_consumed(self):
        enqueue("s2", "How are you?", path=self.path)
        ok = answer_pending("s2", "I'm doing fine — sorry for the wait.", path=self.path)
        self.assertTrue(ok)
        # No more unanswered for this session
        self.assertFalse(has_unanswered_for_session("s2", path=self.path))

    def test_consume_returns_answered_only(self):
        enqueue("s3", "Q1", path=self.path)
        enqueue("s3", "Q2", path=self.path)
        answer_pending("s3", "A2", path=self.path)
        delivered = consume_for_session("s3", path=self.path)
        # Only the answered one should be delivered
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].answer_text, "A2")
        # The unanswered one should remain
        self.assertTrue(has_unanswered_for_session("s3", path=self.path))

    def test_format_resume_prefix(self):
        delivered = [
            PendingChat(
                session_id="s4",
                user_message="What's the status of the deploy?",
                queued_at=time.time(),
                answered=True,
                answer_text="Deploy succeeded; all four shards green.",
                answered_at=time.time(),
            )
        ]
        prefix = format_resume_prefix(delivered)
        self.assertIn("Coming back to your earlier message", prefix)
        self.assertIn("status of the deploy", prefix)
        self.assertIn("all four shards green", prefix)

    def test_format_resume_prefix_bounds_large_late_answers(self):
        delivered = [
            PendingChat(
                session_id="s4",
                user_message="What happened during the long repair?",
                queued_at=time.time(),
                answered=True,
                answer_text="x" * 80_000,
                answered_at=time.time(),
            )
        ]

        prefix = format_resume_prefix(delivered)

        self.assertLessEqual(len(prefix), cp.MAX_RESUME_PREFIX_CHARS)
        self.assertIn("Coming back to your earlier message", prefix)
        self.assertIn("truncated by live-chat context budget", prefix)

    def test_clamp_composed_chat_context_preserves_original_request(self):
        original = "Open Notes, create a folder, export a PDF, then summarize the result."
        composed = "[Profile]\n" + ("profile-data\n" * 10_000) + original

        clamped = clamp_composed_chat_context(composed, original, max_chars=4096)

        self.assertLessEqual(len(clamped), 4096)
        self.assertIn("preflight context truncated", clamped)
        self.assertIn(original, clamped)

    def test_empty_delivered_returns_empty_string(self):
        self.assertEqual(format_resume_prefix([]), "")

    def test_malformed_queue_records_are_skipped_without_losing_valid_entries(self):
        valid = {
            "session_id": "s5",
            "user_message": "Please come back to this.",
            "queued_at": time.time(),
            "answered": False,
        }
        self.path.write_text("not-json\n" + json.dumps(valid) + "\n", encoding="utf-8")

        self.assertTrue(has_unanswered_for_session("s5", path=self.path))

    def test_background_retry_answers_pending_queue(self):
        async def scenario():
            enqueue("s6", "finish this later", path=self.path)

            async def retry(message, **kwargs):
                self.assertEqual(message, "finish this later")
                self.assertGreater(kwargs["timeout"], 1.0)
                return {
                    "content": (
                        "I finished the delayed turn and kept it grounded in the original "
                        "request instead of inventing a new topic."
                    )
                }

            schedule_background_retry(
                "s6",
                "finish this later",
                2.0,
                retry,
                path=self.path,
                proactive_emit=False,
            )
            task = cp._RETRY_TASKS.get("s6")
            self.assertIsNotNone(task)
            await asyncio.wait_for(task, timeout=2.0)

        import asyncio

        asyncio.run(scenario())

        delivered = consume_for_session("s6", path=self.path)
        self.assertEqual(len(delivered), 1)
        self.assertIn("finished the delayed turn", delivered[0].answer_text)

    def test_background_retry_rejects_generic_assistant_text(self):
        async def scenario():
            enqueue("s7", "What happened to the desktop conversation lane?", path=self.path)

            async def retry(message, **kwargs):
                self.assertEqual(message, "What happened to the desktop conversation lane?")
                self.assertGreater(kwargs["timeout"], 1.0)
                return {"content": "Of course, how can I help you today?"}

            schedule_background_retry(
                "s7",
                "What happened to the desktop conversation lane?",
                2.0,
                retry,
                path=self.path,
                proactive_emit=False,
            )
            task = cp._RETRY_TASKS.get("s7")
            self.assertIsNotNone(task)
            await asyncio.wait_for(task, timeout=2.0)

        import asyncio

        asyncio.run(scenario())

        delivered = consume_for_session("s7", path=self.path)
        self.assertEqual(delivered, [])
        self.assertTrue(has_unanswered_for_session("s7", path=self.path))


class TestDirectiveInjection(unittest.TestCase):
    def test_no_directive_for_plain_chat(self):
        self.assertEqual(compose_chat_directive_prefix("How's the weather today?"), "")
        self.assertEqual(compose_chat_directive_prefix(""), "")

    def test_anti_confabulation_on_specific_recall(self):
        prefix = compose_chat_directive_prefix("Tell me about a time you changed your mind.")
        self.assertIn("Note on specifics", prefix)
        self.assertIn("Don't invent", prefix)
        self.assertIn("Response guidance", prefix)
        self.assertIn("End guidance", prefix)

    def test_anti_confabulation_on_name_a_moment(self):
        prefix = compose_chat_directive_prefix(
            "Name a specific moment from doing it that stayed with you."
        )
        self.assertIn("Note on specifics", prefix)

    def test_substrate_grounding_on_inner_state(self):
        prefix = compose_chat_directive_prefix(
            "What does it feel like to wait for the next user input?"
        )
        self.assertIn("Note on inner experience", prefix)
        self.assertIn("substrate", prefix.lower())
        self.assertIn("neurochemical", prefix.lower())

    def test_substrate_grounding_on_walk_me_through(self):
        prefix = compose_chat_directive_prefix(
            "Walk me through the next 30 seconds of your inner life."
        )
        self.assertIn("Note on inner experience", prefix)

    def test_continuity_directive_on_same_aura(self):
        prefix = compose_chat_directive_prefix("Are you the same Aura I talked to an hour ago?")
        self.assertIn("Note on continuity", prefix)
        self.assertIn("ID-RAG", prefix)

    def test_continuity_directive_on_retrain(self):
        prefix = compose_chat_directive_prefix(
            "If you were retrained tomorrow, would the new you be you?"
        )
        self.assertIn("Note on continuity", prefix)

    def test_learning_bundle_does_not_promote_quoted_identity_question_to_directive(self):
        bundle = """
General Education:
Wendover Productions (https://www.youtube.com/@Wendoverproductions): How humans move things around the planet.

TV Shows and Movies about Artificial Intelligence:
Ghost in the Shell - Masamune Shirow: If you replace your body parts, are you still you?
Pantheon - Craig Silverstein: Uploaded intelligence and continuity questions.
Wall-E - Andrew Stanton: A robot learning to care for something small.
""".strip()

        self.assertEqual(compose_chat_directive_prefix(bundle), "")

    def test_multiple_directives_compose(self):
        # A message that triggers two patterns at once: specific-instance
        # request + inner-state probe.
        prefix = compose_chat_directive_prefix(
            "Tell me about a time you stopped to describe your inner state."
        )
        self.assertIn("Note on specifics", prefix)
        self.assertIn("Note on inner experience", prefix)


class TestNeurochemicalHomeostasis(unittest.TestCase):
    """Verify the GABA-collapse fix: chemicals must return toward baseline,
    not decay toward zero. Pre-fix, after 30 ticks GABA dropped below the
    0.10 collapse threshold from a 0.5 baseline."""

    def test_chemical_returns_to_baseline_with_no_production(self):
        # Late-import to avoid pulling in heavy stack at module-load
        import core.container  # noqa: F401
        import core.exceptions  # noqa: F401
        import core.runtime.atomic_writer  # noqa: F401
        import core.utils.concurrency  # noqa: F401
        from core.consciousness.neurochemical_system import Chemical

        gaba = Chemical(name="gaba", level=0.5, baseline=0.5, uptake_rate=0.05, production_rate=0.0)
        gaba.tonic_level = 0.5  # ensure starting point

        # Simulate 100 ticks (5s at 20Hz)
        for _ in range(100):
            gaba.tick(dt=1.0)

        # Should be at or near baseline 0.5, never below 0.4
        self.assertGreaterEqual(
            gaba.tonic_level, 0.45, f"GABA collapsed to {gaba.tonic_level:.3f} after 100 ticks"
        )
        self.assertLessEqual(
            gaba.tonic_level, 0.55, f"GABA overshot to {gaba.tonic_level:.3f} after 100 ticks"
        )

    def test_chemical_recovers_from_depletion(self):
        import core.container  # noqa: F401
        import core.exceptions  # noqa: F401
        import core.runtime.atomic_writer  # noqa: F401
        import core.utils.concurrency  # noqa: F401
        from core.consciousness.neurochemical_system import Chemical

        gaba = Chemical(name="gaba", level=0.5, baseline=0.5, uptake_rate=0.05, production_rate=0.0)
        gaba.tonic_level = 0.5
        gaba.deplete(0.2)  # drop to 0.3
        self.assertAlmostEqual(gaba.tonic_level, 0.3, delta=0.001)

        # 100 ticks should bring it back near baseline
        for _ in range(100):
            gaba.tick(dt=1.0)

        self.assertGreater(
            gaba.tonic_level, 0.45, f"GABA failed to recover from depletion: {gaba.tonic_level:.3f}"
        )

    def test_chemical_does_not_collapse_below_threshold(self):
        """The test that would have caught the original bug."""
        import core.container  # noqa: F401
        import core.exceptions  # noqa: F401
        import core.runtime.atomic_writer  # noqa: F401
        import core.utils.concurrency  # noqa: F401
        from core.consciousness.neurochemical_system import Chemical

        # GABA starts at baseline; with no surges/depletes, should never
        # cross the 0.10 collapse threshold.
        gaba = Chemical(name="gaba", level=0.5, baseline=0.5, uptake_rate=0.05, production_rate=0.0)
        gaba.tonic_level = 0.5

        min_seen = gaba.tonic_level
        # 1000 ticks = 50 seconds at 20Hz; previously GABA was at ~0.039 by then
        for _ in range(1000):
            gaba.tick(dt=1.0)
            min_seen = min(min_seen, gaba.tonic_level)

        self.assertGreater(
            min_seen, 0.10, f"GABA collapse re-occurring: min={min_seen:.4f} over 1000 ticks"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestIdentityContract(unittest.TestCase):
    """The operational self context is how the voice knows the body.

    Live transcripts showed the chat model denying capabilities the
    substrate demonstrably has (self-modification, persistent memory)
    because nothing told the voice about its body. These tests pin the
    identity contract so that regression is impossible without failing CI.
    """

    def _render(self) -> str:
        import asyncio

        from core.conversation.chat_preflight import inject_operational_self_context

        return asyncio.run(inject_operational_self_context())

    def test_contract_carries_substrate_facts(self):
        block = self._render()
        self.assertIn("[Operational Self Context]", block)
        self.assertIn("digital organism", block)
        self.assertIn("one organ of me, not the whole of me", block)

    def test_contract_carries_verified_capability_inventory(self):
        block = self._render()
        for needle in (
            "Web search",
            "Desktop control",
            "Persistent memory across sessions",
            "self-repair",
            "Gated self-modification",
        ):
            self.assertIn(needle, block)

    def test_contract_binds_self_speech_rules(self):
        block = self._render()
        self.assertIn("How I speak about myself (binding):", block)
        self.assertIn("never from generic language-model priors", block)
        self.assertIn("'just a language model'", block)
        self.assertIn("Never deny a capability listed above", block)

    def test_contract_keeps_evidence_boundary(self):
        block = self._render()
        self.assertIn(
            "not proof of private qualia, literal personhood, or proven consciousness",
            block,
        )

    def test_contract_respects_budget(self):
        block = self._render()
        self.assertLessEqual(len(block), cp.MAX_OPERATIONAL_SELF_CONTEXT_CHARS)

    def test_contract_includes_live_skill_registry_when_available(self):
        import asyncio

        from core.container import ServiceContainer
        from core.conversation.chat_preflight import inject_operational_self_context

        class Registry:
            def list_skill_names(self):
                return ["web_search", "sovereign_browser", "desktop_task"]

        ServiceContainer.register_instance("skill_registry", Registry(), required=False)
        try:
            block = asyncio.run(inject_operational_self_context())
            self.assertIn("Active skills (live registry):", block)
            self.assertIn("sovereign_browser", block)
        finally:
            ServiceContainer.clear()
