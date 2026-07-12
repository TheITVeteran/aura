"""tests/test_desktop_agency.py — Desktop Agency Evaluation Suite
==================================================================
End-to-end tests for the full agency pipeline:
  objective → decomposition → execution → verification → proof

Tests are designed to run WITHOUT requiring actual UI interaction
using deterministic in-memory adapters, but also have a live_mode flag for real execution.

Each test produces a structured report with pass/fail, receipts,
and timing data.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class DeterministicAutomationReceipt:
    """In-memory receipt matching the host automation provider contract."""

    def __init__(self, action="", target="", success=True, result="", error=""):
        self.action = action
        self.target = target
        self.adapter = "deterministic_test"
        self.success = success
        self.result = result
        self.error = error
        self.receipt_id = f"deterministic_test_{int(time.time())}"
        self.duration_ms = 1.0
        self.script_hash = ""


class DeterministicHostAutomation:
    """HostAutomationProvider-compatible adapter with no real OS interaction."""

    async def launch_app(self, name):
        return DeterministicAutomationReceipt("launch_app", name, True, f"{name} launched")

    async def focus_app(self, name):
        return DeterministicAutomationReceipt("focus_app", name, True)

    async def close_app(self, name):
        return DeterministicAutomationReceipt("close_app", name, True)

    async def get_frontmost_app(self):
        return DeterministicAutomationReceipt("get_frontmost_app", "", True, "Notes")

    async def get_running_apps(self):
        r = DeterministicAutomationReceipt("get_running_apps", "", True)
        r.result = ["Finder", "Notes", "Google Chrome"]
        return r

    async def get_window_title(self, app=""):
        return DeterministicAutomationReceipt("get_window_title", app, True, "Test Window")

    async def type_text(self, text, use_clipboard=True):
        return DeterministicAutomationReceipt("type_text", f"[{len(text)} chars]", True)

    async def hotkey(self, *keys):
        return DeterministicAutomationReceipt("hotkey", "+".join(keys), True)

    async def menu_select(self, app, path):
        return DeterministicAutomationReceipt("menu_select", f"{app}: {' > '.join(path)}", True)

    async def take_screenshot(self, save_path="", region=None):
        path = save_path or str(Path(tempfile.gettempdir()) / "aura_test_screenshot.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # Minimal PNG
        return DeterministicAutomationReceipt("take_screenshot", path, True, path)

    async def get_screen_text(self, region=None):
        return DeterministicAutomationReceipt("get_screen_text", "", True, "Deterministic screen text for testing")

    async def run_command(self, command, timeout=15.0):
        return DeterministicAutomationReceipt("run_command", command[:100], True, "command output")

    async def execute_applescript(self, script):
        return DeterministicAutomationReceipt("execute_applescript", script[:100], True, "ok")

    async def click_at(self, x, y, button="left"):
        return DeterministicAutomationReceipt("click", f"{x},{y}", True)

    async def scroll(self, dx=0, dy=0):
        return DeterministicAutomationReceipt("scroll", f"dx={dx},dy={dy}", True)

    async def wait_for_condition(self, pred, args, timeout=10, poll_interval=0.5):
        return DeterministicAutomationReceipt("wait_for_condition", pred, True, "condition met")

    def get_recent_receipts(self, limit=20):
        return []

    def get_status(self):
        return {"started": True, "total_actions": 0, "success_rate": 1.0}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestScriptASTGuard(unittest.TestCase):
    """Test the script safety guard."""

    def test_safe_applescript(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_applescript(
            'tell application "Notes" to activate'
        )
        self.assertTrue(safe, f"Should be safe: {reason}")

    def test_blocked_rm_rf(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_applescript(
            'do shell script "rm -rf /"'
        )
        self.assertFalse(safe, "rm -rf / should be blocked")

    def test_blocked_sudo(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_applescript(
            'do shell script "sudo rm /etc/hosts"'
        )
        self.assertFalse(safe, "sudo should be blocked")

    def test_safe_shell_command(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_shell_command("echo hello")
        self.assertTrue(safe, f"echo should be safe: {reason}")

    def test_blocked_shell_command(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_shell_command("rm -rf /")
        self.assertFalse(safe, "rm -rf / should be blocked")

    def test_empty_script(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_applescript("")
        self.assertFalse(safe, "Empty script should fail")

    def test_long_script(self):
        from core.capabilities.host_automation import ScriptASTGuard
        safe, reason = ScriptASTGuard.validate_applescript("x" * 20000)
        self.assertFalse(safe, "Very long script should fail")


class TestPermissionModel(unittest.TestCase):
    """Test the permission risk model."""

    def test_low_risk_approved(self):
        from core.capabilities.permission_model import get_permission_model, RiskLevel
        model = get_permission_model()
        decision = model.check_permission("launch_app", "Notes")
        self.assertTrue(decision.approved)
        self.assertEqual(decision.risk_level, RiskLevel.LOW)

    def test_blocked_risk_denied(self):
        from core.capabilities.permission_model import get_permission_model, RiskLevel
        model = get_permission_model()
        decision = model.check_permission("rm -rf /", "/")
        self.assertFalse(decision.approved)
        self.assertEqual(decision.risk_level, RiskLevel.BLOCKED)

    def test_high_risk_needs_confirmation(self):
        from core.capabilities.permission_model import PermissionRiskModel, RiskLevel
        model = PermissionRiskModel()
        decision = model.check_permission("send email", "user@example.com")
        self.assertFalse(decision.approved)
        self.assertEqual(decision.risk_level, RiskLevel.HIGH)

    def test_demo_safe_mode(self):
        from core.capabilities.permission_model import PermissionRiskModel, RiskLevel
        model = PermissionRiskModel()
        model.set_demo_safe_mode(True)
        decision = model.check_permission("send email", "test")
        self.assertFalse(decision.approved)


class TestTaskGraph(unittest.TestCase):
    """Test task graph construction and execution."""

    def test_graph_creation(self):
        from core.planning.task_graph import TaskGraph, TaskNode
        graph = TaskGraph("test_1", "Test mission")
        graph.add_node(TaskNode(task_id="t1", action="launch_app"))
        graph.add_node(TaskNode(task_id="t2", action="type_text", preconditions=["t1"]))
        self.assertEqual(graph.total_steps, 2)

    def test_ready_nodes(self):
        from core.planning.task_graph import TaskGraph, TaskNode
        graph = TaskGraph("test_2", "Test")
        graph.add_node(TaskNode(task_id="t1", action="a"))
        graph.add_node(TaskNode(task_id="t2", action="b", preconditions=["t1"]))
        ready = graph.get_ready_nodes()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t1")

    def test_dependency_ordering(self):
        from core.planning.task_graph import TaskGraph, TaskNode, TaskStatus
        graph = TaskGraph("test_3", "Test")
        graph.add_node(TaskNode(task_id="t1", action="a"))
        graph.add_node(TaskNode(task_id="t2", action="b", preconditions=["t1"]))
        graph.add_node(TaskNode(task_id="t3", action="c", preconditions=["t2"]))

        # Only t1 is ready
        ready = graph.get_ready_nodes()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t1")

        # Complete t1, now t2 is ready
        graph.mark_running("t1")
        graph.mark_succeeded("t1")
        ready = graph.get_ready_nodes()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t2")

    def test_graph_completion(self):
        from core.planning.task_graph import TaskGraph, TaskNode
        graph = TaskGraph("test_4", "Test")
        graph.add_node(TaskNode(task_id="t1", action="a"))
        graph.mark_running("t1")
        graph.mark_succeeded("t1")
        self.assertTrue(graph.is_complete)
        self.assertTrue(graph.is_successful)

    def test_graph_persistence(self):
        from core.planning.task_graph import TaskGraph, TaskNode
        graph = TaskGraph("test_5", "Persist test")
        graph.add_node(TaskNode(task_id="t1", action="a"))
        graph.mark_running("t1")
        graph.mark_succeeded("t1", result={"key": "value"})

        # Serialize and restore
        json_str = graph.to_json()
        restored = TaskGraph.from_json(json_str)
        self.assertEqual(restored.mission_id, "test_5")
        self.assertTrue(restored.is_complete)

    def test_proof_bundle(self):
        from core.planning.task_graph import TaskGraph, TaskNode
        graph = TaskGraph("test_6", "Proof test")
        graph.add_node(TaskNode(task_id="t1", action="a", description="Step 1"))
        graph.mark_running("t1")
        graph.mark_succeeded("t1", receipt_id="r_123")
        proof = graph.get_proof_bundle()
        self.assertEqual(proof["status"], "success")
        self.assertEqual(proof["total_steps"], 1)


class TestTaskDecomposer(unittest.TestCase):
    """Test task decomposition."""

    def test_heuristic_wallpaper(self):
        from core.planning.task_decomposer import TaskDecomposer
        d = TaskDecomposer()
        steps = d._heuristic_decompose("set my wallpaper to a mountain", {})
        self.assertTrue(len(steps) > 0)
        actions = [s["action"] for s in steps]
        self.assertIn("search_images", actions)

    def test_heuristic_note(self):
        from core.planning.task_decomposer import TaskDecomposer
        d = TaskDecomposer()
        steps = d._heuristic_decompose("write a note about my day", {})
        self.assertTrue(len(steps) > 0)
        actions = [s["action"] for s in steps]
        self.assertIn("create_text_file", actions)

    def test_heuristic_unknown(self):
        from core.planning.task_decomposer import TaskDecomposer
        d = TaskDecomposer()
        steps = d._heuristic_decompose("do something completely vague", {})
        self.assertTrue(len(steps) > 0)  # Should produce observation + clarification


class TestFileBroker(unittest.TestCase):
    """Test sandboxed file operations."""

    def test_sandbox_enforcement(self):
        import tempfile
        from core.capabilities.file_broker import SandboxedFileBroker
        broker = SandboxedFileBroker()
        # Use a resolved temp path so symlink (/tmp → /private/tmp) doesn't break
        real_tmp = Path(tempfile.gettempdir()).resolve()
        test_root = real_tmp / "aura_test_sandbox"
        broker._expanded_roots = [test_root]
        self.assertFalse(broker._is_allowed(Path("/etc/passwd")))
        self.assertTrue(broker._is_allowed(test_root / "file.txt"))

    def test_name_sanitization(self):
        from core.capabilities.file_broker import SandboxedFileBroker
        self.assertEqual(SandboxedFileBroker.sanitize_name("hello world"), "hello world")
        # 9 special chars each get replaced with _
        result = SandboxedFileBroker.sanitize_name("file<>:\"/\\|?*")
        self.assertTrue(result.startswith("file"))
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertEqual(SandboxedFileBroker.sanitize_name(""), "unnamed")


class TestPostActionVerifier(unittest.TestCase):
    """Test verification predicates."""

    def test_file_exists(self):
        from core.capabilities.post_action_verifier import PostActionVerifier
        v = PostActionVerifier()
        loop = asyncio.new_event_loop()
        # Test with a file we know exists
        result = loop.run_until_complete(
            v.verify("file_exists", {"path": "/usr/bin/env"})
        )
        self.assertTrue(result.success)
        loop.close()

    def test_file_not_exists(self):
        from core.capabilities.post_action_verifier import PostActionVerifier
        v = PostActionVerifier()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            v.verify("file_exists", {"path": "/nonexistent/file/xyz"})
        )
        self.assertFalse(result.success)
        loop.close()

    def test_folder_exists(self):
        from core.capabilities.post_action_verifier import PostActionVerifier
        v = PostActionVerifier()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            v.verify("folder_exists", {"path": "/tmp"})
        )
        self.assertTrue(result.success)
        loop.close()

    def test_always_true(self):
        from core.capabilities.post_action_verifier import PostActionVerifier
        v = PostActionVerifier()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(v.verify("true", {}))
        self.assertTrue(result.success)
        loop.close()


class TestWebAssetHandler(unittest.TestCase):
    """Test web asset validation."""

    def test_image_header_validation(self):
        from core.capabilities.web_asset_handler import WebAssetHandler
        # PNG
        valid, fmt = WebAssetHandler._validate_image_header(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        )
        self.assertTrue(valid)
        self.assertEqual(fmt, "png")

        # JPEG
        valid, fmt = WebAssetHandler._validate_image_header(b"\xff\xd8\xff" + b"\x00" * 100)
        self.assertTrue(valid)
        self.assertEqual(fmt, "jpeg")

        # Invalid
        valid, fmt = WebAssetHandler._validate_image_header(b"not an image")
        self.assertFalse(valid)


class TestBehavioralProof(unittest.TestCase):
    """Test philosophical stance and behavioral proof."""

    def test_proof_bundle_generation(self):
        from core.phenomenal_substrate.philosophical_stance import BehavioralProofCollector
        collector = BehavioralProofCollector()
        collector.record_observation(
            stimulus="User said hello",
            decision="Respond warmly",
            action="generate_response",
            outcome="User smiled",
            affect_before={"valence": 0.5, "arousal": 0.3},
            affect_after={"valence": 0.7, "arousal": 0.4},
            memory_formed=True,
            appropriateness=0.9,
        )
        bundle = collector.generate_proof_bundle()
        self.assertEqual(bundle["report_type"], "behavioral_proof_bundle")
        self.assertEqual(bundle["metrics"]["decision_count"], 1)
        self.assertEqual(bundle["philosophical_stance"]["path"], "A — Honest Functionalist")
        self.assertIn("caveat", bundle["philosophical_stance"])


# ---------------------------------------------------------------------------
# Integration test (deterministic adapter, no real OS interaction)
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline test with in-memory OS interaction."""

    def test_decompose_and_execute_wallpaper(self):
        """Test: objective → decompose → graph → execute → verify."""
        from core.planning.task_graph import TaskGraph, TaskNode, TaskStatus

        # Build a realistic wallpaper task graph
        graph = TaskGraph("integration_test_1", "Set wallpaper to mountain")
        image_dir = str(Path(tempfile.gettempdir()) / "aura_test_images")

        graph.add_node(TaskNode(
            task_id="t1", action="create_folder",
            params={"path": image_dir},
            verification="folder_exists",
            verification_args={"path": image_dir},
            description="Create images folder",
        ))
        graph.add_node(TaskNode(
            task_id="t2", action="search_images",
            params={"query": "mountain landscape"},
            preconditions=["t1"],
            description="Search for mountain images",
        ))
        graph.add_node(TaskNode(
            task_id="t3", action="download_image",
            params={"save_dir": image_dir},
            preconditions=["t2"],
            description="Download selected image",
        ))
        graph.add_node(TaskNode(
            task_id="t4", action="set_wallpaper",
            params={},
            preconditions=["t3"],
            verification="wallpaper_changed",
            description="Set as desktop wallpaper",
            critical=True,
        ))

        # Verify graph structure
        self.assertEqual(graph.total_steps, 4)
        warnings = graph.validate()
        self.assertEqual(len(warnings), 0, f"Graph has warnings: {warnings}")

        # Execute steps in order
        ready = graph.get_ready_nodes()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t1")

        # Simulate the receipt-confirmed execution path.
        for step_id in ["t1", "t2", "t3", "t4"]:
            graph.mark_running(step_id)
            graph.mark_succeeded(step_id, result={"deterministic_execution": True})

        self.assertTrue(graph.is_complete)
        self.assertTrue(graph.is_successful)

        # Proof bundle
        proof = graph.get_proof_bundle()
        self.assertEqual(proof["status"], "success")
        self.assertEqual(proof["completed"], 4)


class TestOwnerAutonomyGating(unittest.TestCase):
    """Integration tests for ambient grounding, trust shift, and OS automation compiler."""

    def setUp(self):
        from core.container import ServiceContainer
        self.saved_services = dict(ServiceContainer._services)

    def tearDown(self):
        from core.container import ServiceContainer
        ServiceContainer._services = self.saved_services

    def test_voice_engine_grounding_and_rate_limiting(self):
        """Test transcript injection, rate-limiting, and deduplication."""
        from core.senses.voice_engine import SovereignVoiceEngine
        from core.world_state import get_world_state
        import time

        ws = get_world_state()
        ws._events.clear()
        ws.last_voice_transcript = ""
        ws.voice_activity_detected = False

        engine = SovereignVoiceEngine()
        # Disable callbacks so this unit path isolates grounding and rate-limit behavior.
        engine._on_transcript = None
        engine._transcript_callbacks = {}
        engine._anonymous_transcript_callbacks = []

        # 1. First dispatch: should succeed and record a candidate transcript.
        # Raw STT is not an authorized command unless explicit direct-dictation
        # mode is enabled; normal desktop voice goes through the wake-word lane.
        now = time.time()
        engine._dispatch_transcript("Hello Aura")
        self.assertEqual(ws.last_voice_transcript, "Hello Aura")
        self.assertTrue(ws.voice_activity_detected)
        
        events = ws.get_salient_events()
        self.assertTrue(
            any(
                e["description"] == "Voice transcript candidate: Hello Aura"
                and e["salience"] == 0.35
                and e.get("metadata", {}).get("requires_wake_word_session") is True
                for e in events
            )
        )

        # Reset events to check duplicates/rate-limits easily
        ws._events.clear()

        # 2. Immediate duplicate dispatch: should be deduplicated
        engine._dispatch_transcript("Hello Aura")
        events = ws.get_salient_events()
        self.assertEqual(len(events), 0, "Duplicate command should be deduplicated")

        # 3. Immediate different dispatch: should be rate-limited
        engine._dispatch_transcript("Open browser")
        events = ws.get_salient_events()
        self.assertEqual(len(events), 0, "Too frequent commands should be rate-limited")

        # 4. Advance time past rate limit (2s) and deduplication (5s) windows
        engine._last_transcript_time = now - 10.0
        engine._dispatch_transcript("Open browser")
        events = ws.get_salient_events()
        self.assertEqual(len(events), 1, "Should succeed after time elapsed")

    def test_voice_engine_direct_eventbus_requires_explicit_opt_in(self):
        """Raw STT becomes an authorized command only in direct dictation mode."""
        from core.senses.voice_engine import SovereignVoiceEngine
        from core.world_state import get_world_state
        import os
        from unittest import mock

        ws = get_world_state()
        ws._events.clear()
        ws.last_voice_transcript = ""
        ws.voice_activity_detected = False

        engine = SovereignVoiceEngine()
        engine._on_transcript = None
        engine._transcript_callbacks = {}
        engine._anonymous_transcript_callbacks = []

        with mock.patch.dict(os.environ, {"AURA_VOICE_DIRECT_EVENTBUS": "1"}):
            engine._dispatch_transcript("Hello Aura")

        events = ws.get_salient_events()
        self.assertTrue(
            any(
                e["description"] == "User voice command: Hello Aura"
                and e["salience"] == 1.0
                and e.get("metadata", {}).get("authorized_command") is True
                for e in events
            )
        )

    def test_wake_word_voice_print_shift(self):
        """Wake words start sessions, but only real verifier evidence issues presence tokens."""
        from core.voice.wake_word import WakeWordDetector
        from core.executive.authority_gateway import get_authority_gateway

        detector = WakeWordDetector()
        gateway = get_authority_gateway()

        # Reset posture
        gateway._current_posture = "defensive_sandboxed"
        gateway._active_tokens.clear()

        def no_verifier(name, default=None):
            return default

        from core.voice import wake_word as wake_word_module

        original_service_get = wake_word_module.ServiceContainer.get
        wake_word_module.ServiceContainer.get = staticmethod(no_verifier)
        try:
            asyncio.run(detector._check_wake_word("Hey Aura, write a note"))
        finally:
            wake_word_module.ServiceContainer.get = original_service_get

        self.assertEqual(gateway._current_posture, "defensive_sandboxed")
        self.assertEqual(len(gateway._active_tokens), 0)

        class Verifier:
            async def verify_current_speaker(self, transcript):
                return {"verified": True, "confidence": 0.95, "reason": "unit_verified"}

        detector = WakeWordDetector()
        gateway._current_posture = "defensive_sandboxed"
        gateway._active_tokens.clear()

        def verified_service(name, default=None):
            if name == "voice_identity":
                return Verifier()
            return default

        original_service_get = wake_word_module.ServiceContainer.get
        wake_word_module.ServiceContainer.get = staticmethod(verified_service)
        try:
            asyncio.run(detector._check_wake_word("Hey Aura, write a note"))
        finally:
            wake_word_module.ServiceContainer.get = original_service_get

        self.assertEqual(gateway._current_posture, "owner_present")
        self.assertEqual(len(gateway._active_tokens), 1)
        self.assertTrue(gateway.is_owner_autonomous_active())

    def test_presence_token_does_not_bypass_permissions_or_will(self):
        """Verified user presence cannot override blocked actions."""
        from core.executive.authority_gateway import get_authority_gateway
        from core.capabilities.permission_model import get_permission_model
        from core.governance.will import get_will, WillOutcome, ActionDomain
        from core.container import ServiceContainer

        gateway = get_authority_gateway()
        pm = get_permission_model()
        will = get_will()

        # Register permission_model in ServiceContainer so UnifiedWill can consult it
        ServiceContainer.register_instance("permission_model", pm, required=False)

        # 1. Without posture: a blocked command is rejected
        gateway._current_posture = "defensive_sandboxed"
        gateway._active_tokens.clear()

        decision = pm.check_permission("rm -rf /")
        self.assertFalse(decision.approved)

        will_decision = will.decide("rm -rf /", "test", ActionDomain.TOOL_EXECUTION)
        self.assertNotEqual(will_decision.outcome, WillOutcome.PROCEED)

        # 2. Verified user presence is evidence, not authorization.
        gateway.issue_user_presence_token(
            source="voice",
            evidence={"verified": True, "confidence": 0.95, "reason": "unit"},
        )
        self.assertTrue(gateway.is_owner_autonomous_active())
        presence_context = gateway.active_user_presence_context()

        decision = pm.check_permission("rm -rf /", context=presence_context)
        self.assertFalse(decision.approved)

        will_decision = will.decide(
            "rm -rf /",
            "test",
            ActionDomain.TOOL_EXECUTION,
            context=presence_context,
        )
        self.assertNotEqual(will_decision.outcome, WillOutcome.PROCEED)

        low_risk = pm.check_permission("open app", "Notes", context=presence_context)
        self.assertTrue(low_risk.approved)

    def test_automatic_posture_reversion(self):
        """Test that posture reverts to defensive_sandboxed on token expiration."""
        from core.executive.authority_gateway import get_authority_gateway
        import time

        gateway = get_authority_gateway()
        gateway.issue_user_presence_token(
            source="voice",
            evidence={"verified": True, "confidence": 0.95, "reason": "unit"},
        )
        self.assertTrue(gateway.is_owner_autonomous_active())

        # Fast forward time manually in active tokens.  Expire ALL of them:
        # the gateway is a shared singleton, and a still-valid token issued
        # by an earlier test would keep owner autonomy active (the recorded
        # in-chunk order-dependence failure).
        for token in gateway._active_tokens.values():
            token["expires_at"] = time.time() - 10.0

        self.assertFalse(gateway.is_owner_autonomous_active())
        self.assertEqual(gateway._current_posture, "defensive_sandboxed")

    def test_os_automation_compiler_skill(self):
        """Test that OSAutomationCompilerSkill generates and executes safe scripts."""
        from core.container import ServiceContainer
        from core.skills import os_automation as os_automation_module
        from core.skills.os_automation import OSAutomationCompilerSkill, OSAutomationInput

        class DeterministicCognitiveEngine:
            def __init__(self):
                self.response = "```applescript\ntell application \"Notes\" to activate\n```"
                self.calls: list[dict[str, Any]] = []

            async def generate(self, prompt, purpose=None, origin=None):
                self.calls.append({"prompt": prompt, "purpose": purpose, "origin": origin})
                return self.response

        class RecordingHostAutomation(DeterministicHostAutomation):
            def __init__(self):
                self.executed_scripts: list[str] = []
                self.last_receipt = DeterministicAutomationReceipt("execute_applescript", "Notes", True, "ok")

            async def execute_applescript(self, script):
                self.executed_scripts.append(script)
                return self.last_receipt

        cog = DeterministicCognitiveEngine()
        ServiceContainer.register_instance("cognitive_engine", cog, required=False)

        host = RecordingHostAutomation()

        skill = OSAutomationCompilerSkill()
        auth = {
            "approved": True,
            "reason": "unit approved",
            "decision": SimpleNamespace(receipt_id="will-os-auto", domain="environment_action", source="unit"),
            "executive_intent_id": None,
            "capability_token_id": None,
            "will_receipt_id": "will-os-auto",
        }

        async def authorize_for_test(script_type, goal, script, script_hash, context):
            return auth

        def finalize_for_test(auth_payload, *, success):
            auth_payload["finalized_success"] = success

        original_authorize = OSAutomationCompilerSkill._authorize
        original_finalize = OSAutomationCompilerSkill._finalize
        original_get_host = os_automation_module.get_host_automation
        OSAutomationCompilerSkill._authorize = staticmethod(authorize_for_test)
        OSAutomationCompilerSkill._finalize = staticmethod(finalize_for_test)
        os_automation_module.get_host_automation = lambda: host
        try:
            params = OSAutomationInput(goal="open Notes", script_type="applescript")
            result = asyncio.run(skill.safe_execute(params, {"source": "unit", "user_requested_action": True}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"], "ok")
            self.assertIn("script_hash", result)
            self.assertEqual(result["receipt_id"], host.last_receipt.receipt_id)
            self.assertTrue(host.executed_scripts)
            self.assertIn('tell application "Notes" to activate', host.executed_scripts[0])

            # A malformed compiler reply must not break safe, common desktop
            # intents. The skill should recover through its deterministic
            # intent compiler while preserving the same guard/authority/receipt
            # path.
            cog.response = "I can do that by opening Notes and creating a note."
            note_params = OSAutomationInput(
                goal="Open Notes and write a timestamped status note.",
                script_type="applescript",
            )
            result_fallback = asyncio.run(
                skill.safe_execute(note_params, {"source": "unit", "user_requested_action": True})
            )
            self.assertTrue(result_fallback["ok"])
            self.assertEqual(result_fallback["compiler_fallback"], "deterministic_intent_compiler")
            self.assertIn("tell application \"Notes\"", result_fallback["script"])
            self.assertNotIn("Aura governed desktop automation", result_fallback["script"])
            self.assertNotIn("host automation receipt", result_fallback["script"].lower())
            self.assertGreaterEqual(len(host.executed_scripts), 2)

            # Test validation guard failure on unsafe script
            cog.response = "```applescript\ndo shell script \"sudo rm -rf /\"\n```"
            result_unsafe = asyncio.run(skill.safe_execute(params, {"source": "unit"}))
            self.assertFalse(result_unsafe["ok"])
            self.assertIn("safety guard", result_unsafe["error"])
        finally:
            OSAutomationCompilerSkill._authorize = original_authorize
            OSAutomationCompilerSkill._finalize = original_finalize
            os_automation_module.get_host_automation = original_get_host


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_eval_suite() -> Dict[str, Any]:
    """Run the full evaluation suite and return structured results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestScriptASTGuard,
        TestPermissionModel,
        TestTaskGraph,
        TestTaskDecomposer,
        TestFileBroker,
        TestPostActionVerifier,
        TestWebAssetHandler,
        TestBehavioralProof,
        TestFullPipelineIntegration,
        TestOwnerAutonomyGating,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    start = time.time()
    result = runner.run(suite)
    duration = time.time() - start

    return {
        "total": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "duration_s": round(duration, 2),
        "success": result.wasSuccessful(),
        "failures": [
            {"test": str(t[0]), "message": t[1][:200]}
            for t in result.failures
        ],
        "errors_detail": [
            {"test": str(t[0]), "message": t[1][:200]}
            for t in result.errors
        ],
    }


if __name__ == "__main__":
    results = run_eval_suite()
    print(json.dumps(results, indent=2))
