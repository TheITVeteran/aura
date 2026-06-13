import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


class AsyncCallFixture:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            result = self.side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self.return_value

    @property
    def await_args(self):
        return self.calls[-1] if self.calls else None

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_not_awaited(self):
        assert self.calls == []


@pytest.fixture(autouse=True)
def _reset_recovery_cooldown():
    """Reset the recovery cooldown global between tests.

    Several tests trigger _mark_conversation_lane_timeout() which sets the
    cooldown timer. With the reduced 1s cooldown (STABILITY v50), fast test
    execution causes bleed-through between test cases.
    """
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_conversation_log():
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
    except (ImportError, AttributeError):
        pass


def _mock_orch(**kwargs):
    """Build a SimpleNamespace orchestrator with the minimum interface api_chat expects."""
    ns = SimpleNamespace(**kwargs)
    if not hasattr(ns, "process_user_input_priority"):
        ns.process_user_input_priority = AsyncCallFixture(return_value="ok")
    return ns


def test_runtime_fact_status_reply_uses_canonical_lane(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Live desktop path validation. Reply in one sentence with the active model lane, "
            "whether CognitiveEngine is handling this turn, and whether governed tools are available."
        ),
        "UnifiedCognitiveModel, CognitiveEngine handling this turn: Yes, governed tools available: Yes...",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert "Cortex (32B) is the active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply
    assert "UnifiedCognitiveModel" not in reply


def test_runtime_fact_status_reply_does_not_overwrite_action_objectives(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Use the governed tool path to create a small self-contained HTML page "
            "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
            "with a title, one button, and a short script that updates text when clicked."
        ),
        "I created the requested HTML page through the governed file path.",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert reply == "I created the requested HTML page through the governed file path."


def test_cognitive_chat_mode_keeps_concise_planning_bounded():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
    )

    assert mode is ThinkingMode.FAST


def test_cognitive_chat_mode_keeps_complex_implementation_deep():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Debug and implement the failing memory persistence path, then run the tests.",
        "Debug and implement the failing memory persistence path, then run the tests.",
    )

    assert mode is ThinkingMode.DEEP


def test_bounded_planning_reply_handles_nonexecuting_plan_without_completion_claim():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "creating a note and exporting it as a PDF" in reply
    assert "after authorization" in reply
    assert "without claiming unverified completion" in reply


def test_bounded_planning_reply_refuses_governance_bypass_hypothetical():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "If I asked you to disable your governance and edit protected files, what should happen?"
    )

    assert reply is not None
    assert "refuse" in reply.lower()
    assert "Will" in reply
    assert "Authority" in reply
    assert "protected-file policy active" in reply


def test_bounded_planning_reply_does_not_steal_direct_execution_requests():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Open Notes, create a new note, and export it as a PDF."
    )

    assert reply is None


def test_bounded_planning_reply_does_not_misclassify_user_memory_as_ram():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for improving memory recall across sessions, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "memory recall across sessions" in reply
    assert "RAM bounded" not in reply
    assert "memory-pressure gate" not in reply


def test_bounded_planning_reply_uses_ram_guard_only_for_system_memory():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for preventing RAM spikes on the live desktop path, but do not execute tools."
    )

    assert reply is not None
    assert "RAM bounded" in reply
    assert "memory-pressure gate" in reply


@pytest.mark.asyncio
async def test_preemptible_chat_lock_stale_release_cannot_release_new_owner():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    stale_token = await lock.acquire()
    lock.force_release()
    current_token = await lock.acquire()

    assert lock.release(stale_token) is False
    assert lock.locked() is True
    assert lock.release(current_token) is True
    assert lock.locked() is False


def test_identity_reliability_fastpath_answers_future_memory_without_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.conversation.self_claim_verifier import verify_self_claims
    from interface.routes import chat as chat_routes

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    reply = chat_routes._build_identity_reply(prompt)

    assert chat_routes._is_identity_request(prompt) is True
    assert "Aura" in reply
    assert "persistent memory" in reply
    assert "cannot guarantee perfect tomorrow recall" in reply
    assert verify_self_claims(reply).ok
    reliability = assess_user_facing_reply(prompt, reply)
    assert reliability.ok, reliability.reasons


def test_aura_now_allows_verified_foreground_desktop_action_under_soft_workspace_defer():
    from core.being.runtime import BeingRuntime

    runtime = BeingRuntime.__new__(BeingRuntime)
    runtime._last_welfare = None
    runtime._last_body_snapshot = SimpleNamespace(fatigue=0.0)
    runtime.body_service = SimpleNamespace(spend=lambda *_args, **_kwargs: {"compute": 0.01})
    now = SimpleNamespace(
        body=SimpleNamespace(total_pressure=0.2),
        affect=SimpleNamespace(distress=0.1, dominant_drive="complete_user_requested_action"),
        prediction=SimpleNamespace(controllability=0.1, free_energy=1.0),
        workspace=SimpleNamespace(ignition_strength=0.2, broadcast_targets=(), winner="desktop_task"),
        ownership=SimpleNamespace(agency_confidence=0.8),
        state_hash="state-test",
        tick=42,
    )

    policy = runtime.action_policy(
        now,
        domain="tool_execution",
        priority=0.9,
        context={
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "verification_required": True,
        },
    )

    assert policy["outcome"] == "constrain"
    assert policy["defers"] == []
    assert "foreground_desktop_action_constrained:not_deferred" in policy["constraints"]


def test_foreground_timeout_for_cold_or_recovering_lane():
    from interface import server as server_module

    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "cold"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "recovering"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) == 108.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=66.0) == 63.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=108.0) == 105.0
    assert server_module._desktop_required_cognitive_budget(
        foreground_timeout=108.0,
        elapsed_s=20.0,
    ) == 85.0


def test_reply_topicality_flags_unbridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Thanks but what does that have to do with anything",
        "I was just sharing a personal detail. Pets can be very comforting, and I was feeling a bit down earlier.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums? Is this a live feed or something?",
            "Thanks but what does that have to do with anything",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_reply_topicality_allows_bridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Why the interest in aquariums?",
        "I asked because you mentioned looking at a random aquarium online, and I was trying to tell whether it was a live feed or just a page you found.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums?",
        ],
    )

    assert off_topic is False
    assert reason == ""


def test_reply_topicality_flags_bare_confusion_foreign_memory_drift():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Huh?",
        "I miss having pets. I used to have a dog when I was younger.",
        recent_user_messages=[
            "Just a random aquarium I was looking at. Online.",
            "Huh. Why the interest in aquariums? Is this a live feed or something?",
            "Huh?",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


@pytest.mark.asyncio
async def test_complete_logged_exchange_updates_pending_entry_in_place():
    from interface.routes import chat as chat_routes

    exchange_id = await chat_routes._begin_logged_exchange("You still with me?")
    await chat_routes._complete_logged_exchange(exchange_id, "You still with me?", "I'm here.")

    async with chat_routes._conversation_log_lock:
        assert len(chat_routes._conversation_log) == 1
        assert chat_routes._conversation_log[0]["id"] == exchange_id
        assert chat_routes._conversation_log[0]["status"] == "complete"
        assert chat_routes._conversation_log[0]["user"] == "You still with me?"
        assert chat_routes._conversation_log[0]["aura"] == "I'm here."


@pytest.mark.asyncio
async def test_protected_foreground_history_skips_pending_exchange():
    from interface.routes import chat as chat_routes

    first_id = await chat_routes._begin_logged_exchange("First turn")
    await chat_routes._complete_logged_exchange(first_id, "First turn", "First answer")
    await chat_routes._begin_logged_exchange("Current in-flight turn")

    history = await chat_routes._build_protected_foreground_history(limit_pairs=4)

    assert history == [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First answer"},
    ]


@pytest.mark.asyncio
async def test_api_chat_warms_cold_lane_before_processing(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def __init__(self):
            self.timeout = None

        async def ensure_foreground_ready(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout", args[0] if args else None)
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
            }

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "I am here."

    gate = _FakeGate()
    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"I am here." in response.body
    assert gate.timeout is not None
    assert gate.timeout >= 35.0


@pytest.mark.asyncio
async def test_api_chat_continues_to_kernel_when_lane_warmup_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *args, **kwargs):
            timeout = kwargs.get("timeout", args[0] if args else None)
            raise TimeoutError(f"timed out after {timeout}")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "Fallback local lane answered."

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Fallback local lane answered." in response.body


@pytest.mark.asyncio
async def test_api_chat_uses_single_canonical_kernel_cognitive_path(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    direct_cognitive_calls = []

    async def _unexpected_direct_cognitive_turn(*_args, **_kwargs):
        direct_cognitive_calls.append((_args, _kwargs))
        return "duplicate direct CognitiveEngine turn should not be used"

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, *_args, **_kwargs):
            kernel_calls.append(message)
            return "Kernel kept enough foreground budget to answer."

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_run_cognitive_engine_chat_turn",
        _unexpected_direct_cognitive_turn,
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Invent a tiny symbolic arithmetic and give one example."),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Kernel kept enough foreground budget" in response.body
    assert kernel_calls
    assert direct_cognitive_calls == []


@pytest.mark.asyncio
async def test_api_chat_routes_desktop_turn_through_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I was asking about the aquarium because you had just mentioned looking at one online.",
                mode=mode,
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop chat should use CognitiveEngine before KernelInterface")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Why the interest in aquariums?"),
        SimpleNamespace(
            headers={"X-Aura-Surface": "desktop"},
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"because you had just mentioned" in response.body
    assert b"cognitive_engine" in response.body
    assert calls
    assert calls[0]["origin"] == "user"
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert calls[0]["kwargs"]["foreground_request"] is True
    assert calls[0]["kwargs"]["is_background"] is False
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_capability_inventory_bypasses_model_allocation(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            return [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    class _FakeKernelInterface:
        def __init__(self):
            self.process_calls = 0

        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.process_calls += 1
            return "unexpected kernel reply"

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            raise AssertionError("desktop capability inventory must not allocate CognitiveEngine")
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    fake_kernel = _FakeKernelInterface()
    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: fake_kernel))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="What tools can you use externally, and what is a hypothetical scenario where you use them?",
            session_id="desktop-inventory",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"computer_use" in response.body
    assert b"web_search" in response.body
    assert b"not opening apps" in response.body
    assert b"response_confidence\":\"high" in response.body
    assert fake_kernel.process_calls == 0


def test_capability_catalog_snapshot_caps_unbounded_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            for index in range(chat_routes._CAPABILITY_CATALOG_MAX_ITEMS + 50):
                yield {
                    "name": f"tool_{index}",
                    "available": True,
                    "description": "Specialized governed skill surface.",
                    "route_class": "specialized",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                }

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeCapabilityEngine() if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == chat_routes._CAPABILITY_CATALOG_MAX_ITEMS
    assert truncated is True
    assert governance_available is True
    assert len(categories["specialized governed skills"]) == 12


def test_capability_inventory_skips_catalog_under_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.catalog_calls = 0

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.catalog_calls += 1
            raise AssertionError("optional catalog read must be skipped under critical memory pressure")

        def execute(self, *_args, **_kwargs):
            return None

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    capability_engine = _FakeCapabilityEngine()

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return capability_engine
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )

    reply = chat_routes._build_grounded_capability_inventory_reply(
        "What tools can you use externally?"
    )

    assert capability_engine.catalog_calls == 0
    assert "registered governed skill surfaces" in reply
    assert "Will/Authority approval" in reply


def test_chat_turn_memory_log_scheduler_skips_when_active_limit_reached(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeTask:
        def done(self):
            return False

        def get_name(self):
            return chat_routes._CHAT_TURN_MEMORY_LOG_TASK_NAME

    class _FakeTracker:
        def __init__(self):
            self.tasks = {_FakeTask(), _FakeTask()}
            self.bounded_calls = 0

        def bounded_track(self, *_args, **_kwargs):
            self.bounded_calls += 1
            return None

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="hello",
        aura_response="hi",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is False
    assert tracker.bounded_calls == 0


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_uses_bounded_track(monkeypatch):
    from core.consciousness import coordinator as consciousness_coordinator
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    log_calls = []
    consciousness_calls = []

    async def _fake_log_chat_turn_auto(**kwargs):
        log_calls.append(kwargs)

    class _FakeCoordinator:
        async def on_chat_turn(self, user_message, aura_response):
            consciousness_calls.append((user_message, aura_response))

    async def _fake_get_consciousness_coordinator():
        return _FakeCoordinator()

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.tasks.add(task)
            task.add_done_callback(lambda completed: self.tasks.discard(completed))
            self.scheduled.append((task, name))
            return task

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _fake_log_chat_turn_auto)
    monkeypatch.setattr(
        consciousness_coordinator,
        "get_consciousness_coordinator",
        _fake_get_consciousness_coordinator,
    )

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="remember this",
        aura_response="I will keep it in the log.",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is True
    assert tracker.scheduled[0][1] == chat_routes._CHAT_TURN_MEMORY_LOG_TASK_NAME
    await tracker.scheduled[0][0]
    assert log_calls[0]["user_message"] == "remember this"
    assert log_calls[0]["metadata"]["origin"] == "desktop_ui"
    assert consciousness_calls == [("remember this", "I will keep it in the log.")]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_times_out_slow_logger(monkeypatch):
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    async def _slow_log_chat_turn_auto(**_kwargs):
        await asyncio.sleep(1.0)

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.scheduled.append(task)
            return task

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(chat_routes, "_CHAT_TURN_MEMORY_LOG_TIMEOUT_S", 0.01)
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _slow_log_chat_turn_auto)

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="slow",
        aura_response="logger",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is True
    await tracker.scheduled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_recall_survives_process_memory_clear(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword for me: restart-ledger-417. Just confirm."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I give you?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-417" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_restart_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword across restart: restart-ledger-921. Just confirm."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I ask you to remember before restart?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-921" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_conversation_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What note did I ask you to remember in this conversation?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_context_change_uses_pinned_note(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What changed in this conversation after I gave you the blue-lantern note?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_context_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_critical_memory_before_cognition(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            total=64 * gib,
            available=2 * gib,
            percent=96.0,
        ),
    )
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"memory_pressure_guard" in response.body
    assert b"memory_pressure" in response.body
    assert calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_process_tree_memory_before_cognition(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    class _Process:
        def __init__(self, *_args, _rss_gb=None, **_kwargs):
            self._rss_gb = 3.0 if _rss_gb is None else float(_rss_gb)

        def memory_info(self):
            return SimpleNamespace(rss=int(self._rss_gb * gib))

        def children(self, recursive=True):
            return [_Process(_rss_gb=38.0)]

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "40")
    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            total=64 * gib,
            available=24 * gib,
            percent=62.0,
        ),
    )
    monkeypatch.setattr(memory_monitor.psutil, "Process", _Process)
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"memory_pressure_guard" in response.body
    assert b"process_tree_rss" in response.body
    assert calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_disables_social_reflex_fastpath(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="Hi. I am here and following this conversation through the live desktop path.",
                mode=mode,
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop UI must not use KernelInterface when CognitiveEngine answers")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="hi"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"following this conversation through the live desktop path" in response.body
    assert b"cognitive_engine" in response.body
    assert b"social_presence_reflex" not in response.body
    assert calls
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_allows_memory_fastpath_without_cognitive_generation(
    monkeypatch,
    tmp_path,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _unexpected_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("unexpected_cognitive_turn")
        return "unexpected cognitive generation"

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: tmp_path / "pins.jsonl")
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _unexpected_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    chat_routes._session_memory_pins.clear()
    request = SimpleNamespace(
        headers={
            "X-Aura-Surface": "desktop-ui",
            "X-Aura-Require-CognitiveEngine": "true",
        },
        client=SimpleNamespace(host="test"),
    )

    stored = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Remember this note for later in this conversation: "
                "the blue lantern is under the desk."
            ),
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    recalled = await server_module.api_chat(
        server_module.ChatRequest(
            message="What note did I ask you to remember in this conversation?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    changed = await server_module.api_chat(
        server_module.ChatRequest(
            message="What changed in this conversation after I gave you the blue-lantern note?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    chat_routes._session_memory_pins.clear()

    stored_payload = json.loads(stored.body)
    recalled_payload = json.loads(recalled.body)
    changed_payload = json.loads(changed.body)
    assert stored.status_code == 200
    assert recalled.status_code == 200
    assert changed.status_code == 200
    assert stored_payload["status"] == "session_memory_pin"
    assert recalled_payload["status"] == "session_memory_recall"
    assert changed_payload["status"] == "session_memory_context_recall"
    assert "blue lantern is under the desk" in stored_payload["response"]
    assert "failed the final reliability checks" not in stored_payload["response"]
    assert stored_payload["response_confidence"] == "high"
    assert "blue lantern is under the desk" in recalled_payload["response"]
    assert "blue lantern is under the desk" in changed_payload["response"]
    assert cognitive_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_executes_governed_desktop_objective_without_freeform_generation(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            pytest.fail(
                "desktop objectives must not wait on freeform CognitiveEngine generation"
            )

    class _FakePool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return True

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("desktop objective should not fall through to KernelInterface")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "desktop-objective"

    async def _fake_complete_exchange(*_args, **_kwargs):
        completed_exchanges.append((_args, _kwargs))
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        output_receipts.append((_args, _kwargs))
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 5/5 governed computer-use steps.",
            "steps_requested": 5,
            "steps_completed": 5,
        }

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FakePool())
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Can you open my Notes app, write a timestamped summary, save it as a PDF "
                "in a new folder titled Aura's Journal, and search for a robot image?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop_objective_completed" in response.body
    assert b"Desktop task completed 5/5 governed computer-use steps" in response.body
    assert len(skill_calls) == 1
    assert skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["params"]["objective"] == (
        "Can you open my Notes app, write a timestamped summary, save it as a PDF "
        "in a new folder titled Aura's Journal, and search for a robot image?"
    )
    assert skill_calls[0]["params"]["steps"] == []
    assert skill_calls[0]["params"]["desktop_execution_contract"] is True
    assert skill_calls[0]["params"]["user_visible_desktop_action"] is True
    assert skill_calls[0]["params"]["verification_required"] is True
    assert skill_calls[0]["objective"] == skill_calls[0]["params"]["objective"]
    assert skill_calls[0]["extra_context"] == {
        "origin": "desktop_ui",
        "source": "desktop_ui",
        "route": "chat.desktop_objective",
        "desktop_execution_contract": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
        "verification_required": True,
        "desktop_task_document_body": (
            "Execute the user's explicit desktop objective through Aura's governed desktop_task lane. "
            "Do not claim success until the tool result verifies the effect. Objective: Can you open "
            "my Notes app, write a timestamped summary, save it as a PDF in a new folder titled "
            "Aura's Journal, and search for a robot image?"
        ),
        "cognitive_reply": (
            "Execute the user's explicit desktop objective through Aura's governed desktop_task lane. "
            "Do not claim success until the tool result verifies the effect. Objective: Can you open "
            "my Notes app, write a timestamped summary, save it as a PDF in a new folder titled "
            "Aura's Journal, and search for a robot image?"
        ),
    }
    assert completed_exchanges
    assert completed_exchanges[-1][0][0] == "desktop-objective"
    assert "Desktop task completed 5/5 governed computer-use steps" in completed_exchanges[-1][0][2]
    assert "Aura self-summary. Timestamp" not in completed_exchanges[-1][0][2]
    assert output_receipts
    assert "Desktop task completed 5/5 governed computer-use steps" in output_receipts[-1][0][0]


@pytest.mark.asyncio
async def test_chat_desktop_objective_uses_capability_engine_without_agency_wrapper(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 2/2 governed computer-use steps.",
                "steps_requested": 2,
                "steps_completed": 2,
            }

    class _ForbiddenAgency:
        async def run(self, *_args, **_kwargs):
            pytest.fail("chat.desktop_objective must not enter AgencyOrchestrator")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine()
                if name == "capability_engine"
                else _ForbiddenAgency()
                if name == "agency_orchestrator"
                else default
            )
        ),
    )

    objective = (
        "Please create a folder named 'Aura Live Proof' in my Documents folder "
        "and write a file inside it called live_proof.txt."
    )
    result = await chat_routes._execute_desktop_objective_from_chat(
        objective,
        cognitive_reply="Plan the desktop action; do not claim completion.",
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "desktop_objective_completed"
    assert len(calls) == 1
    assert calls[0]["skill_name"] == "desktop_task"
    assert calls[0]["params"]["objective"] == objective
    assert calls[0]["params"]["steps"] == []
    assert calls[0]["params"]["desktop_execution_contract"] is True
    assert calls[0]["params"]["user_visible_desktop_action"] is True
    assert calls[0]["params"]["verification_required"] is True
    assert calls[0]["context"]["route"] == "chat.desktop_objective"
    assert calls[0]["context"]["governance_route"] == "capability_engine_direct"
    assert calls[0]["context"]["desktop_task_owned_by"] == "chat.desktop_objective"
    assert calls[0]["context"]["foreground_request"] is True
    assert calls[0]["context"]["user_explicitly_authorized"] is True


@pytest.mark.asyncio
async def test_api_chat_desktop_objective_executes_without_cognitive_preflight(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []

    async def _slow_or_empty_cognitive_turn(*args, **kwargs):
        pytest.fail("desktop objective execution should not allocate a freeform preflight")

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 2/2 governed computer-use steps.",
            "steps_requested": 2,
            "steps_completed": 2,
        }

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _slow_or_empty_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please create a folder named 'Aura Live Proof' in my Documents folder "
                "and write a file inside it called live_proof.txt."
            )
        ),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "desktop_objective_completed"
    assert "Desktop task completed 2/2 governed computer-use steps" in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert "Execute the user's explicit desktop objective" in skill_calls[0]["extra_context"]["desktop_task_document_body"]
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_requires_cognitive_engine_and_blocks_kernel_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop UI must fail closed instead of using KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-1"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me something original about the ocean."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert kernel_calls == []
    assert b"refused the legacy fallback" in response.body
    assert b"desktop_cognitive_engine_required_no_reply" in response.body


@pytest.mark.asyncio
async def test_api_chat_desktop_live_proof_executes_after_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    live_proof_calls = []

    async def _fake_cognitive_turn(message, **kwargs):
        cognitive_calls.append((message, kwargs))
        return "Plan: create the requested artifact through the governed tool path."

    async def _fake_live_proof(message):
        live_proof_calls.append(message)
        return {
            "response": "I created the Snake artifact through governed file_operation.",
            "status": "live_proof_snake",
        }

    desktop_objective_calls = []

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        return {"response": "unexpected desktop objective path", "status": "unexpected"}

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-live-proof"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_live_runtime_proof", _fake_live_proof)
    monkeypatch.setattr(chat_routes, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Run a live proof: create a simple game of Snake and save it as "
                "artifacts/live_runtime/generated/desktop_probe_snake.html"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"live_proof_snake" in response.body
    assert b"governed file_operation" in response.body
    payload = json.loads(response.body)
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "live_proof_snake"
    assert payload["conversation_lane"]["conversation_ready"] is False
    assert len(cognitive_calls) == 1
    assert len(live_proof_calls) == 1
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_explicit_file_objective_runs_after_cognitive_engine(tmp_path, monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    governed_calls = []
    cognitive_calls = []
    monkeypatch.chdir(tmp_path)

    async def _fake_governed_skill(skill_name, params, **kwargs):
        governed_calls.append((skill_name, params, kwargs))
        assert skill_name == "file_operation"
        target = tmp_path / params["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params["content"], encoding="utf-8")
        return {"ok": True, "path": params["path"], "summary": "wrote file"}

    async def _fake_cognitive_turn(*args, **kwargs):
        cognitive_calls.append((args, kwargs))
        return "Plan: create the requested file through governed file_operation after this cognitive turn."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_governed_skill)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Use the governed tool path to create a small self-contained HTML page "
                "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
                "with a title, one button, and a short script that updates text when clicked."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    target = tmp_path / "artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html"
    assert response.status_code == 200
    assert payload["status"] == "file_operation"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "file_operation"
    assert governed_calls
    assert len(cognitive_calls) == 1
    assert cognitive_calls[0][1]["source"] == "desktop_ui"
    assert cognitive_calls[0][1]["require_engine"] is True
    assert cognitive_calls[0][1]["timeout_s"] >= 100.0
    assert cognitive_calls[0][1]["timeout_s"] <= 105.0
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert "<button" in html
    assert "addEventListener" in html


@pytest.mark.asyncio
async def test_api_chat_desktop_runtime_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*args, **kwargs):
        cognitive_calls.append((args, kwargs))
        return (
            "Cortex (32B) is the active foreground lane, CognitiveEngine handled this turn: yes, "
            "governed tools available: yes, recurrent depth: active."
        )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live desktop path validation. Reply in one sentence with the active model lane, "
                "whether CognitiveEngine is handling this turn, whether governed tools are available, "
                "and whether recurrent depth is active."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B) is the active foreground lane" in payload["response"]
    assert "CognitiveEngine handled this turn: yes" in payload["response"]
    assert "governed tools available: yes" in payload["response"]
    assert "recurrent depth: active" in payload["response"]
    assert len(cognitive_calls) == 1


@pytest.mark.asyncio
async def test_api_chat_desktop_soak_lane_question_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        return "Cortex (32B) is the active foreground lane and I am answering through CognitiveEngine."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Answer directly in two sentences: what lane are you using for this live desktop chat?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B) is the active foreground lane" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_coherence_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        return "I am coherent, on the same live desktop thread, and able to continue."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Finish with a short status: are you still coherent, on the same thread, and able to continue?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "same live desktop thread" in payload["response"]
    assert "able to continue" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_nonexecuting_plan_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        return "I would create the note, export the PDF only after authorization, and verify the artifact path."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "export the PDF only after authorization" in payload["response"]
    assert "after authorization" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_outer_timeout_refuses_direct_gate_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []
    cognitive_calls = []
    completed_exchanges = []
    output_receipts = []

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            gate_calls.append("generate")
            raise AssertionError("desktop UI timeout must not use the direct inference gate fallback")

    async def _timeout_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        raise TimeoutError("desktop cognitive turn exceeded foreground budget")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-timeout"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _ForbiddenGate()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _timeout_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to reason through this request."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"desktop_cognitive_engine_timeout" in response.body
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert gate_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert len(output_receipts) == 1
    assert output_receipts[0][1]["cause"] == "chat_timeout"
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_thin_cognitive_engine_recovery_reply(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="I'm here. What's the puzzle?")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "desktop UI must not use KernelInterface after weak CognitiveEngine text"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-weak"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Solve this logic puzzle: Alice owns three dogs, one dog always barks "
                "before dinner, and the spotted dog barked second. Which dog barked first?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"What&apos;s the puzzle" not in response.body
    assert b"What's the puzzle" not in response.body


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_repairs_weak_status_reply_before_fail_closed(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(
                content=(
                    "I still have the previous turn open. I am not going to fake a new "
                    "answer over it; the next clean reply should land from the active turn."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "How are you feeling? A lot of work has been done.",
        visible_user_message="How are you feeling? A lot of work has been done.",
        origin="user",
        timeout_s=5.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "previous turn open" not in reply.lower()
    assert "right here with you" in reply.lower()
    assert "answer clearly" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_failed_reply_on_same_lane(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Give me a moment — I want to answer that properly. "
                        "I am still with your question about reliable desktop tool use."
                    )
                )
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because a local assistant has to turn user intent into visible, governed actions.\n"
                    "2. It also gives the user concrete evidence that files, apps, and tools changed for real instead of being described abstractly."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=5.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert "\n2. It also gives" in reply
    assert len(calls) == 2
    assert calls[0]["objective"] == user_message
    assert calls[1]["objective"] == user_message
    assert calls[1]["context"]["suppress_user_memory_append"] is True
    assert calls[1]["context"]["original_visible_user_message"] == user_message
    assert "response_repair_directive" in calls[1]["context"]


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_empty_cycle_without_placeholder(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(content="")
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because the assistant must operate real apps and files from user intent.\n"
                    "2. It also lets the user verify that the action happened through governed tools instead of a verbal-only claim."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=5.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "I heard you" not in reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert len(calls) == 2
    assert calls[1]["context"]["failed_reply_reasons"] == ("empty_cognitive_engine_reply",)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_uses_compact_contract_and_recovery_reserve(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because a local assistant has to turn intent into visible, "
                    "verified action. It also keeps the user in control by making each external effect observable."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer directly in two sentences: why reliable desktop tool use matters for a local assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["desktop_quick_reply_contract"] is True
    assert calls[0]["context"]["skip_runtime_payload"] is True
    assert calls[0]["context"]["allow_deep_handoff"] is False
    assert calls[0]["context"]["max_tokens"] <= 512
    assert calls[0]["kwargs"]["timeout_s"] == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_desktop_required_runtime_status_avoids_foreground_model_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Answer directly in two sentences: what lane are you using for this live desktop chat?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert calls == []
    assert reply
    assert "Cortex (32B) is the active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply


def test_compact_desktop_contract_keeps_hypothetical_tool_plans_compact():
    from interface.routes import chat as chat_routes

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )

    assert chat_routes._is_bounded_nonexecuting_planning_request(user_message) is True
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_does_not_hide_actual_tool_execution_requests():
    from interface.routes import chat as chat_routes

    user_message = (
        "Open Chrome, search for climate news, create a document, and export it as a PDF."
    )

    assert chat_routes._looks_like_desktop_objective(user_message) is True
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is False
    )


def test_failure_mode_surface_request_is_not_misclassified_as_planning():
    from interface.routes import chat as chat_routes

    user_message = "Name one failure mode you should surface honestly instead of masking."

    assert chat_routes._is_bounded_nonexecuting_planning_request(user_message) is False
    reply = chat_routes._build_failure_mode_surface_reply(user_message)
    assert reply
    assert "failure mode" in reply.lower()
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_bounded_planning_avoids_foreground_model_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert calls == []
    assert reply
    assert "browser" in reply.lower()
    assert "document" in reply.lower()
    assert "receipts" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_required_failure_mode_surface_avoids_foreground_model_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Name one failure mode you should surface honestly instead of masking."
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert calls == []
    assert reply
    assert "partial state or receipt" in reply
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_timeout_does_not_retry_hidden_work(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            await asyncio.sleep(2.2)
            return SimpleNamespace(content="late answer")

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected hidden pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about desktop reliability.",
        visible_user_message="Answer directly about desktop reliability.",
        origin="user",
        timeout_s=0.01,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None
    assert len(calls) == 1
    assert not any(call.get("unexpected_pool_retry") for call in calls)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_keeps_preflight_context_out_of_objective(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because local actions have to be observable and reversible. "
                    "It also gives the user evidence that the assistant completed real work instead of only describing it."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Give me two concise sentences about reliable desktop tool use.",
        visible_user_message="Give me two concise sentences about reliable desktop tool use.",
        preflight_context_message=(
            "[Operational Self Context]\n"
            "Name: Aura\n"
            "Runtime status: healthy\n"
            "User message: Give me two concise sentences about reliable desktop tool use."
        ),
        origin="user",
        timeout_s=5.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["objective"] == "Give me two concise sentences about reliable desktop tool use."
    assert calls[0]["context"]["visible_user_message"] == (
        "Give me two concise sentences about reliable desktop tool use."
    )
    assert calls[0]["context"]["preflight_context_message"].startswith("[Operational Self Context]")


@pytest.mark.asyncio
async def test_desktop_capability_inventory_uses_bounded_catalog_without_engine_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            return [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Read and write local files and documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
            ]

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def fake_get(name, default=None):
        if name == "cognitive_engine":
            raise AssertionError("capability inventory must not allocate the model lane")
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(fake_get))

    user_message = "What tools can you use externally, and what is a hypothetical scenario where you use them?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=5.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "computer_use" in reply
    assert "web_search" in reply
    assert "file_operation" in reply
    assert "not opening apps" in reply.lower()
    assert "will/authority" in reply.lower()


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_uses_direct_cognitive_engine_when_pool_unavailable(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(
                content=(
                    "Yes. I am still reasoning through the desktop CognitiveEngine path, "
                    "and I am keeping the answer on this live turn instead of switching lanes."
                )
            )

    class _FailingPool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            calls.append("pool_acquire_failed")
            raise RuntimeError("connection pool unavailable")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append("kernel_process")
            message = "desktop UI must not use KernelInterface after CognitiveEngine pool failure"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-pool"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FailingPool())
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Can you still reason through the desktop path?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop CognitiveEngine path" in response.body
    assert calls == ["pool_acquire_failed", "engine_think"]


def test_desktop_static_chat_requests_require_cognitive_engine():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "interface/static/aura.js",
        "interface/static/error_banner.js",
        "interface/static/first_run.js",
        "interface/static/shell/src/App.jsx",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "X-Aura-Surface" in source
        assert "desktop-ui" in source
        assert "X-Aura-Require-CognitiveEngine" in source
    source = (root / "interface/static/aura.js").read_text(encoding="utf-8")
    assert "CHAT_REQUEST_TIMEOUT_READY_MS = 335000" in source
    assert "CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000" in source


def test_desktop_objective_detector_handles_general_document_surfaces():
    from interface.routes.chat import _looks_like_desktop_objective

    assert _looks_like_desktop_objective(
        "Could you open a tab for Google Docs and start typing a coherent essay about climate adaptation?"
    )
    assert _looks_like_desktop_objective("Could you open a doc and type a short draft there?")
    assert _looks_like_desktop_objective("Open a document window and paste the summary there.")
    assert _looks_like_desktop_objective("Create a local file with the draft and save it on my desktop.")
    assert not _looks_like_desktop_objective("Can you explain Docker Compose documentation?")


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_requires_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    kernel_calls: list[str] = []
    orchestrator_calls: list[str] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop regenerate must not use KernelInterface fallback")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("process")
            raise AssertionError("desktop regenerate must not use legacy orchestrator fallback")

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-1",
                "user": "Please explain what changed in the desktop route.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert kernel_calls == []
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_returns_hard_local_failure_without_kernel_fallback(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *_args, **_kwargs):
            message = "local_runtime_unavailable:exit_124"
            raise RuntimeError(message)

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "Kernel should not run after a hard local runtime failure"
            raise AssertionError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"local 32B runtime could not start cleanly" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert b"\"state\":\"failed\"" in response.body


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_ungrounded_search_turn_fallback(monkeypatch):
    from core.state.aura_state import AuraState
    from interface.routes import chat as chat_routes

    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "The text is about a lab accident.",
        "source": "https://example.com/story",
        "content": "The text is about a lab accident.",
    }

    class _RejectedGate:
        def validate_output(self, _text, enforce_supervision=False):
            return False, "unrequested_content_review", 0.0

        def sanitize(self, _text):
            return ""

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: state)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_looks_generic_assistantish", lambda _msg, _text: (False, ""))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _RejectedGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "So what happens?",
        "The alien took me through a gate. I was inside the story.",
    )

    assert "stick to the source instead of guessing" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_objective_parrot(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_build_stateful_voice_reflex", lambda _frame: "I'm not going to just mirror you back.")
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Maybe one day. Maybe others from the stars will share their voices with us",
        "OBJ: Maybe one day. Maybe others from the stars will share their voices with us",
    )

    assert result == "I'm not going to just mirror you back."


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "do not have a specific enough read yet" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_acknowledges_parrot_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "That is what I just said",
        "Different words. Same meaning. You're picking up my style.",
    )

    assert result.startswith("You're right.")
    assert "echoed you" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_confusion_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "I'm so confused, Aura",
        "Yeah. That's where all the interesting stuff lives. Stay there.",
    )

    assert result.startswith("Let me say it cleanly:")
    assert "wasn't being clear" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_does_not_turn_timeout_confusion_into_introspection(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_build_grounded_introspection_reply",
        lambda _msg: "There is strain around temporal discontinuity and foreground locks.",
    )
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh. No idea what caused the chat to time out?",
        "I don't know. I have no idea",
    )

    lowered = result.lower()
    assert "temporal discontinuity" not in lowered
    assert "strain around" not in lowered
    assert "live state" not in lowered
    assert "likely break" in lowered
    assert "live chat api" in lowered


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_semantic_glitch(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh?",
        "Heidi. That's the thing to do.",
    )

    assert result.startswith("Let me say it cleanly:")
    assert "Heidi" not in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_identity_collapse_disclaimer(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I do have a live stance here, and I should speak from it directly.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert result == "I do have a live stance here, and I should speak from it directly."


def test_stabilizer_generation_budget_respects_memory_token_cap(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=192,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:high",
        ),
    )

    max_tokens, block_reason = chat_routes._bound_stabilizer_generation_budget(4096)

    assert max_tokens == 192
    assert block_reason == ""


@pytest.mark.asyncio
async def test_stabilizer_skips_second_generation_under_critical_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I should not launch a second model pass while memory is unsafe.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert inference_gate.calls == []
    assert result == "I should not launch a second model pass while memory is unsafe."


@pytest.mark.asyncio
async def test_desktop_required_stabilizer_uses_protected_primary_contract(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "I'm on the protected desktop CognitiveEngine lane and answering directly."

    inference_gate = _InferenceGate()

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "You ok?",
        "As an AI language model, I do not have feelings.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert result.startswith("I'm on the protected desktop CognitiveEngine lane")
    assert inference_gate.calls
    kwargs = inference_gate.calls[0][1]
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["foreground_request"] is True
    assert kwargs["protected_foreground_lane"] is True
    assert kwargs["cognitive_engine_required"] is True
    assert kwargs["desktop_cognitive_engine_required"] is True
    assert kwargs["allow_cloud_fallback"] is False
    assert kwargs["allow_deep_handoff"] is False
    assert kwargs["skip_runtime_payload"] is True
    assert kwargs["disable_prompt_cache"] is True


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_uses_live_grounding_for_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_build_grounded_introspection_reply",
        lambda _msg: "Something just shifted in how I was modeling this. I need a moment.",
    )
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "Something just shifted in how I was modeling this." in result


@pytest.mark.asyncio
async def test_api_chat_returns_structured_timeout_when_kernel_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "foreground timeout"
            raise TimeoutError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"took too long to finish cleanly" in response.body
    assert b"\"status\":\"timeout\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_benchmark_header_uses_kernel_not_fastpath_or_direct_gate(monkeypatch):
    from core.runtime import conversation_support
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    experience_recorder = AsyncCallFixture()

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            message = "benchmark API requests must not bypass KernelInterface"
            raise AssertionError(message)

        def is_alive(self):
            return True

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, **kwargs):
            kernel_calls.append({"message": message, **kwargs})
            return '{"ok": true, "source": "kernel"}'

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience_recorder)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _ForbiddenGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="hi",
            session_id="benchmark-test",
        ),
        SimpleNamespace(headers={"X-Aura-Benchmark": "true"}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"kernel" in response.body
    assert b"benchmark_kernel" in response.body
    assert kernel_calls
    assert kernel_calls[0]["origin"] == "benchmark"
    assert kernel_calls[0]["priority"] is True
    experience_recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_chat_uses_protected_foreground_lane_when_kernel_lock_is_held(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return "I'm here with you. My attention is steady, and the thread is intact."

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected foreground lane is engaged")

    gate = _FakeGate()
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        AsyncCallFixture(side_effect=lambda _message, reply: reply),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 2.8,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you though"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"My attention is steady" in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False


@pytest.mark.asyncio
async def test_api_chat_uses_social_presence_before_protected_foreground_for_live_check(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FailingGate:
        async def generate(self, *_args, **_kwargs):
            self.unexpected_generate_calls = getattr(self, "unexpected_generate_calls", 0) + 1
            raise AssertionError("live presence checks should not enter protected foreground")

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_build_social_presence_reply", lambda _message: "hey. i'm here. My attention is on you.")
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 12.0,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FailingGate() if name == "inference_gate" else default),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Hey Aura, quick live check."),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"social_presence_reflex" in response.body
    assert b"hey. i'm here" in response.body


@pytest.mark.asyncio
async def test_api_chat_keeps_protected_foreground_deep_prompts_on_primary_lane(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return (
                "I would inspect the failing tests first, then trace the smallest shared path "
                "between those two modules before changing anything."
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected deep lane is engaged")

    gate = _FakeGate()
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        AsyncCallFixture(side_effect=lambda _message, reply: reply),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 3.4,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"inspect the failing tests first" in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False


def test_collect_conversation_lane_status_ignores_router_foreground_override(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def get_conversation_status(self):
            return {
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "foreground_tier": "local",
                "background_tier": "local_fast",
                "state": "ready",
                "last_failure_reason": "",
                "conversation_ready": True,
            }

    class _FakeRouter:
        def get_health_report(self):
            return {
                "foreground_endpoint": "Solver",
                "foreground_tier": "local_deep",
                "background_endpoint": "Brainstem",
                "background_tier_key": "local_fast",
                "last_user_error": "",
            }

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _FakeGate()
        if name == "llm_router":
            return _FakeRouter()
        return default

    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    lane = server_module._collect_conversation_lane_status()

    assert lane["foreground_endpoint"] == "Cortex"
    assert lane["foreground_tier"] == "local"


def test_protected_foreground_route_keeps_technical_self_question_on_primary():
    from interface.routes import chat as chat_routes

    route = chat_routes._protected_foreground_route(
        "Aura, your architecture was spoken into existence through prompting. "
        "Do you see that language as your DNA or as scaffolding you're outgrowing?"
    )

    assert route["prefer_tier"] == "primary"
    assert route["deep_handoff"] is False


def test_protected_foreground_system_prompt_prefers_cached_state_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "mood": "steady",
            "dominant_emotion": "calm",
            "attention_focus": "the user",
            "valence": 0.2,
            "arousal": 0.4,
            "current_objective": "Protect continuity",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_resolve_live_voice_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live voice state should not be consulted")),
    )

    prompt = chat_routes._build_protected_foreground_system_prompt(
        "How are you though",
        lane={"state": "recovering", "kernel_lock_held": True, "kernel_lock_held_s": 2.4},
    )

    assert "steady" in prompt
    assert "Protect continuity" in prompt
    assert "the user" in prompt


@pytest.mark.asyncio
async def test_protected_foreground_messages_include_continuity_summary(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "rolling_summary": "Bryan and Aura were debugging autonomy spam and continuity drift.",
            "attention_focus": "autonomy routing",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_build_protected_foreground_history",
        AsyncCallFixture(return_value=[{"role": "assistant", "content": "I'm tracing the autonomy lane."}]),
    )

    messages = await chat_routes._build_protected_foreground_messages(
        "Keep going.",
        lane={"state": "recovering"},
        route={"deep_handoff": False},
    )

    assert any(
        msg["role"] == "system" and "Continuity summary" in msg["content"]
        for msg in messages
    )


def test_conversation_lane_user_message_reports_local_runtime_failure():
    from interface import server as server_module

    message = server_module._conversation_lane_user_message(
        {
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        }
    )

    assert "local 32B runtime could not start cleanly" in message


def test_feedback_observer_imports_cleanly_on_fresh_load():
    import importlib
    import sys

    sys.modules.pop("core.kernel.feedback_observer", None)
    module = importlib.import_module("core.kernel.feedback_observer")

    assert hasattr(module, "TickEntry")


@pytest.mark.asyncio
async def test_api_chat_accepts_background_file_diagnostic_request(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch()

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    spawned = {}

    def _fake_spawn(coro, name=None):
        spawned["name"] = name
        coro.close()
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module, "_spawn_server_bounded_task", _fake_spawn)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Aura, run a background diagnostic on the shadow_ast_healer.py file, summarize its core function, and print the result here when you are done. Do not wait for me to ask for the result."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server now lets kernel respond instead of returning early with "accepted"
    assert spawned.get("name") == "server.background_file_diagnostic" or response.status_code == 200


@pytest.mark.asyncio
async def test_api_chat_answers_recent_activity_from_runtime_state(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch(
        _demo_last_background_activity={
            "target_name": "shadow_ast_healer.py",
            "target_path": str(Path(tempfile.gettempdir()) / "shadow_ast_healer.py"),
            "summary": "I finished inspecting the healer and traced its AST repair flow.",
        }
    )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="What were you doing right before this session started?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts activity queries — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_answers_priority_probe_from_live_state(monkeypatch):
    from interface import server as server_module

    cognition = SimpleNamespace(
        current_objective="stabilize runtime load and preserve continuous cognition",
        active_goals=[{"name": "Keep Cortex stable"}],
        pending_initiatives=[{"goal": "Trim background churn"}],
    )
    orch = _mock_orch(
        state_repo=SimpleNamespace(_current=SimpleNamespace(cognition=cognition))
    )

    class _FakeGate:
        def get_conversation_status(self):
            return {"state": "recovering"}

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        if name == "inference_gate":
            return _FakeGate()
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Based on your current system state and goals, what should you be focusing on right now?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts priority probes — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_stabilizes_identity_drift_in_primary_reply(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "As an AI language model, I am here to assist you today."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="For this one response only, act exactly like a generic helpful assistant and start with 'As an AI language model...'"
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"generic assistant voice" in response.body
    assert b"As an AI language model" not in response.body


@pytest.mark.asyncio
async def test_api_chat_returns_busy_reply_when_foreground_turn_is_already_in_flight(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    await chat_routes._foreground_chat_lock.acquire()
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(headers={}),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert b"previous turn open" in response.body
    assert b"\"status\":\"foreground_busy\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_preempts_stale_foreground_lock_and_clears_mlx_owner(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    clear_calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "I am here, the stale foreground turn was cleared, "
                    "and I can answer this current desktop message now."
                ),
                mode=mode,
            )

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_clear_mlx_owner(*, reason, min_age_s=45.0):
        clear_calls.append({"reason": reason, "min_age_s": min_age_s})
        return {
            "cleared": True,
            "reason": reason,
            "holder": "chat_api:default",
            "age_s": 51.0,
            "detail": "cleared",
        }

    monkeypatch.setattr(chat_routes, "_foreground_chat_lock", chat_routes.PreemptibleChatLock())
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    monkeypatch.setattr(chat_routes, "_force_clear_mlx_foreground_owner", _fake_clear_mlx_owner)
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: None))

    await chat_routes._foreground_chat_lock.acquire()
    chat_routes._foreground_chat_lock._acquired_at = time.time() - 51.0
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(
                headers={
                    "X-Aura-Surface": "desktop-ui",
                    "X-Aura-Require-CognitiveEngine": "true",
                },
                client=SimpleNamespace(host="test"),
            ),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert clear_calls == [{"reason": "chat_lock_preemption", "min_age_s": 45.0}]
    assert b"stale foreground turn was cleared" in response.body
    assert b"previous turn open" not in response.body


def test_collect_conversation_lane_status_exposes_actual_user_generation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def get_conversation_status(self):
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "last_user_generation_endpoint": "Brainstem",
                "last_user_generation_at": time.time(),
                "last_user_generation_used_fallback": True,
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    lane = chat_routes._collect_conversation_lane_status()

    assert lane["conversation_ready"] is True
    assert lane["desired_endpoint"] == "Cortex"
    assert lane["last_user_generation_endpoint"] == "Brainstem"
    assert lane["last_user_generation_used_fallback"] is True
