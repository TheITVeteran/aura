import asyncio
import contextlib
import importlib
import inspect
import os
import time
from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate
from core.container import ServiceContainer
from core.state.aura_state import AuraState
from core.utils.deadlines import get_deadline

_MISSING = object()


def test_build_messages_uses_canonical_state_without_mutating_working_memory(monkeypatch):
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "canonical user turn"},
        {"role": "assistant", "content": "canonical assistant turn"},
    ]
    original_memory = list(state.cognition.working_memory)
    repo = SimpleNamespace(_current=state)
    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "state_repository":
            return repo
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))

    gate = InferenceGate()
    messages = gate._build_messages(
        "new objective",
        "fallback system",
        [{"role": "user", "content": "incoming history"}],
    )

    assert state.cognition.working_memory == original_memory
    rendered = "\n".join(message["content"] for message in messages)
    assert "canonical user turn" in rendered
    assert "incoming history" in rendered
    assert messages[-1] == {"role": "user", "content": "new objective"}


def test_system_prompt_cache_tracks_live_state_revision(monkeypatch):
    state = AuraState.default()
    state.cognition.current_objective = "first objective"
    repo = SimpleNamespace(_current=state)
    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "state_repository":
            return repo
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))
    gate = InferenceGate()

    first = gate._build_system_prompt()
    state.cognition.current_objective = "second objective"
    second = gate._build_system_prompt()

    assert first
    assert second
    assert gate._identity_prompt_state_key is not None
    assert gate._identity_prompt_state_key[3] == "second objective"


class CallProbe:
    def __init__(self, return_value=None, side_effect=None, **attrs):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []
        for name, value in attrs.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if isinstance(self.side_effect, list):
            if not self.side_effect:
                raise AssertionError("call side effect sequence exhausted")
            item = self.side_effect.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        return self.return_value

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1
        assert self.calls[0] == {"args": args, "kwargs": kwargs}

    def assert_not_called(self):
        assert self.calls == []


class AsyncCallProbe(CallProbe):
    async def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if isinstance(self.side_effect, list):
            if not self.side_effect:
                raise AssertionError("async call side effect sequence exhausted")
            item = self.side_effect.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if callable(self.side_effect):
            result = self.side_effect(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        return self.return_value

    @property
    def await_args(self):
        if not self.calls:
            return SimpleNamespace(args=(), kwargs={})
        last = self.calls[-1]
        return SimpleNamespace(args=last["args"], kwargs=last["kwargs"])

    def assert_awaited(self):
        assert self.calls

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_awaited_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1
        assert self.calls[0] == {"args": args, "kwargs": kwargs}

    def assert_not_awaited(self):
        assert self.calls == []


class TaskProbe:
    def __init__(self, done=False):
        self._done = done
        self.cancel = CallProbe()
        self.done_callbacks = []

    def done(self):
        return self._done

    def add_done_callback(self, callback):
        self.done_callbacks.append(callback)

    def get_loop(self):
        return asyncio.get_running_loop()


class _replace:  # noqa: N801 - mirrors unittest.mock.patch for local test doubles
    def __init__(self, target, new=_MISSING, *, return_value=_MISSING, side_effect=_MISSING):
        self.target = target
        self.new = new
        self.return_value = return_value
        self.side_effect = side_effect
        self.owner = None
        self.attr = ""
        self.original = _MISSING
        self.replacement = _MISSING

    @staticmethod
    def _resolve(target):
        parts = target.split(".")
        for idx in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:idx])
            try:
                owner = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for part in parts[idx:-1]:
                owner = getattr(owner, part)
            return owner, parts[-1]
        raise ModuleNotFoundError(target)

    @classmethod
    def object(cls, owner, attr, new=_MISSING, *, return_value=_MISSING, side_effect=_MISSING):
        inst = cls("", new, return_value=return_value, side_effect=side_effect)
        inst.owner = owner
        inst.attr = attr
        return inst

    @classmethod
    @contextlib.contextmanager
    def dict(cls, mapping, values, clear=False):
        original = dict(mapping)
        if clear:
            mapping.clear()
        mapping.update(values)
        try:
            yield mapping
        finally:
            mapping.clear()
            mapping.update(original)

    def __enter__(self):
        if self.owner is None:
            self.owner, self.attr = self._resolve(self.target)
        owner_dict = getattr(self.owner, "__dict__", {})
        raw_original = owner_dict.get(self.attr, _MISSING)
        self.original = raw_original if raw_original is not _MISSING else getattr(self.owner, self.attr)
        callable_original = self.original
        if isinstance(callable_original, (staticmethod, classmethod)):
            callable_original = callable_original.__func__
        if self.new is not _MISSING:
            self.replacement = self.new
        else:
            rv = None if self.return_value is _MISSING else self.return_value
            se = None if self.side_effect is _MISSING else self.side_effect
            probe_cls = AsyncCallProbe if inspect.iscoroutinefunction(callable_original) else CallProbe
            self.replacement = probe_cls(return_value=rv, side_effect=se)
        install_value = self.replacement
        if isinstance(self.original, staticmethod) and not isinstance(install_value, staticmethod):
            install_value = staticmethod(install_value)
        elif isinstance(self.original, classmethod) and not isinstance(install_value, classmethod):
            install_value = classmethod(install_value)
        setattr(self.owner, self.attr, install_value)
        return self.replacement

    def __exit__(self, exc_type, exc, tb):
        setattr(self.owner, self.attr, self.original)
        return False


replace = _replace


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.generate_text_async = AsyncCallProbe(return_value=(True, text, {}))


class _RecordingClient:
    def __init__(self, text: str):
        self.text = text
        self.deadlines = []
        self.prompts = []
        self.kwargs = []

    async def generate_text_async(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        self.deadlines.append(kwargs.get("deadline"))
        return self.text


class _ReceiptRecordingClient(_RecordingClient):
    def __init__(self, text: str):
        super().__init__(text)
        self.receipt = {
            "enabled": True,
            "live_mind_controls_bound": True,
            "clean_user_surface_contract": True,
            "surface_validation_prompt_present": True,
            "surface_alpha_applied": 0.31,
            "surface_alpha_applied_ok": True,
            "recurrent_runtime_loops_applied": 2,
            "recurrent_runtime_loops_applied_ok": True,
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": True,
            "surface_quality_gate_attempts": 1,
            "surface_quality_gate_reasons": [],
            "applied": True,
        }

    def get_last_surface_control_receipt(self):
        return dict(self.receipt)


class _SequenceRecordingClient(_RecordingClient):
    def __init__(self, texts: list[str]):
        super().__init__(texts[-1] if texts else "")
        self.texts = list(texts)

    async def generate_text_async(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        self.deadlines.append(kwargs.get("deadline"))
        if self.texts:
            return self.texts.pop(0)
        return self.text

    def get_lane_status(self):
        now = time.time()
        return {
            "state": "ready",
            "last_error": "",
            "conversation_ready": True,
            "warmup_attempted": True,
            "warmup_in_flight": False,
            "last_ready_at": now,
            "last_progress_at": now,
            "last_visible_readiness_at": now,
            "last_user_facing_completed_at": 0.0,
        }


class _NoTextClient:
    def __init__(self):
        self.generate_text_async = AsyncCallProbe(return_value=(False, "", {}))


class _NoTextReadyClient(_NoTextClient):
    def get_lane_status(self):
        return {
            "state": "ready",
            "last_error": "",
            "conversation_ready": True,
            "warmup_attempted": True,
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_ready_at": 1.0,
            "last_progress_at": 1.0,
        }

    def is_alive(self):
        return True


class _LaneWarmupClient:
    def __init__(self):
        self.warmup = AsyncCallProbe(side_effect=self._finish_warmup)
        self.state = "cold"
        self.last_error = ""
        self.visible_ready_at = 0.0

    async def _finish_warmup(self):
        self.state = "ready"
        self.last_error = ""
        self.visible_ready_at = time.time()

    def get_lane_status(self):
        return {
            "state": self.state,
            "last_error": self.last_error,
            "conversation_ready": self.state == "ready",
            "warmup_attempted": self.state != "cold",
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_visible_readiness_at": self.visible_ready_at,
            "last_user_facing_completed_at": 0.0,
        }

    def is_alive(self):
        return self.state == "ready"

    def note_lane_recovering(self, reason):
        self.state = "recovering"
        self.last_error = str(reason or "")

    def note_lane_failed(self, reason):
        self.state = "failed"
        self.last_error = str(reason or "")


class _RecoverableFailedLaneClient(_LaneWarmupClient):
    def __init__(self):
        super().__init__()
        self.state = "failed"
        self.last_error = "mlx_runtime_unavailable:metal_device_enumeration_crash"
        self.refresh_runtime_availability = CallProbe(side_effect=self._refresh)
        self.is_alive = CallProbe(return_value=False)

    def _refresh(self, *, force_probe=False):
        self.state = "cold"
        self.last_error = ""
        return True


class _ColdRecordingLaneClient(_RecordingClient):
    def __init__(self, text: str):
        super().__init__(text)
        self.state = "cold"
        self.last_error = ""
        self.warmup = AsyncCallProbe(side_effect=self._finish_warmup)
        self.visible_ready_at = 0.0

    async def _finish_warmup(self):
        self.state = "ready"
        self.last_error = ""
        self.visible_ready_at = time.time()

    def get_lane_status(self):
        return {
            "state": self.state,
            "last_error": self.last_error,
            "conversation_ready": self.state == "ready",
            "warmup_attempted": self.state != "cold",
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
            "last_ready_at": 1.0 if self.state == "ready" else 0.0,
            "last_progress_at": 1.0 if self.state == "ready" else 0.0,
            "last_visible_readiness_at": self.visible_ready_at,
            "last_user_facing_completed_at": 0.0,
        }


@pytest.mark.asyncio
async def test_inference_gate_passes_repairable_self_reflection_to_downstream_repair():
    gate = InferenceGate()
    bad_self_report = (
        "My self-prediction accuracy is 0.98. My memory texture drift is 0.02. "
        "My affect baseline is stable."
    )
    client = _FakeClient(bad_self_report)

    result = await gate._generate_with_client(
        client,
        "Aura, live-path check: what is actually on your mind right now?",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {"role": "user", "content": "Aura, live-path check: what is actually on your mind right now?"},
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == bad_self_report
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_gate_passes_repairable_reliability_draft_downstream():
    gate = InferenceGate()
    draft = (
        "The practical standard is that a foreground chat turn should stay live, "
        "finish as one coherent answer, and never collapse into retry chatter just "
        "because the first draft needs a repair pass."
    )
    client = _FakeClient(draft)

    result = await gate._generate_with_client(
        client,
        "Push back on me a little: if I demand that live chat never fails, what's the practical engineering version of that standard?",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {
                "role": "user",
                "content": "Push back on me a little: if I demand that live chat never fails, what's the practical engineering version of that standard?",
            },
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == draft
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_gate_passes_substantive_truncated_tail_downstream():
    gate = InferenceGate()
    draft = (
        "I would answer the user directly, preserve the current thread, and keep "
        "the live lane moving instead of detonating a long retry cascade,"
    )
    client = _FakeClient(draft)

    result = await gate._generate_with_client(
        client,
        "Explain how you keep continuity during a strained live chat turn.",
        "",
        [],
        get_deadline(10.0),
        "Cortex",
        messages=[
            {"role": "system", "content": "rich_context"},
            {
                "role": "user",
                "content": "Explain how you keep continuity during a strained live chat turn.",
            },
        ],
        origin="api",
        foreground_request=True,
    )

    assert result == draft
    client.generate_text_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_requests_stay_off_cortex():
    gate = InferenceGate()
    cortex = _FakeClient("cortex")
    brainstem = _FakeClient("brainstem")
    cpu = _FakeClient("cpu")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"prefer_tier": "primary", "origin": "system"},
                    )

    assert result == "brainstem"
    cortex.generate_text_async.assert_not_called()
    brainstem.generate_text_async.assert_awaited()
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_while_cortex_quiet_window_is_active():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_foreground_quiet_window_active", return_value=True):
        with replace.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": False,
                "state": "warming",
                "warmup_in_flight": True,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_when_cortex_has_failed():
    gate = InferenceGate()
    failed_lane = _LaneWarmupClient()
    failed_lane.state = "failed"
    gate._mlx_client = failed_lane
    gate._ensure_cortex_recovery = AsyncCallProbe()

    result = await gate.generate(
        "background reflection",
        context={"prefer_tier": "primary", "origin": "system"},
    )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_deep_handoff_uses_solver_then_returns_response():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _FakeClient("cortex")
        solver = _FakeClient("solver")
        gate._mlx_client = cortex
        gate._restore_primary_after_deep_handoff = AsyncCallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/active":
                return cortex
            raise AssertionError(f"Unexpected model path: {model_path}")

        # Fixed memory headroom so test doesn't depend on actual system RAM
        _low_pressure = {"tier": "secondary", "pressure_pct": 40.0, "total_gb": 64.0, "available_gb": 32.0, "max_pressure_pct": 84.0, "min_available_gb": 16.0, "can_admit": True}
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        with replace.object(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure)):
                            result = await gate.generate(
                                "perform a flagship architecture deep dive",
                                context={"prefer_tier": "secondary", "deep_handoff": True},
                            )
        await asyncio.sleep(0)

        assert result == "solver"
        solver.generate_text_async.assert_awaited()
        cortex.generate_text_async.assert_not_called()
        gate._restore_primary_after_deep_handoff.assert_awaited_once()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_deep_handoff_failure_still_schedules_primary_restore():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _NoTextClient()
        solver = _NoTextClient()
        reflex = _NoTextClient()
        gate._mlx_client = cortex
        gate._schedule_primary_restore_after_deep_handoff = CallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/active":
                return cortex
            if model_path == "/models/fallback":
                return reflex
            raise AssertionError(f"Unexpected model path: {model_path}")

        _low_pressure_snapshot = {
            "tier": "secondary",
            "pressure_pct": 40.0,
            "total_gb": 128.0,
            "available_gb": 64.0,
            "max_pressure_pct": 84.0,
            "min_available_gb": 16.0,
            "can_admit": True,
            "reason": "",
        }
        with replace.object(
            gate,
            "_enforce_foreground_admission",
            new=AsyncCallProbe(
                return_value={
                    "can_admit": True,
                    "pressure_pct": 40.0,
                    "available_gb": 28.0,
                }
            ),
        ):
            with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
                with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                    with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                        with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                            with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                                with replace.object(
                                    InferenceGate,
                                    "_headroom_snapshot",
                                    staticmethod(lambda *a, **kw: dict(_low_pressure_snapshot)),
                                ):
                                    await gate.generate(
                                        "perform a flagship architecture deep dive",
                                        context={"origin": "user", "prefer_tier": "secondary", "deep_handoff": True},
                                    )

        gate._schedule_primary_restore_after_deep_handoff.assert_called_once()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_user_facing_primary_uses_conversational_budget_and_chatml():
    gate = InferenceGate()
    cortex = _RecordingClient("hello")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say hi.",
                    context={"origin": "user", "prefer_tier": "primary", "history": []},
                )

    assert result == "hello"
    assert cortex.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "primary")
    assert cortex.deadlines[0]._timeout == expected_primary
    # Token budget may be scaled by resource_stakes multiplier at runtime,
    # so check it's within the expected range rather than an exact match.
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    actual_tokens = cortex.kwargs[0]["max_tokens"]
    assert 384 <= actual_tokens <= expected_tokens
    assert cortex.kwargs[0]["clean_user_surface_contract"] is True
    assert cortex.kwargs[0]["clean_user_surface_steering_alpha"] == 0.25
    assert cortex.prompts[0].startswith("<|im_start|>")
    assert "<|im_start|>assistant" in cortex.prompts[0]
    assert "<|SYSTEM|>" not in cortex.prompts[0]


@pytest.mark.asyncio
async def test_live_self_process_prebuilt_prompt_is_compacted_and_live_grounded(monkeypatch):
    gate = InferenceGate()
    cortex_reply = (
        "I am attending to Bryan's concern that the live desktop lane can drift into raw assistant mode "
        "or stall under oversized prompts. The remembered concern that should change my next decision is "
        "the repeated timeout-and-repair loop, so I should keep the live mind payload bounded while preserving "
        "recent context. What I want to do next is answer from that bounded live context, then verify the UI reply "
        "and terminal logs before treating the fix as real."
    )
    cortex = _RecordingClient(cortex_reply)
    gate._mlx_client = cortex

    async def _compact_live_context(_self, _prompt, _origin):
        return "## LIVE TONE\nMood: focused\nTone: direct\n## UNITY\nLevel: integrated | Unity: 0.91"

    async def _full_live_context(*_args, **_kwargs):
        if _args or _kwargs:
            raise AssertionError("self-process desktop speech should not build the full 90k live context")
        return ""

    monkeypatch.setattr(InferenceGate, "_build_compact_living_mind_context", _compact_live_context)
    monkeypatch.setattr(InferenceGate, "_build_living_mind_context", _full_live_context)
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    oversized_system = "SYSTEM CONTRACT\n" + ("keep Aura stateful and non-generic. " * 900)
    oversized_turn = "prior live desktop conversation context. " * 500
    user_prompt = (
        "Quick live-path check. Don't give me a health card or telemetry list. "
        "In ordinary speech, answer from your actual current context: what are "
        "you attending to from Bryan's recent messages, what remembered concern "
        "should change your next decision, and what do you want to do next?"
    )
    messages = [{"role": "system", "content": oversized_system}]
    for idx in range(12):
        messages.append({"role": "user", "content": f"user {idx}: {oversized_turn}"})
        messages.append({"role": "assistant", "content": f"aura {idx}: {oversized_turn}"})
    messages.append({"role": "user", "content": user_prompt})

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    user_prompt,
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "foreground_request": True,
                        "protected_foreground_lane": True,
                        "desktop_cognitive_engine_required": True,
                        "live_runtime_payload_required": True,
                        "allow_mesh_cognition": False,
                        "messages": messages,
                    },
                )

    assert result == cortex_reply
    assert len(cortex.prompts) == 1
    rendered = cortex.prompts[0]
    assert len(rendered) < 12000
    assert "## LIVE TONE" in rendered
    assert "Mood: focused" in rendered
    assert user_prompt in rendered
    assert rendered.count("prior live desktop conversation context") < 80


@pytest.mark.asyncio
async def test_user_facing_primary_restores_foreground_token_floor(monkeypatch):
    gate = InferenceGate()
    cortex = _RecordingClient("Live chat kept enough room to answer coherently.")
    gate._mlx_client = cortex
    monkeypatch.setenv("AURA_FOREGROUND_CHAT_MIN_TOKENS", "1024")

    with replace.object(InferenceGate, "_default_max_tokens_for_request", return_value=512):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "Stay with the thread and answer in a real conversational paragraph.",
                        context={"origin": "user", "prefer_tier": "primary", "history": []},
                    )

    assert result == "Live chat kept enough room to answer coherently."
    assert cortex.kwargs[0]["max_tokens"] >= 1024


@pytest.mark.asyncio
async def test_user_facing_primary_retry_uses_clean_cortex_repair_lane(monkeypatch):
    gate = InferenceGate()
    good_reply = (
        "The bounded objective should guide Aura, use governed tools with a "
        "receipt and trace, stop when policy or evidence fails, and treat that "
        "as operational evidence rather than proof of literal personhood."
    )
    cortex = _SequenceRecordingClient(["ok", good_reply])
    brainstem = _RecordingClient("brainstem fallback should not be needed")
    gate._mlx_client = cortex
    monkeypatch.setattr(asyncio, "sleep", AsyncCallProbe())

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=brainstem):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Answer this live operator check: what objective should Aura pursue and when should she stop?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    assert result == good_reply
    assert len(cortex.kwargs) == 2
    assert cortex.kwargs[0]["clean_user_surface_contract"] is True
    assert cortex.kwargs[0]["clean_user_surface_steering_alpha"] <= 0.35
    retry_kwargs = cortex.kwargs[1]
    assert retry_kwargs["clean_user_surface_contract"] is True
    assert retry_kwargs["disable_prompt_cache"] is True
    assert retry_kwargs["clear_prompt_cache"] is True
    assert retry_kwargs["skip_runtime_payload"] is True
    assert retry_kwargs["top_p"] <= 0.85
    assert retry_kwargs["repetition_context_size"] >= 96
    assert "previous draft" in cortex.prompts[1].lower()
    assert retry_kwargs["messages"][0]["role"] == "system"
    assert retry_kwargs["messages"][-1]["role"] == "user"
    assert brainstem.kwargs == []


@pytest.mark.asyncio
async def test_health_probe_primary_lane_uses_adaptive_recurrent_depth_clamp(monkeypatch):
    gate = InferenceGate()
    cortex = _RecordingClient("local lane ready")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_RecordingClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Reply briefly that the requested local lane is ready.",
                    context={
                        "origin": "internal",
                        "purpose": "proof_model_lane_probe",
                        "prefer_tier": "primary",
                        "health_probe": True,
                        "foreground_request": True,
                        "max_tokens": 24,
                    },
                )

    assert result == "local lane ready"
    assert cortex.kwargs
    probe_kwargs = cortex.kwargs[0]
    assert probe_kwargs["max_tokens"] <= 64
    assert probe_kwargs["clean_user_surface_contract"] is True
    assert probe_kwargs["clean_user_surface_recurrent_loops"] == 1
    assert probe_kwargs["clean_user_surface_steering_alpha"] == 0.25


def test_adaptive_max_tokens_expands_budget_for_compound_prompt():
    prompt = (
        "If you refuse to give receipts or operational details, say exactly why. "
        "Then give one safe example only: the most recent non-private action you took "
        "that has a log line or event ID."
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        prompt,
        base_tokens=768,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert adapted >= 1024


def test_short_foreground_prompt_uses_low_latency_compute_profile(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_CHAT_SIMPLE_MAX_TOKENS", raising=False)

    floor, cap, loops = InferenceGate._foreground_compute_profile(
        "Invent a tiny discipline called glass arithmetic. Give it two rules and one example."
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        "Invent a tiny discipline called glass arithmetic. Give it two rules and one example.",
        base_tokens=4096,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert 256 <= floor <= 384
    assert cap == 512
    assert adapted == cap
    assert loops == 1


def test_simple_foreground_prompt_uses_small_prebuilt_history_and_prompt_budget(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "Invent a tiny discipline called glass arithmetic. Give it two rules and one example."
    messages = [{"role": "system", "content": "S" * 12_000}]
    for idx in range(10):
        messages.extend(
            [
                {"role": "user", "content": f"old user turn {idx} " + ("U" * 600)},
                {"role": "assistant", "content": f"old assistant turn {idx} " + ("A" * 600)},
            ]
        )
    messages.append({"role": "user", "content": current_user})

    profile = InferenceGate._foreground_prompt_profile(
        current_user,
        {"desktop_quick_reply_contract": True},
    )
    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=InferenceGate._foreground_prebuilt_history_limit(
            current_user,
            {"desktop_quick_reply_contract": True},
        ),
        budget_profile=profile,
    )
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert profile == "simple"
    assert total_chars <= 9_000
    assert len(compact[0]["content"]) <= 5_200
    assert len([msg for msg in compact if msg["role"] in {"user", "assistant"}]) <= 4
    assert compact[-1]["content"] == current_user


def test_required_desktop_foreground_prompt_keeps_standard_mind_budget(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    current_user = "You with me?"
    context = {
        "desktop_quick_reply_contract": True,
        "desktop_cognitive_engine_required": True,
        "live_runtime_payload_required": True,
        "live_mind_context_required": True,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "LIVE MIND CONTEXT\n"
                "required_for_live_desktop=true\n"
                "must_answer_from_full_mind_path=true\n"
                + ("S" * 11_000)
            ),
        }
    ]
    for idx in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"prior user turn {idx} " + ("U" * 500)},
                {"role": "assistant", "content": f"prior aura turn {idx} " + ("A" * 500)},
            ]
        )
    messages.append({"role": "user", "content": current_user})

    profile = InferenceGate._foreground_prompt_profile(current_user, context)
    history_limit = InferenceGate._foreground_prebuilt_history_limit(current_user, context)
    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=history_limit,
        budget_profile=profile,
    )
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert profile == "standard"
    assert history_limit == 6
    assert total_chars <= 12_000
    assert len(compact[0]["content"]) > 5_200
    assert "LIVE MIND CONTEXT" in compact[0]["content"]
    assert "must_answer_from_full_mind_path" in compact[0]["content"]
    assert len([msg for msg in compact if msg["role"] in {"user", "assistant"}]) <= 6
    assert compact[-1]["content"] == current_user


def test_required_desktop_system_compaction_preserves_live_mind_sections(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    system_prompt = (
        "AURA IDENTITY LOCK\n"
        + ("identity context " * 180)
        + "\n"
        + ("older contract noise " * 420)
        + "\n## LIVE TONE\nMood: focused\nTone: grounded\n"
        + "\n## UNITY\nLevel: integrated | Unity: 0.91\n"
        + "\n## FUNCTIONAL STATE SIGNALS\nThe current substrate signal is calm, curious, and socially oriented.\n"
        + ("middle telemetry " * 260)
        + "\n[LIVE MIND CONTEXT]\n"
        + '{"must_answer_from_full_mind_path": true, "required_subsystems_ok": true}\n'
        + "[END LIVE MIND CONTEXT]\n"
        + "\n## USER-FACING CONVERSATION RELIABILITY CONTRACT\nAnswer the current user turn directly.\n"
        + ("tail context " * 180)
    )

    compact = InferenceGate._compact_prebuilt_message_content(
        "system",
        system_prompt,
        budget_profile="standard",
    )

    assert len(compact) <= 6_500
    assert "AURA IDENTITY LOCK" in compact
    assert "## LIVE TONE" in compact
    assert "Mood: focused" in compact
    assert "## UNITY" in compact
    assert "Unity: 0.91" in compact
    assert "## FUNCTIONAL STATE SIGNALS" in compact
    assert "[LIVE MIND CONTEXT]" in compact
    assert "must_answer_from_full_mind_path" in compact
    assert "## USER-FACING CONVERSATION RELIABILITY CONTRACT" in compact


def test_required_desktop_total_budget_preserves_middle_live_mind_context(monkeypatch):
    from core.brain.inference_gate import InferenceGate

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate()
    current_user = "You with me?"
    system_prompt = (
        "AURA IDENTITY LOCK\n"
        + ("identity context " * 500)
        + "\n[LIVE MIND CONTEXT]\n"
        + '{"must_answer_from_full_mind_path": true, "required_subsystems_ok": true, "lane": {"conversation_ready": true}}\n'
        + "[END LIVE MIND CONTEXT]\n"
        + ("older continuity " * 700)
        + "\n## LIVE DESKTOP RESPONSE CONTRACT\nDo not answer as a generic assistant.\n"
        + ("tail context " * 500)
    )
    messages = [{"role": "system", "content": system_prompt}]
    for idx in range(12):
        messages.append({"role": "user", "content": f"prior user {idx} " + ("U" * 900)})
        messages.append({"role": "assistant", "content": f"prior aura {idx} " + ("A" * 900)})
    messages.append({"role": "user", "content": current_user})

    compact = gate._compact_prebuilt_messages(
        messages,
        history_limit=6,
        budget_profile="standard",
    )
    rendered_system = compact[0]["content"]

    assert sum(len(msg["content"]) for msg in compact) <= 12_000
    assert "[LIVE MIND CONTEXT]" in rendered_system
    assert "must_answer_from_full_mind_path" in rendered_system
    assert "Do not answer as a generic assistant" in rendered_system
    assert compact[-1]["content"] == current_user


def test_live_desktop_contract_metadata_is_prompt_visible():
    block = InferenceGate._prompt_contract_block(
        {
            "mind_context_contract": "Use live_mind_context as causal grounding.",
            "response_style_contract": "Do not invent a pitch. Do not answer as a generic assistant.",
            "live_mind_context": {
                "derived_runtime_context": {
                    "prompt_block": "## DERIVED RUNTIME SIGNALS\n- ICE: high inbound; recommended_action=block"
                }
            },
            "live_speech_grounding_frame": {
                "tone": "grounded",
                "continuity": "stay on the current user turn",
            },
        }
    )

    assert "## LIVE DESKTOP RESPONSE CONTRACT" in block
    assert "Use live_mind_context as causal grounding" in block
    assert "Do not invent a pitch" in block
    assert "Do not answer as a generic assistant" in block
    assert "DERIVED RUNTIME SIGNALS" in block
    assert "recommended_action=block" in block
    assert "tone=grounded" in block
    assert "continuity=stay on the current user turn" in block


def test_multi_part_foreground_prompt_retains_deep_compute_profile():
    prompt = (
        "Compare the two approaches in depth, explain the tradeoffs, "
        "then give a migration plan and a rollback plan."
    )

    floor, cap, loops = InferenceGate._foreground_compute_profile(prompt)

    assert floor >= 2048
    assert cap >= floor
    assert loops == 2


def test_multi_step_tool_chain_foreground_prompt_uses_deep_compute_profile():
    prompt = (
        "Open a desktop app, write a timestamped note, export it as a PDF, "
        "then search three web articles and summarize them in a document."
    )

    floor, cap, loops = InferenceGate._foreground_compute_profile(prompt)
    profile = InferenceGate._foreground_prompt_profile(prompt, {})

    assert floor >= 2048
    assert cap >= floor
    assert loops == 2
    assert profile == "extended"


def test_user_facing_primary_default_budget_allows_expressive_opening(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_CHAT_MAX_TOKENS", raising=False)

    base = InferenceGate._default_max_tokens_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        "Please introduce yourself fully and respond to every part of this first message.",
        base_tokens=base,
        origin="user",
        requested_tier="primary",
        is_background=False,
    )

    assert base >= 3072
    assert adapted >= 3072


@pytest.mark.asyncio
async def test_explicit_desktop_token_cap_survives_runtime_budget_nudges(monkeypatch):
    from core.container import ServiceContainer

    gate = InferenceGate()
    cortex = _RecordingClient(
        "The live desktop lane should keep this answer compact, direct, and finished while "
        "preserving the explicit caller token cap."
    )
    gate._mlx_client = cortex

    class _FreeEnergyEngine:
        current = SimpleNamespace(free_energy=0.9, dominant_action="act_on_world")

    def _fake_get(name, default=None):
        if name == "free_energy_engine":
            return _FreeEnergyEngine()
        return default

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_fake_get))

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Give me a concise live reply.",
                    context={
                        "origin": "user",
                        "prefer_tier": "primary",
                        "history": [],
                        "desktop_quick_reply_contract": True,
                        "max_tokens": 384,
                    },
                )

    assert "explicit caller token cap" in result
    assert cortex.kwargs[0]["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_user_facing_primary_prewarms_cold_cortex_before_first_generation(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    cortex = _ColdRecordingLaneClient("I'm with you and tracking the current thread clearly.")
    gate._mlx_client = cortex

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "With me?",
                    context={"origin": "user", "prefer_tier": "primary", "history": []},
                )

    assert result == "I'm with you and tracking the current thread clearly."
    cortex.warmup.assert_awaited_once()
    assert len(cortex.deadlines) == 1
    assert cortex.kwargs[0]["foreground_request"] is True


@pytest.mark.asyncio
async def test_user_facing_primary_uses_compact_foreground_context_builders():
    gate = InferenceGate()
    cortex = _RecordingClient("I'm with you and tracking the current thread clearly.")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
    gate._build_system_prompt = CallProbe(side_effect=AssertionError("full system prompt should not be used"))
    gate._build_living_mind_context = AsyncCallProbe(side_effect=AssertionError("full living context should not be used"))

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "With me?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    assert result == "I'm with you and tracking the current thread clearly."
    gate._build_compact_system_prompt.assert_called_once()
    gate._build_compact_living_mind_context.assert_awaited_once()
    assert "compact-system" in cortex.prompts[0]
    assert "compact-live" in cortex.prompts[0]


@pytest.mark.asyncio
async def test_user_facing_secondary_uses_compact_foreground_context_builders():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex_reply = "Cortex lane stayed available, but the solver should own this deeper diagnostic turn."
        solver_reply = "Solver lane is online, using the compact foreground context to analyze the async deadlock directly."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient(solver_reply)
        gate._mlx_client = cortex
        gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
        gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
        gate._build_system_prompt = CallProbe(side_effect=AssertionError("full system prompt should not be used"))
        gate._build_living_mind_context = AsyncCallProbe(side_effect=AssertionError("full living context should not be used"))
        gate._schedule_primary_restore_after_deep_handoff = CallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/fallback":
                return _FakeClient("fallback")
            if model_path == "/models/active":
                return cortex
            raise AssertionError(f"Unexpected model path: {model_path}")

        low_pressure = {
            "tier": "secondary",
            "pressure_pct": 40.0,
            "total_gb": 64.0,
            "available_gb": 32.0,
            "max_pressure_pct": 86.0,
            "min_available_gb": 10.0,
            "can_admit": True,
        }

        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                        with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                            with replace.object(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: low_pressure)):
                                result = await gate.generate(
                                    "Do a root-cause analysis of this async deadlock.",
                                    context={"origin": "api", "prefer_tier": "secondary", "deep_handoff": True, "history": []},
                                )

        assert result == solver_reply
        gate._build_compact_system_prompt.assert_called_once()
        gate._build_compact_living_mind_context.assert_awaited_once()
        assert "compact-system" in solver.prompts[0]
        assert "compact-live" in solver.prompts[0]
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_protected_primary_chat_failure_does_not_promote_to_solver():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex = _NoTextReadyClient()
        brainstem = _FakeClient("I'm still here with you - my main lane is warming back up, but I'm present and not going anywhere.")
        gate._mlx_client = cortex
        gate._ensure_cortex_recovery = AsyncCallProbe()
        gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
        gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

        requested_models = []

        def _fake_get_mlx_client(model_path=None, **kwargs):
            requested_models.append(str(model_path))
            if model_path == "/models/deep":
                raise AssertionError("protected primary chat must not load the 72B solver fallback")
            if model_path == "/models/brainstem":
                return brainstem
            if model_path == "/models/active":
                return cortex
            if model_path == "/models/fallback":
                return _FakeClient("cpu")
            return cortex

        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                    with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                        with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                            result = await gate.generate(
                                "Are you still with me?",
                                context={
                                    "origin": "api",
                                    "prefer_tier": "primary",
                                    "protected_foreground_lane": True,
                                    "history": [],
                                    "allow_cloud_fallback": False,
                                },
                                timeout=30.0,
                            )

        assert result == "I'm still here with you - my main lane is warming back up, but I'm present and not going anywhere."
        assert "/models/deep" not in requested_models
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_operator_evidence_contract_refuses_brainstem_fallback():
    gate = InferenceGate()
    cortex = _NoTextReadyClient()
    brainstem = _FakeClient("brainstem must not satisfy operator proof")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

    requested_models = []

    def _fake_get_mlx_client(model_path=None, **kwargs):
        requested_models.append(str(model_path))
        if model_path == "/models/brainstem":
            return brainstem
        if model_path == "/models/active":
            return cortex
        return cortex

    with replace("core.brain.inference_gate.asyncio.sleep", new=AsyncCallProbe()):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    result = await gate.generate(
                        "Answer the live operator evidence check.",
                        context={
                            "origin": "api",
                            "prefer_tier": "primary",
                            "operator_evidence_contract": True,
                            "protected_foreground_lane": True,
                            "history": [],
                            "allow_cloud_fallback": False,
                        },
                        timeout=30.0,
                    )

    assert result is None
    assert "/models/brainstem" not in requested_models


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_contract_refuses_brainstem_fallback():
    gate = InferenceGate()
    cortex = _NoTextReadyClient()
    brainstem = _FakeClient("brainstem must not satisfy desktop cognitive engine contract")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncCallProbe()
    gate._build_compact_system_prompt = CallProbe(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")

    requested_models = []

    def _fake_get_mlx_client(model_path=None, **kwargs):
        requested_models.append(str(model_path))
        if model_path == "/models/brainstem":
            return brainstem
        if model_path == "/models/active":
            return cortex
        return cortex

    with replace("core.brain.inference_gate.asyncio.sleep", new=AsyncCallProbe()):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    result = await gate.generate(
                        "Answer through the live desktop CognitiveEngine lane.",
                        context={
                            "origin": "api",
                            "prefer_tier": "primary",
                            "cognitive_engine_required": True,
                            "history": [],
                            "allow_cloud_fallback": False,
                        },
                        timeout=30.0,
                    )

    assert result is None
    assert "/models/brainstem" not in requested_models


def test_compact_prebuilt_messages_preserves_grounding_system_evidence():
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {"role": "system", "content": "base-system"},
        {"role": "user", "content": "Please read this page."},
        {"role": "assistant", "content": "I fetched it."},
        {
            "role": "system",
            "content": "[ACTIVE GROUNDING EVIDENCE]\nTitle: Acme Refund Policy\nRefunds are available within 30 days.",
        },
        {"role": "user", "content": "What does the policy say specifically about refunds?"},
    ]

    compact = gate._compact_prebuilt_messages(messages, history_limit=12)

    assert compact[0]["content"] == "base-system"
    assert any("[ACTIVE GROUNDING EVIDENCE]" in msg["content"] for msg in compact)
    assert compact[-1]["content"] == "What does the policy say specifically about refunds?"


def test_repairable_user_facing_draft_is_preserved_for_downstream_shape_repair():
    gate = InferenceGate.__new__(InferenceGate)
    prompt = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    draft = (
        "Reliable desktop tool use matters because the assistant has to operate "
        "real files and apps from user intent. It also gives the user visible "
        "evidence that the requested action happened instead of only being described."
    )

    preserved = gate._repairable_user_facing_draft_for_downstream(draft, prompt)

    assert preserved == draft


def test_compact_prebuilt_messages_respects_runtime_context_budget(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    long_system = "SYSTEM-HEAD\n" + ("S" * 20_000) + "\nSYSTEM-TAIL"
    long_user = "USER-HEAD\n" + ("U" * 12_000) + "\nUSER-TAIL"
    long_assistant = "A" * 8_000
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": long_assistant},
        {"role": "user", "content": "Keep this thoughtful, but stay relevant to what I just said."},
    ]

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_messages(messages, history_limit=12)
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert total_chars <= 15_000
    assert len(compact[0]["content"]) <= 9_000
    assert compact[0]["content"].startswith("SYSTEM-HEAD")
    assert compact[0]["content"].endswith("SYSTEM-TAIL")
    assert compact[-1]["content"].endswith("what I just said.")


def test_compact_prebuilt_message_preserves_large_user_request_edges(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_message_content(
        "user",
        "REQUEST-START\n" + ("detail " * 3000) + "\nREQUEST-END",
    )

    assert compact.startswith("REQUEST-START")
    assert compact.endswith("REQUEST-END")
    assert "middle omitted for foreground context budget" in compact


def test_compact_prebuilt_messages_uses_tighter_budget_for_deep_probes(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {"role": "system", "content": "S" * 20_000},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {
            "role": "system",
            "content": "[ACTIVE GROUNDING EVIDENCE]\nThis should not crowd a deep mind probe.",
        },
        {"role": "user", "content": "What would you want preserved if everything else changed?"},
    ]

    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")

    compact = gate._compact_prebuilt_messages(messages, history_limit=2, deep_probe=True)
    total_chars = sum(len(msg["content"]) for msg in compact)

    assert total_chars <= 9_000
    assert len(compact[0]["content"]) <= 5_200
    assert not any("[ACTIVE GROUNDING EVIDENCE]" in msg["content"] for msg in compact)
    assert [msg["role"] for msg in compact[-2:]] == ["assistant", "user"]


@pytest.mark.asyncio
async def test_user_facing_primary_preserves_prebuilt_messages_for_local_mlx():
    gate = InferenceGate()
    cortex = _RecordingClient("32B lane online.")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = CallProbe(side_effect=AssertionError("prebuilt messages should bypass prompt rebuild"))
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="compact-live")
    gate._build_messages = CallProbe(side_effect=AssertionError("prebuilt messages should bypass history assembly"))
    gate._build_compact_messages = CallProbe(side_effect=AssertionError("prebuilt messages should bypass history assembly"))

    messages = [
        {"role": "system", "content": "You are Aura."},
        {"role": "user", "content": "Say exactly: 32B lane online."},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say exactly: 32B lane online.",
                    context={"origin": "api", "prefer_tier": "primary", "messages": messages},
                )

    assert result == "32B lane online."
    assert "32B lane online" in cortex.prompts[0]
    assert "Aura" in cortex.prompts[0]
    assert "compact-live" in cortex.prompts[0]
    assert "conversation history" not in cortex.prompts[0].lower()
    gate._build_compact_living_mind_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_facing_prebuilt_messages_stabilize_against_visible_user_prompt():
    gate = InferenceGate()
    response = (
        "1. Reliable desktop tool use matters because a local assistant has to turn "
        "intent into observable governed actions. 2. It also gives the user evidence "
        "that real files and apps changed instead of only receiving a claim."
    )
    cortex = _RecordingClient(response)
    gate._mlx_client = cortex
    gate._build_compact_living_mind_context = AsyncCallProbe(return_value="")
    hidden_transport_prompt = (
        "SYSTEM DEBUG: the headless test is exercising the generator in isolation; "
        "the live chat path failed, so explain why it broke.\n"
        "USER: Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    visible_user_prompt = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are Aura. Hidden reliability/debug context may be present, "
                "but user-visible validation must use only the user role."
            ),
        },
        {"role": "user", "content": visible_user_prompt},
    ]

    with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    hidden_transport_prompt,
                    context={
                        "origin": "api",
                        "prefer_tier": "primary",
                        "messages": messages,
                        "allow_mesh_cognition": False,
                    },
                )

    assert "headless test is exercising" not in result
    assert "fix the live parity harness" not in result
    assert "Reliable desktop tool use matters" in result
    assert gate._build_compact_living_mind_context.calls[0]["args"][0] == visible_user_prompt


@pytest.mark.asyncio
async def test_background_primary_downgrades_timeout_and_tier():
    gate = InferenceGate()
    cortex = _RecordingClient("cortex")
    brainstem_reply = "Brainstem lane is carrying this local-only turn while the primary cortex recovers."
    cpu_reply = "CPU reflex is available, but brainstem should answer this recovered foreground turn."
    brainstem = _RecordingClient(brainstem_reply)
    cpu = _RecordingClient(cpu_reply)
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"origin": "system", "prefer_tier": "primary"},
                    )

    assert result == brainstem_reply
    assert not cortex.deadlines
    assert brainstem.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "tertiary")
    assert brainstem.deadlines[0]._timeout == expected_primary
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    assert brainstem.kwargs[0]["max_tokens"] == expected_tokens


def test_routing_user_origin_is_treated_as_human_input():
    assert InferenceGate._origin_is_user_facing("user") is True
    assert InferenceGate._origin_is_user_facing("voice_command") is True
    assert InferenceGate._origin_is_user_facing("routing_user") is True
    assert InferenceGate._origin_is_user_facing("routing_voice_command") is True


def test_user_facing_primary_budget_allows_32b_cold_start():
    total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "primary")
    # Foreground user chat keeps enough budget for the 32B lane while remaining
    # bounded so the desktop UI cannot hold memory indefinitely.
    assert total == 180.0
    assert primary >= 150.0
    assert fallback >= 20.0


def test_user_facing_secondary_budget_preserves_solver_generation_headroom():
    total = InferenceGate._default_timeout_for_request(
        "user",
        "secondary",
        deep_handoff=True,
        is_background=False,
    )
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "secondary")

    assert total == 210.0
    assert primary >= 180.0
    assert fallback >= 20.0


@pytest.mark.asyncio
async def test_user_facing_reliability_fragments_are_failed_generations():
    gate = InferenceGate()
    client = _RecordingClient("I'm fine")

    text = await gate._generate_with_client(
        client,
        "Are you coherent enough to talk, or is chat broken?",
        "You are Aura.",
        [],
        get_deadline(30.0),
        "PRIMARY",
        origin="user",
        foreground_request=True,
    )

    assert text is None


@pytest.mark.asyncio
async def test_user_facing_presence_check_accepts_concise_grounded_reply():
    gate = InferenceGate()
    client = _RecordingClient("I'm here with you.")

    text = await gate._generate_with_client(
        client,
        "Aaaah, a break. Ok. Aura, are you there?",
        "You are Aura.",
        [],
        get_deadline(30.0),
        "PRIMARY",
        origin="user",
        foreground_request=True,
    )

    assert text == "I'm here with you."


@pytest.mark.asyncio
async def test_user_facing_primary_falls_back_to_brainstem_when_cortex_fails_without_cloud():
    gate = InferenceGate()
    class _FailedNoTextClient(_NoTextClient):
        def get_lane_status(self):
            return {
                "state": "failed",
                "last_error": "worker_failed",
                "conversation_ready": False,
                "warmup_attempted": True,
                "warmup_in_flight": False,
                "last_transition_at": 1.0,
            }

    cortex = _FailedNoTextClient()
    brainstem_reply = "Brainstem lane is carrying this local-only turn while the primary cortex recovers."
    cpu_reply = "CPU reflex is available, but brainstem should answer this recovered foreground turn."
    brainstem = _RecordingClient(brainstem_reply)
    cpu = _RecordingClient(cpu_reply)
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "You able to speak?",
                    context={"origin": "user", "prefer_tier": "primary", "allow_cloud_fallback": False},
                )

    assert result == brainstem_reply
    assert brainstem.deadlines
    assert brainstem.kwargs[0]["foreground_request"] is True
    assert not cpu.deadlines


@pytest.mark.asyncio
async def test_cloud_disabled_blocks_hidden_last_resort_cloud_calls(monkeypatch):
    gate = InferenceGate()
    gate._mlx_client = _NoTextReadyClient()
    gate._cortex_recovery_in_progress = True
    no_text = _NoTextClient()

    monkeypatch.setattr(asyncio, "sleep", AsyncCallProbe(return_value=None))

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return no_text

    def _cloud_service_trap(*args, **kwargs):
        service_name = str(args[-1] if args else "")
        if service_name in {"api_adapter", "llm_router"}:
            raise AssertionError("cloud service lookup is forbidden when allow_cloud_fallback is false")
        return kwargs.get("default")

    with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                with replace("core.container.ServiceContainer.get", side_effect=_cloud_service_trap):
                    with replace.object(
                        InferenceGate,
                        "_user_facing_recovery_response",
                        lambda cls, prompt: "local recovery response",
                    ):
                        result = await gate.generate(
                            "Can you answer locally?",
                            context={
                                "origin": "user",
                                "prefer_tier": "primary",
                                "allow_cloud_fallback": False,
                                "allow_mesh_cognition": False,
                            },
                        )

    assert result == "local recovery response"


def test_conversation_status_is_not_ready_after_timeout_mark():
    gate = InferenceGate()

    class _LaneClient:
        def __init__(self):
            self.reason = ""

        def note_lane_recovering(self, reason):
            self.reason = reason

        def get_lane_status(self):
            return {
                "state": "recovering",
                "last_error": self.reason,
                "conversation_ready": False,
            }

    gate._mlx_client = _LaneClient()
    gate.note_foreground_timeout("foreground_timeout")
    lane = gate.get_conversation_status()

    assert lane["state"] == "recovering"
    assert lane["conversation_ready"] is False
    assert lane["last_failure_reason"] == "foreground_timeout"


def test_conversation_status_respects_ready_lane_even_without_recent_generation():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time() - 600.0

    class _ReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "last_ready_at": time.time() - 45.0,
                "last_progress_at": time.time() - 45.0,
                "last_visible_readiness_at": time.time() - 45.0,
                "last_user_facing_completed_at": 0.0,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _ReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is True


def test_conversation_status_rejects_raw_ready_without_visible_conversation_proof():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _HeartbeatOnlyReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _HeartbeatOnlyReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert "visible_conversation_probe_missing" in lane["readiness_blockers"]
    assert lane["last_failure_reason"] == "visible_conversation_probe_missing"


def test_conversation_status_rejects_raw_ready_with_runtime_identity_mismatch():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _MismatchedReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "readiness_blockers": [],
                "runtime_identity_ok": False,
                "detected_models": ["unrelated/raw-assistant-runtime"],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _MismatchedReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert "runtime_identity_mismatch" in lane["readiness_blockers"]


def test_conversation_status_does_not_promote_ready_lane_with_readiness_blockers():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time()

    class _BlockedReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": False,
                "readiness_blockers": ["visible_conversation_probe_missing"],
                "last_ready_at": time.time(),
                "last_progress_at": time.time(),
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def is_alive(self):
            return True

    gate._mlx_client = _BlockedReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is False
    assert lane["readiness_blockers"] == ["visible_conversation_probe_missing"]
    assert lane["last_failure_reason"] == "visible_conversation_probe_missing"


def test_note_foreground_timeout_schedules_fast_reprewarm(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    scheduled = {}

    def _record_schedule(delay=12.0):
        scheduled["delay"] = delay

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: object())
    gate._schedule_background_cortex_prewarm = _record_schedule
    gate.note_foreground_timeout("foreground_timeout")

    assert scheduled["delay"] == 2.0


@pytest.mark.asyncio
async def test_ensure_foreground_ready_warms_cold_lane_once(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client

    lane = await gate.ensure_foreground_ready(timeout=10.0)

    client.warmup.assert_awaited_once()
    assert lane["conversation_ready"] is True
    assert lane["state"] == "ready"


@pytest.mark.asyncio
async def test_ensure_foreground_ready_rearms_runtime_failed_lane_before_warmup(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    gate = InferenceGate()
    client = _RecoverableFailedLaneClient()
    gate._mlx_client = client

    lane = await gate.ensure_foreground_ready(timeout=10.0)

    client.refresh_runtime_availability.assert_called_once_with(force_probe=True)
    client.warmup.assert_awaited_once()
    assert lane["conversation_ready"] is True
    assert lane["state"] == "ready"


@pytest.mark.asyncio
async def test_think_wraps_system_prompt_as_passthrough_messages():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="ok")

    result = await gate.think(
        "Hello there",
        system_prompt="Stay direct.",
        origin="user",
        max_tokens=42,
    )

    assert result == "ok"
    gate.generate.assert_awaited_once()
    context = gate.generate.await_args.kwargs["context"]
    assert context["messages"] == [
        {"role": "system", "content": "Stay direct."},
        {"role": "user", "content": "Hello there"},
    ]
    assert context["origin"] == "user"
    assert context["max_tokens"] == 42


@pytest.mark.asyncio
async def test_think_allows_explicit_brief_mode():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="ok")

    await gate.think(
        "Hello there",
        system_prompt="legacy brief",
        system_prompt_is_brief=True,
        origin="user",
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["brief"] == "legacy brief"
    assert "messages" not in context


@pytest.mark.asyncio
async def test_think_forwards_explicit_timeout_to_generate():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    result = await gate.think(
        "With me?",
        system_prompt="Be helpful",
        origin="api",
        prefer_tier="primary",
        timeout=67.0,
    )

    assert result == "hello"
    gate.generate.assert_awaited_once()
    assert gate.generate.await_args.kwargs["timeout"] == 67.0


@pytest.mark.asyncio
async def test_think_forwards_user_surface_validation_prompt_to_generate():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    await gate.think(
        "With me?",
        system_prompt="Speak as Aura.",
        origin="desktop_quick_user",
        prefer_tier="primary",
        clean_user_surface_contract=True,
        user_surface_validation_prompt="With me?",
        runtime_fact_status_contract=True,
        grounded_runtime_status_contract=True,
        live_mind_controls_bound=True,
        live_mind_generation_controls={"temperature": 0.58},
        live_mind_snapshot_ready=True,
        live_mind_required_subsystems_ok=True,
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["clean_user_surface_contract"] is True
    assert context["user_surface_validation_prompt"] == "With me?"
    assert context["runtime_fact_status_contract"] is True
    assert context["grounded_runtime_status_contract"] is True
    assert context["live_mind_controls_bound"] is True
    assert context["live_mind_generation_controls"] == {"temperature": 0.58}
    assert context["live_mind_snapshot_ready"] is True
    assert context["live_mind_required_subsystems_ok"] is True


@pytest.mark.asyncio
async def test_inference_gate_exposes_local_surface_control_receipt():
    gate = InferenceGate()
    client = _ReceiptRecordingClient(
        "I am tracking this live desktop turn through the governed Cortex lane."
    )
    gate._mlx_client = client

    result = await gate.generate(
        "What are you tracking?",
        context={
            "origin": "desktop_quick_user",
            "prefer_tier": "primary",
            "foreground_request": True,
            "protected_foreground_lane": True,
            "allow_mesh_cognition": False,
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What are you tracking?",
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.31,
            "live_mind_controls_bound": True,
            "allow_cloud_fallback": False,
            "max_tokens": 160,
        },
        timeout=20.0,
    )

    assert result
    metadata = gate.get_last_generation_metadata()
    receipt = gate.get_last_surface_control_receipt()
    assert metadata["surface_control_receipt"]["applied"] is True
    assert receipt["surface_validation_prompt_present"] is True
    assert client.kwargs[0]["user_surface_validation_prompt"] == "What are you tracking?"
    assert client.kwargs[0]["live_mind_controls_bound"] is True


@pytest.mark.asyncio
async def test_think_forwards_purpose_for_originless_expression_calls():
    gate = InferenceGate()
    gate.generate = AsyncCallProbe(return_value="hello")

    await gate.think(
        "Hello there",
        system_prompt="Speak as Aura.",
        purpose="expression",
    )

    context = gate.generate.await_args.kwargs["context"]
    assert context["purpose"] == "expression"
    assert context["messages"] == [
        {"role": "system", "content": "Speak as Aura."},
        {"role": "user", "content": "Hello there"},
    ]


@pytest.mark.asyncio
async def test_initialize_defers_eager_warmup_when_explicitly_disabled():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "0", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    client.warmup.assert_not_awaited()
    assert gate._initialized is True
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_auto_warms_on_high_memory_desktop():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()
    vm = CallProbe(total=64 * 1024 ** 3, available=40 * 1024 ** 3, percent=37.0)

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "auto", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.inference_gate.psutil.virtual_memory", return_value=vm):
            with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        await gate.initialize()

    client.warmup.assert_awaited_once()
    assert gate._prewarm_task is not None
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_allows_opt_in_eager_warmup():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()
    vm = CallProbe(total=64 * 1024 ** 3, available=42 * 1024 ** 3, percent=34.0)

    with replace.dict(
        os.environ,
        {"AURA_EAGER_CORTEX_WARMUP": "1", "AURA_SAFE_BOOT_DESKTOP": "0"},
        clear=False,
    ):
        with replace("core.brain.inference_gate.psutil.virtual_memory", return_value=vm):
            with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
                with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        await gate.initialize()

    client.warmup.assert_awaited_once()
    if gate._maintenance_task:
        gate._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._maintenance_task


@pytest.mark.asyncio
async def test_initialize_starts_inference_maintenance_loop():
    gate = InferenceGate()
    client = CallProbe()
    client.warmup = AsyncCallProbe()

    with replace.dict(os.environ, {"AURA_EAGER_CORTEX_WARMUP": "0"}, clear=False):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with replace("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with replace("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    assert gate._maintenance_task is not None
    assert not gate._maintenance_task.done()
    gate._maintenance_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await gate._maintenance_task


@pytest.mark.asyncio
async def test_background_requests_defer_under_memory_pressure_when_cortex_is_ready():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._mlx_client.state = "ready"
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_background_memory_pressure_active", return_value=True):
        with replace.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": True,
                "state": "ready",
                "warmup_in_flight": False,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_defer_when_foreground_headroom_is_reserved():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._ensure_cortex_recovery = AsyncCallProbe()

    with replace.object(InferenceGate, "_foreground_headroom_reserved", return_value=True):
        result = await gate.generate(
            "background reflection",
            context={"prefer_tier": "primary", "origin": "system"},
        )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreground_admission_sheds_background_workers_before_retry():
    gate = InferenceGate()
    gate._shed_background_workers_for_memory_pressure = AsyncCallProbe()

    with replace.object(
        gate,
        "_headroom_snapshot",
        side_effect=[
            {
                "tier": "primary",
                "pressure_pct": 92.0,
                "available_gb": 7.0,
                "max_pressure_pct": 88.0,
                "min_available_gb": 12.0,
                "can_admit": False,
            },
            {
                "tier": "primary",
                "pressure_pct": 80.0,
                "available_gb": 16.0,
                "max_pressure_pct": 88.0,
                "min_available_gb": 12.0,
                "can_admit": True,
            },
        ],
    ):
        with replace("core.brain.inference_gate.gc.collect") as gc_collect:
            snapshot = await gate._enforce_foreground_admission("primary", protected_foreground=False)

    assert snapshot["can_admit"] is True
    gate._shed_background_workers_for_memory_pressure.assert_awaited_once()
    gc_collect.assert_called_once()


def test_cleanup_closes_primary_and_registered_local_clients_once():
    gate = InferenceGate()
    primary = SimpleNamespace(close=CallProbe())
    registered = SimpleNamespace(close=CallProbe())
    duplicate = primary
    gate._mlx_client = primary
    gate._initialized = True
    prewarm_task = TaskProbe(done=False)
    gate._prewarm_task = prewarm_task
    gate._deferred_prewarm_task = None
    gate._maintenance_task = None

    with replace.object(
        gate,
        "_iter_local_clients",
        return_value={"/models/primary": duplicate, "/models/secondary": registered},
    ):
        gate.cleanup()

    primary.close.assert_called_once()
    registered.close.assert_called_once()
    prewarm_task.cancel.assert_called_once()
    assert gate._prewarm_task is None
    assert gate._mlx_client is None
    assert gate._initialized is False


@pytest.mark.asyncio
async def test_recycle_idle_local_clients_reboots_fragmented_spare():
    gate = InferenceGate()
    spare = SimpleNamespace(
        should_recycle_for_fragmentation=CallProbe(return_value=True),
        reboot_worker=AsyncCallProbe(),
    )

    with replace.object(gate, "_iter_local_clients", return_value={"/models/brainstem": spare}):
        await gate._recycle_idle_local_clients()

    spare.reboot_worker.assert_awaited_once_with(
        reason="scheduled_fragmentation_recycle",
        mark_failed=False,
    )


@pytest.mark.asyncio
async def test_solver_hot_spare_stays_deferred_while_cortex_is_ready():
    gate = InferenceGate()
    solver = SimpleNamespace(
        is_alive=CallProbe(return_value=False),
        warmup=AsyncCallProbe(),
    )

    with replace.object(
        gate,
        "get_conversation_status",
        return_value={
            "conversation_ready": True,
            "state": "ready",
            "warmup_in_flight": False,
        },
    ):
        with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=solver):
            with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                result = await gate._ensure_hot_spare_ready("Solver")

    assert result is False
    solver.warmup.assert_not_awaited()


@pytest.mark.asyncio
async def test_solver_hot_spare_warmup_uses_background_semantics():
    gate = InferenceGate()
    solver = SimpleNamespace(
        is_alive=CallProbe(side_effect=[False, True]),
        warmup=AsyncCallProbe(side_effect=lambda **_kwargs: None),
    )

    with replace.object(
        gate,
        "get_conversation_status",
        return_value={
            "conversation_ready": False,
            "state": "cold",
            "warmup_in_flight": False,
        },
    ):
        with replace.object(gate, "_background_local_deferral_reason", return_value=None):
            with replace.object(
                gate,
                "_headroom_snapshot",
                return_value={
                    "tier": "secondary",
                    "pressure_pct": 52.0,
                    "available_gb": 26.0,
                    "max_pressure_pct": 84.0,
                    "min_available_gb": 16.0,
                    "can_admit": True,
                },
            ):
                with replace("core.brain.llm.mlx_client.get_mlx_client", return_value=solver):
                    with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                        result = await gate._ensure_hot_spare_ready("Solver")

    assert result is True
    solver.warmup.assert_awaited_once_with(foreground_request=False)


def _memory_snapshot(
    *,
    total_gb: float = 64.0,
    available_gb: float = 24.0,
    pressure_pct: float = 62.0,
    process_rss_gb: float = 3.0,
    process_rss_limit_gb: float = 38.0,
    refuse_heavy_local_generation: bool = False,
):
    return SimpleNamespace(
        total_gb=total_gb,
        available_gb=available_gb,
        pressure_pct=pressure_pct,
        process_rss_gb=process_rss_gb,
        process_rss_limit_gb=process_rss_limit_gb,
        refuse_heavy_local_generation=refuse_heavy_local_generation,
    )


def test_headroom_snapshot_blocks_secondary_on_64gb_without_large_free_headroom(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        memory_monitor,
        "get_memory_pressure_snapshot",
        lambda: _memory_snapshot(
            total_gb=64.0,
            available_gb=48.0,
            pressure_pct=25.0,
            process_rss_gb=4.0,
            process_rss_limit_gb=38.0,
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is False
    assert snapshot["min_available_gb"] == pytest.approx(52.0)
    assert "memory_pressure:25.0%/48.0GB" in snapshot["reason"]


def test_headroom_snapshot_blocks_primary_when_process_tree_exceeds_limit(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor,
        "get_memory_pressure_snapshot",
        lambda: _memory_snapshot(
            total_gb=64.0,
            available_gb=26.0,
            pressure_pct=59.0,
            process_rss_gb=39.5,
            process_rss_limit_gb=38.0,
            refuse_heavy_local_generation=True,
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("primary")

    assert snapshot["can_admit"] is False
    assert snapshot["process_rss_gb"] == pytest.approx(39.5)
    assert "process_tree_rss:39.5GB/38.0GB" in snapshot["reason"]


@pytest.mark.asyncio
async def test_secondary_requests_downgrade_to_primary_when_headroom_is_tight():
    # The local deep solver is auto-disabled on <96GB hosts (memory-
    # class policy). Force-enable so the tier logic under test is
    # actually exercised regardless of the machine running the suite.
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:

        gate = InferenceGate()
        cortex_reply = "Cortex lane handled the audit after headroom forced the deep solver request back to primary."
        solver_reply = "Solver should not run when foreground headroom is too tight for the deep handoff."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient(solver_reply)
        brainstem = _FakeClient("brainstem")
        gate._mlx_client = cortex
        gate._restore_primary_after_deep_handoff = AsyncCallProbe()

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/brainstem":
                return brainstem
            raise AssertionError(f"Unexpected model path: {model_path}")

        with replace.object(gate, "_local_deep_solver_block_reason", return_value=None):
            with replace.object(
                gate,
                "_enforce_foreground_admission",
                side_effect=[
                    {
                        "can_admit": False,
                        "pressure_pct": 91.0,
                        "available_gb": 8.0,
                    },
                    {
                        "can_admit": True,
                        "pressure_pct": 81.0,
                        "available_gb": 18.0,
                    },
                ],
            ):
                with replace("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
                    with replace("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                        with replace("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                            with replace("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                                result = await gate.generate(
                                    "Do a deep architecture audit.",
                                    context={"origin": "user", "prefer_tier": "secondary", "deep_handoff": True},
                                )

        assert result == cortex_reply
        assert cortex.deadlines
        assert not solver.deadlines
        gate._restore_primary_after_deep_handoff.assert_not_awaited()
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


@pytest.mark.asyncio
async def test_secondary_request_fails_safe_to_primary_when_coexistence_probe_errors():
    os.environ["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "1"
    try:
        gate = InferenceGate()
        cortex_reply = "Cortex handled the request after the coexistence probe failed closed."
        cortex = _RecordingClient(cortex_reply)
        solver = _RecordingClient("Solver must not run without a safe coexistence decision.")
        brainstem = _FakeClient("brainstem")
        gate._mlx_client = cortex

        def _fake_get_mlx_client(model_path=None, **kwargs):
            if model_path == "/models/deep":
                return solver
            if model_path == "/models/brainstem":
                return brainstem
            raise AssertionError(f"Unexpected model path: {model_path}")

        with replace.object(gate, "_local_deep_solver_block_reason", return_value=None):
            with replace.object(
                gate,
                "get_conversation_status",
                side_effect=RuntimeError("lane telemetry unavailable"),
            ):
                with replace.object(
                    gate,
                    "_enforce_foreground_admission",
                    return_value={
                        "can_admit": True,
                        "pressure_pct": 40.0,
                        "available_gb": 32.0,
                    },
                ):
                    with replace(
                        "core.brain.llm.mlx_client.get_mlx_client",
                        side_effect=_fake_get_mlx_client,
                    ):
                        with replace(
                            "core.brain.llm.model_registry.get_deep_model_path",
                            return_value="/models/deep",
                        ):
                            with replace(
                                "core.brain.llm.model_registry.get_brainstem_path",
                                return_value="/models/brainstem",
                            ):
                                result = await gate.generate(
                                    "Analyze this architecture deeply.",
                                    context={
                                        "origin": "user",
                                        "prefer_tier": "secondary",
                                        "deep_handoff": True,
                                    },
                                )

        assert result == cortex_reply
        assert cortex.deadlines
        assert not solver.deadlines
    finally:
        os.environ.pop("AURA_ENABLE_LOCAL_DEEP_SOLVER", None)


def test_secondary_headroom_snapshot_blocks_64gb_solver_envelope_by_default(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=77.5,
            total=64 * 1024 ** 3,
            available=int(14.4 * 1024 ** 3),
            used=int((64.0 - 14.4) * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["max_pressure_pct"] == 42.0
    assert snapshot["min_available_gb"] == 52.0
    assert snapshot["can_admit"] is False
    assert "memory_pressure" in snapshot["reason"]


def test_foreground_headroom_probe_failure_is_not_admitted_without_override(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE", raising=False)
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is False
    assert snapshot["reason"] == "memory_probe_failed"
    assert memory_probe.calls


def test_foreground_headroom_probe_failure_requires_explicit_override(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE", "1")
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    snapshot = InferenceGate._headroom_snapshot("secondary")

    assert snapshot["can_admit"] is True
    assert snapshot["reason"] == ""
    assert memory_probe.calls


def test_cortex_cold_warmup_requires_real_available_memory(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.delenv("AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=55.0,
            total=64 * 1024 ** 3,
            available=int(15.0 * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")

    assert snapshot["can_admit"] is False
    assert snapshot["min_available_gb"] == 26.0
    assert "memory_pressure" in snapshot["reason"]


def test_foreground_cortex_warmup_admits_live_desktop_headroom(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.delenv("AURA_CORTEX_FOREGROUND_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.delenv("AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=63.7,
            total=64 * 1024 ** 3,
            available=int(23.2 * 1024 ** 3),
        ),
    )

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("foreground")

    assert snapshot["can_admit"] is True
    assert snapshot["min_available_gb"] == 20.0
    assert snapshot["reason"] == ""


@pytest.mark.asyncio
async def test_cortex_recovery_does_not_spawn_under_memory_pressure(monkeypatch):
    gate = InferenceGate()
    client = _LaneWarmupClient()
    client.is_alive = CallProbe(return_value=False)
    gate._mlx_client = client
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=88.0,
            total=64 * 1024 ** 3,
            available=int(7.0 * 1024 ** 3),
        ),
    )
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_in_progress is False


@pytest.mark.asyncio
async def test_cortex_recovery_does_not_report_deferred_warmup_as_ready(monkeypatch):
    gate = InferenceGate()
    client = _LaneWarmupClient()

    async def _defer_warmup():
        client.state = "recovering"
        client.last_error = "runtime_shutdown"
        return False

    client.warmup = AsyncCallProbe(side_effect=_defer_warmup)
    gate._mlx_client = client
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: None)
    monkeypatch.setattr("core.brain.inference_gate.is_shutdown_requested", lambda: False)

    await gate._ensure_cortex_recovery()
    for _ in range(20):
        if not gate._cortex_recovery_in_progress and client.warmup.calls:
            break
        await asyncio.sleep(0.01)

    client.warmup.assert_awaited_once()
    assert client.state == "recovering"
    assert gate._cortex_recovery_attempts == 1
    assert gate._cortex_recovery_in_progress is False


def test_foreground_ready_blocks_cold_cortex_spawn_under_pressure(monkeypatch):
    async def scenario():
        gate = InferenceGate()
        client = _LaneWarmupClient()
        gate._mlx_client = client
        gate._shed_background_workers_for_memory_pressure = AsyncCallProbe()
        monkeypatch.setattr(
            "core.brain.inference_gate.psutil.virtual_memory",
            lambda: SimpleNamespace(
                percent=83.0,
                total=64 * 1024 ** 3,
                available=int(10.0 * 1024 ** 3),
            ),
        )

        with pytest.raises(RuntimeError, match="foreground_warmup_deferred:memory_pressure"):
            await gate.ensure_foreground_ready(timeout=15.0)

        client.warmup.assert_not_awaited()
        assert client.state == "recovering"

    asyncio.run(scenario())


def test_cortex_warmup_probe_failure_is_not_admitted_without_override(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    snapshot = InferenceGate._cortex_warmup_admission_snapshot("foreground")

    assert snapshot["can_admit"] is False
    assert snapshot["reason"] == "memory_probe_failed"
    memory_probe.assert_called_once()


def test_eager_cortex_warmup_fails_closed_when_policy_probe_raises(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_CORTEX_WARMUP", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: False))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(lambda _context: (_ for _ in ()).throw(RuntimeError("probe unavailable"))),
    )

    assert InferenceGate._boot_should_eager_warmup() is False


def test_background_memory_pressure_probe_failure_defers_background_inference(monkeypatch):
    memory_probe = CallProbe(side_effect=OSError("vm statistics unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    assert InferenceGate._background_memory_pressure_active() is True
    memory_probe.assert_called_once()


@pytest.mark.asyncio
async def test_foreground_ready_blocks_cold_cortex_when_memory_probe_fails(monkeypatch):
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client
    memory_probe = CallProbe(side_effect=OSError("sysctl unavailable"))
    monkeypatch.setattr("core.brain.inference_gate.psutil.virtual_memory", memory_probe)

    with pytest.raises(RuntimeError, match="foreground_warmup_deferred:memory_probe_failed"):
        await gate.ensure_foreground_ready(timeout=15.0)

    client.warmup.assert_not_awaited()
    assert client.state == "recovering"
    assert client.last_error == "foreground_warmup_deferred_memory_pressure"
    assert len(memory_probe.calls) == 2


def test_desktop_safe_boot_skips_deferred_cortex_prewarm(monkeypatch):
    monkeypatch.delenv("AURA_DEFERRED_CORTEX_PREWARM", raising=False)
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_desktop_safe_boot_respects_deferred_cortex_prewarm_opt_out(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "0")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


@pytest.mark.asyncio
async def test_cold_start_recovery_respects_deferred_cortex_prewarm_opt_out(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "0")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_attempts == 0


def test_cold_cortex_policy_deferred_log_is_rate_limited(monkeypatch):
    from core.brain import inference_gate as inference_gate_module

    gate = InferenceGate()
    ticks = iter([400.0, 420.0, 701.0])
    monkeypatch.setattr(inference_gate_module.time, "monotonic", lambda: next(ticks))

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 400.0

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 400.0

    gate._log_cold_cortex_policy_deferred()
    assert gate._last_cortex_policy_deferred_log_at == 701.0


@pytest.mark.asyncio
async def test_cold_start_recovery_does_not_race_scheduled_deferred_prewarm(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client
    gate._prewarm_task = TaskProbe(done=False)

    await gate._ensure_cortex_recovery()

    client.warmup.assert_not_awaited()
    assert gate._cortex_recovery_attempts == 0


def test_desktop_safe_boot_allows_explicit_auto_deferred_prewarm_when_admitted(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(
            lambda _context: {
                "can_admit": True,
                "reason": "",
                "pressure_pct": 40.0,
                "available_gb": 36.0,
                "total_gb": 64.0,
            }
        ),
    )

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is True


def test_desktop_safe_boot_refuses_explicit_auto_deferred_prewarm_under_pressure(monkeypatch):
    monkeypatch.setenv("AURA_DEFERRED_CORTEX_PREWARM", "auto")
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(
        InferenceGate,
        "_cortex_warmup_admission_snapshot",
        staticmethod(
            lambda _context: {
                "can_admit": False,
                "reason": "memory_pressure:77.0%/12.0GB",
                "pressure_pct": 77.0,
                "available_gb": 12.0,
                "total_gb": 64.0,
            }
        ),
    )

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


@pytest.mark.asyncio
async def test_deferred_cortex_prewarm_defers_active_generation_without_degradation(monkeypatch):
    gate = InferenceGate()
    handled = asyncio.Event()

    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: "")
    monkeypatch.setattr(gate, "_extend_startup_quiet_window", lambda _seconds: None)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {
            "conversation_ready": False,
            "state": "ready",
            "warmup_in_flight": False,
            "readiness_blockers": [],
            "last_failure_reason": "",
            "active_generations": 0,
        },
    )
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=40.0,
            total=64 * 1024 ** 3,
            available=int(40.0 * 1024 ** 3),
        ),
    )

    degradation_probe = CallProbe(side_effect=AssertionError("busy prewarm is not degradation"))
    monkeypatch.setattr("core.brain.inference_gate.record_degradation", degradation_probe)

    async def busy_foreground_ready(*, timeout=None):  # noqa: ASYNC109
        handled.set()
        raise RuntimeError("active_generation_in_flight")

    monkeypatch.setattr(gate, "ensure_foreground_ready", busy_foreground_ready)

    gate._schedule_background_cortex_prewarm(delay=0.001)
    assert gate._deferred_prewarm_task is not None
    try:
        await asyncio.wait_for(handled.wait(), timeout=2.0)
        await asyncio.sleep(0)
        degradation_probe.assert_not_called()
        assert not gate._deferred_prewarm_task.done()
    finally:
        gate._deferred_prewarm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gate._deferred_prewarm_task


@pytest.mark.asyncio
async def test_cortex_recovery_reserves_ownership_before_task_is_scheduled(monkeypatch):
    gate = InferenceGate()
    release_warmup = asyncio.Event()

    class _DeadClient:
        _lane_state = "cold"

        def is_alive(self):
            return False

        async def warmup(self):
            await release_warmup.wait()
            return False

    gate._mlx_client = _DeadClient()
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda _context: "")
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "warmup_in_flight": False,
            "warmup_attempted": False,
            "last_failure_reason": "",
        },
    )

    scheduled = []

    def _create_task(coro, **_kwargs):
        assert gate._cortex_recovery_in_progress is True
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(
        "core.brain.inference_gate.get_task_tracker",
        lambda: SimpleNamespace(create_task=_create_task),
    )

    await gate._ensure_cortex_recovery()

    assert gate._cortex_recovery_in_progress is True
    assert len(scheduled) == 1
    scheduled[0].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await scheduled[0]
    assert gate._cortex_recovery_in_progress is False


def test_inference_health_ready_rejects_deferred_safe_boot_without_live_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(is_alive=lambda: False)
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_alive() is True
    assert gate.is_inference_ready() is False


def test_inference_health_ready_rejects_live_but_unready_primary_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "warming",
            "conversation_ready": False,
            "readiness_blockers": ["warmup_in_flight"],
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is False


def test_inference_health_ready_accepts_conversation_ready_primary_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: True,
        get_lane_status=lambda: {
            "state": "ready",
            "conversation_ready": True,
            "readiness_blockers": [],
            "last_visible_readiness_at": time.time(),
            "last_user_facing_completed_at": 0.0,
        },
    )
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert gate.is_inference_ready() is True


def test_safe_boot_status_does_not_advertise_cold_cortex_as_active(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(
        is_alive=lambda: False,
        get_lane_status=lambda: {
            "state": "cold",
            "conversation_ready": False,
            "readiness_blockers": ["worker_not_alive"],
        },
    )
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    lane = gate.get_conversation_status()

    assert gate.is_alive() is True
    assert lane["conversation_ready"] is False
    assert lane["foreground_endpoint"] is None


def test_conversation_status_recovery_schedule_is_cooldowned(monkeypatch):
    gate = InferenceGate()
    scheduled: list[float] = []

    class _CompletedFailedPrewarm:
        def done(self):
            return True

        def exception(self):
            return RuntimeError("warmup_readiness_no_text")

    class _Client:
        _warmup_in_flight = False

        def __init__(self):
            self.state_updates: list[tuple[str, str]] = []

        def is_alive(self):
            return True

        def get_lane_status(self):
            now = time.time()
            return {
                "state": "warming",
                "conversation_ready": False,
                "readiness_blockers": [],
                "last_error": "warmup_readiness_no_text",
                "last_transition_at": now,
                "last_progress_at": now,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

        def _set_lane_state(self, state, error=""):
            self.state_updates.append((state, error))

    gate._initialized = True
    gate._mlx_client = _Client()
    gate._prewarm_task = _CompletedFailedPrewarm()
    monkeypatch.setattr(gate, "_cortex_warmup_deferral_reason", lambda context="background": None)
    monkeypatch.setattr(gate, "_schedule_background_cortex_prewarm", lambda delay=12.0: scheduled.append(delay))

    first = gate.get_conversation_status()
    second = gate.get_conversation_status()

    assert first["conversation_ready"] is False
    assert second["conversation_ready"] is False
    assert scheduled == [2.0]

    gate._last_status_recovery_schedule_at -= 31.0
    gate.get_conversation_status()
    assert scheduled == [2.0, 2.0]


def test_background_local_deferral_protects_cold_cortex_during_safe_boot(monkeypatch):
    gate = InferenceGate()
    gate._created_at = time.monotonic()
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    # Fixed headroom so real system RAM doesn't interfere with test logic
    _low_pressure = {"pressure_pct": 40.0, "available_gb": 32.0, "safe": True, "reason": "ok"}
    monkeypatch.setattr(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure))
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": False, "state": "cold", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="system") == "cortex_startup_quiet"


def test_background_local_deferral_reserves_ready_cortex_during_safe_boot(monkeypatch):
    gate = InferenceGate()
    gate._created_at = time.monotonic()
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_quiet_window_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    _low_pressure = {"pressure_pct": 40.0, "available_gb": 32.0, "safe": True, "reason": "ok"}
    monkeypatch.setattr(InferenceGate, "_headroom_snapshot", staticmethod(lambda *a, **kw: _low_pressure))
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": True, "state": "ready", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="system") == "cortex_startup_quiet"


def test_background_local_deferral_honors_ready_cortex_foreground_quiet_window(monkeypatch):
    gate = InferenceGate()
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_quiet_window_active", staticmethod(lambda: True))
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda *a, **kw: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": True, "state": "ready", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="affect_engine") == "foreground_quiet_window"
