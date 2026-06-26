import pytest

from core.brain.llm_health_router import HealthAwareLLMRouter


class EndpointClient:
    def __init__(self, response: str):
        self.response = response
        self.failure: Exception | None = None
        self.calls = []

    async def think(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.failure is not None:
            raise self.failure
        return self.response


class InferenceGateRecorder:
    def get_conversation_status(self):
        return {
            "conversation_ready": False,
            "state": "warming",
            "warmup_in_flight": True,
        }

    def _background_local_deferral_reason(self, origin=None):
        return "cortex_startup_quiet"


class ContextGenerateClient:
    def __init__(self, response: str = "context response"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, context=None):
        self.calls.append({"prompt": prompt, "context": dict(context or {})})
        return self.response


class ReceiptGenerateClient:
    async def generate(self, prompt, **kwargs):
        return "receipt-bound response"

    def get_last_surface_control_receipt(self):
        return {
            "enabled": True,
            "live_mind_controls_bound": True,
            "clean_user_surface_contract": True,
            "surface_alpha_applied": 0.28,
            "surface_alpha_applied_ok": True,
            "recurrent_runtime_loops_applied": 2,
            "recurrent_runtime_loops_applied_ok": True,
            "applied": True,
        }


class ReceiptThinkClient:
    async def think(self, prompt, **kwargs):
        return "receipt-bound think response"

    def get_last_surface_control_receipt(self):
        return {
            "enabled": True,
            "live_mind_controls_bound": True,
            "clean_user_surface_contract": True,
            "surface_validation_prompt_present": True,
            "surface_alpha_applied": 0.28,
            "surface_alpha_applied_ok": True,
            "recurrent_runtime_loops_applied": 2,
            "recurrent_runtime_loops_applied_ok": True,
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": True,
            "surface_quality_gate_attempts": 1,
            "surface_quality_gate_reasons": [],
            "applied": True,
        }


@pytest.fixture
def router_clients():
    router = HealthAwareLLMRouter()
    clients = {
        "cortex": EndpointClient("32B response"),
        "solver": EndpointClient("72B response"),
        "brainstem": EndpointClient("7B response"),
        "api": EndpointClient("API response"),
    }
    router.register(
        name="Cortex",
        url="internal",
        model="cortex-32b",
        is_local=True,
        tier="local",
        client=clients["cortex"],
    )
    router.register(
        name="Solver",
        url="internal",
        model="solver-72b",
        is_local=True,
        tier="local_deep",
        client=clients["solver"],
    )
    router.register(
        name="Brainstem",
        url="internal",
        model="brainstem-7b",
        is_local=True,
        tier="local_fast",
        client=clients["brainstem"],
    )
    router.register(
        name="Gemini-Fast",
        url="cloud",
        model="gemini-2.0-flash",
        is_local=False,
        tier="api_fast",
        client=clients["api"],
    )
    return router, clients


@pytest.mark.asyncio
async def test_primary_tier_excludes_solver_lane(router_clients):
    router, clients = router_clients
    clients["cortex"].failure = RuntimeError("32B failed")

    result = await router.generate_with_metadata("Hello", prefer_tier="primary")

    assert result["endpoint"] == "all_failed"
    assert clients["solver"].calls == []


@pytest.mark.asyncio
async def test_secondary_tier_requires_explicit_deep_handoff(router_clients):
    router, clients = router_clients

    result = await router.generate_with_metadata(
        "Complex task",
        prefer_tier="secondary",
        deep_handoff=True,
    )

    assert result["endpoint"] == "Solver"
    assert result["text"] == "72B response"
    assert len(clients["solver"].calls) == 1


@pytest.mark.asyncio
async def test_no_tier_preference_defaults_to_primary_without_solver(router_clients):
    router, clients = router_clients
    clients["cortex"].failure = RuntimeError("32B failed")
    clients["api"].failure = RuntimeError("API failed")

    result = await router.generate_with_metadata("Hello")

    assert result["endpoint"] == "all_failed"
    assert clients["solver"].calls == []


@pytest.mark.asyncio
async def test_live_benchmark_requests_stay_on_cortex_lane(router_clients):
    router, clients = router_clients

    result = await router.generate_with_metadata(
        "Repair this multi-file traceback and emit only the patched artifact.",
        prefer_tier="secondary",
        deep_handoff=True,
        origin="benchmark",
        purpose="benchmark_evaluation",
        benchmark_request=True,
        skip_runtime_payload=True,
    )

    assert result["endpoint"] == "Cortex"
    assert result["text"] == "32B response"
    assert clients["solver"].calls == []


@pytest.mark.asyncio
async def test_router_surfaces_mlx_surface_control_receipt():
    router = HealthAwareLLMRouter()
    client = ReceiptGenerateClient()
    router.register(
        name="Cortex",
        url="internal",
        model="cortex-32b",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "Hello",
        prefer_tier="primary",
        origin="user",
        skip_runtime_payload=True,
        clean_user_surface_contract=True,
        live_mind_controls_bound=True,
    )

    assert result["surface_control_receipt"]["live_mind_controls_bound"] is True
    assert result["surface_control_receipt"]["applied"] is True

    text = await router.think(
        "Hello again",
        prefer_tier="primary",
        origin="user",
        skip_runtime_payload=True,
        clean_user_surface_contract=True,
        live_mind_controls_bound=True,
    )

    assert text == "receipt-bound response"
    assert router.get_last_generation_metadata()["surface_control_receipt"]["applied"] is True


@pytest.mark.asyncio
async def test_router_surfaces_inference_gate_surface_control_receipt_from_think_client():
    router = HealthAwareLLMRouter()
    client = ReceiptThinkClient()
    router.register(
        name="Cortex",
        url="internal",
        model="cortex-32b",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "Hello",
        prefer_tier="primary",
        origin="desktop_quick_user",
        skip_runtime_payload=True,
        clean_user_surface_contract=True,
        user_surface_validation_prompt="Hello",
        live_mind_controls_bound=True,
    )

    assert result["text"] == "receipt-bound think response"
    assert result["surface_control_receipt"]["applied"] is True
    assert result["surface_control_receipt"]["surface_validation_prompt_present"] is True


@pytest.mark.asyncio
async def test_foreground_primary_skips_brainstem_and_uses_cloud_fallback(router_clients):
    router, clients = router_clients
    clients["cortex"].failure = RuntimeError("32B failed")

    result = await router.generate_with_metadata(
        "Hello",
        prefer_tier="primary",
        origin="user",
        allow_cloud_fallback=True,
    )

    assert result["endpoint"] == "Gemini-Fast"
    assert result["text"] == "API response"
    assert clients["brainstem"].calls == []


@pytest.mark.asyncio
async def test_gui_report_prefers_last_foreground_endpoint_over_background(router_clients, monkeypatch):
    router, _clients = router_clients

    await router.generate("Hello", prefer_tier="primary", origin="user")
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "0")
    await router.generate(
        "Idle thought",
        prefer_tier="tertiary",
        origin="system",
        is_background=True,
    )

    report = router.get_health_report()
    assert report["current_tier"] == "Cortex (32B)"
    assert report["active_endpoint"] == "Cortex"
    assert report["background_endpoint"] == "Brainstem"


@pytest.mark.asyncio
async def test_background_quiet_window_blocks_brainstem_until_cortex_ready(router_clients, monkeypatch):
    router, clients = router_clients
    gate = InferenceGateRecorder()

    def get_service(cls, name, default=None):
        if name == "inference_gate":
            return gate
        return default

    monkeypatch.setattr(HealthAwareLLMRouter, "_foreground_quiet_window_active", lambda self: True)
    monkeypatch.setattr("core.container.ServiceContainer.get", classmethod(get_service))

    result = await router.generate_with_metadata(
        "Idle thought",
        prefer_tier="tertiary",
        origin="system",
        is_background=True,
    )

    assert result["endpoint"] == "suppressed"
    assert result["error"] in {
        "foreground_quiet_window",
        "background_deferred:cortex_startup_quiet",
        "background_deferred:conversation_lane_warming",
    }
    assert clients["brainstem"].calls == []


@pytest.mark.asyncio
async def test_gui_report_mapping(router_clients):
    router, _clients = router_clients

    await router.generate("Hello", prefer_tier="primary")
    report = router.get_health_report()
    assert report["current_tier"] == "Cortex (32B)"
    assert report["active_endpoint"] == "Cortex"

    await router.generate("Hello", prefer_tier="secondary", deep_handoff=True)
    report = router.get_health_report()
    assert report["current_tier"] == "Solver (72B)"


@pytest.mark.asyncio
async def test_router_preserves_clean_surface_contract_for_context_generate_client():
    router = HealthAwareLLMRouter()
    client = ContextGenerateClient("32B clean surface")
    router.register(
        name="Cortex",
        url="internal",
        model="cortex-32b",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "Write a direct user-visible answer.",
        prefer_tier="primary",
        origin="user",
        purpose="chat",
        foreground_request=True,
        clean_user_surface_contract=True,
        clean_user_surface_recurrent_loops=1,
        clean_user_surface_steering_alpha=0.25,
        operator_evidence_contract=True,
        skip_runtime_payload=True,
    )

    assert result["endpoint"] == "Cortex"
    assert result["text"] == "32B clean surface"
    assert len(client.calls) == 1
    context = client.calls[0]["context"]
    assert context["origin"] == "user"
    assert context["prefer_tier"] == "primary"
    assert context["foreground_request"] is True
    assert context["operator_evidence_contract"] is True
    assert context["clean_user_surface_contract"] is True
    assert context["clean_user_surface_recurrent_loops"] == 1
    assert context["clean_user_surface_steering_alpha"] == 0.25
