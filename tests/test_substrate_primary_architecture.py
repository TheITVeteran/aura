from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_observation_vector_is_grounded_and_state_dependent():
    from core.brain.llm.sensorimotor_grounding import observation_to_vector

    visual = observation_to_vector(
        {"source": "camera", "summary": "bright window", "confidence": 0.9, "energy": 0.4},
        dim=64,
    )
    audio = observation_to_vector(
        {"source": "microphone", "transcript": "Bryan said run the tests", "confidence": 0.8, "rms": 0.2},
        dim=64,
    )

    assert visual.shape == (64,)
    assert audio.shape == (64,)
    assert float(np.linalg.norm(visual)) > 0.0
    assert float(np.linalg.norm(audio)) > 0.0
    assert not np.allclose(visual, audio)


def test_continuous_substrate_accepts_sensor_observations():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate

    substrate = ContinuousSubstrate()
    substrate.inject_observation(
        {"source": "screen", "summary": "terminal test output changed", "confidence": 0.7, "energy": 0.3}
    )
    for _ in range(8):
        substrate._step_once()

    summary = substrate.get_state_summary()
    assert summary["grounded_observation"] is True
    assert summary["last_observation_source"] == "screen"
    assert float(np.linalg.norm(substrate.get_state_vector())) > 0.0


def test_substrate_token_generator_changes_output_under_lesion():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.substrate_token_generator import SubstrateTokenGenerator

    substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate, threshold=1.0)

    substrate.inject_input(np.ones(64, dtype=np.float32) * 0.55)
    for _ in range(12):
        substrate._step_once()
    intact = generator.generate("continue the repair loop", force=True)

    substrate._state = np.zeros(64, dtype=np.float32)
    lesioned = generator.generate("continue the repair loop", force=True)

    assert intact.used_substrate is True
    assert lesioned.used_substrate is True
    assert intact.logits_checksum != lesioned.logits_checksum
    assert intact.token_ids != lesioned.token_ids


def test_substrate_token_generator_falls_back_on_high_prediction_error():
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.substrate_token_generator import SubstrateTokenGenerator

    substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate, threshold=0.05)

    result = generator.generate(
        "Explain a multi-file architecture migration with hidden external baselines and receipts?",
        max_tokens=16,
    )

    assert result.used_substrate is False
    assert result.fallback_reason == "prediction_error_exceeded"
    assert result.prediction_error > result.threshold


@pytest.mark.asyncio
async def test_llm_router_uses_substrate_before_transformer(monkeypatch):
    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.llm_router import IntelligentLLMRouter
    from core.container import ServiceContainer

    ServiceContainer.clear()
    substrate = ContinuousSubstrate()
    substrate.inject_input(np.ones(64, dtype=np.float32) * 0.6)
    for _ in range(12):
        substrate._step_once()
    ServiceContainer.register_instance("continuous_substrate", substrate, required=False)

    monkeypatch.setenv("AURA_SUBSTRATE_PRIMARY", "1")
    router = IntelligentLLMRouter()
    text = await router.think("quiet status", force_substrate=True, max_tokens=8, origin="user")

    assert text.startswith("Substrate path:")
    assert router.last_user_tier == "substrate"


@pytest.mark.asyncio
async def test_online_lora_governor_blocks_when_training_is_running(
    tmp_path,
    resource_observer,
):
    from core.adaptation.online_lora_governor import OnlineLoRAGovernor
    from core.runtime.resource_observation import ProcessObservation

    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=123,
                ppid=1,
                create_time=1.0,
                status="running",
                name="python",
                cmdline=("python", "-m", "mlx_lm", "lora", "--train"),
                rss_bytes=1024,
            )
        ]
    )
    governor = OnlineLoRAGovernor(
        receipt_path=tmp_path / "receipts.jsonl",
        observer=resource_observer,
    )

    receipt = await governor.maybe_update_from_reflection("I noticed a repair pattern.")

    assert receipt.status == "blocked_existing_training"
    assert "pid=123" in receipt.reason


@pytest.mark.asyncio
async def test_online_lora_governor_collects_without_false_update_status(
    tmp_path,
    resource_observer,
):
    from core.adaptation.online_lora_governor import OnlineLoRAGovernor
    from core.container import ServiceContainer

    ServiceContainer.clear()
    resource_observer.configure_processes([])
    governor = OnlineLoRAGovernor(
        receipt_path=tmp_path / "receipts.jsonl",
        observer=resource_observer,
    )

    receipt = await governor.maybe_update_from_reflection("I noticed a repair pattern.")

    assert receipt.status == "queued_collect_only"
    assert "no canonical learning owner" in receipt.reason


@pytest.mark.asyncio
async def test_default_goal_seeder_creates_tool_attached_goals(tmp_path):
    from core.goals.default_goals import DEFAULT_AUTONOMY_GOALS, seed_default_autonomy_goals
    from core.goals.goal_engine import GoalEngine

    engine = GoalEngine(db_path=str(tmp_path / "goals.sqlite3"))
    seeded = await seed_default_autonomy_goals(engine)

    assert len(seeded) == len(DEFAULT_AUTONOMY_GOALS)
    assert all(goal["status"] == "in_progress" for goal in seeded)
    assert all(goal["required_tools"] for goal in seeded)


@pytest.mark.asyncio
async def test_overt_action_loop_executes_verifies_and_receipts(tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": "Run a light environment self-audit",
                    "source": "test_goal",
                    "urgency": 0.6,
                    "metadata": {"required_skills": ["environment_info"]},
                },
                will_receipt_id="will-test-1",
            )

    class FakeEngine:
        def __init__(self):
            self.calls = []

        async def execute(self, skill_name, params, context=None):
            self.calls.append((skill_name, params, dict(context or {})))
            return {
                "ok": True,
                "summary": "Environment audit completed.",
                "result": {"hostname": "test-host"},
            }

    fake_engine = FakeEngine()
    receipt_store = ReceiptStore(tmp_path / "receipts")
    orchestrator = SimpleNamespace(status=SimpleNamespace(state="running"))
    loop = OvertActionLoop(
        orchestrator=orchestrator,
        capability_engine=fake_engine,
        synthesizer=FakeSynth(),
        receipt_store=receipt_store,
        state_provider=lambda: SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[])),
    )
    loop._record_life_trace = lambda result, raw: setattr(result, "life_trace_id", "life-test-1")

    result = await loop.run_once(force=True)

    assert result["status"] == "verified"
    assert result["skill"] == "environment_info"
    assert result["verified"] is True
    assert result["tool_receipt_id"].startswith("tool_execution-")
    assert result["autonomy_receipt_id"].startswith("autonomy-")
    assert loop.status()["actions_verified"] == 1
    assert fake_engine.calls
    _, _, context = fake_engine.calls[0]
    assert context["origin"] == "overt_action_loop"
    assert context["authorization"] == "governed_autonomous_overt_action"
    assert context["requested_authority_scope"].startswith("overt_action_loop:")
    assert context["requested_authority_scope"].endswith(":environment_info")
    assert context["orchestrator"] is orchestrator
    assert context["action_selection"]["provenance"] == "structured:environment_info"
    assert context["action_expectation"]["required_evidence"] == ["result"]
    assert result["expectation_verdict"]["passed"] is True
    tool_receipt = receipt_store.get(result["tool_receipt_id"])
    assert tool_receipt.verification_evidence["expectation_verdict"]["passed"] is True


@pytest.mark.asyncio
async def test_overt_action_loop_quarantines_retained_memory_prose_instead_of_searching(
    tmp_path,
):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    objective = (
        "Small thing to remember: Bryan's dog is named Biscuit. "
        "[RETAINED MEMORY EVIDENCE] source=durable_memory_search confidence=0.98"
    )

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={"goal": objective, "source": "durable_memory", "urgency": 0.7},
                will_receipt_id="will-memory-evidence",
            )

    class FakeEngine:
        def __init__(self):
            self.calls = []

        async def execute(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True}

    engine = FakeEngine()
    loop = OvertActionLoop(
        capability_engine=engine,
        synthesizer=FakeSynth(),
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        state_provider=lambda: SimpleNamespace(
            cognition=SimpleNamespace(pending_initiatives=[])
        ),
    )

    result = await loop.run_once(force=True)

    assert result["status"] == "skipped"
    assert result["error"] == (
        "initiative_not_actionable:missing_structured_action_contract"
    )
    assert result["selection_provenance"] == "unstructured"
    assert result["next_step_hint"] == "require_structured_action_contract"
    assert result["autonomy_receipt_id"].startswith("autonomy-")
    receipt = loop.receipt_store.get(result["autonomy_receipt_id"])
    assert receipt.metadata["status"] == "skipped"
    assert receipt.metadata["selection_reason"] == "missing_structured_action_contract"
    assert engine.calls == []
    assert loop.status()["actions_started"] == 0


def test_overt_action_selection_requires_explicit_web_intent_or_structured_skill(tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    loop = OvertActionLoop(receipt_store=ReceiptStore(tmp_path / "receipts"))

    natural = loop._choose_skill_and_params(
        {"goal": "Search the web for current macOS release notes"},
        {},
    )
    structured = loop._choose_skill_and_params(
        {
            "goal": "Recall the durable memory search result about Biscuit",
            "metadata": {
                "required_skills": ["web_search"],
                "params": {"query": "current veterinary guidance"},
            },
        },
        {},
    )
    incidental = loop._choose_skill_and_params(
        {"goal": "Review source=durable_memory_search evidence about Biscuit"},
        {},
    )

    assert natural.actionable is True
    assert natural.skill == "web_search"
    assert natural.params["query"] == "current macOS release notes"
    assert natural.provenance == "natural_language:explicit_web_search"
    assert structured.actionable is True
    assert structured.skill == "web_search"
    assert structured.params["query"] == "current veterinary guidance"
    assert structured.provenance == "structured:web_search"
    assert incidental.actionable is False
    assert incidental.reason == "missing_structured_action_contract"


@pytest.mark.asyncio
async def test_initiative_synthesis_preserves_structured_pending_action_contract(
    monkeypatch,
):
    from core.initiative_synthesis import InitiativeSynthesizer

    synth = InitiativeSynthesizer()
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            pending_initiatives=[
                {
                    "goal": "Check current release status",
                    "source": "operator_goal",
                    "required_skills": ["web_search"],
                    "params": {"query": "current Aura release status"},
                }
            ]
        )
    )
    monkeypatch.setattr(
        "core.initiative_synthesis.ServiceContainer.get",
        lambda _name, default=None: default,
    )

    await synth._gather_system_impulses(state)

    matching = [item for item in synth._impulse_queue if item.source == "operator_goal"]
    assert len(matching) == 1
    assert matching[0].metadata["required_skills"] == ["web_search"]
    assert matching[0].metadata["params"] == {
        "query": "current Aura release status"
    }


def test_initiative_synthesis_quarantines_retained_memory_evidence():
    from core.initiative_synthesis import InitiativeSynthesizer

    synth = InitiativeSynthesizer()

    accepted = synth.submit(
        "Remember Biscuit [RETAINED MEMORY EVIDENCE] "
        "source=durable_memory_search confidence=0.98",
        "conversation_memory",
    )

    assert accepted is False
    assert synth._impulse_queue == []


@pytest.mark.asyncio
async def test_overt_action_loop_rejects_shallow_success_without_expected_evidence(tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": "Run a light environment self-audit",
                    "source": "test_goal",
                    "urgency": 0.6,
                    "metadata": {"required_skills": ["environment_info"]},
                },
                will_receipt_id="will-test-shallow",
            )

    class ShallowEngine:
        async def execute(self, skill_name, params, context=None):
            return {"ok": True, "summary": "Environment audit fired."}

    receipt_store = ReceiptStore(tmp_path / "receipts")
    loop = OvertActionLoop(
        capability_engine=ShallowEngine(),
        synthesizer=FakeSynth(),
        receipt_store=receipt_store,
        state_provider=lambda: SimpleNamespace(
            cognition=SimpleNamespace(pending_initiatives=[])
        ),
    )
    loop._record_life_trace = lambda result, raw: setattr(
        result,
        "life_trace_id",
        "life-test-shallow",
    )

    result = await loop.run_once(force=True)

    assert result["verified"] is False
    assert result["status"] == "success_unverified"
    assert result["expectation_verdict"]["missing_evidence"] == ["result"]
    assert result["next_step_hint"] == "repeat_overt_environment_info_with_evidence"
    assert loop.status()["actions_verified"] == 0
    tool_receipt = receipt_store.get(result["tool_receipt_id"])
    assert tool_receipt.status == "success_unverified"
    assert tool_receipt.verification_evidence["expectation_verdict"]["passed"] is False


@pytest.mark.asyncio
async def test_overt_action_loop_applies_expectations_to_actuator_path(
    tmp_path,
    monkeypatch,
):
    from core.actuators.actuator_registry import ActuatorResult
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    class FakeActuator:
        def validate_params(self, params):
            return True

    class FakeRegistry:
        def get_actuator(self, name):
            return FakeActuator() if name == "code_execution" else None

        async def execute_action_async(self, name, params, *, context=None):
            return ActuatorResult(
                success=True,
                message="Sandbox launch returned success without output evidence.",
                updates={},
            )

    registry = FakeRegistry()
    monkeypatch.setattr(
        "core.actuators.actuator_registry.get_actuator_registry",
        lambda: registry,
    )

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": "Execute a bounded diagnostic calculation",
                    "source": "test_goal",
                    "urgency": 0.6,
                    "metadata": {
                        "required_skills": ["code_execution"],
                        "params": {"code": "print(2 + 2)"},
                    },
                },
                will_receipt_id="will-test-actuator",
            )

    receipt_store = ReceiptStore(tmp_path / "receipts")
    loop = OvertActionLoop(
        capability_engine=SimpleNamespace(),
        synthesizer=FakeSynth(),
        receipt_store=receipt_store,
        state_provider=lambda: SimpleNamespace(
            cognition=SimpleNamespace(pending_initiatives=[])
        ),
    )
    loop._record_life_trace = lambda result, raw: setattr(
        result,
        "life_trace_id",
        "life-test-actuator",
    )

    result = await loop.run_once(force=True)

    assert result["skill"] == "code_execution"
    assert result["verified"] is False
    assert result["status"] == "success_unverified"
    assert result["expectation_verdict"]["missing_evidence"] == ["updates"]
    assert result["next_step_hint"] == (
        "verify_overt_code_execution_applied_updates"
    )
    receipt = receipt_store.get(result["tool_receipt_id"])
    assert receipt.status == "success_unverified"


@pytest.mark.asyncio
async def test_overt_action_loop_waits_without_intrinsic_initiative(monkeypatch, tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    class EmptySynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(winner=None, will_receipt_id="")

    monkeypatch.delenv("AURA_OVERT_ACTION_FALLBACK", raising=False)
    loop = OvertActionLoop(
        capability_engine=SimpleNamespace(),
        synthesizer=EmptySynth(),
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        state_provider=lambda: SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[])),
    )

    result = await loop.run_once(force=True)

    assert result["status"] == "skipped"
    assert result["error"] == "no_authorized_initiative"
    assert loop.status()["actions_started"] == 0


@pytest.mark.asyncio
async def test_overt_action_loop_failed_action_records_retry_hint(monkeypatch, tmp_path):
    from core.container import ServiceContainer
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore
    from core.self_model import SelfModel

    monkeypatch.setattr("core.self_model.DATA_FILE", tmp_path / "self_model.json")

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": "Repair the codebase without disrupting the user",
                    "source": "test_goal",
                    "urgency": 0.6,
                    "metadata": {"required_skills": ["auto_refactor"]},
                },
                will_receipt_id="will-test-2",
            )

    class FakeEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": False,
                "status": "blocked_by_user_advocate",
                "error": "User advocate blocked: missing benefit",
            }

    ServiceContainer.clear()
    self_model = SelfModel(id="overt-action-self-model")
    ServiceContainer.register_instance("self_model", self_model)
    loop = OvertActionLoop(
        capability_engine=FakeEngine(),
        synthesizer=FakeSynth(),
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        state_provider=lambda: SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[])),
    )
    loop._record_life_trace = lambda result, raw: setattr(result, "life_trace_id", "life-test-2")

    result = await loop.run_once(force=True)

    assert result["status"] == "failed"
    assert result["next_step_hint"] == "retry_with_explicit_user_benefit_and_non_mutating_scope"
    assert result["autonomy_receipt_id"].startswith("autonomy-")
    lessons = self_model.beliefs.get("runtime_lessons") or []
    assert lessons
    assert lessons[-1]["source"] == "overt_action_loop"
    assert "retry_with_explicit_user_benefit" in lessons[-1]["lesson"]
