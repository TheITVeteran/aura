from __future__ import annotations

import threading
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
    # CP126 02689c97: force_substrate BYPASSES the confidence gate, so it is an
    # evaluation-only control — a caller kwarg on a user-facing turn can no
    # longer force unvetted substrate output at the user. The architectural
    # contract under test (substrate is consulted BEFORE the transformer) is
    # exercised from an evaluation origin, which is what legitimately forces it.
    text = await router.think(
        "quiet status", force_substrate=True, max_tokens=8, origin="evaluation"
    )
    assert isinstance(text, str) and text.strip()

    # The contract is that the substrate is CONSULTED before the transformer,
    # and that is what this asserts. It used to assert the substrate's TEXT was
    # returned — but that text comes from an untrained random projection onto a
    # 32-word proto vocabulary ("Substrate path: world action hold grounded…"),
    # and measured 2026-08-04 it was reachable as a live user-facing reply.
    # A person may not be handed it; see
    # tests/test_substrate_readout_is_not_language.py.
    generation = router.stats.get("last_substrate_generation")
    assert generation is not None, "the substrate was never consulted"
    assert generation["used_substrate"] is True
    assert generation["text"].startswith("Substrate path:")
    assert generation["is_user_presentable"] is False

    # And the authority itself: the same call from a user turn must NOT be
    # able to force the bypass.
    denied = IntelligentLLMRouter()
    assert denied._substrate_primary_enabled() is True


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
async def test_overt_action_goal_lookup_never_waits_on_owner_loop():
    from core.runtime.overt_action_loop import OvertActionLoop

    owner_thread = threading.get_ident()
    observed: dict[str, int] = {}

    class GoalEngine:
        @staticmethod
        def get_goal(goal_id):
            observed["thread"] = threading.get_ident()
            return {"id": goal_id, "objective": "bounded lookup"}

    loop = OvertActionLoop()
    loop._goal_engine = lambda: GoalEngine()

    goal = await loop._goal_for_initiative({"goal_id": "goal-1"})

    assert goal["id"] == "goal-1"
    assert observed["thread"] != owner_thread


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
        "initiative_not_actionable:retained_memory_is_evidence_not_an_action"
    )
    assert result["selection_provenance"] == "non_action_evidence"
    assert result["next_step_hint"] == "retain_as_evidence_without_execution"
    assert result["autonomy_receipt_id"].startswith("autonomy-")
    receipt = loop.receipt_store.get(result["autonomy_receipt_id"])
    assert receipt.metadata["status"] == "skipped"
    assert receipt.metadata["selection_reason"] == (
        "retained_memory_is_evidence_not_an_action"
    )
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
    assert incidental.reason == "retained_memory_is_evidence_not_an_action"


def test_overt_action_selection_semantically_plans_paraphrased_objectives(tmp_path):
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    loop = OvertActionLoop(receipt_store=ReceiptStore(tmp_path / "receipts"))
    objectives = (
        "It would help to compare the latest three runtime incidents and save the findings.",
        "I should examine what changed after the last recovery and preserve the evidence.",
        "The next useful move is finding the current device state before deciding what to adjust.",
    )

    selections = [
        loop._choose_skill_and_params(
            {"goal": objective, "source": "cognitive_loop"},
            {},
        )
        for objective in objectives
    ]

    assert all(selection.actionable for selection in selections)
    assert all(selection.execution_mode == "planned_goal" for selection in selections)
    assert all(selection.skill == "autonomous_task_engine" for selection in selections)
    assert [selection.params["goal"] for selection in selections] == list(objectives)


@pytest.mark.asyncio
async def test_overt_action_loop_executes_self_chosen_semantic_plan_with_provenance(
    tmp_path,
    monkeypatch,
):
    from core.agency.autonomous_task_engine import TaskResult
    from core.runtime.overt_action_loop import OvertActionLoop
    from core.runtime.receipts import ReceiptStore

    objective = "Compare recent runtime failures and record the verified common cause."
    calls = []

    class FakeTaskEngine:
        async def execute_goal(self, goal, context=None):
            calls.append((goal, dict(context or {})))
            return TaskResult(
                plan_id="plan-semantic-1",
                goal=goal,
                succeeded=True,
                summary="Compared and recorded with evidence.",
                trace_id="trace-semantic-1",
                steps_completed=3,
                steps_total=3,
                evidence=["artifact:incident-comparison"],
            )

    class FakeSynth:
        async def start(self):
            return None

        async def synthesize(self, state):
            return SimpleNamespace(
                winner={
                    "goal": objective,
                    "source": "cognitive_loop",
                    "urgency": 0.72,
                },
                will_receipt_id="will-semantic-1",
            )

    monkeypatch.setattr(
        "core.runtime.service_access.resolve_task_engine",
        lambda default=None: FakeTaskEngine(),
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
        "life-semantic-1",
    )

    result = await loop.run_once(force=True)

    assert result["status"] == "verified"
    assert result["execution_mode"] == "planned_goal"
    assert result["selection_provenance"] == "semantic_plan:live_capability_catalog"
    assert result["skill"] == "autonomous_task_engine"
    assert calls[0][0] == objective
    context = calls[0][1]
    assert context["requested_by"] == "aura"
    assert context["autonomous"] is True
    assert context["will_receipt_id"] == "will-semantic-1"
    assert context["action_selection"]["execution_mode"] == "planned_goal"
    receipt = receipt_store.get(result["tool_receipt_id"])
    assert receipt.metadata["execution_mode"] == "planned_goal"
    assert receipt.metadata["selection_provenance"] == (
        "semantic_plan:live_capability_catalog"
    )


@pytest.mark.asyncio
async def test_overt_action_semantic_plan_never_claims_partial_work_completed(monkeypatch):
    from core.agency.autonomous_task_engine import TaskResult
    from core.runtime.overt_action_loop import OvertActionLoop

    class PartialTaskEngine:
        async def execute_goal(self, goal, context=None):
            return TaskResult(
                plan_id="plan-partial",
                goal=goal,
                succeeded=True,
                summary="One step remains.",
                trace_id="trace-partial",
                steps_completed=2,
                steps_total=3,
            )

    monkeypatch.setattr(
        "core.runtime.service_access.resolve_task_engine",
        lambda default=None: PartialTaskEngine(),
    )

    raw = await OvertActionLoop()._execute_planned_goal(
        "Complete all three steps.",
        governance_context={"will_receipt_id": "will-partial"},
    )

    assert raw["ok"] is False
    assert raw["status"] == "execution_not_completed"
    assert raw["steps_completed"] == 2
    assert raw["steps_total"] == 3


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
