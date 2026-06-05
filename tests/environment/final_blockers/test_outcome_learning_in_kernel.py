import pytest

from core.environment.environment_kernel import EnvironmentKernel
from core.environment.command import ActionIntent, CommandSpec, CommandStep
from core.environment.parsed_state import ParsedState


class StateCompilerRecorder:
    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def compile(self, observation):
        self.calls.append(observation)
        return self.states.pop(0)


class CommandCompilerRecorder:
    def __init__(self, environment_id):
        self.environment_id = environment_id
        self.calls = []

    def compile(self, intent, *, trace_id="", receipt_id=None):
        self.calls.append({"intent": intent, "trace_id": trace_id, "receipt_id": receipt_id})
        return CommandSpec(
            command_id="cmd_1",
            environment_id=self.environment_id,
            intent=intent,
            preconditions=[],
            steps=[CommandStep(kind="observe", value="execute_intent")],
            expected_effects=[intent.expected_effect],
        )


class ApprovingGovernanceBridge:
    async def decide_action(self, intent):
        return type(
            "Decision",
            (),
            {
                "approved": True,
                "will_receipt_id": "will-test",
                "authority_receipt_id": "auth-test",
                "executive_intent_id": None,
                "capability_token_id": None,
                "reason": "approved",
            },
        )()


@pytest.mark.asyncio
async def test_kernel_observes_parses_before_and_after_execute(fake_adapter):
    fake_adapter.screens = ["before", "after"]
    kernel = EnvironmentKernel(adapter=fake_adapter)
    parsed_before = ParsedState(
        environment_id=kernel.environment_id,
        sequence_id=1,
        self_state={"local_coordinates": (10, 10)},
    )
    parsed_after = ParsedState(
        environment_id=kernel.environment_id,
        sequence_id=2,
        self_state={"local_coordinates": (11, 10)},
    )
    kernel.state_compiler = StateCompilerRecorder([parsed_before, parsed_after])
    kernel.command_compiler = CommandCompilerRecorder(kernel.environment_id)
    kernel.governance_bridge = ApprovingGovernanceBridge()
    
    await kernel.start(run_id="test_run")
    
    intent = ActionIntent(name="move_east", expected_effect="position_changed")
    frame = await kernel.step(intent=intent)
    
    assert len(kernel.state_compiler.calls) == 2
    assert len(kernel.command_compiler.calls) == 1
    assert frame.outcome_assessment is not None
    assert "position_changed" in frame.outcome_assessment.observed_events
