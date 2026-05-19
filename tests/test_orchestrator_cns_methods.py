from types import SimpleNamespace

import pytest

from core.orchestrator_methods import OrchestratorCNSMixin


class _Emitter:
    def __init__(self):
        self.events = []

    def emit(self, event, text, **kwargs):
        self.events.append((event, text, kwargs))


class _CNS:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def process_stimulus(self, _message):
        if self.error is not None:
            raise self.error
        return self.response


class _CNSHost(OrchestratorCNSMixin):
    def __init__(self, cns=None, cognitive_engine=None):
        self.cns = cns
        self.cognitive_engine = cognitive_engine
        self.emitter = _Emitter()
        self.executed = []

    async def _execute_task(self, payload):
        self.executed.append(payload)
        return {"accepted": True}


class _FallbackEngine:
    async def process(self, message, _orchestrator):
        return f"fallback:{message}"


@pytest.mark.asyncio
async def test_cns_executes_skill_plan():
    host = _CNSHost(
        cns=_CNS(
            {
                "execution": {
                    "neuron": SimpleNamespace(id="skill:web_search", name="Web Search"),
                    "synapse": SimpleNamespace(intent_pattern="search"),
                }
            }
        )
    )

    result = await host.process_user_input_cns("search for Aura")

    assert result["ok"] is True
    assert result["status"] == "executed"
    assert result["skill"] == "web_search"
    assert host.executed == [{"skill": "web_search", "params": {"query": "search for Aura"}}]


@pytest.mark.asyncio
async def test_cns_falls_back_when_no_neural_path():
    host = _CNSHost(cns=_CNS({}), cognitive_engine=_FallbackEngine())

    result = await host.process_user_input_cns("hello")

    assert result == {
        "ok": True,
        "status": "fallback",
        "reason": "no_neural_path",
        "result": "fallback:hello",
    }


@pytest.mark.asyncio
async def test_cns_failure_is_structured_and_non_executing():
    host = _CNSHost(cns=_CNS(error=TimeoutError("cns offline")))

    result = await host.process_user_input_cns("run something")

    assert result["ok"] is False
    assert result["error"] == "cns_processing_failed"
    assert host.executed == []
