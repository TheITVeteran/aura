import json
import pytest

from core.actuators.actuator_validator import ActuatorCodeValidator, ValidationResult
from core.actuators.actuator_synthesis import ActuatorSynthesizer


class RunUntrustedProbe:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, script, **kwargs):
        self.calls.append({"script": script, "kwargs": kwargs})
        return dict(self.response)


class LocalBrainProbe:
    def __init__(self, response):
        self.response = response
        self.generate_calls = []
        self.close_calls = 0

    async def generate(self, prompt, **kwargs):
        self.generate_calls.append({"prompt": prompt, "kwargs": kwargs})
        return dict(self.response)

    async def close(self):
        self.close_calls += 1


# Clean compliant code string for tests
SAFE_CODE = """
class SafeActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "safe_actuator"

    @property
    def description(self) -> str:
        return "A safe test actuator"

    def validate_params(self, params: dict) -> bool:
        return True

    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

# Malicious code strings
BAD_IMPORT_CODE = """
import os
class MaliciousActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "bad"
    @property
    def description(self) -> str:
        return "bad"
    def validate_params(self, params: dict) -> bool:
        return True
    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

BAD_BUILTIN_CODE = """
class MaliciousActuator(BaseActuator):
    @property
    def name(self) -> str:
        eval("print('evil')")
        return "bad"
    @property
    def description(self) -> str:
        return "bad"
    def validate_params(self, params: dict) -> bool:
        return True
    def execute(self, params: dict) -> ActuatorResult:
        return ActuatorResult(True, "Success", {})
"""

MISSING_METHOD_CODE = """
class BadActuator(BaseActuator):
    @property
    def name(self) -> str:
        return "bad"
    # Missing description, validate_params, execute
"""


def test_validate_ast_safe():
    res = ActuatorCodeValidator.validate_ast(SAFE_CODE)
    assert res.success
    assert res.details["class_name"] == "SafeActuator"


def test_validate_ast_forbidden_import():
    res = ActuatorCodeValidator.validate_ast(BAD_IMPORT_CODE)
    assert not res.success
    assert "Forbidden import" in res.error


def test_validate_ast_forbidden_builtin():
    res = ActuatorCodeValidator.validate_ast(BAD_BUILTIN_CODE)
    assert not res.success
    assert "Forbidden function call" in res.error


def test_validate_ast_missing_methods():
    res = ActuatorCodeValidator.validate_ast(MISSING_METHOD_CODE)
    assert not res.success
    assert "missing required methods" in res.error


def test_validate_ast_syntax_error():
    res = ActuatorCodeValidator.validate_ast("class SyntaxErrorActuator: def syntax_err:")
    assert not res.success
    assert "Syntax error" in res.error


def test_validate_sandbox_success(monkeypatch):
    import core.sandbox.runner as sandbox_runner

    run_untrusted = RunUntrustedProbe(
        {
            "status": "completed",
            "stdout": json.dumps(
                {
                    "success": True,
                    "name": "safe_actuator",
                    "description": "desc",
                    "validate_empty": True,
                    "has_test_params": True,
                }
            ),
        }
    )

    monkeypatch.setattr(sandbox_runner, "run_untrusted", run_untrusted)
    res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)

    assert res.success
    assert res.details["name"] == "safe_actuator"
    assert len(run_untrusted.calls) == 1


def test_validate_sandbox_timeout_or_error(monkeypatch):
    import core.sandbox.runner as sandbox_runner

    monkeypatch.setattr(
        sandbox_runner,
        "run_untrusted",
        RunUntrustedProbe({
            "status": "timeout",
            "stderr": "Execution timed out",
            "stdout": "",
        }),
    )

    res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)
    assert not res.success
    assert "Sandbox failed with status: timeout" in res.error


def test_validate_sandbox_json_decode_error(monkeypatch):
    import core.sandbox.runner as sandbox_runner

    monkeypatch.setattr(
        sandbox_runner,
        "run_untrusted",
        RunUntrustedProbe({
            "status": "completed",
            "stdout": "Not JSON Output",
        }),
    )

    res = ActuatorCodeValidator.validate_sandbox(SAFE_CODE)
    assert not res.success
    assert "invalid JSON" in res.error


def test_validate_causal_successful_compilation(monkeypatch):
    import core.sandbox.runner as sandbox_runner

    responses = [
        {
            "status": "completed",
            "stdout": json.dumps(
                {
                    "success": True,
                    "name": "safe_actuator",
                    "description": "desc",
                    "validate_empty": True,
                    "has_test_params": True,
                    "test_params": {},
                }
            ),
        },
        {
            "status": "completed",
            "stdout": json.dumps(
                {
                    "success": True,
                    "message": "Success",
                    "updates": {},
                }
            ),
        },
    ]
    calls = []

    def run_untrusted(script, **kwargs):
        calls.append({"script": script, "kwargs": kwargs})
        return responses.pop(0)

    monkeypatch.setattr(sandbox_runner, "run_untrusted", run_untrusted)
    res = ActuatorCodeValidator.validate_causal(SAFE_CODE)

    assert res.success
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_actuator_synthesizer_synthesis(monkeypatch, tmp_path):
    import core.actuators.actuator_synthesis as synthesis_module
    from core.actuators.actuator_synthesis import SynthesisRequest

    brain = LocalBrainProbe({"response": "```python\n" + SAFE_CODE + "\n```"})
    monkeypatch.setattr(synthesis_module, "LocalBrain", lambda: brain)

    monkeypatch.setattr(
        ActuatorCodeValidator,
        "validate_ast",
        classmethod(lambda cls, source: ValidationResult(True, details={"class_name": "SafeActuator"})),
    )
    monkeypatch.setattr(
        ActuatorCodeValidator,
        "validate_sandbox",
        classmethod(
            lambda cls, source: ValidationResult(
                True,
                details={"name": "safe_actuator", "description": "A safe test actuator"},
            )
        ),
    )
    monkeypatch.setattr(
        ActuatorCodeValidator,
        "validate_causal",
        classmethod(lambda cls, source: ValidationResult(True)),
    )

    synthesizer = ActuatorSynthesizer(output_dir=str(tmp_path))

    async def governance_approve(actuator_name, source_code, request):
        return True

    synthesizer._governance_approve = governance_approve

    req = SynthesisRequest(problem_description="Read temp from sensor")
    result = await synthesizer.request_synthesis(req)

    assert result is not None
    assert result.name == "safe_actuator"
    assert len(brain.generate_calls) == 1
    assert brain.close_calls == 1
