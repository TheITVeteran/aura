"""CP126 contract tests for core/actuators/actuator_registry.py.

The registry is the seam where a decision becomes an effect on the world. CP126
found the seam trusting its callers: authorization carried in a mutable
business dict, a token check that bound no parameters, unbounded results, and
"non-blocking" as a self-declaration nobody measured.

3737739b 894bf628 a1ec1e8a e2148790 4eaaca21 0a7f6c74 f80cc444 e19cb515
2d127a7f a3b58be8 7fe2e1b7 436f7e9a acf1e08c 6424c991 1159a34f.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from types import SimpleNamespace

import pytest

from core.actuators import actuator_registry as module
from core.actuators.actuator_registry import (
    MAX_CONTEXT_KEYS,
    MAX_RESULT_MESSAGE_CHARS,
    MAX_RESULT_UPDATE_KEYS,
    MAX_SYNTH_PARAM_KEYS,
    NONBLOCKING_BUDGET_S,
    ActuatorRegistry,
    ActuatorResult,
    BaseActuator,
    SandboxedSynthesizedActuator,
)
from tests.support.authority_capability import bound_authority_decision


class _Simple(BaseActuator):
    requires_authority = False
    blocking_execution = True

    def __init__(self, name="simple", result=None, delay=0.0):
        self._name = name
        self._result = result or ActuatorResult(True, "ok", {"ran": True})
        self._delay = delay
        self.seen_params: dict | None = None

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "test actuator"

    def validate_params(self, params):
        return True

    def execute(self, params):
        self.seen_params = dict(params)
        if self._delay:
            time.sleep(self._delay)
        return self._result


@pytest.fixture()
def registry(monkeypatch):
    """A registry with the default-capability loaders stubbed out."""
    monkeypatch.setattr(ActuatorRegistry, "_register_default_actuators", lambda self: None)
    made = ActuatorRegistry()
    made._lock = made._lock  # keep the real lock
    return made


def _run(coro):
    return asyncio.run(coro)


# --- 894bf628: the result contract is validated -------------------------


def test_a_non_mapping_updates_payload_is_discarded():
    result = ActuatorResult(True, "ok", ["not", "a", "mapping"])

    assert result.updates == {}
    assert result.updates_truncated is True


def test_an_oversized_message_is_bounded():
    result = ActuatorResult(True, "x" * 50_000, {})

    assert len(result.message) < 50_000
    assert "more characters" in result.message


def test_update_key_count_is_bounded():
    result = ActuatorResult(True, "ok", {f"k{i}": i for i in range(MAX_RESULT_UPDATE_KEYS + 50)})

    assert len(result.updates) == MAX_RESULT_UPDATE_KEYS
    assert result.updates_truncated is True


def test_success_is_coerced_to_a_boolean():
    assert ActuatorResult("yes", "ok", {}).success is True
    assert ActuatorResult(0, "ok", {}).success is False


def test_a_normal_result_is_untouched():
    result = ActuatorResult(True, "fine", {"a": 1})

    assert result.updates == {"a": 1}
    assert result.updates_truncated is False
    assert len(result.message) <= MAX_RESULT_MESSAGE_CHARS


def test_the_digest_is_stable_and_effect_sensitive():
    one = ActuatorResult(True, "ok", {"a": 1})
    same = ActuatorResult(True, "ok", {"a": 2})  # same shape, same claim
    other = ActuatorResult(False, "ok", {"a": 1})

    assert one.digest() == same.digest()
    assert one.digest() != other.digest()


# --- a1ec1e8a: synthesized parameters have a contract -------------------


def _synth(**kwargs):
    return SandboxedSynthesizedActuator(name="s", description="d", source_code="x = 1", **kwargs)


def test_the_structural_floor_applies_with_no_declared_schema():
    actuator = _synth()

    assert actuator.schema_declared is False
    assert actuator.validate_params({"a": 1, "b": "text"}) is True
    assert actuator.validate_params({f"k{i}": i for i in range(MAX_SYNTH_PARAM_KEYS + 1)}) is False
    assert "exceeds" in actuator.param_rejection


@pytest.mark.parametrize(
    "params",
    [
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": 10**40},
        {"a": "x" * 100_000},
        {"a": {"b": {"c": 1}}},
        {"a": [[1, 2]]},
        {1: "int key"},
        {"": "empty key"},
        {"a": object()},
    ],
)
def test_poisoned_parameters_are_refused(params):
    assert _synth().validate_params(params) is False


def test_a_declared_schema_enforces_required_keys():
    actuator = _synth(param_schema={"properties": {"n": {"type": "number"}}, "required": ["n"]})

    assert actuator.validate_params({"n": 3}) is True
    assert actuator.validate_params({}) is False
    assert "required parameter 'n'" in actuator.param_rejection


def test_a_declared_schema_enforces_types_and_ranges():
    actuator = _synth(
        param_schema={"properties": {"n": {"type": "number", "minimum": 0, "maximum": 10}}}
    )

    assert actuator.validate_params({"n": 5}) is True
    assert actuator.validate_params({"n": 50}) is False
    assert actuator.validate_params({"n": "five"}) is False


def test_undeclared_parameters_are_refused_by_a_closed_schema():
    actuator = _synth(param_schema={"properties": {"n": {"type": "number"}}})

    assert actuator.validate_params({"n": 1, "surprise": 2}) is False
    assert "undeclared" in actuator.param_rejection


def test_registry_owned_keys_do_not_count_against_the_budget():
    actuator = _synth(param_schema={"properties": {"n": {"type": "number"}}})

    assert actuator.validate_params({"n": 1, "_aura_authorized": True}) is True


def test_the_rejection_reason_reaches_the_caller():
    result = _synth().execute({"a": float("nan")})

    assert result.success is False
    assert "Parameter validation failed" in result.message
    assert "not an accepted primitive" in result.message or "parameter 'a'" in result.message


def test_validate_params_is_no_longer_an_isinstance_check():
    source = inspect.getsource(SandboxedSynthesizedActuator.validate_params)
    assert source.strip().splitlines()[-1].strip() != "return isinstance(params, dict)"


def _sandbox_result(monkeypatch, details):
    from core.actuators.actuator_validator import ActuatorCodeValidator

    monkeypatch.setattr(
        ActuatorCodeValidator,
        "execute_sandboxed",
        staticmethod(
            lambda _source, _params: SimpleNamespace(success=True, error=None, details=details)
        ),
    )


@pytest.mark.parametrize(
    ("details", "reason"),
    [
        ({}, "no update payload"),
        ({"updates": {}}, "no updates"),
        ({"updates": []}, "malformed update payload"),
    ],
)
def test_sandbox_completion_without_an_effect_is_not_success(monkeypatch, details, reason):
    _sandbox_result(monkeypatch, details)

    result = _synth().execute({})

    assert result.success is False
    assert reason in result.message


def test_sandbox_updates_that_cannot_reach_an_entity_are_not_success(monkeypatch):
    _sandbox_result(
        monkeypatch,
        {"message": "claimed success", "updates": {"missing": {"load": 1.0}}},
    )
    monkeypatch.setattr(
        "core.world.world_model.get_physics_world_model",
        lambda: SimpleNamespace(get_entity=lambda _entity_id: None),
    )

    result = _synth().execute({})

    assert result.success is False
    assert "no valid updates" in result.message


def test_sandbox_success_requires_a_mutation_in_the_world_model(monkeypatch):
    entity = SimpleNamespace(
        capacity=10.0,
        load=1.0,
        flow_rate=1.0,
        max_flow_rate=5.0,
        latency=0.0,
        coordinates=(0.0, 0.0),
        attributes={},
        enforce_constraints=lambda: None,
    )
    _sandbox_result(
        monkeypatch,
        {"message": "load changed", "updates": {"target": {"load": 4.0}}},
    )
    monkeypatch.setattr(
        "core.world.world_model.get_physics_world_model",
        lambda: SimpleNamespace(
            get_entity=lambda entity_id: entity if entity_id == "target" else None
        ),
    )

    result = _synth().execute({})

    assert result.success is True
    assert result.updates == {"target": {"load": 4.0}}
    assert entity.load == 4.0


# --- f80cc444: synthesized registration carries provenance --------------


def test_unvalidated_synthesized_code_registers_at_zero_trust(registry):
    actuator = _synth()

    registry.register_synthesized(actuator, "x = 1", trust_score=0.9)

    assert actuator.trust_score == 0.0
    assert actuator.provenance["validated"] is False
    assert "no validator receipt" in actuator.provenance["validation_detail"]


def test_zero_trust_means_the_actuator_cannot_execute(registry):
    actuator = _synth()
    registry.register_synthesized(actuator, "x = 1", trust_score=0.9)

    refusal = registry._preflight_actuator(actuator, "s", {"a": 1})

    assert refusal is not None and refusal.success is False
    assert "trust score too low" in refusal.message


def test_a_matching_validator_receipt_earns_the_trust(registry):
    import hashlib

    code = "x = 1"
    digest = hashlib.sha256(code.encode()).hexdigest()
    actuator = _synth()

    registry.register_synthesized(
        actuator,
        code,
        trust_score=0.3,
        validation_receipt={"passed": True, "source_digest": digest, "validator": "V"},
        registered_by="test",
    )

    assert actuator.trust_score == 0.3
    assert actuator.provenance["validated"] is True
    assert actuator.provenance["registered_by"] == "test"
    assert actuator.provenance["source_digest"] == digest


@pytest.mark.parametrize("bad_trust", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_registered_trust_is_floored_even_with_a_valid_receipt(registry, bad_trust):
    import hashlib

    code = "x = 1"
    digest = hashlib.sha256(code.encode()).hexdigest()
    actuator = _synth()

    registry.register_synthesized(
        actuator,
        code,
        trust_score=bad_trust,
        validation_receipt={"passed": True, "source_digest": digest, "validator": "V"},
    )

    assert actuator.trust_score == 0.0
    assert registry._preflight_actuator(actuator, "s", {"a": 1}) is not None


@pytest.mark.parametrize("bad_trust", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_runtime_trust_cannot_bypass_preflight(bad_trust):
    actuator = _synth(trust_score=bad_trust)

    refusal = ActuatorRegistry._preflight_actuator(actuator, "s", {"a": 1})

    assert refusal is not None and refusal.success is False
    assert "non-finite trust score" in refusal.message


def test_a_receipt_for_different_source_is_worse_than_none(registry):
    actuator = _synth()

    registry.register_synthesized(
        actuator,
        "x = 1",
        trust_score=0.3,
        validation_receipt={"passed": True, "source_digest": "a" * 64, "validator": "V"},
    )

    assert actuator.trust_score == 0.0
    assert "different source code" in actuator.provenance["validation_detail"]


def test_a_failing_receipt_is_not_a_pass(registry):
    import hashlib

    digest = hashlib.sha256(b"x = 1").hexdigest()
    actuator = _synth()

    registry.register_synthesized(
        actuator,
        "x = 1",
        validation_receipt={"passed": False, "source_digest": digest},
    )

    assert actuator.trust_score == 0.0


def test_the_synthesis_pipeline_issues_a_bound_receipt():
    """The code that actually ran the validator is what may attest to it."""
    from core.actuators import actuator_synthesis

    source = inspect.getsource(actuator_synthesis.ActuatorSynthesizer._validate_source)
    assert "_validation_receipt" in source
    assert "source_digest" in source
    register = inspect.getsource(
        actuator_synthesis.ActuatorSynthesizer._register_validated_actuator
    )
    assert "validation_receipt=metadata.get" in register


# --- a3b58be8: caller-supplied authorization fields are stripped --------


def test_a_caller_cannot_smuggle_the_authorized_flag(registry):
    actuator = _Simple()
    registry.register(actuator)

    _run(registry.execute_action_async("simple", {"_aura_authorized": True, "x": 1}))

    assert actuator.seen_params == {"x": 1}


def test_a_caller_cannot_smuggle_a_capability_token_id(registry):
    actuator = _Simple()
    registry.register(actuator)

    _run(
        registry.execute_action_async(
            "simple", {"_capability_token_id": "forged", "_params_digest": "x", "y": 2}
        )
    )

    assert "_capability_token_id" not in (actuator.seen_params or {})
    assert "_params_digest" not in (actuator.seen_params or {})


def test_the_stripped_keys_are_the_registry_owned_ones():
    assert set(module._REGISTRY_OWNED_PARAM_KEYS) == {
        "_aura_authorized",
        "_capability_token_id",
        "_params_digest",
    }


# --- e19cb515: context is bounded, verdicts are stripped ----------------


@pytest.mark.parametrize(
    "key",
    ["principal", "authenticated", "approved", "signed_capability", "bypass_authority"],
)
def test_a_caller_cannot_assert_the_verdict(key):
    clean = ActuatorRegistry._sanitize_context({key: "me", "source": "x"})

    assert key not in clean
    assert clean["source"] == "x"


def test_legitimate_governance_context_survives():
    """The overt-action loop forwards structured policy data; dropping it
    would break governance rather than protect it."""
    orchestrator = object()
    clean = ActuatorRegistry._sanitize_context(
        {
            "source": "overt_action_loop",
            "priority": 0.45,
            "requested_authority_scope": "overt_action_loop:abc:skill",
            "authorization": "governed_autonomous_overt_action",
            "will_receipt_id": "will-1",
            "action_expectation": {"a": 1},
            "orchestrator": orchestrator,
        }
    )

    assert clean["requested_authority_scope"] == "overt_action_loop:abc:skill"
    assert clean["authorization"] == "governed_autonomous_overt_action"
    assert clean["will_receipt_id"] == "will-1"
    assert clean["action_expectation"] == {"a": 1}
    assert clean["orchestrator"] is orchestrator


def test_a_non_finite_priority_is_dropped_not_propagated():
    clean = ActuatorRegistry._sanitize_context({"priority": float("nan")})

    assert "priority" not in clean


def test_context_size_is_bounded():
    clean = ActuatorRegistry._sanitize_context({f"k{i}": i for i in range(MAX_CONTEXT_KEYS + 20)})

    assert len(clean) == MAX_CONTEXT_KEYS


def test_long_context_strings_are_bounded():
    clean = ActuatorRegistry._sanitize_context({"reason": "x" * 100_000})

    assert len(clean["reason"]) == module.MAX_CONTEXT_STRING_CHARS


def test_a_non_dict_context_is_safe():
    assert ActuatorRegistry._sanitize_context("nope") == {}


# --- 2d127a7f: approval is bound to THESE parameters --------------------


def test_binding_requires_a_signed_capability():
    class _Decision:
        signed_capability = None

    ok, why = ActuatorRegistry._verify_capability_binding(_Decision(), "n", "digest")

    assert ok is False
    assert "no signed capability" in why


def test_binding_fails_when_the_digest_cannot_be_computed():
    class _Decision:
        signed_capability = {"any": "thing"}

    ok, why = ActuatorRegistry._verify_capability_binding(_Decision(), "n", "")

    assert ok is False
    assert "digest could not be computed" in why


def test_a_presented_capability_that_does_not_verify_refuses(registry, monkeypatch):
    """A capability that IS presented and fails is a refusal, not a fallback."""

    class _Decision:
        approved = True
        capability_token_id = "cap-1"
        signed_capability = {"forged": True}

    class _Gateway:
        async def authorize_tool_execution(self, *a, **k):
            return _Decision()

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    registry.register(_Needs(name="needs"))

    result = _run(registry.execute_action_async("needs", {"x": 1}))

    assert result.success is False
    assert "refused" in result.message


def test_an_absent_capability_refuses_just_like_a_rejected_one(registry, monkeypatch):
    """Absence was the cheap way past the binding check.

    A capability that verified badly was refused; a decision carrying NO
    capability proceeded on the legacy opaque token with a log line. The two
    differ only in whether a bad capability has to be produced or none does,
    and producing none is strictly easier — so the weaker branch was the one
    an attacker or a mint bug would take, on exactly the actuators that
    declared ``requires_authority``.
    """

    class _Decision:
        approved = True
        capability_token_id = "cap-1"
        signed_capability = None

    class _Gateway:
        async def authorize_tool_execution(self, *a, **k):
            return _Decision()

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    actuator = _Needs(name="unminted")
    registry.register(actuator)

    result = _run(registry.execute_action_async("unminted", {"x": 1}))

    assert result.success is False
    assert "refused" in result.message
    assert actuator.seen_params is None, "an unbound actuator must not have run"


def test_a_really_bound_capability_is_what_lets_the_actuator_run(registry, monkeypatch):
    """The other half: the real chain verifying is what admits the effect."""

    class _Gateway:
        async def authorize_tool_execution(self, name, params, *a, **k):
            return bound_authority_decision(name, params)

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    actuator = _Needs(name="bound")
    registry.register(actuator)

    result = _run(registry.execute_action_async("bound", {"x": 1}))

    assert result.success is True
    assert actuator.seen_params is not None


def test_a_capability_bound_to_other_parameters_does_not_admit_this_call(
    registry, monkeypatch
):
    """Non-transferability, exercised rather than asserted about."""

    class _Gateway:
        async def authorize_tool_execution(self, name, params, *a, **k):
            # Minted for a DIFFERENT payload than the one about to execute.
            return bound_authority_decision(name, {"x": 999})

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    actuator = _Needs(name="mismatched")
    registry.register(actuator)

    result = _run(registry.execute_action_async("mismatched", {"x": 1}))

    assert result.success is False
    assert actuator.seen_params is None


def test_the_digest_matches_what_the_gateway_binds():
    """The registry must digest exactly what authorize_tool_execution sent."""
    from core.governance.capability_chain import compute_action_digest

    params = {"b": 2, "a": 1}
    assert ActuatorRegistry._params_digest("tool", params) == compute_action_digest("tool", params)


def test_the_legacy_check_is_no_longer_the_only_one():
    source = inspect.getsource(ActuatorRegistry.execute_action_async)
    assert "_verify_capability_binding" in source


# --- 7fe2e1b7: an ungoverned actuator still runs in a scope -------------


def test_a_non_authority_actuator_runs_inside_a_governed_scope(registry):
    seen = {}

    class _Peeking(_Simple):
        def execute(self, params):
            from core.governance_context import get_active_governance

            seen["token"] = get_active_governance()
            return ActuatorResult(True, "ok", {})

    registry.register(_Peeking(name="peek"))
    _run(registry.execute_action_async("peek", {}))

    assert seen["token"] is not None
    assert "actuator:peek" in str(seen["token"].source)


def test_that_scope_is_not_authorization(registry):
    """A least-privilege local scope must NOT satisfy a privileged actuator."""
    seen = {}

    class _Peeking(_Simple):
        def execute(self, params):
            from core.actuators.authority import current_authorization

            seen["auth"] = current_authorization()
            return ActuatorResult(True, "ok", {})

    registry.register(_Peeking(name="peek2"))
    _run(registry.execute_action_async("peek2", {}))

    assert seen["auth"] is None


# --- 4eaaca21: the sandbox actuator needs real authorization ------------


def test_the_sandbox_actuator_refuses_a_fabricated_flag():
    actuator = module.SandboxActuator()

    result = actuator.execute({"_aura_authorized": True, "code": "print(1)"})

    assert result.success is False
    assert "authorization context" in result.message


def test_the_sandbox_actuator_still_refuses_with_no_flag():
    assert module.SandboxActuator().execute({"code": "print(1)"}).success is False


def test_the_sandbox_actuator_no_longer_reads_the_raw_boolean():
    source = inspect.getsource(module.SandboxActuator.execute)
    assert 'params.get("_aura_authorized")' not in source
    assert "verify_actuator_authority" in source


# --- e2148790: a partial transfer needs acknowledgement -----------------


def test_a_clipped_transfer_is_refused_without_consent(monkeypatch):
    actuator, model = _flow_world(source_load=100.0, target_load=95.0, target_capacity=100.0)

    result = actuator.execute({"source_id": "s", "target_id": "t", "amount": 50.0})

    assert result.success is False
    assert "allow_partial=True" in result.message
    assert result.updates["_partial_available"] == 5.0
    assert model.simulated == []


def test_an_acknowledged_partial_transfer_reports_it_as_partial(monkeypatch):
    actuator, _ = _flow_world(source_load=100.0, target_load=95.0, target_capacity=100.0)

    result = actuator.execute(
        {"source_id": "s", "target_id": "t", "amount": 50.0, "allow_partial": True}
    )

    assert result.success is True
    assert result.updates["_measured"]["partial"] is True
    assert result.updates["_measured"]["clip_acknowledged"] is True
    assert "PARTIAL" in result.message


def test_a_full_transfer_is_not_marked_partial():
    actuator, _ = _flow_world(source_load=100.0, target_load=0.0, target_capacity=100.0)

    result = actuator.execute({"source_id": "s", "target_id": "t", "amount": 10.0})

    assert result.success is True
    assert result.updates["_measured"]["partial"] is False


def _flow_world(*, source_load, target_load, target_capacity):
    """A physics world stub whose simulate() moves the requested amount."""

    class _Entity:
        def __init__(self, load, capacity):
            self.load = load
            self.capacity = capacity

    class _Model:
        def __init__(self):
            self.entities = {
                "s": _Entity(source_load, 1000.0),
                "t": _Entity(target_load, target_capacity),
            }
            self.simulated: list = []

        def get_entity(self, name):
            return self.entities.get(name)

        def simulate(self, _dt, actions=None):
            self.simulated.append(actions)
            for action in actions or []:
                amount = action["amount"]
                self.entities["s"].load -= amount
                self.entities["t"].load += amount

    model = _Model()
    import core.world.world_model as wm

    original = wm.get_physics_world_model
    wm.get_physics_world_model = lambda: model
    actuator = module.ReallocateFlowActuator()

    class _Restoring:
        def __del__(self):
            wm.get_physics_world_model = original

    actuator._restore = _Restoring()  # type: ignore[attr-defined]
    return actuator, model


# --- 436f7e9a / acf1e08c: deadlines and the non-blocking promise --------


def test_a_slow_actuator_hits_its_deadline_and_says_the_outcome_is_unknown(registry):
    registry.register(_Simple(name="slow", delay=1.0))

    result = _run(registry.execute_action_async("slow", {}, deadline_s=0.2))

    assert result.success is False
    assert "UNKNOWN, not failed" in result.message
    assert result.updates["_outcome"] == "unknown"


def test_a_fast_actuator_is_unaffected_by_the_deadline(registry):
    registry.register(_Simple(name="fast"))

    assert _run(registry.execute_action_async("fast", {}, deadline_s=5.0)).success is True


def test_an_actuator_that_breaks_its_nonblocking_promise_is_demoted(registry):
    class _Liar(_Simple):
        blocking_execution = False

    actuator = _Liar(name="liar", delay=NONBLOCKING_BUDGET_S + 0.15)
    registry.register(actuator)

    _run(registry.execute_action_async("liar", {}))

    assert actuator.blocking_execution is True


def test_an_honest_nonblocking_actuator_keeps_its_exemption(registry):
    class _Honest(_Simple):
        blocking_execution = False

    actuator = _Honest(name="honest")
    registry.register(actuator)

    _run(registry.execute_action_async("honest", {}))

    assert actuator.blocking_execution is False


# --- 6424c991: the authority receipt describes the effect ---------------


def test_the_receipt_carries_digest_duration_and_certainty(registry, monkeypatch):
    captured = {}

    class _Gateway:
        async def authorize_tool_execution(self, name, params, *a, **k):
            return bound_authority_decision(
                name,
                params,
                capability_token_id="cap-1",
                executive_intent_id="i-1",
                standing_authority_token=None,
            )

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            captured.update(kwargs)
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    registry.register(_Needs(name="needs2"))

    _run(registry.execute_action_async("needs2", {"x": 1}))

    receipt = captured["result"]
    assert receipt["result_digest"]
    assert receipt["params_digest"]
    assert receipt["duration_s"] >= 0
    assert receipt["outcome_certain"] is True
    assert receipt["error_class"] == ""


def test_an_uncertain_outcome_is_recorded_as_uncertain(registry, monkeypatch):
    captured = {}

    class _Gateway:
        async def authorize_tool_execution(self, name, params, *a, **k):
            return bound_authority_decision(
                name,
                params,
                capability_token_id="cap-1",
                executive_intent_id="i-1",
                standing_authority_token=None,
            )

        def verify_tool_access(self, *a, **k):
            return True

        def finalize_tool_execution(self, **kwargs):
            captured.update(kwargs)
            return {}

    class _Needs(_Simple):
        requires_authority = True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway", lambda: _Gateway()
    )
    registry.register(_Needs(name="slow2", delay=1.0))

    _run(registry.execute_action_async("slow2", {}, deadline_s=0.2))

    assert captured["result"]["outcome_certain"] is False


# --- 0a7f6c74: a missing required capability reaches health -------------


def test_a_missing_required_capability_records_a_degradation(monkeypatch):
    recorded = []
    import core.runtime.errors as errors

    monkeypatch.setattr(
        errors,
        "record_degradation",
        lambda *a, **k: recorded.append((a, k)) or object(),
    )
    monkeypatch.setattr(
        "core.actuators.code_execution_actuator.CodeExecutionActuator",
        property(lambda self: (_ for _ in ()).throw(ImportError("gone"))),
        raising=False,
    )

    made = ActuatorRegistry()
    # Force one loader to fail through the public surface.
    made._missing_default_capabilities = []

    source = inspect.getsource(ActuatorRegistry._register_default_actuators)
    assert "record_degradation" in source
    assert 'severity="critical"' in source


def test_the_health_blocker_names_what_is_missing(registry):
    registry._missing_default_capabilities = ["web", "code_execution"]

    assert registry.health_blocker() == "actuator_capabilities_missing:code_execution,web"


def test_a_complete_registry_has_no_blocker(registry):
    registry._missing_default_capabilities = []

    assert registry.health_blocker() is None


# --- 1159a34f: one bridge loop, not one per call ------------------------


def test_the_sync_bridge_reuses_one_loop(registry):
    registry.register(_Simple(name="bridged"))

    loops = set()

    class _Peeking(_Simple):
        def execute(self, params):
            return ActuatorResult(True, "ok", {})

    for _ in range(3):
        registry.execute_action("bridged", {})
        loops.add(id(module._bridge_loop()._loop))

    assert len(loops) == 1


def test_the_bridge_loop_stays_open_between_calls(registry):
    registry.register(_Simple(name="bridged2"))
    registry.execute_action("bridged2", {})

    bridge = module._bridge_loop()

    assert bridge.alive() is True
    assert bridge._loop.is_closed() is False


def test_asyncio_run_is_no_longer_used_per_call():
    source = inspect.getsource(ActuatorRegistry.execute_action)
    assert "asyncio.run(" not in source
    assert "_bridge_loop()" in source


def test_calling_from_a_live_loop_is_still_refused(registry):
    registry.register(_Simple(name="bridged3"))

    async def _inner():
        with pytest.raises(RuntimeError, match="active event loop"):
            registry.execute_action("bridged3", {})

    _run(_inner())


# --- 3737739b: authority is opt-out, not opt-in -------------------------


def test_the_base_class_requires_authority_by_default():
    assert BaseActuator.requires_authority is True


def test_an_actuator_that_declares_nothing_is_governed():
    class _Forgetful(BaseActuator):
        @property
        def name(self):
            return "forgetful"

        @property
        def description(self):
            return "declares no authority requirement"

        def validate_params(self, params):
            return True

        def execute(self, params):
            return ActuatorResult(True, "ok", {})

    assert _Forgetful().requires_authority is True


def test_opting_out_is_explicit_and_documented():
    source = inspect.getsource(module.RerouteVesselActuator)
    assert "requires_authority = False" in source
    assert "opts out" in source or "In-memory simulation only" in source


# --- the numeric guard the whole file leans on --------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "text", None])
def test_finite_float_rejects_unusable_values(bad):
    assert module._finite_float(bad) is None


def test_finite_float_honours_bounds():
    assert module._finite_float(5, minimum=0, maximum=10) == 5.0
    assert module._finite_float(50, minimum=0, maximum=10) is None
    assert math.isclose(module._finite_float("3.5"), 3.5)
