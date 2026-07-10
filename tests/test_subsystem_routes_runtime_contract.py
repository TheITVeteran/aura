import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from interface.routes import subsystems


class PayloadRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def test_skill_execute_payload_normalizer_unwraps_live_desktop_envelope():
    params, context = subsystems._normalize_skill_execute_payload(
        {
            "input": {
                "action": "write",
                "path": "artifacts/live_runtime/generated/probe.txt",
                "content": "ok",
            },
            "context": {"route": "desktop-ui.live_probe"},
            "foreground_request": True,
        }
    )

    assert params == {
        "action": "write",
        "path": "artifacts/live_runtime/generated/probe.txt",
        "content": "ok",
    }
    assert context["route"] == "desktop-ui.live_probe"
    assert context["foreground_request"] is True


def test_skill_execute_payload_normalizer_preserves_authority_context():
    params, context = subsystems._normalize_skill_execute_payload(
        {
            "input": {"target": "Authorized Notes Export Utility"},
            "context": {
                "route": "desktop-ui.program_dna",
                "scoped_authority": "custom_scope",
            },
            "explicit_authorization": "operator_probe",
        }
    )

    assert params == {"target": "Authorized Notes Export Utility"}
    assert context["route"] == "desktop-ui.program_dna"
    assert context["scoped_authority"] == "custom_scope"
    assert context["explicit_authorization"] == "operator_probe"


def test_skill_execute_authority_context_defaults_for_authenticated_api_call():
    context = subsystems._apply_skill_execute_authority_context(
        "program_dna_reconstruct",
        {},
    )

    assert context["origin"] == "live_skill_api"
    assert context["route"] == "api.skill.execute"
    assert context["foreground_request"] is True
    assert context["user_requested_action"] is True
    assert context["user_explicitly_authorized"] is True
    assert context["explicit_authorization"] == "internal_authenticated_skill_execute"
    assert context["scoped_authority"] == "api_skill_execute:api.skill.execute:program_dna_reconstruct"


def test_skill_execute_authority_context_preserves_existing_scope():
    context = subsystems._apply_skill_execute_authority_context(
        "file_operation",
        {"route": "desktop-ui.file", "scoped_authority": "existing_scope"},
    )

    assert context["route"] == "desktop-ui.file"
    assert context["scoped_authority"] == "existing_scope"


def test_skill_execute_payload_normalizer_preserves_direct_params():
    params, context = subsystems._normalize_skill_execute_payload(
        {"action": "exists", "path": "README.md"}
    )

    assert params == {"action": "exists", "path": "README.md"}
    assert context == {}


def test_skill_execute_payload_normalizer_moves_expectation_fields_to_context():
    expectation = {
        "objective": "write and verify the file",
        "required_evidence": ["sha256", "effect_verified"],
        "repair_hint": "retry_verified_write",
        "rollback_hint": "restore_previous_file_version",
        "allow_partial": False,
    }

    params, context = subsystems._normalize_skill_execute_payload(
        {
            "input": {"action": "write", "path": "probe.txt", "content": "ok"},
            "action_expectation": expectation,
        }
    )

    assert params == {"action": "write", "path": "probe.txt", "content": "ok"}
    assert context["action_expectation"] == expectation


@pytest.mark.asyncio
async def test_terminal_send_preserves_client_input_errors():
    with pytest.raises(HTTPException) as caught:
        await subsystems.api_terminal_send(PayloadRequest({"text": "   "}))

    assert caught.value.status_code == 400
    assert caught.value.detail == "text required"


@pytest.mark.asyncio
async def test_skill_execute_returns_structured_failure_for_router_runtime_error(monkeypatch):
    recorded = []

    class Router:
        def __init__(self):
            self.calls = []

        async def route_execution(self, skill_name, params, engine, *, context=None):
            self.calls.append((skill_name, params, engine, context))
            raise RuntimeError(f"{skill_name} route unavailable")

    class Engine:
        pass

    def service_get(name, default=None):
        if name == "intent_router":
            return Router()
        if name == "capability_engine":
            return Engine()
        return default

    monkeypatch.setattr(subsystems.ServiceContainer, "get", staticmethod(service_get))
    monkeypatch.setattr(
        subsystems,
        "record_degradation",
        lambda subsystem, error: recorded.append((subsystem, str(error))),
    )

    response = await subsystems.api_skill_execute("research", {"query": "hello"}, None, None)
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload == {"ok": False, "error": "research route unavailable"}
    assert recorded == [("subsystems", "research route unavailable")]


@pytest.mark.asyncio
async def test_skill_execute_forwards_scoped_authority_to_router(monkeypatch):
    recorded_context = {}

    class Router:
        async def route_execution(self, skill_name, params, engine, *, context=None):
            recorded_context.update(context or {})
            return {"ok": True, "skill": skill_name, "params": params}

    class Engine:
        pass

    def service_get(name, default=None):
        if name == "intent_router":
            return Router()
        if name == "capability_engine":
            return Engine()
        return default

    monkeypatch.setattr(subsystems.ServiceContainer, "get", staticmethod(service_get))

    response = await subsystems.api_skill_execute(
        "program_dna_reconstruct",
        {"target": "Authorized Notes Export Utility"},
        None,
        None,
    )
    payload = json.loads(response.body)

    assert payload["ok"] is True
    assert recorded_context["scoped_authority"] == (
        "api_skill_execute:api.skill.execute:program_dna_reconstruct"
    )
    assert recorded_context["user_explicitly_authorized"] is True


@pytest.mark.asyncio
async def test_skill_execute_derives_and_forwards_runtime_expectation(monkeypatch):
    from core.runtime.skill_contract import ActionExpectation

    recorded_context = {}

    class Router:
        async def route_execution(self, skill_name, params, engine, *, context=None):
            recorded_context.update(context or {})
            return {"ok": True, "skill": skill_name}

    class Engine:
        skills = {}

        @staticmethod
        def action_expectation_for(skill_name, params, context):
            return ActionExpectation(
                objective="write and verify probe.txt",
                required_evidence=["sha256", "effect_verified"],
                repair_hint="retry_verified_write",
                rollback_hint="restore_previous_file_version",
                allow_partial=False,
            )

    def service_get(name, default=None):
        if name == "intent_router":
            return Router()
        if name == "capability_engine":
            return Engine()
        return default

    monkeypatch.setattr(subsystems.ServiceContainer, "get", staticmethod(service_get))

    response = await subsystems.api_skill_execute(
        "file_operation",
        {"action": "write", "path": "probe.txt", "content": "ok"},
        None,
        None,
    )

    assert response.status_code == 200
    assert recorded_context["action_expectation"]["required_evidence"] == [
        "sha256",
        "effect_verified",
    ]
    assert recorded_context["action_expectation"]["rollback_hint"] == (
        "restore_previous_file_version"
    )


@pytest.mark.asyncio
async def test_skill_execute_blocks_consequential_operation_without_expectation(
    monkeypatch,
):
    router_calls = []

    class Router:
        async def route_execution(self, *args, **kwargs):
            router_calls.append((args, kwargs))
            return {"ok": True}

    class Engine:
        skills = {"personality": SimpleNamespace()}

        @staticmethod
        def resolve_skill_name(skill_name):
            return skill_name

        @staticmethod
        def action_expectation_for(skill_name, params, context):
            return None

        @staticmethod
        def _effect_scope_for_execution(skill_name, meta, params, context):
            return "state_mutation"

    def service_get(name, default=None):
        if name == "intent_router":
            return Router()
        if name == "capability_engine":
            return Engine()
        return default

    monkeypatch.setattr(subsystems.ServiceContainer, "get", staticmethod(service_get))

    response = await subsystems.api_skill_execute(
        "personality",
        {"action": "update", "trait": "precision"},
        None,
        None,
    )
    payload = json.loads(response.body)

    assert response.status_code == 422
    assert payload["status"] == "action_expectation_required"
    assert payload["effect_scope"] == "state_mutation"
    assert payload["required_contract"]["rollback_hint"]
    assert router_calls == []
