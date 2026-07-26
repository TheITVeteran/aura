"""CP126 hardening contracts for core/brain/llm/function_calling_adapter.py.

This adapter is the mind→body bridge, so: arguments are validated BEFORE
dispatch, execution is deadline-bounded, results are redacted and size-bounded,
tool definitions are structurally validated, the ungoverned direct path is
opt-in, and exception text is summarized rather than echoed to the model.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import core.brain.llm.function_calling_adapter as fca
from core.brain.llm.function_calling_adapter import (
    FunctionCallingAdapter,
    _redact,
    _serialize_result,
)
from core.capability_engine import CapabilityEngine


class DemoInput(BaseModel):
    action: str


def _engine_adapter(execute=None, *, input_model=DemoInput):
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {"demo_skill": SimpleNamespace(input_model=input_model)}
    if execute is not None:
        engine.execute = execute
    return FunctionCallingAdapter(engine)


# ── 820e9983: execute_tool validates before dispatching ───────────────────


@pytest.mark.asyncio
async def test_invalid_args_never_reach_the_tool():
    called = []

    async def _exec(name, args, ctx):
        called.append(args)
        return {"ok": True}

    adapter = _engine_adapter(_exec)
    out = await adapter.execute_tool("demo_skill", {})  # missing required 'action'
    assert out.startswith("Error:")
    assert called == []  # the tool was never invoked


@pytest.mark.asyncio
async def test_valid_args_are_the_validated_ones():
    seen = {}

    async def _exec(name, args, ctx):
        seen.update(args)
        return {"ok": True}

    adapter = _engine_adapter(_exec)
    out = await adapter.execute_tool("demo_skill", {"action": "open"})
    assert seen == {"action": "open"}
    assert json.loads(out)["ok"] is True


# ── b4e37de4: structural floor even without a schema model ────────────────


def test_non_mapping_args_refused():
    adapter = _engine_adapter(input_model=None)
    v = adapter.validate_tool_args("demo_skill", ["not", "a", "dict"])
    assert v["valid"] is False and "Validation Error" in v["error"]


def test_schema_less_skill_is_flagged_unenforced():
    adapter = _engine_adapter(input_model=None)
    v = adapter.validate_tool_args("demo_skill", {"anything": 1})
    assert v["valid"] is True and v["schema_enforced"] is False


# ── c7971b5c: validation errors are summarized, not echoed verbatim ───────


def test_validation_error_keeps_contract_but_bounds_detail():
    adapter = _engine_adapter()
    v = adapter.validate_tool_args("demo_skill", {})
    assert v["valid"] is False
    assert "Validation Error" in v["error"]           # existing contract preserved
    assert len(v["error"]) < fca._MAX_ERROR_CHARS + 120


# ── 297f2287: execution is deadline-bounded ───────────────────────────────


@pytest.mark.asyncio
async def test_hung_tool_hits_the_deadline():
    async def _hang(name, args, ctx):
        await asyncio.sleep(30)

    adapter = _engine_adapter(_hang)
    out = await adapter.execute_tool("demo_skill", {"action": "x"}, deadline_s=0.05)
    assert "deadline" in out and "uncertain" in out


# ── 31f805b3: broadened error contract ────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [OSError("disk"), TypeError("bad"), ValueError("nope"), KeyError("k")])
async def test_common_failures_stay_in_the_string_contract(exc):
    async def _boom(name, args, ctx):
        raise exc

    adapter = _engine_adapter(_boom)
    out = await adapter.execute_tool("demo_skill", {"action": "x"})
    assert isinstance(out, str) and out.startswith("Error:")


@pytest.mark.asyncio
async def test_cancellation_still_propagates():
    async def _cancel(name, args, ctx):
        raise asyncio.CancelledError()

    adapter = _engine_adapter(_cancel)
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute_tool("demo_skill", {"action": "x"})


# ── 1f879409: serialization failure after execution is reported, not raised ─


@pytest.mark.asyncio
async def test_non_serializable_result_does_not_raise():
    class Weird:
        pass

    async def _weird(name, args, ctx):
        return {"obj": Weird()}

    adapter = _engine_adapter(_weird)
    out = await adapter.execute_tool("demo_skill", {"action": "x"})
    assert isinstance(out, str)  # default=str handles it; never raises


def test_serialize_result_handles_unserializable_container():
    class Boom:
        def __repr__(self):
            return "<boom>"

    text = _serialize_result({"k": Boom()})
    assert isinstance(text, str) and "boom" in text.lower()


# ── d7eb693e: results are redacted and size-bounded ───────────────────────


def test_result_secrets_are_redacted():
    out = _serialize_result({"api_key": "sk-live-123", "url": "https://u:p@h/x", "ok": True})
    assert "sk-live-123" not in out
    assert "u:p@h" not in out


def test_result_is_size_bounded():
    out = _serialize_result({"blob": "A" * 200000})
    assert len(out) <= fca._MAX_RESULT_CHARS + 200


def test_redact_handles_bytes():
    assert _redact({"data": b"12345"})["data"] == "<5 bytes>"


# ── de629d00: tool definitions are structurally validated ─────────────────


def test_malformed_and_duplicate_tool_definitions_are_refused():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {}
    engine.get_tool_definitions = lambda: [
        "not-a-dict",
        {"no_function_key": True},
        {"function": {"name": "good", "description": "d"}},
        {"function": {"name": "good", "description": "DUPLICATE"}},
        {"function": {"name": ""}},
    ]
    defs = FunctionCallingAdapter(engine).get_tool_definitions()
    assert list(defs) == ["good"]
    assert defs["good"]["description"] == "d"  # first wins; duplicate refused


# ── c7c582cf: ungoverned direct execution is opt-in ───────────────────────


@pytest.mark.asyncio
async def test_direct_execution_refused_without_opt_in(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNGOVERNED_TOOL_EXECUTION", raising=False)
    called = []

    class _LegacyRegistry:
        skills = {}

        def load_skill(self, name):
            called.append(name)
            return SimpleNamespace(execute=lambda a, c: None)

    adapter = FunctionCallingAdapter(_LegacyRegistry())
    out = await adapter.execute_tool("demo_skill", {"action": "x"})
    assert "ungoverned direct execution is disabled" in out
    # load_skill is only reached by validate (not by the execution path)
    assert out.startswith("Error:")


# ── 7c8082f5: legacy schema honors declared types/optionality ─────────────


def test_legacy_schema_uses_declared_types_and_optionality():
    class _LegacyRegistry:
        skills = {
            "s": SimpleNamespace(
                description="d",
                inputs={
                    "count": {"type": "integer", "description": "how many"},
                    "note": {"type": "string", "default": "hi"},
                },
            )
        }

    defs = FunctionCallingAdapter(_LegacyRegistry()).get_tool_definitions()
    props = defs["s"]["parameters"]["properties"]
    assert props["count"]["type"] == "integer"   # not coerced to string
    assert defs["s"]["parameters"]["required"] == ["count"]  # 'note' has a default
