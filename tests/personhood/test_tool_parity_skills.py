"""tests/personhood/test_tool_parity_skills.py
===========================================
Unit and integration test suite verifying the five new tool-parity skills:
  1. CodeREPLSkill
  2. ImageGenSkill
  3. XToolsSkill
  4. RenderBridgeSkill
  5. VoiceOutputSkill
"""

import pytest
import os
import asyncio
from pathlib import Path

from core.skills.code_repl import CodeREPLSkill, CodeREPLInput
from core.skills.image_gen import ImageGenSkill, ImageGenInput
from core.skills.x_tools import XToolsSkill, XToolsInput
from core.skills.render_bridge import RenderBridgeSkill, RenderBridgeInput
from core.skills.voice_output import VoiceOutputSkill, VoiceOutputInput
from core.capability_engine import CapabilityEngine


def test_skills_instantiation_and_metadata():
    """Verify that all five new skills can be successfully instantiated and have correct metadata."""
    # 1. CodeREPLSkill
    repl = CodeREPLSkill()
    assert repl.name == "code_repl"
    assert repl.input_model == CodeREPLInput
    assert "sandboxed_compute" in repl.effect_scope

    # 2. ImageGenSkill
    image_gen = ImageGenSkill()
    assert image_gen.name == "image_gen"
    assert image_gen.input_model == ImageGenInput

    # 3. XToolsSkill
    x_tools = XToolsSkill()
    assert x_tools.name == "x_tools"
    assert x_tools.input_model == XToolsInput

    # 4. RenderBridgeSkill
    render_bridge = RenderBridgeSkill()
    assert render_bridge.name == "render_bridge"
    assert render_bridge.input_model == RenderBridgeInput

    # 5. VoiceOutputSkill
    voice_output = VoiceOutputSkill()
    assert voice_output.name == "voice_output"
    assert voice_output.input_model == VoiceOutputInput


@pytest.mark.asyncio
async def test_code_repl_execution():
    """Verify that CodeREPLSkill successfully executes Python code and captures output."""
    repl = CodeREPLSkill()

    # Simple arithmetic that works in the restricted sandbox (no imports needed)
    params = CodeREPLInput(
        code="x = 2 + 2\nprint(x)",
        timeout=10,
        capture_files=False
    )

    result = await repl.execute(params, {})

    # The sandbox runner captures stdout inside its own JSON protocol,
    # so the output appears in the result's stdout field.
    assert result.get("ok") is True
    assert result.get("returncode") == 0
    # Verify stdout capture via at least one backend
    stdout = result.get("stdout", "")
    assert "4" in stdout


@pytest.mark.asyncio
async def test_capability_engine_trigger_routing():
    """Verify capability engine successfully maps trigger phrases to the new skills."""
    engine = CapabilityEngine()
    # CapabilityEngine.__init__ calls reload_skills() + _load_default_trigger_patterns() already

    # Verify our 5 new skills exist in capability engine skills dictionary
    assert "code_repl" in engine.skills
    assert "image_gen" in engine.skills
    assert "x_tools" in engine.skills
    assert "render_bridge" in engine.skills
    assert "voice_output" in engine.skills

    # Verify custom trigger mapping matches intent
    repl_triggers = engine.detect_intent("run this python code please")
    assert "code_repl" in repl_triggers

    image_triggers = engine.detect_intent("generate an image of a futuristic skyline")
    assert "image_gen" in image_triggers

    x_triggers = engine.detect_intent("search twitter for the latest AI trends")
    assert "x_tools" in x_triggers

    render_triggers = engine.detect_intent("display chart with the summary results")
    assert "render_bridge" in render_triggers

    voice_triggers = engine.detect_intent("text to speech this paragraph")
    assert "voice_output" in voice_triggers
