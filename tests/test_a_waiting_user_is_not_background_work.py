"""The build the user is waiting for lost its own admission contest.

Live 2026-07-27. "Reverse-engineer a clean-room 2048 and place it on my
Desktop." The build failed with::

    RuntimeError: LLM returned no Python source; the model returned nothing at
    all.

The model returned nothing because it was never asked. ``LLMCodeGenerator`` set
``self.is_background = True`` in its constructor, unconditionally, and the
router defers background inference while a foreground turn holds the local
substrate. The foreground turn holding it was the chat turn waiting on the
build. A self-deadlock, reported as a model failure.

Two defects, and both are classes rather than instances:

* **priority is the caller's to declare.** Only the caller knows whether a
  human is sitting there. The generator's constructor cannot know it, so it
  must not decide it — ``context["is_background"]`` wins over the default.
* **an empty return must carry its reason.** The router's deferral is correct
  behaviour; returning a bare ``""`` for it is not, because every caller then
  invents a cause. The reason is now recorded where the failing caller can read
  it, so the error names admission instead of blaming the model.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from core.brain.llm import deferral_record
from core.brain.llm.code_generator import LLMCodeGenerator

REPO = Path(__file__).resolve().parents[1]


class _RecordingRouter:
    """Stands in for the router; remembers how the request was tagged."""

    def __init__(self, reply: str = "```python\nx = 1\n```") -> None:
        self.reply = reply
        self.seen: dict[str, object] = {}

    async def think(self, prompt: str, **kwargs: object) -> str:
        self.seen = dict(kwargs)
        return self.reply


def _generate(router: _RecordingRouter, context: dict[str, object]) -> str:
    generator = LLMCodeGenerator(router=router)
    return asyncio.run(generator.generate_async("build a thing", context=context))


# ── The caller decides ────────────────────────────────────────────────────

def test_a_caller_can_declare_the_user_is_waiting() -> None:
    router = _RecordingRouter()
    _generate(router, {"is_background": False})
    assert router.seen.get("is_background") is False


def test_the_default_still_holds_for_callers_that_say_nothing() -> None:
    """Autonomous self-improvement really is background; do not swing it."""
    router = _RecordingRouter()
    _generate(router, {})
    assert router.seen.get("is_background") is True


def test_a_caller_can_still_declare_itself_background() -> None:
    router = _RecordingRouter()
    _generate(router, {"is_background": True})
    assert router.seen.get("is_background") is True


# ── Every reconstruction lane declares itself foreground ──────────────────

@pytest.mark.parametrize(
    "module",
    ["core/self_improvement/program_dna.py", "core/self_improvement/program_materialization.py"],
)
def test_every_reconstruction_synthesis_is_foreground(module: str) -> None:
    """A reconstruction exists because someone asked for it, by definition.

    Asserted structurally rather than by grep so a new synthesis site added
    later cannot quietly reintroduce the deadlock.
    """
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    contexts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "origin"
            for key in node.keys
            if key is not None
        )
        and any(
            isinstance(key, ast.Constant) and key.value == "system_prompt"
            for key in node.keys
            if key is not None
        )
    ]
    assert contexts, f"no synthesis context found in {module}"
    for context in contexts:
        declared = {
            key.value: value
            for key, value in zip(context.keys, context.values, strict=False)
            if isinstance(key, ast.Constant)
        }
        assert "is_background" in declared, "a reconstruction context must declare its lane"
        assert declared["is_background"].value is False


# ── An empty generation names its real cause ──────────────────────────────

def test_an_unexplained_empty_generation_still_says_so() -> None:
    deferral_record.reset_for_test()
    with pytest.raises(RuntimeError, match="returned nothing at all"):
        _generate(_RecordingRouter(reply=""), {})


def test_a_deferred_generation_blames_admission_not_the_model() -> None:
    deferral_record.reset_for_test()
    deferral_record.record_deferral(
        origin="program_dna_reconstruction", reason="foreground turn holds the local substrate"
    )
    with pytest.raises(RuntimeError, match="deferred, not run"):
        _generate(_RecordingRouter(reply=""), {})
    deferral_record.reset_for_test()


def test_a_stale_deferral_never_explains_an_unrelated_failure() -> None:
    """A confident wrong cause is worse than an absent one."""
    deferral_record.reset_for_test()
    deferral_record.record_deferral(origin="metabolic", reason="something much earlier")
    import time

    assert deferral_record.explain_empty_generation(now=time.time() + 3600) == ""
    deferral_record.reset_for_test()


def test_what_the_model_actually_said_survives_into_the_error() -> None:
    """The other undiagnosable ending: prose reaches the parser as "code".

    An apology parses as nothing, the SyntaxError said only "invalid syntax",
    and the sentence that would have explained everything was discarded.
    """
    deferral_record.reset_for_test()
    deferral_record.record_deferral(origin="metabolic", reason="deferred much earlier")
    with pytest.raises(RuntimeError, match="I cannot help with that"):
        _generate(_RecordingRouter(reply="I cannot help with that."), {})
    deferral_record.reset_for_test()
