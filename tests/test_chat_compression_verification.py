"""The revision pass has to see what it is revising, and be checked before it wins."""

from __future__ import annotations

import pytest

from core.context.chat_compression import (
    SNAPSHOT_OPEN,
    ChatCompressionService,
    CompressionStatus,
    estimate_tokens_for_messages,
    response_text,
)

SNAPSHOT = SNAPSHOT_OPEN + "\n" + ("detail line with /a/real/path.py\n" * 40) + "</state_snapshot>"


def _history(n: int = 120) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 200}
        for i in range(n)
    ]


class _Brain:
    """A brain whose two calls can be answered independently."""

    def __init__(self, summary=SNAPSHOT, revision=SNAPSHOT, shape="dict"):
        self.summary = summary
        self.revision = revision
        self.shape = shape
        self.prompts: list[str] = []

    async def generate(self, prompt, system_prompt=None, options=None):
        self.prompts.append(prompt)
        text = self.summary if len(self.prompts) == 1 else self.revision
        if self.shape == "str":
            return text
        if self.shape == "content":
            return type("R", (), {"content": text})()
        return {"response": text}


async def _compress(brain, history=None):
    service = ChatCompressionService(temp_dir="/tmp/claude-501/compression-test")
    history = history or _history()
    return await service.compress(
        history=history,
        model_token_limit=1000,
        current_token_count=estimate_tokens_for_messages(history),
        brain=brain,
        force=True,
    )


# ── the defect ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_revision_pass_is_shown_the_snapshot_and_the_history():
    """It used to be a bare instruction, so the model critiqued what it could not see."""
    brain = _Brain()
    await _compress(brain)

    assert len(brain.prompts) == 2
    revision_prompt = brain.prompts[1]
    assert SNAPSHOT in revision_prompt, "the snapshot under review must be attached"
    assert "turn 0" in revision_prompt, "the history it must cover must be attached"


@pytest.mark.asyncio
async def test_a_revision_that_is_not_a_snapshot_is_refused():
    """"Looks good, nothing missing" replaced the compressed memory with a note about it."""
    brain = _Brain(revision="Looks good — nothing important was missing.")
    new_history, info = await _compress(brain)

    assert info.status is CompressionStatus.COMPRESSED
    assert new_history[0]["content"] == SNAPSHOT


@pytest.mark.asyncio
async def test_a_revision_that_collapses_the_snapshot_is_refused():
    """A revision is meant to add detail; one that loses half has summarised the summary."""
    brain = _Brain(revision=SNAPSHOT_OPEN + "\nshort\n</state_snapshot>")
    new_history, _info = await _compress(brain)
    assert new_history[0]["content"] == SNAPSHOT


@pytest.mark.asyncio
async def test_a_genuine_improvement_is_accepted():
    better = SNAPSHOT + "\n" + SNAPSHOT_OPEN + " extra recovered detail </state_snapshot>"
    brain = _Brain(revision=better)
    new_history, _info = await _compress(brain)
    assert new_history[0]["content"] == better


@pytest.mark.asyncio
async def test_an_empty_revision_keeps_the_first_pass():
    brain = _Brain(revision="")
    new_history, _info = await _compress(brain)
    assert new_history[0]["content"] == SNAPSHOT


@pytest.mark.asyncio
async def test_a_failing_revision_pass_keeps_the_first_pass():
    class Half(_Brain):
        async def generate(self, prompt, system_prompt=None, options=None):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return {"response": SNAPSHOT}
            raise ConnectionError("revision call died")

    new_history, info = await _compress(Half())
    assert info.status is CompressionStatus.COMPRESSED
    assert new_history[0]["content"] == SNAPSHOT


# ── the brain contract ───────────────────────────────────────────────────


def test_response_text_accepts_every_live_generate_shape():
    assert response_text({"response": " a "}) == "a"
    assert response_text({"content": "b"}) == "b"
    assert response_text("c") == "c"
    assert response_text(type("R", (), {"content": "d"})()) == "d"
    assert response_text(None) == ""
    assert response_text({}) == ""
    assert response_text(42) == ""


@pytest.mark.asyncio
async def test_a_brain_returning_a_bare_string_still_compresses():
    """CognitiveEngine.generate returns str; the service assumed a mapping."""
    new_history, info = await _compress(_Brain(shape="str"))
    assert info.status is CompressionStatus.COMPRESSED
    assert new_history[0]["content"] == SNAPSHOT


class _Broken:
    async def generate(self, *_args, **_kwargs):
        raise AttributeError("router half-built")


@pytest.mark.asyncio
async def test_a_brain_that_raises_an_attribute_error_degrades_instead_of_escaping():
    """The old except tuple was network shapes only, so type errors escaped compress()."""
    new_history, info = await _compress(_Broken())
    assert info.status in {CompressionStatus.NOOP, CompressionStatus.CONTENT_TRUNCATED}
    assert new_history is None or isinstance(new_history, list)


@pytest.mark.asyncio
async def test_a_failed_brain_still_reclaims_oversized_tool_output():
    """Truncation is the half of compression that does not need a model at all."""
    history = _history(60) + [
        {"role": "tool", "content": "[TOOL RESULT]\n" + "\n".join(f"line {i} " + "d" * 40 for i in range(6000))}
    ]
    new_history, info = await _compress(_Broken(), history=history)
    assert info.status is CompressionStatus.CONTENT_TRUNCATED
    assert info.new_token_count < info.original_token_count
    assert "Output truncated" in new_history[-1]["content"]
    assert "Full output saved to:" in new_history[-1]["content"]


@pytest.mark.asyncio
async def test_no_brain_falls_back_to_truncation_rather_than_failing():
    history = _history(60) + [
        {"role": "tool", "content": "[TOOL RESULT]\n" + "\n".join(f"line {i} " + "d" * 40 for i in range(6000))}
    ]
    new_history, info = await _compress(None, history=history)
    assert info.status is CompressionStatus.CONTENT_TRUNCATED
    assert isinstance(new_history, list)


# ── the guarantees the service already offered ───────────────────────────


@pytest.mark.asyncio
async def test_a_compression_that_would_inflate_is_discarded():
    brain = _Brain(summary="y" * 400_000, revision="y" * 400_000)
    new_history, info = await _compress(brain)
    assert new_history is None
    assert info.status is CompressionStatus.FAILED_INFLATED


@pytest.mark.asyncio
async def test_an_empty_summary_is_reported_rather_than_installed():
    brain = _Brain(summary="")
    new_history, info = await _compress(brain)
    assert new_history is None
    assert info.status is CompressionStatus.FAILED_EMPTY_SUMMARY


def test_two_different_objects_no_longer_share_the_context_manager_name():
    """`ServiceContainer.get` and the orchestrator attribute returned different classes."""
    from pathlib import Path

    import core.orchestrator.initializers.core_baseline as module

    baseline = Path(module.__file__).read_text(encoding="utf-8")
    assert 'register_runtime_service("context_window_manager"' in baseline
    assert 'register_runtime_service("context_manager"' not in baseline
    assert "orchestrator.context_manager = ContextWindowManager" not in baseline


def test_the_verification_prompt_carries_its_subject():
    """A bare instruction let the model critique what it could not see."""
    from core.context.chat_compression import VERIFICATION_PROMPT

    assert "{snapshot}" in VERIFICATION_PROMPT
    assert "{history}" in VERIFICATION_PROMPT


def test_the_uncalled_prompt_assembler_is_gone():
    """Dead code on a now-live class reads as the supported path."""
    from core.context.context_manager import ContextWindowManager

    assert not hasattr(ContextWindowManager, "build_prompt")


def test_the_window_manager_still_exposes_the_reachable_entry_point():
    from core.context.context_manager import ContextWindowManager

    assert hasattr(ContextWindowManager, "compress_if_needed")
    manager = ContextWindowManager(model_name="Cortex")
    assert manager._raw_limit == 32_000


@pytest.mark.asyncio
async def test_under_threshold_is_a_noop():
    service = ChatCompressionService(temp_dir="/tmp/claude-501/compression-test")
    history = _history(4)
    new_history, info = await service.compress(
        history=history,
        model_token_limit=1_000_000,
        current_token_count=estimate_tokens_for_messages(history),
        brain=_Brain(),
    )
    assert new_history is None
    assert info.status is CompressionStatus.NOOP
