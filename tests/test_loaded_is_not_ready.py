"""Readiness, adapter provenance, and the listener's lifecycle.

"loaded" meant `mlx_lm.load` returned two objects. It did not mean the
tokenizer belongs to the model, or that the pair can produce a single token —
and every caller reads `loaded` as readiness, including the fallback that
decides whether to swap lanes, which would happily swap TO a lane that could
not answer either.

The adapter gate was the existence of `adapter_config.json`. A filename is not
a provenance.

And the listener could be started twice by concurrent callers, had no shutdown
path at all, and left its handle set after dying so it could never be restarted.
"""
from __future__ import annotations

import json

import pytest

from core.brain.llm.nucleus_manager import (
    _CANCELLED_LOAD_DRAIN_S,
    _LISTENER_STOP_TIMEOUT_S,
    _adapter_compatibility,
    _readiness_probe,
)


class _Tokenizer:
    vocab_size = 32_000

    @staticmethod
    def encode(text):
        return [1, 2, 3]


class _Model:
    class args:
        vocab_size = 32_000


# ── readiness ─────────────────────────────────────────────────────────────


def test_a_working_pair_is_ready():
    verdict = _readiness_probe(_Model(), _Tokenizer())

    assert verdict["ready"] is True
    assert "encoded 3 tokens" in verdict["reason"]


@pytest.mark.parametrize(("model", "tokenizer"), [(None, _Tokenizer()), (_Model(), None)])
def test_a_missing_half_is_not_ready(model, tokenizer):
    assert _readiness_probe(model, tokenizer)["ready"] is False


def test_two_bare_objects_are_not_ready():
    """This is exactly what `load` returning successfully used to prove."""
    assert _readiness_probe(object(), object())["ready"] is False


def test_a_tokenizer_that_raises_is_not_ready():
    class _Broken:
        @staticmethod
        def encode(_text):
            raise RuntimeError("no vocab loaded")

    verdict = _readiness_probe(_Model(), _Broken())
    assert verdict["ready"] is False
    assert "RuntimeError" in verdict["reason"]


def test_a_tokenizer_that_produces_nothing_is_not_ready():
    class _Empty:
        vocab_size = 10

        @staticmethod
        def encode(_text):
            return []

    assert _readiness_probe(_Model(), _Empty())["ready"] is False


def test_a_tokenizer_that_can_emit_ids_the_model_lacks_is_not_ready():
    """The mismatch that produces garbage rather than an error."""

    class _Wide:
        vocab_size = 99_000

        @staticmethod
        def encode(_text):
            return [1]

    verdict = _readiness_probe(_Model(), _Wide())
    assert verdict["ready"] is False
    assert "exceeds model vocab" in verdict["reason"]


# ── adapter provenance ────────────────────────────────────────────────────


def _adapter(tmp_path, config: dict | None, *, weights: bool = True):
    directory = tmp_path / "adapter"
    directory.mkdir(exist_ok=True)
    if config is not None:
        (directory / "adapter_config.json").write_text(json.dumps(config))
    if weights:
        (directory / "adapters.safetensors").write_bytes(b"\x00")
    return str(directory)


def test_a_matching_adapter_is_attached(tmp_path):
    path = _adapter(tmp_path, {"base_model_name_or_path": "/elsewhere/Qwen2.5-32B-Instruct"})

    verdict = _adapter_compatibility(path, "/models/Qwen2.5-32B-Instruct")
    assert verdict["compatible"] is True


def test_an_adapter_for_a_different_base_is_refused(tmp_path):
    path = _adapter(tmp_path, {"base_model_name_or_path": "Llama-3-8B"})

    verdict = _adapter_compatibility(path, "/models/Qwen2.5-32B-Instruct")
    assert verdict["compatible"] is False
    assert "trained against" in verdict["reason"]


def test_a_config_without_weights_is_refused(tmp_path):
    path = _adapter(tmp_path, {"base_model_name_or_path": "Qwen2.5-32B-Instruct"}, weights=False)

    assert _adapter_compatibility(path, "/models/Qwen2.5-32B-Instruct")["compatible"] is False


def test_a_config_naming_no_base_is_refused(tmp_path):
    """Compatibility that cannot be established is not compatibility."""
    path = _adapter(tmp_path, {"lora_layers": 8})

    verdict = _adapter_compatibility(path, "/models/Qwen2.5-32B-Instruct")
    assert verdict["compatible"] is False
    assert "names no base model" in verdict["reason"]


def test_an_unreadable_config_is_refused(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{not json")
    (directory / "adapters.safetensors").write_bytes(b"\x00")

    assert _adapter_compatibility(str(directory), "/models/x")["compatible"] is False


def test_a_missing_directory_is_refused(tmp_path):
    assert _adapter_compatibility(str(tmp_path / "nope"), "/models/x")["compatible"] is False


# ── listener lifecycle ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_starts_create_one_listener():
    import asyncio

    from core.brain.llm.nucleus_manager import NucleusManager

    created: list[str] = []

    class _Sub:
        @staticmethod
        async def get():
            await asyncio.sleep(3600)

    class _Bus:
        @staticmethod
        async def subscribe(_topic):
            created.append(_topic)
            return _Sub()

    manager = NucleusManager.__new__(NucleusManager)
    manager.bus = _Bus()
    manager._listener_task = None
    manager._listener_subscription = None
    manager._listener_lock = asyncio.Lock()
    manager._running = True

    await asyncio.gather(*(manager.ensure_listener_started() for _ in range(5)))
    await asyncio.sleep(0.05)

    assert len(created) == 1, created
    await manager.stop_listener()
    assert manager._listener_task is None


@pytest.mark.asyncio
async def test_a_dead_listener_can_be_restarted():
    """The handle was never cleared, so a listener that died stayed "started"
    forever and update handling never came back."""
    import asyncio

    from core.brain.llm.nucleus_manager import NucleusManager

    starts: list[int] = []

    class _Sub:
        @staticmethod
        async def get():
            # Outside every class the listener's handlers catch, which is the
            # case that used to leave the handle set forever.
            raise ZeroDivisionError("worker died in a way the handler never caught")

    class _Bus:
        @staticmethod
        async def subscribe(_topic):
            starts.append(1)
            return _Sub()

    manager = NucleusManager.__new__(NucleusManager)
    manager.bus = _Bus()
    manager._listener_task = None
    manager._listener_subscription = None
    manager._listener_lock = asyncio.Lock()
    manager._running = True

    await manager.ensure_listener_started()
    await asyncio.sleep(0.05)
    assert manager._listener_task is None, "the dead listener still holds the handle"

    await manager.ensure_listener_started()
    await asyncio.sleep(0.05)
    assert len(starts) == 2


def test_the_lifecycle_bounds_are_named():
    assert _LISTENER_STOP_TIMEOUT_S > 0
    assert _CANCELLED_LOAD_DRAIN_S > 0
