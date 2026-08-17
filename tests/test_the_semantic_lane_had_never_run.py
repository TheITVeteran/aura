"""The keyword probe was not a fallback. It was the only path.

`_signal_self_model` scores how much a turn is about Aura's own identity. It
has two lanes: a keyword list, and embedding similarity against the live
canonical self block. The docstring calls the keyword probe "the
no-vector-organ fallback".

It read the self block like this:

    self_service = _get_service("canonical_self")
    reader = getattr(self_service, "get_context_block", None)

`get_context_block` is a method of `CanonicalSelfEngine`. The
`canonical_self` service key holds the `CanonicalSelf` **dataclass** that the
engine publishes on every tick. So `reader` was None on every turn of every
session, the block was always empty, and the semantic lane never ran once —
while the receipt said `method=keyword_terms` as though a measurement had
been taken and come back negative.

The abstraction was right and the seam named the wrong object. These tests
hold the seam to the object that can actually answer.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.brain import cognitive_ingress
from core.self.canonical_self import CanonicalSelf, CanonicalSelfEngine

SOURCE = pathlib.Path("core/brain/cognitive_ingress.py")


class _Resolver:
    """A service registry holding exactly what the live tick registers."""

    def __init__(self, engine=None, snapshot=None):
        self._services = {}
        if engine is not None:
            self._services["canonical_self_engine"] = engine
        if snapshot is not None:
            self._services["canonical_self"] = snapshot

    def __call__(self, name, default=None):
        return self._services.get(name, default)


@pytest.fixture
def engine():
    return CanonicalSelfEngine()


def _install(monkeypatch, resolver):
    monkeypatch.setattr(cognitive_ingress, "_get_service", resolver)


# ── the seam reaches the object that owns the method ────────────────────────


def test_the_dataclass_the_tick_registers_cannot_answer(engine):
    """The premise, stated as a test so it cannot quietly stop being true."""
    snapshot = engine.get_self()

    assert isinstance(snapshot, CanonicalSelf)
    assert not hasattr(snapshot, "get_context_block"), (
        "if the dataclass grows this method, the old seam starts working and "
        "this test should be re-read rather than deleted"
    )
    assert callable(getattr(engine, "get_context_block", None))


def test_the_block_is_read_from_the_engine(monkeypatch, engine):
    _install(monkeypatch, _Resolver(engine=engine, snapshot=engine.get_self()))

    block = cognitive_ingress._canonical_self_context_block()

    assert block, "the semantic lane got no self block from a live engine"
    assert "CANONICAL SELF-MODEL" in block


def test_the_dataclass_alone_yields_nothing_and_says_so(monkeypatch, engine):
    """Registering only the snapshot is the live wiring that was broken."""
    _install(monkeypatch, _Resolver(snapshot=engine.get_self()))

    assert cognitive_ingress._canonical_self_context_block() == ""


def test_an_absent_service_yields_nothing(monkeypatch):
    _install(monkeypatch, _Resolver())

    assert cognitive_ingress._canonical_self_context_block() == ""


# ── the lane is reachable, which is what was never true ─────────────────────


def test_the_semantic_method_is_reachable_with_an_embedder(monkeypatch, engine):
    """Whether it fires depends on the embedder. Whether it CAN fire must not
    depend on a service key that holds the wrong object."""
    _install(monkeypatch, _Resolver(engine=engine, snapshot=engine.get_self()))
    monkeypatch.setattr(cognitive_ingress, "_embedding_similarity", lambda _a, _b: 0.72)

    signal = cognitive_ingress._signal_self_model("should someone retrain your reward pathway?")

    assert "embedding_cosine_vs_canonical_self" in signal.detail
    assert signal.present is True


def test_without_an_embedder_the_receipt_says_keywords(monkeypatch, engine):
    """The honest fallback, named as such on a turn that does fire."""
    _install(monkeypatch, _Resolver(engine=engine, snapshot=engine.get_self()))
    monkeypatch.setattr(cognitive_ingress, "_embedding_similarity", lambda _a, _b: None)

    signal = cognitive_ingress._signal_self_model("what are your values")

    assert signal.present is True
    assert "method=keyword_terms" in signal.detail


def test_a_mundane_turn_stays_absent(monkeypatch, engine):
    """Identity relevance is a claim, and an ordinary turn does not make it."""
    _install(monkeypatch, _Resolver(engine=engine, snapshot=engine.get_self()))
    monkeypatch.setattr(cognitive_ingress, "_embedding_similarity", lambda _a, _b: 0.30)

    signal = cognitive_ingress._signal_self_model("what is the weather in Denver")

    assert signal.present is False


def test_a_missing_block_is_recorded_rather_than_scored_as_irrelevant(monkeypatch, engine):
    """An empty block read as "not about identity", which is a measurement
    this function did not make."""
    recorded = []
    _install(monkeypatch, _Resolver(snapshot=engine.get_self()))
    monkeypatch.setattr(
        cognitive_ingress,
        "record_degradation",
        lambda subsystem, exc, **kwargs: recorded.append((subsystem, kwargs)),
    )

    cognitive_ingress._canonical_self_context_block()

    assert recorded, "a dead semantic lane produced no record at all"
    subsystem, kwargs = recorded[0]
    assert subsystem == "cognitive_ingress.canonical_self"
    assert "keywords alone" in kwargs["action"]
    # The record names which keys were tried, so the next reader can see
    # whether the service was absent or present-but-wrong.
    assert "canonical_self_engine" in kwargs["action"]


# ── a structural gate, so the seam cannot drift back ────────────────────────


def test_the_engine_key_is_tried_before_the_snapshot_key():
    """Order matters: the snapshot key is the one that cannot answer."""
    assert cognitive_ingress._CANONICAL_SELF_READERS[0] == "canonical_self_engine"
    assert "canonical_self" in cognitive_ingress._CANONICAL_SELF_READERS


def test_no_caller_reads_get_context_block_off_the_snapshot_key():
    """The exact shape of the original defect, as a parse-tree check."""
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") != "_get_service":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "canonical_self":
            continue
        # Reaching the snapshot key is fine inside the reader that also tries
        # the engine; it is not fine as a standalone lookup elsewhere.
        assert node.lineno < cognitive_ingress._signal_self_model.__code__.co_firstlineno, (
            f"{SOURCE}:{node.lineno} looks up the canonical_self snapshot "
            "outside the reader that falls back to the engine"
        )
