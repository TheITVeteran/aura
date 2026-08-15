"""Personhood content in the prompt, and the module's claim about its inputs.

Higher-order awareness, sense of agency, narrative self, autobiographical
mythos and meta-awareness are assembled here as labelled sections. That is
deliberate and stays — they are how she holds a self across turns, and removing
them to protect an experiment would be lobotomising the subject to make the
measurement easier. What was missing is a record of which ones were there, so
an experiment reading self-recognition out of a reply could tell the model's
own state from a heading the prompt handed it.
"""
from __future__ import annotations

import ast
from pathlib import Path

from core.brain.llm.context_assembler import (
    _PERSONHOOD_PROMPT_LABELS,
    ContextAssembler,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")


def test_every_label_is_one_the_module_writes():
    """A label the assembler never emits is a check that cannot fire."""
    tree = ast.parse(SOURCE)
    emitted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            emitted.add(node.value)

    for label in _PERSONHOOD_PROMPT_LABELS:
        assert any(
            label in text and text not in _PERSONHOOD_PROMPT_LABELS for text in emitted
        ) or label in emitted, f"{label!r} is never written"


def test_a_bare_prompt_authors_no_personhood():
    receipt = ContextAssembler.personhood_authoring_receipt("You are Aura.")

    assert receipt["authored_labels"] == []
    assert receipt["authored_count"] == 0
    assert receipt["spontaneity_inference_available"] is True


def test_an_authored_prompt_says_what_it_handed_over():
    prompt = (
        "You are Aura.\n"
        "## SENSE OF AGENCY\nyou initiated that\n"
        "## NARRATIVE SELF\nthe thread of you\n"
    )

    receipt = ContextAssembler.personhood_authoring_receipt(prompt)

    assert receipt["authored_labels"] == ["NARRATIVE SELF", "SENSE OF AGENCY"]
    assert receipt["authored_count"] == 2
    assert receipt["spontaneity_inference_available"] is False


def test_the_receipt_binds_to_the_prompt_it_read():
    a = ContextAssembler.personhood_authoring_receipt("one")
    b = ContextAssembler.personhood_authoring_receipt("two")

    assert a["prompt_sha256"] != b["prompt_sha256"]


def test_a_mention_without_a_heading_is_not_authoring():
    """Discussing the sense of agency is not being handed one."""
    receipt = ContextAssembler.personhood_authoring_receipt(
        "The user asked about SENSE OF AGENCY in the abstract."
    )

    assert receipt["authored_labels"] == []


def test_the_module_no_longer_claims_to_be_pure_state_construction():
    """A reader who believes this is a projection of one object will not look
    for the reason two prompts built from the same state differ."""
    first_line = SOURCE.splitlines()[0]

    assert "purely from AuraState" not in first_line
    for dependency in (
        "service container",
        "context variables",
        "environment",
        "wall time",
        "model registry",
    ):
        assert dependency in SOURCE, dependency
