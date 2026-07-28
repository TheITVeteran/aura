"""Protected status decides whether repair may touch a subsystem.

CP126 5b472fda: bare substring containment failed in BOTH directions —
"will" matched "goodwill_tracker" (suppressing repair on something never
protected) while a genuinely protected component with a novel name matched
nothing and got no protection at all.
"""
from __future__ import annotations

import pytest

from core.adaptation.adaptive_immunity import get_adaptive_immune_system


@pytest.fixture(scope="module")
def immune():
    return get_adaptive_immune_system()


@pytest.mark.parametrize(
    "name",
    ["will", "identity", "soul", "executive", "continuity", "sovereignty"],
)
def test_a_protected_token_is_protected(immune, name):
    assert immune._is_protected_subsystem(name) is True


@pytest.mark.parametrize(
    "name",
    ["executive_core", "identity_guard", "will.decide", "self_model_debug_dump"],
)
def test_a_protected_token_inside_a_longer_name_still_matches(immune, name):
    assert immune._is_protected_subsystem(name) is True


@pytest.mark.parametrize(
    "name",
    ["goodwill_tracker", "executor_pool", "memory", "willow", "identityless"],
)
def test_a_mere_substring_is_not_protection(immune, name):
    """The false positives: repair was suppressed on things never protected."""
    assert immune._is_protected_subsystem(name) is False


def test_a_multi_token_hint_needs_all_its_tokens(immune):
    assert immune._is_protected_subsystem("memory_guard") is True
    assert immune._is_protected_subsystem("guard") is False
    assert immune._is_protected_subsystem("canonical_self") is True


def test_dotted_names_tokenize(immune):
    assert immune._is_protected_subsystem("core.will.engine") is True
    assert immune._is_protected_subsystem("core.willpower.engine") is False


@pytest.mark.parametrize("value", ["", None, "   "])
def test_an_empty_subsystem_is_not_protected(immune, value):
    assert immune._is_protected_subsystem(value or "") is False


def test_case_is_ignored(immune):
    assert immune._is_protected_subsystem("Executive_Core") is True
