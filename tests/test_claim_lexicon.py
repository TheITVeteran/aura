"""tests/test_claim_lexicon.py — a loaded name must say what it measures.

Operationally: this measures whether every module whose FILENAME carries a term
that asserts a conclusion — consciousness, qualia, phenomenal, personhood,
sentience, subjective, volition, strange-loop, emergent, AGI, soul, free-will,
self-aware — states in its module docstring what it operationally measures.

A reader meets this project through filenames. `test_consciousness_guarantee.py`
and `core/consciousness/qualia_engine.py` assert conclusions, and they are met
long before any ledger that qualifies them. TESTING.md previously said "the
filenames are historical", which was true and did not help: the name is what
travels into a screenshot, a search result, a review, a conversation.

The gate is ratcheted rather than absolute. Fixing every remaining file in one
pass would produce a batch of hurried one-liners, which is worse than the
problem it solves — the point of the line is that someone had to finish the
sentence "this measures ..." honestly.
"""

from __future__ import annotations

import json

import pytest

from tools.check_claim_lexicon import (
    BASELINE_PATH,
    DEFINITION_MARKERS,
    LOADED_TERMS,
    has_operational_definition,
    load_baseline,
    loaded_terms_in,
    scan,
)


def test_the_debt_is_at_or_below_its_baseline():
    """The ratchet. This is the gate's actual job."""
    report = scan()
    baseline = load_baseline()
    assert baseline is not None, f"no baseline at {BASELINE_PATH}"
    assert report["missing_count"] <= baseline, (
        f"{report['missing_count']} loaded-name files lack an operational "
        f"definition, above the baseline of {baseline}. Add a line beginning "
        f"{DEFINITION_MARKERS[0]!r} to the module docstring: "
        + ", ".join(item["file"] for item in report["missing_definition"][:5])
    )


def test_the_baseline_may_only_shrink():
    """A baseline that can rise is a baseline that will."""
    baseline = load_baseline()
    report = scan()
    assert baseline is not None
    assert baseline >= report["missing_count"], "baseline is stale-low"
    # And it must have actually come down from where this started: 61 files
    # carried a loaded name with no definition when the gate was written.
    assert baseline <= 61, (
        f"the baseline is {baseline}, at or above the 61 this started at — it "
        "has been raised rather than reduced"
    )


def test_the_loudest_names_have_definitions():
    """The files the criticism named specifically, pinned individually.

    A ratchet lets debt sit. These are the ones that must not be sitting: they
    are the modules and batteries an outside reader is most likely to meet
    first and most likely to misread.
    """
    report = scan()
    compliant = set(report["compliant"])
    for required in (
        "core/consciousness/qualia_engine.py",
        "core/consciousness/qualia_synthesizer.py",
        "core/consciousness/somatic_qualia.py",
        "core/cognitive/strange_loop.py",
        "core/autonomy/personhood_engine.py",
        "core/consciousness/consciousness_bridge.py",
        "tests/test_consciousness_guarantee.py",
        "tests/test_consciousness_guarantee_advanced.py",
        "tests/test_personhood_battery.py",
    ):
        assert required in compliant, (
            f"{required} carries a loaded term in its name and does not say "
            "what it operationally measures"
        )


# ── the matcher ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("qualia_engine.py", ["qualia"]),
        ("test_consciousness_guarantee.py", ["consciousness"]),
        ("strange_loop.py", ["strange_loop"]),
        ("test_sentient_unity.py", ["sentien"]),
        # Not matches: ordinary engineering words that happen to contain a term.
        ("imagination.py", []),
        ("mind_tick.py", []),
        ("memory_manager.py", []),
        ("pagination.py", []),
    ],
)
def test_terms_match_name_tokens_not_substrings(filename, expected):
    """`imagination.py` contains the letters of "agi" and asserts nothing.

    A gate that cannot tell those apart spends its credibility on false
    positives and gets switched off — which is how the repository ends up with
    no gate at all.
    """
    assert loaded_terms_in(filename) == expected


def test_a_docstring_without_the_marker_does_not_count():
    """Atmosphere is not a definition.

    The marker exists because "this is not a claim of consciousness" is a
    disclaimer, and a disclaimer does not tell a reader what the file measures.
    """
    assert has_operational_definition(
        "A careful and honest module docstring that disclaims everything."
    ) is False
    assert has_operational_definition(None) is False
    assert has_operational_definition("") is False


@pytest.mark.parametrize("marker", DEFINITION_MARKERS)
def test_either_marker_satisfies_the_gate(marker):
    assert has_operational_definition(f"Header.\n\n{marker} entropy over a vector.")


def test_the_term_list_covers_what_the_criticism_named():
    """The vocabulary is the gate's judgement; state it explicitly."""
    for term in ("consciousness", "qualia", "phenomenal", "personhood", "volition", "agi"):
        assert term in LOADED_TERMS


def test_the_baseline_file_is_well_formed():
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload["max_missing_definition"], int)
    assert "never raise" in payload["description"].lower()
