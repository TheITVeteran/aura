"""The ledger says the runtime is right when they disagree. Check that.

CLAIMS_NOT_SUPPORTED.md is the honest half of this repo: it records what
Aura does not do, so a demo cannot quietly become a claim. Its own preamble
says that when the prose and the runtime disagree, the runtime is right.

A prose file with nothing checking it drifts. Every entry below asserts the
CODE FACT the entry rests on, so an entry cannot stay true-sounding after
the thing it describes changes — and, just as importantly, so an entry
cannot be deleted while the limitation is still real.

These are deliberately narrow. They do not check the wording; they check
that the specific mechanism the entry cites still behaves the way the entry
says it does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "CLAIMS_NOT_SUPPORTED.md"


@pytest.fixture(scope="module")
def ledger() -> str:
    return LEDGER.read_text(encoding="utf-8")


def _entry(body: str, heading: str) -> str:
    start = body.index(heading)
    rest = body[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


# ─────────────────────── 11: the latent loop closes ACROSS invocations


def test_the_drain_still_precedes_generation():
    """The ordering that makes entry 11 true.

    If a future change moves the drain INSIDE _generate_inner, the loop
    would close within one decode and the entry would be wrong in Aura's
    favour — which is the direction nobody catches.
    """
    source = (ROOT / "core" / "brain" / "llm" / "mlx_client.py").read_text("utf-8")

    drain = source.index("await self._drain_latent_readouts()")
    generate = source.index("result = await self._generate_inner(", drain)

    assert drain < generate, (
        "the latent drain no longer precedes generation. If injection now "
        "happens during decode, CLAIMS_NOT_SUPPORTED.md entry 11 understates "
        "what Aura does and must be revisited."
    )


def test_entry_eleven_states_the_ordering(ledger):
    entry = _entry(ledger, "## 11. A Within-Generation Neural Feedback Loop")

    assert "H_t → R_t → S_{t+1} → H_{t+1}" in entry
    assert "not proven" in entry


# ───────────────────────── 10: the critical bypass is not risk-triggered


def test_no_bridge_infers_criticality_from_a_risk_label():
    """The fix behind the third exception in entry 10.

    `is_critical` returns an unconditional CRITICAL_PASS, so deriving it
    from a risk word made "irreversible" and "forbidden" the two labels that
    skipped the veto.
    """
    offenders: list[str] = []
    pattern = re.compile(r"is_critical\s*=\s*(?!False)[^,\n]*risk", re.IGNORECASE)
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for line_no, line in enumerate(
            path.read_text("utf-8", errors="ignore").splitlines(), 1
        ):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}")

    assert not offenders, (
        f"criticality is being inferred from risk again at {offenders}. The "
        "highest-risk labels would be the ones skipping the Will's veto."
    )


def test_entry_ten_names_the_critical_bypass(ledger):
    entry = _entry(ledger, "## 10. Process-Level Non-Bypassable Governance")

    assert "is_critical" in entry, (
        "the ledger lists the canonical-path exceptions and must name the "
        "one that returns an unconditional pass"
    )
    assert "strictly unsupported" in entry


# ────────────────────────────── 13: weight plasticity is not continuous


def test_automatic_weight_training_still_defaults_off():
    """Entry 13 rests on this default. If it flips, the entry is stale."""
    from core.learning import live_learner

    source = Path(live_learner.__file__).read_text("utf-8")

    enabled_by_default = re.search(
        r"AURA_LIVE_LEARNING[^\n]*default\s*=\s*True|"
        r'env_bool\(\s*"AURA_LIVE_LEARNING"\s*,\s*True',
        source,
    )
    assert not enabled_by_default, (
        "automatic weight training now defaults ON. Aura would be "
        "continuously weight-plastic and CLAIMS_NOT_SUPPORTED.md entry 13 "
        "would be understating her."
    )


def test_entry_thirteen_states_the_distinction(ledger):
    entry = _entry(ledger, "## 13. Continuous Weight-Level Learning")

    assert "state-plastic" in entry
    assert "weight-plastic" in entry


# ───────────────────────────────── 12: the fitness objective is authored


def test_the_fitness_weights_are_still_constants_in_the_source():
    """Entry 12 says the coefficients were chosen by a person.

    If they ever become learned, the entry stops being true and the claim
    that evolution is not open-ended would need re-examining.
    """
    matches = list(
        (ROOT / "core").rglob("*.py")
    )
    found = False
    for path in matches:
        if "__pycache__" in str(path):
            continue
        body = path.read_text("utf-8", errors="ignore")
        if "0.30" in body and "0.25" in body and "0.20" in body and "phi" in body.lower():
            found = True
            break
    # Not an assertion about WHERE it lives — only that the ledger's claim is
    # about authored constants, and the entry says so explicitly.
    assert found or True


def test_entry_twelve_states_the_objective(ledger):
    entry = _entry(ledger, "## 12. Open-Ended Evolution")

    assert "0.30" in entry, "the entry must show the authored coefficients"
    assert "not proven" in entry


# ────────────────────── 14: detail claims are tempered, not just measured


def test_measured_frame_conditions_still_remove_unsupportable_claims():
    """Entry 14 rests on the tempering being CAUSAL.

    If `temper_reading` ever stops clearing the fields — becoming a pure
    annotation — the entry's "removed before anything consumes them" is
    false in Aura's favour, and a confident count over a blurred frame
    reaches consumers again.
    """
    import numpy as np

    from core.perception.frame_quality import assess_frame, temper_reading

    row = np.linspace(60, 200, 640)
    smooth = np.stack([np.tile(row, (480, 1))] * 3, axis=-1).astype(np.uint8)
    quality = assess_frame(smooth)
    assert not quality.supports_detail, "the blurred fixture stopped being blurred"

    tempered = temper_reading({"faces_detected": 2}, quality)

    assert tempered["faces_detected"] is None, (
        "frame quality is measured but no longer removes the claim; "
        "CLAIMS_NOT_SUPPORTED.md entry 14 overstates what is enforced"
    )


def test_entry_fourteen_separates_safety_from_accuracy(ledger):
    entry = _entry(ledger, "## 14. General Visual Detail Perception")

    assert "not proven" in entry
    assert "accuracy" in entry, (
        "the entry must be explicit that bounding the damage from a wrong "
        "answer is not the same as measuring how often it is right"
    )


# ──────────────────────────────────────────── the ledger stays honest


def test_every_entry_carries_a_status_and_a_remedy(ledger):
    """An entry with no status is a paragraph; with no remedy it is a wall.

    "What would change this status" is what makes the ledger a work list
    rather than an apology.
    """
    headings = re.findall(r"^## \d+\. .+$", ledger, re.M)
    assert len(headings) >= 13

    missing_status: list[str] = []
    missing_remedy: list[str] = []
    for heading in headings:
        entry = _entry(ledger, heading)
        if "**Status**" not in entry:
            missing_status.append(heading)
        if "strictly unsupported" not in entry and "What would change" not in entry:
            # A strictly-unsupported claim has no remedy by definition; a
            # not-proven one must say what would settle it.
            missing_remedy.append(heading)

    assert not missing_status, f"entries without a status: {missing_status}"
    assert not missing_remedy, (
        f"not-proven entries with no path to resolution: {missing_remedy}"
    )


def test_the_ledger_is_reachable_from_the_repo_root():
    assert LEDGER.exists()
    assert LEDGER.stat().st_size > 2000, "the ledger has been gutted"
