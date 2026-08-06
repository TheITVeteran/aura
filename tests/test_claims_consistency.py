"""CLAIMS_MATRIX.md must not contradict itself, and neither must the docs citing it.

Found by hand on 2026-08-06, which is the problem: claim 14's summary row read
`blocked` while its own detail section, forty lines below in the same file,
read `locally demonstrated`. `docs/CLAIM_BOUNDARIES.md` was worse — section A
said full validation "requires independent, adversarial, out-of-distribution
evaluation beyond this repository", and section B, immediately below it,
granted the AGI-candidate label on nine batteries that all run inside this
repository. One of the nine was a result that `docs/DNU_BASELINE_FAIRNESS_
AUDIT.md` had explicitly retracted for that exact use, cited by name.

None of that was caught by anything. A claims file that disagrees with itself
is worse than no claims file: it lets any reader find a sentence supporting
whatever they already believed, and it lets the optimistic sentence be the one
that gets quoted.

The two tables use different vocabularies for the same axis — `locally
demonstrated`/`pass`, `not proven`/`blocked` — which is why an eyeball sweep
missed the drift for so long, and why the mapping lives in code here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX = REPO_ROOT / "CLAIMS_MATRIX.md"

#: Two vocabularies, one axis. `partial` is deliberately its own value: a claim
#: that is partly evidenced is neither proven nor blocked, and collapsing it
#: into either direction is how a half-result gets quoted as a whole one.
STATUS_EQUIVALENCE = {
    "causally demonstrated": "supported",
    "locally demonstrated": "supported",
    "pass": "supported",
    "not proven": "open",
    "blocked": "open",
    "partial": "partial",
}


def _canonical(raw: str) -> str | None:
    text = raw.replace("`", "").replace("*", "").split("—")[0].strip().lower()
    return STATUS_EQUIVALENCE.get(text)


def _claim_id(cell: str) -> str | None:
    match = re.match(r"\*{0,2}(\d+[a-z]?)[.\s]", cell.strip())
    return match.group(1) if match else None


@pytest.fixture(scope="module")
def matrix_text() -> str:
    return MATRIX.read_text(encoding="utf-8")


def _summary_table(text: str) -> dict[str, str]:
    """Rows shaped `| **N. Name** | `status` | evidence |`."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            continue
        claim = _claim_id(cells[1])
        if claim and cells[2].startswith("`"):
            found.setdefault(claim, cells[2])
    return found


def _falsifier_table(text: str) -> dict[str, str]:
    """Rows shaped `| N | name | definition | evidence | command | status | falsifier |`."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 8 and re.fullmatch(r"\d+[a-z]?", cells[1]):
            found[cells[1]] = cells[6]
    return found


def _detail_sections(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        heading = re.match(r"###\s*(\d+[a-z]?)\.", line)
        if heading:
            current = heading.group(1)
            continue
        if current and "**Classification**" in line:
            found[current] = line.split(":", 1)[1]
            current = None
    return found


def test_every_claim_agrees_with_itself(matrix_text: str) -> None:
    """The defect this file was written for: one claim, two verdicts."""
    summary = _summary_table(matrix_text)
    falsifier = _falsifier_table(matrix_text)
    detail = _detail_sections(matrix_text)

    disagreements: list[str] = []
    for claim in sorted(set(summary) | set(falsifier) | set(detail), key=_sort_key):
        seen = {
            source: _canonical(value)
            for source, value in (
                ("summary table", summary.get(claim)),
                ("falsifier table", falsifier.get(claim)),
                ("detail section", detail.get(claim)),
            )
            if value is not None
        }
        distinct = {v for v in seen.values() if v is not None}
        if len(distinct) > 1:
            rendered = ", ".join(f"{k}={v}" for k, v in seen.items())
            disagreements.append(f"claim {claim}: {rendered}")

    assert not disagreements, (
        "CLAIMS_MATRIX.md contradicts itself. A reader will quote whichever "
        "verdict suits them:\n  " + "\n  ".join(disagreements)
    )


def _sort_key(claim: str) -> tuple[int, str]:
    return (int(re.match(r"\d+", claim).group()), claim)


def test_every_status_uses_a_known_vocabulary(matrix_text: str) -> None:
    """An unrecognised status silently drops out of the agreement check above."""
    unknown: list[str] = []
    for label, table in (
        ("summary table", _summary_table(matrix_text)),
        ("falsifier table", _falsifier_table(matrix_text)),
        ("detail section", _detail_sections(matrix_text)),
    ):
        for claim, raw in table.items():
            if _canonical(raw) is None:
                unknown.append(f"{label} claim {claim}: {raw.strip()[:60]!r}")

    assert not unknown, (
        "unrecognised claim status — add it to STATUS_EQUIVALENCE with its "
        "meaning, or fix the typo. An unknown status is not checked against "
        "anything:\n  " + "\n  ".join(unknown)
    )


def test_every_claim_has_a_recorded_falsifier(matrix_text: str) -> None:
    """A claim with no way to lose is not a claim.

    31a, 31b, 32 and 33 were added to the summary table without falsifier rows.
    They are the newest and most-cited claims in the file, and nothing recorded
    what would refute them.
    """
    summary = _summary_table(matrix_text)
    falsifier = _falsifier_table(matrix_text)

    missing = sorted(set(summary) - set(falsifier), key=_sort_key)

    assert not missing, (
        "claim(s) with no falsifier row: "
        + ", ".join(missing)
        + ". Record what would refute it, or the row is an assertion."
    )


def test_falsifier_rows_are_not_empty(matrix_text: str) -> None:
    """A present-but-blank falsifier passes the check above while saying nothing."""
    thin: list[str] = []
    for line in matrix_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 9 and re.fullmatch(r"\d+[a-z]?", cells[1]):
            if len(cells[7]) < 40:
                thin.append(f"claim {cells[1]}: {cells[7]!r}")

    assert not thin, (
        "falsifier text too thin to name a failure mode:\n  " + "\n  ".join(thin)
    )


def test_retracted_evidence_is_not_cited_as_licensing(matrix_text: str) -> None:
    """The AGI-candidate defect, held closed.

    `docs/CLAIM_BOUNDARIES.md` licensed the label on a battery that
    `docs/DNU_BASELINE_FAIRNESS_AUDIT.md` had retracted for that exact purpose.
    The audit's conclusion is unambiguous and was ignored for months, so the
    words that grant the label are checked against the words that withdraw it.
    """
    boundaries = (REPO_ROOT / "docs" / "CLAIM_BOUNDARIES.md").read_text(encoding="utf-8")

    agi_section = re.search(
        r"###\s*B\.\s*AGI-Candidate Architecture(.*?)(?=\n###\s|\Z)",
        boundaries,
        re.DOTALL,
    )
    assert agi_section, "CLAIM_BOUNDARIES.md section B not found — did it move?"
    body = agi_section.group(1)

    status = re.search(r"\*\*Status\*\*:\s*(.+)", body)
    assert status, "section B has no Status line"
    assert _canonical(status.group(1)) == "open", (
        "the AGI-candidate label is marked as anything other than open. It is "
        "not licensable by any battery in this repository — every one of them "
        "is written by the author of the system under test. See the section's "
        "own text for why the previous licensing condition failed."
    )
