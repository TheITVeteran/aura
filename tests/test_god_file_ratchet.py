"""The largest files may shrink. They may not grow.

This test exists because the check that was supposed to do this job ran on
zero files. `core/architecture_quality/gate.py` implements a growth ratchet
with a policy default that forbids growth — and its loop iterates
`changed_paths`, which the only production caller never passes. The loop body
never executed. Under that gate `interface/routes/chat.py` reached 24,658
lines and `core/brain/llm/mlx_client.py` gained 3,135.

So the first test here is not about file sizes at all. It is the assertion
that this ratchet examines a non-empty set of files, because that is the
specific way the previous one failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.god_file_ratchet import (
    DEFAULT_THRESHOLD,
    RATCHET_PATH,
    load_ratchet,
    measure,
    violations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ratchet() -> dict[str, object]:
    return load_ratchet()


@pytest.fixture(scope="module")
def current(ratchet: dict[str, object]) -> dict[str, int]:
    return measure(REPO_ROOT, int(ratchet["threshold"]))


def test_the_ratchet_actually_examines_files(ratchet: dict[str, object]) -> None:
    """The failure mode of the gate this replaces: a check that checked nothing.

    An empty scope passes every other assertion in this file vacuously, which
    is exactly how a growth ratchet can sit in the tree for months reporting
    success while the files it names double in size.
    """
    recorded: dict[str, int] = ratchet["files"]  # type: ignore[assignment]
    assert recorded, "ratchet records no files — it cannot be ratcheting anything"
    assert len(recorded) > 50, (
        f"only {len(recorded)} files recorded; the scan is probably not reaching "
        "the source tree"
    )


def test_no_oversized_file_grew(
    ratchet: dict[str, object], current: dict[str, int]
) -> None:
    recorded: dict[str, int] = ratchet["files"]  # type: ignore[assignment]
    grew, _ = violations(recorded, current, int(ratchet["threshold"]))

    assert not grew, (
        "oversized file(s) grew. These files are already past the structural "
        "threshold; adding to them makes the largest problem in the codebase "
        "larger. Put the new code in a module of its own, or shrink the file "
        "first:\n  " + "\n  ".join(grew)
    )


def test_no_new_file_crossed_the_threshold(
    ratchet: dict[str, object], current: dict[str, int]
) -> None:
    recorded: dict[str, int] = ratchet["files"]  # type: ignore[assignment]
    _, appeared = violations(recorded, current, int(ratchet["threshold"]))

    assert not appeared, (
        "file(s) newly crossed the oversized threshold:\n  "
        + "\n  ".join(appeared)
        + "\n\nSplit it, or record it deliberately with "
        "`python tools/god_file_ratchet.py --write`."
    )


def test_growth_is_detected() -> None:
    """The ratchet must be able to fail, or it is decoration."""
    recorded = {"core/example.py": 2000}
    grew, appeared = violations(recorded, {"core/example.py": 2001}, DEFAULT_THRESHOLD)

    assert grew and not appeared
    assert "+1" in grew[0]


def test_shrinkage_is_never_a_violation() -> None:
    recorded = {"core/example.py": 2000}
    grew, appeared = violations(recorded, {"core/example.py": 900}, DEFAULT_THRESHOLD)

    assert not grew and not appeared


def test_deleted_file_is_not_a_violation() -> None:
    """Removing an oversized file is the outcome this ratchet wants."""
    grew, appeared = violations({"core/gone.py": 2000}, {}, DEFAULT_THRESHOLD)

    assert not grew and not appeared


def test_new_oversized_file_is_detected() -> None:
    grew, appeared = violations({}, {"core/fresh.py": 1900}, DEFAULT_THRESHOLD)

    assert appeared and not grew


def test_recorded_sizes_are_not_below_reality_by_accident(
    ratchet: dict[str, object], current: dict[str, int]
) -> None:
    """Every recorded file is at or above the threshold it was recorded for.

    A recorded entry under the threshold means the ratchet file was hand-edited
    into a state the generator would never produce, which usually means someone
    lowered a number to make a failure go away rather than shrinking the file.
    """
    threshold = int(ratchet["threshold"])
    recorded: dict[str, int] = ratchet["files"]  # type: ignore[assignment]
    too_small = {
        path: size
        for path, size in recorded.items()
        if size <= threshold and path in current
    }

    assert not too_small, (
        f"recorded sizes at or below the {threshold}-line threshold: {too_small}"
    )


def test_ratchet_file_is_valid_and_declares_its_schema() -> None:
    data = json.loads(RATCHET_PATH.read_text(encoding="utf-8"))

    assert data["schema"] == "aura.god_file_ratchet.v1"
    assert int(data["threshold"]) == DEFAULT_THRESHOLD
    assert isinstance(data["files"], dict)
    assert all(isinstance(v, int) and v > 0 for v in data["files"].values())


def test_threshold_matches_the_architecture_baseline() -> None:
    """Two definitions of "oversized" that disagree would let a file sit in the
    gap: over one threshold, under the other, flagged by neither."""
    baseline = json.loads(
        (REPO_ROOT / "config" / "aura_architecture_quality_baseline.json").read_text(
            encoding="utf-8"
        )
    )

    assert int(baseline["god_file_threshold"]) == DEFAULT_THRESHOLD
