"""God-file ratchet — the largest files may shrink, and may not grow.

WHY THIS EXISTS SEPARATELY FROM THE ARCHITECTURE GATE
`core/architecture_quality/gate.py` already implements this check, with a
policy knob (`max_line_growth_for_large_file`) and a default that forbids
growth. It examined zero files on every run. The block iterates
`changed_paths`, and the only production caller —
`tools/closeout/architecture_quality_gate.py` — calls `evaluate_reports()`
without them, so the loop ran over an empty tuple and reported a pass.
`interface/routes/chat.py` reached 24,658 lines under a gate configured to
prevent exactly that; `core/brain/llm/mlx_client.py` gained 3,135.

That fail-open is fixed. This ratchet still exists beside it for two reasons
that are not redundancy:

1.  The architecture baseline is Ed25519-signed and needs an external key plus
    a migration receipt to regenerate. It is currently stale — the gate fails
    on four unrelated counts before reaching file sizes — so the growth
    finding is buried in a step that is already red. A ratchet nobody can read
    the output of does not ratchet.
2.  This one runs as an ordinary test, in the ordinary suite, with no key
    material. The cost of enforcement should not be a signing ceremony.

WHAT IT DOES NOT DO
It does not require anyone to refactor a 24,000-line file today. It requires
that the file not get worse. Decomposition is the goal; a floor under the debt
is the thing that can be enforced starting now, and the numbers in
`config/god_file_ratchet.json` only ever move down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_PATH = REPO_ROOT / "config" / "god_file_ratchet.json"

#: Matches `god_file_threshold` in the architecture baseline. A file over this
#: is structurally oversized by the repo's own standing definition.
DEFAULT_THRESHOLD = 1500

INCLUDE_ROOTS = ("core", "interface", "infrastructure", "slo", "tools")

_SKIP_PARTS = (
    "__pycache__",
    ".venv",
    ".git",
    "node_modules",
    ".claude",
    "worktrees",
)


def source_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for include in INCLUDE_ROOTS:
        base = root / include
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            found.append(path)
    return found


def line_count(path: Path) -> int:
    """Physical lines. Not statements, not tokens.

    Deliberately the crudest possible metric: it cannot be argued with, it
    cannot be gamed by reformatting toward one-liners without making the file
    worse in a way a reviewer will see, and every developer can predict it.
    """
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def measure(root: Path, threshold: int) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in source_files(root):
        count = line_count(path)
        if count > threshold:
            sizes[path.relative_to(root).as_posix()] = count
    return dict(sorted(sizes.items()))


def load_ratchet(path: Path = RATCHET_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def violations(
    recorded: dict[str, int],
    current: dict[str, int],
    threshold: int,
) -> tuple[list[str], list[str]]:
    """(growth, newly-oversized). Shrinkage is never a violation."""
    grew: list[str] = []
    for path, allowed in sorted(recorded.items()):
        now = current.get(path)
        if now is None:
            continue  # deleted or renamed — that is a shrink, not a regression
        if now > allowed:
            grew.append(f"{path}: {allowed} -> {now} (+{now - allowed})")

    appeared = [
        f"{path}: {size} lines (threshold {threshold})"
        for path, size in sorted(current.items())
        if path not in recorded
    ]
    return grew, appeared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "regenerate the ratchet. REFUSES to record any file larger than its "
            "current recorded size — the file only moves down."
        ),
    )
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    existing = load_ratchet() if RATCHET_PATH.exists() else {}
    threshold = int(existing.get("threshold", DEFAULT_THRESHOLD))
    recorded: dict[str, int] = dict(existing.get("files", {}))  # type: ignore[arg-type]
    current = measure(root, threshold)

    grew, appeared = violations(recorded, current, threshold)

    if args.write:
        # The only-shrinks property, enforced in the writer rather than trusted
        # to reviewers: a regeneration run on a dirty tree must not quietly
        # bless growth that happened since the last one.
        if grew:
            print("REFUSING to write: these files grew since the ratchet was set.", file=sys.stderr)
            for item in grew:
                print(f"  {item}", file=sys.stderr)
            print(
                "\nShrink them, or record the growth deliberately by editing "
                "config/god_file_ratchet.json by hand so the diff is reviewable.",
                file=sys.stderr,
            )
            return 1
        merged = {path: min(size, recorded.get(path, size)) for path, size in current.items()}
        payload = {
            "schema": "aura.god_file_ratchet.v1",
            "threshold": threshold,
            "note": (
                "Physical line counts for files over the threshold. These numbers "
                "may only decrease. Regenerate with `python tools/god_file_ratchet.py "
                "--write`, which refuses to record growth."
            ),
            "files": dict(sorted(merged.items())),
        }
        RATCHET_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RATCHET_PATH} ({len(merged)} files over {threshold} lines)")
        return 0

    if not grew and not appeared:
        total = sum(current.values())
        print(f"god-file ratchet OK — {len(current)} files over {threshold} lines, {total} lines total")
        return 0

    if grew:
        print(f"{len(grew)} oversized file(s) GREW:")
        for item in grew:
            print(f"  {item}")
    if appeared:
        print(f"{len(appeared)} file(s) newly crossed {threshold} lines:")
        for item in appeared:
            print(f"  {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
