"""core/self_modification/behavioral_gate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Behavioral validation for self-modification patches: a patch must prove
behavior against the repo's real tests, not merely parse.

The hole this closes (external review, July 3): sandbox validation ran
syntax/py_compile/Pyright, then looked for test files INSIDE the sandbox
— which never contains the repo's tests/ tree — and on finding none
marked unit tests passed by default. A patch that compiles cleanly but
poisons logic sailed through with zero behavioral evidence.

Mechanism:
1. IMPACTED-TEST SELECTION — repo tests are matched to the target module
   by stem (tests/**/test_<stem>*.py) plus a bounded import scan (tests
   that import the module directly). Deterministic, no LLM.
2. CLONED-REPO EXECUTION — the repo is cloned copy-on-write (APFS
   clonefile via `cp -c`; falls back to a plain copy elsewhere), the
   patch is applied to the CLONE, and the impacted tests run there with
   the repo venv. The live tree is never touched; every other module the
   tests import is the real one, so integration behavior is exercised.
3. FAIL-CLOSED COVERAGE — a core/ patch with ZERO impacted tests is not
   silently promotable: the gate returns covered=False and the caller
   must route it to shadow/approval instead of auto-promotion.
"""
from __future__ import annotations

import logging
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Aura.SelfMod.BehavioralGate")

MAX_IMPACTED_TESTS = 6          # bounded: the closest tests, not the world
IMPORT_SCAN_CAP = 400           # test files scanned for direct imports
DEFAULT_TIMEOUT_S = 180.0

# Directories never cloned (caches and state, not behavior).
_CLONE_EXCLUDES = (".git", ".venv", "__pycache__", ".claude", "data", "artifacts", "models")


@dataclass
class BehavioralVerdict:
    """Outcome of the behavioral gate for one patch."""
    passed: bool                 # tests ran and were green
    covered: bool                # at least one impacted test existed
    tests: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "covered": self.covered,
            "tests": self.tests,
            "duration_s": round(self.duration_s, 2),
            "detail": self.detail[:500],
        }


def select_impacted_tests(target_file: str, repo_root: Path) -> list[Path]:
    """Deterministically map a module to the repo tests that exercise it."""
    repo_root = Path(repo_root)
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        return []
    stem = Path(target_file).stem
    module_dotted = str(Path(target_file).with_suffix("")).replace("/", ".")

    impacted: dict[Path, int] = {}  # path → priority (lower = stronger match)

    # 1. Stem match: tests/test_<stem>.py and tests/**/test_<stem>*.py
    for candidate in sorted(tests_dir.rglob(f"test_{stem}*.py")):
        impacted.setdefault(candidate, 0)

    # 2. Bounded import scan: tests that import the module directly.
    import_re = re.compile(
        rf"^\s*(?:from\s+{re.escape(module_dotted)}\s+import|import\s+{re.escape(module_dotted)}\b)",
        re.MULTILINE,
    )
    scanned = 0
    for candidate in sorted(tests_dir.rglob("test_*.py")):
        if candidate in impacted:
            continue
        if scanned >= IMPORT_SCAN_CAP:
            break
        scanned += 1
        try:
            if import_re.search(candidate.read_text(encoding="utf-8", errors="replace")):
                impacted.setdefault(candidate, 1)
        except OSError:
            continue

    ranked = sorted(impacted.items(), key=lambda kv: (kv[1], str(kv[0])))
    return [path for path, _prio in ranked[:MAX_IMPACTED_TESTS]]


async def _clone_repo(repo_root: Path, clone_dir: Path) -> bool:
    """Copy-on-write clone of the repo (APFS `cp -c`; plain copy fallback)."""
    import asyncio

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    await asyncio.to_thread(clone_dir.mkdir, parents=True, exist_ok=True)
    entries = await asyncio.to_thread(
        lambda: [
            entry for entry in repo_root.iterdir()
            if entry.name not in _CLONE_EXCLUDES
        ]
    )
    for flags in (["-Rc"], ["-R"]):  # clonefile first, plain copy fallback
        result = await get_subprocess_gateway().run_async(
            ["cp", *flags, *[str(entry) for entry in entries], str(clone_dir)],
            capture_output=True,
            timeout=120,
            source="core.self_modification.behavioral_gate.clone",
        )
        if result.returncode == 0:
            return True
    return False


async def run_behavioral_gate(
    target_file: str,
    patched_source: str,
    *,
    repo_root: Path | str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> BehavioralVerdict:
    """Prove a patch against the repo's real impacted tests in a CoW clone.

    Never touches the live tree. Returns covered=False when no impacted
    tests exist — the caller decides what uncovered means (for core/
    patches: not auto-promotable).
    """
    import asyncio

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    started = time.monotonic()
    repo_root = await asyncio.to_thread(lambda: Path(repo_root or ".").resolve())
    impacted = await asyncio.to_thread(select_impacted_tests, target_file, repo_root)
    if not impacted:
        return BehavioralVerdict(
            passed=False,
            covered=False,
            detail=(
                f"no impacted tests found for {target_file} — behavioral "
                "equivalence cannot be demonstrated; not auto-promotable"
            ),
            duration_s=time.monotonic() - started,
        )

    with tempfile.TemporaryDirectory(prefix="aura_behavioral_gate_") as tmp:
        clone_dir = Path(tmp) / "clone"
        if not await _clone_repo(repo_root, clone_dir):
            return BehavioralVerdict(
                passed=False,
                covered=True,
                tests=[str(t.relative_to(repo_root)) for t in impacted],
                detail="repo clone failed; failing closed",
                duration_s=time.monotonic() - started,
            )

        # Apply the patch to the CLONE only.
        patched_path = clone_dir / target_file
        await asyncio.to_thread(patched_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            patched_path.write_text, patched_source, encoding="utf-8"
        )

        rel_tests = [str(t.relative_to(repo_root)) for t in impacted]
        result = await get_subprocess_gateway().run_async(
            [
                sys.executable, "-m", "pytest", "-q",
                "-p", "no:cacheprovider",
                *rel_tests,
            ],
            capture_output=True,
            timeout=timeout_s,
            cwd=clone_dir,
            source="core.self_modification.behavioral_gate.pytest",
        )
        tail = str(result.stdout or "")[-1500:] + str(result.stderr or "")[-500:]
        passed = result.returncode == 0
        if passed:
            logger.info(
                "Behavioral gate PASSED for %s (%d impacted tests)",
                target_file, len(rel_tests),
            )
        else:
            logger.warning(
                "Behavioral gate FAILED for %s: %s", target_file, tail[-300:],
            )
        return BehavioralVerdict(
            passed=passed,
            covered=True,
            tests=rel_tests,
            detail=tail if not passed else "impacted tests green",
            duration_s=time.monotonic() - started,
        )
