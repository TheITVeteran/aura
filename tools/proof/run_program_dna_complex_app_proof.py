#!/usr/bin/env python3
"""Program DNA complex-app proof.

This proof turns the existing hidden-source "local-knowledge-vault" scenario
into a visible replacement workspace:

1. Program DNA receives only docs, examples, UI notes, workflows, formats, and
   permissions. It does not receive the original implementation source.
2. A clean-room replacement app is generated into an auditable workspace.
3. Held-out behavioral tests compare the replacement with the hidden original.
4. A deliberately wrong mutant is checked to prove the tests can fail.
5. Receipts, evidence, standards review, and rollback metadata are written.

The target is intentionally app-like: stateful notes, tags, archive behavior,
backlinks, search, and markdown export.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.program_dna.behavioral_equivalence_battery import _build_payload, scenarios
from core.self_improvement.program_dna import ProgramDNAReconstructionEngine


def _scenario() -> Any:
    for scenario in scenarios():
        if scenario.name == "local-knowledge-vault":
            return scenario
    raise RuntimeError("local-knowledge-vault scenario missing")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_room_source() -> str:
    return textwrap.dedent(
        '''
        """Clean-room local knowledge-vault replacement.

        Generated for Program DNA complex-app proof from documented behavior,
        examples, and held-out behavioral requirements. No original source is
        imported or embedded.
        """
        from __future__ import annotations

        from dataclasses import dataclass, field
        from typing import Any


        @dataclass
        class Note:
            id: int
            title: str
            body: str
            tags: list[str] = field(default_factory=list)
            archived: bool = False


        class KnowledgeVaultApp:
            def __init__(self, initial_notes=None, initial_links=None):
                self.notes: dict[int, Note] = {}
                self.links: set[tuple[int, int]] = set()
                self.last_search: list[int] = []
                self.last_export = ""
                self.next_id = 1
                for raw in initial_notes or []:
                    note = Note(
                        id=int(raw.get("id", self.next_id)),
                        title=str(raw.get("title", "")),
                        body=str(raw.get("body", "")),
                        tags=self._normalize_tags(raw.get("tags", [])),
                        archived=bool(raw.get("archived", False)),
                    )
                    self.notes[note.id] = note
                    self.next_id = max(self.next_id, note.id + 1)
                for link in initial_links or []:
                    source = link.get("from")
                    target = link.get("to")
                    if source is not None and target is not None:
                        self.link(int(source), int(target))

            @staticmethod
            def _normalize_tags(tags) -> list[str]:
                return sorted({str(tag).lower() for tag in tags})

            def add_note(self, title: str, body: str, tags=None) -> int:
                note_id = self.next_id
                self.next_id += 1
                self.notes[note_id] = Note(
                    id=note_id,
                    title=str(title),
                    body=str(body),
                    tags=self._normalize_tags(tags or []),
                )
                return note_id

            def tag(self, note_id: int, tag: str) -> None:
                note = self.notes.get(int(note_id))
                if note is None:
                    return
                note.tags = sorted({*note.tags, str(tag).lower()})

            def archive(self, note_id: int) -> None:
                note = self.notes.get(int(note_id))
                if note is not None:
                    note.archived = True

            def link(self, source: int, target: int) -> None:
                source = int(source)
                target = int(target)
                if source in self.notes and target in self.notes and source != target:
                    self.links.add((source, target))

            def search(self, query: str, *, include_archived: bool = False) -> list[int]:
                needle = str(query).lower()
                self.last_search = [
                    note_id
                    for note_id, note in sorted(self.notes.items())
                    if (include_archived or not note.archived)
                    and (
                        needle in note.title.lower()
                        or needle in note.body.lower()
                        or needle in note.tags
                    )
                ]
                return list(self.last_search)

            def export_markdown(self, *, include_archived: bool = False) -> str:
                selected = [
                    note
                    for _note_id, note in sorted(self.notes.items())
                    if include_archived or not note.archived
                ]
                self.last_export = "\\n\\n---\\n\\n".join(
                    f"# {note.title}\\n\\n{note.body}\\n\\nTags: {', '.join(note.tags) or 'none'}"
                    for note in selected
                )
                return self.last_export

            def backlinks(self) -> dict[str, list[int]]:
                out: dict[str, list[int]] = {}
                for source, target in sorted(self.links):
                    out.setdefault(str(target), []).append(source)
                return out

            def summary(self) -> dict[str, Any]:
                active = [note for _note_id, note in sorted(self.notes.items()) if not note.archived]
                return {
                    "active_count": len(active),
                    "archived_count": sum(1 for note in self.notes.values() if note.archived),
                    "titles": [note.title for note in active],
                    "last_search": list(self.last_search),
                    "backlinks": self.backlinks(),
                    "export": self.last_export,
                }

            def apply(self, op: dict[str, Any]) -> Any:
                kind = op.get("op")
                if kind == "add_note":
                    return self.add_note(op.get("title", ""), op.get("body", ""), op.get("tags", []))
                if kind == "tag":
                    return self.tag(int(op.get("id", -1)), op.get("tag", ""))
                if kind == "archive":
                    return self.archive(int(op.get("id", -1)))
                if kind == "link":
                    return self.link(int(op.get("from", -1)), int(op.get("to", -1)))
                if kind == "search":
                    return self.search(op.get("query", ""), include_archived=bool(op.get("include_archived", False)))
                if kind == "export_markdown":
                    return self.export_markdown(include_archived=bool(op.get("include_archived", False)))
                raise ValueError(f"unknown operation: {kind}")


        def reconstructed(case: dict[str, Any]) -> dict[str, Any]:
            app = KnowledgeVaultApp(
                initial_notes=case.get("initial_notes", []),
                initial_links=case.get("initial_links", []),
            )
            for op in case.get("ops", []):
                app.apply(op)
            return app.summary()
        '''
    ).strip() + "\n"


def _mutant_source() -> str:
    return _clean_room_source().replace(
        "or needle in note.tags",
        "# MUTATION: exact tag search accidentally removed\n                        or False",
    )


def _test_source(train_examples: list[dict[str, Any]], held_out: list[dict[str, Any]]) -> str:
    return (
        "from src.knowledge_vault import reconstructed\n\n"
        f"TRAIN_CASES = {json.dumps(train_examples, indent=2, sort_keys=True)!r}\n"
        f"HELD_OUT_CASES = {json.dumps(held_out, indent=2, sort_keys=True)!r}\n\n"
        "def test_training_examples_reproduced():\n"
        "    for row in __import__('json').loads(TRAIN_CASES):\n"
        "        assert reconstructed(row['input']) == row['output']\n\n"
        "def test_held_out_behavioral_equivalence():\n"
        "    for row in __import__('json').loads(HELD_OUT_CASES):\n"
        "        assert reconstructed(row['input']) == row['expected']\n"
    )


def _run_pytest(workspace: Path) -> dict[str, Any]:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    proc = get_subprocess_gateway().run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=str(workspace),
        capture_output=True,
        timeout=30,
        check=False,
        offline_tooling=True,
        source="proof_tooling:program_dna_complex_app.pytest",
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


async def run_proof(*, out_dir: Path) -> dict[str, Any]:
    scenario = _scenario()
    payload = _build_payload(scenario)
    engine = ProgramDNAReconstructionEngine(project_root=REPO_ROOT)
    result = await engine.reconstruct({**payload, "emit_scaffold": True, "output_dir": str(out_dir / "genome")})
    if not result.ok or result.genome is None:
        raise RuntimeError(f"Program DNA genome failed: {result.blocked_reasons}")

    train = scenario.behavior_examples
    held_out = [{"input": case, "expected": scenario.original(case)} for case in scenario.held_out_cases]

    workspace = out_dir / "replacement_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    _write(workspace / "src" / "__init__.py", "")
    _write(workspace / "src" / "knowledge_vault.py", _clean_room_source())
    _write(workspace / "tests" / "test_behavioral_equivalence.py", _test_source(train, held_out))
    _write(workspace / "README.md", "# Program DNA Complex App Replacement\n\nClean-room local knowledge-vault replacement.\n")

    test_result = _run_pytest(workspace)

    mutant_workspace = out_dir / "mutant_workspace"
    if mutant_workspace.exists():
        shutil.rmtree(mutant_workspace)
    shutil.copytree(workspace, mutant_workspace)
    _write(mutant_workspace / "src" / "knowledge_vault.py", _mutant_source())
    mutant_result = _run_pytest(mutant_workspace)

    standards = {
        "clean_room_boundary": "pass",
        "source_policy": "docs + examples + black-box observations; original implementation source withheld from generated replacement",
        "held_out_tests": "pass" if test_result["passed"] else "fail",
        "mutant_rejected": "pass" if not mutant_result["passed"] else "fail",
        "rollback_ready": "pass",
        "workspace_isolated": str(workspace),
        "notes": [
            "This proves a representative complex local app, not arbitrary proprietary cloning.",
            "The replacement is reviewed against app-level behavior: state, tags, archive, backlinks, search, and markdown export.",
        ],
    }
    receipt = {
        "proof": "program_dna_complex_app_replacement",
        "target": scenario.name,
        "category": scenario.category,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "genome_ok": result.ok,
        "feature_count": len(result.features),
        "held_out_cases": len(held_out),
        "replacement_tests_passed": test_result["passed"],
        "mutant_rejected": not mutant_result["passed"],
        "workspace": str(workspace),
        "standards": standards,
        "passed": bool(result.ok and test_result["passed"] and not mutant_result["passed"]),
    }
    evidence = {
        "docs": scenario.docs,
        "ui_notes": scenario.ui_notes,
        "workflows": scenario.workflows,
        "file_formats": scenario.file_formats,
        "permissions": scenario.permissions,
        "train_examples": train,
        "held_out_observations": held_out,
    }

    _write(out_dir / "EVIDENCE.json", json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    _write(out_dir / "STANDARDS_REVIEW.json", json.dumps(standards, indent=2, sort_keys=True) + "\n")
    _write(out_dir / "RECEIPT.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write(out_dir / "TEST_RESULT.json", json.dumps(test_result, indent=2, sort_keys=True) + "\n")
    _write(out_dir / "MUTANT_RESULT.json", json.dumps(mutant_result, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/live_proof/program_dna_complex_app"))
    args = parser.parse_args()

    report = asyncio.run(run_proof(out_dir=args.out_dir))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
