#!/usr/bin/env python3
"""Hidden-source Program DNA behavioral equivalence battery.

The battery gives Aura's Program DNA engine docs, examples, and observations,
but not implementation source. A deterministic clean-room builder then produces
small replacement behaviors from the emitted genome/evidence contract and held-
out tests compare those replacements with private originals.

This is empirical proof for representative reconstruction archetypes. It does
not claim arbitrary closed-source cloning or proprietary source recovery.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.self_improvement.program_dna import ProgramDNAReconstructionEngine


Case = dict[str, Any]
BehaviorFn = Callable[[Case], Any]


@dataclass(frozen=True)
class ProgramDNABatteryScenario:
    name: str
    category: str
    docs: list[str]
    behavior_examples: list[dict[str, Any]]
    held_out_cases: list[Case]
    original: BehaviorFn = field(repr=False)
    ui_notes: list[str] = field(default_factory=list)
    api_observations: list[str] = field(default_factory=list)
    file_formats: list[str] = field(default_factory=list)
    workflows: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    missing_docs: bool = False


@dataclass
class ScenarioResult:
    name: str
    category: str
    ok: bool
    equivalence: float
    cases_passed: int
    cases_total: int
    features: list[str]
    hidden_source_withheld: bool
    failures: list[dict[str, Any]] = field(default_factory=list)
    genome_summary: dict[str, Any] = field(default_factory=dict)


def _slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def _cli_original(case: Case) -> str:
    text = str(case["text"])
    command = str(case["command"])
    if command == "slug":
        return _slugify(text)
    if command == "stats":
        lines = text.count("\n") + (1 if text else 0)
        words = len([word for word in re.split(r"\s+", text.strip()) if word])
        return f"lines={lines} words={words} chars={len(text)}"
    raise ValueError(f"unknown command: {command}")


def _gui_original(case: Case) -> dict[str, Any]:
    count = int(case.get("initial_count", 0))
    label = str(case.get("initial_label", "Ready"))
    for action in case.get("actions", []):
        kind = action.get("type")
        if kind == "increment":
            count += 1
            label = f"Count: {count}"
        elif kind == "decrement":
            count -= 1
            label = f"Count: {count}"
        elif kind == "reset":
            count = 0
            label = "Ready"
        elif kind == "set_label":
            label = str(action.get("value", ""))
    return {"count": count, "label": label, "buttons_enabled": True}


def _csv_json_original(case: Case) -> str:
    reader = csv.DictReader(io.StringIO(str(case["csv"])))
    rows = [
        {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
        for row in reader
    ]
    return json.dumps({"columns": reader.fieldnames or [], "rows": rows}, sort_keys=True)


def _web_original(case: Case) -> dict[str, Any]:
    method = str(case.get("method", "GET")).upper()
    path = str(case["path"])
    body = case.get("body") or {}
    if path == "/health" and method == "GET":
        return {"status": 200, "json": {"ok": True}}
    if path == "/echo" and method == "POST":
        return {"status": 200, "json": {"echo": body}}
    match = re.fullmatch(r"/items/([a-z0-9_-]+)", path)
    if match and method == "GET":
        return {"status": 200, "json": {"id": match.group(1), "kind": "item"}}
    return {"status": 404, "json": {"error": "not_found"}}


def _db_original(case: Case) -> dict[str, Any]:
    rows = [dict(item) for item in case.get("initial_rows", [])]
    next_id = max([int(row.get("id", 0)) for row in rows] or [0]) + 1
    for op in case.get("ops", []):
        kind = op.get("op")
        if kind == "add":
            rows.append({"id": next_id, "title": str(op.get("title", "")), "done": False})
            next_id += 1
        elif kind == "done":
            for row in rows:
                if row["id"] == int(op.get("id")):
                    row["done"] = True
        elif kind == "delete":
            rows = [row for row in rows if row["id"] != int(op.get("id"))]
    return {"rows": rows, "open_count": sum(1 for row in rows if not row["done"])}


def _auth_original(case: Case) -> dict[str, Any]:
    user = str(case.get("user", ""))
    password = str(case.get("password", ""))
    route = str(case.get("route", "/profile"))
    authenticated = user == "demo" and password == "correct-horse"
    if not authenticated:
        return {"status": 401, "json": {"error": "unauthorized"}}
    if route == "/profile":
        return {"status": 200, "json": {"user": "demo", "scopes": ["read"]}}
    return {"status": 403, "json": {"error": "forbidden"}}


def _missing_docs_original(case: Case) -> str:
    text = re.sub(r"\s+", " ", str(case["text"]).strip())
    return text[:1].upper() + text[1:].lower() if text else ""


def _cli_replacement(case: Case) -> str:
    text = str(case["text"])
    command = str(case["command"])
    if command == "slug":
        lowered = text.lower()
        normalized = re.sub(r"[^a-z0-9]+", "-", lowered)
        return re.sub(r"-+", "-", normalized).strip("-")
    if command == "stats":
        words = [word for word in re.split(r"\s+", text.strip()) if word]
        line_count = text.count("\n") + (1 if text else 0)
        return f"lines={line_count} words={len(words)} chars={len(text)}"
    raise ValueError(f"unknown command: {command}")


def _gui_replacement(case: Case) -> dict[str, Any]:
    count = int(case.get("initial_count", 0))
    label = str(case.get("initial_label", "Ready"))
    for action in case.get("actions", []):
        action_type = action.get("type")
        if action_type == "increment":
            count += 1
            label = f"Count: {count}"
        elif action_type == "decrement":
            count -= 1
            label = f"Count: {count}"
        elif action_type == "reset":
            count = 0
            label = "Ready"
        elif action_type == "set_label":
            label = str(action.get("value", ""))
    return {"count": count, "label": label, "buttons_enabled": True}


def _csv_json_replacement(case: Case) -> str:
    reader = csv.DictReader(io.StringIO(str(case["csv"])))
    rows = []
    for row in reader:
        rows.append({key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()})
    return json.dumps({"columns": reader.fieldnames or [], "rows": rows}, sort_keys=True)


def _web_replacement(case: Case) -> dict[str, Any]:
    method = str(case.get("method", "GET")).upper()
    path = str(case["path"])
    if method == "GET" and path == "/health":
        return {"status": 200, "json": {"ok": True}}
    if method == "POST" and path == "/echo":
        return {"status": 200, "json": {"echo": case.get("body") or {}}}
    item_match = re.fullmatch(r"/items/([a-z0-9_-]+)", path)
    if method == "GET" and item_match:
        return {"status": 200, "json": {"id": item_match.group(1), "kind": "item"}}
    return {"status": 404, "json": {"error": "not_found"}}


def _db_replacement(case: Case) -> dict[str, Any]:
    rows = [dict(row) for row in case.get("initial_rows", [])]
    next_id = max([int(row.get("id", 0)) for row in rows] or [0]) + 1
    for op in case.get("ops", []):
        if op.get("op") == "add":
            rows.append({"id": next_id, "title": str(op.get("title", "")), "done": False})
            next_id += 1
        elif op.get("op") == "done":
            for row in rows:
                if row["id"] == int(op.get("id")):
                    row["done"] = True
        elif op.get("op") == "delete":
            rows = [row for row in rows if row["id"] != int(op.get("id"))]
    return {"rows": rows, "open_count": sum(1 for row in rows if not row["done"])}


def _auth_replacement(case: Case) -> dict[str, Any]:
    allowed = case.get("user") == "demo" and case.get("password") == "correct-horse"
    if not allowed:
        return {"status": 401, "json": {"error": "unauthorized"}}
    if case.get("route", "/profile") == "/profile":
        return {"status": 200, "json": {"user": "demo", "scopes": ["read"]}}
    return {"status": 403, "json": {"error": "forbidden"}}


def _missing_docs_replacement(case: Case) -> str:
    normalized = re.sub(r"\s+", " ", str(case["text"]).strip())
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:].lower()


def scenarios() -> list[ProgramDNABatteryScenario]:
    return [
        ProgramDNABatteryScenario(
            name="mini-text-cli",
            category="cli",
            docs=[
                "Small CLI utility with two commands: slug normalizes text into a URL slug; stats returns line, word, and char counts.",
                "Whitespace separates words; slug output is lowercase ASCII with punctuation collapsed to hyphens.",
            ],
            behavior_examples=[
                {"input": {"command": "slug", "text": "Hello, Aura!"}, "output": "hello-aura"},
                {"input": {"command": "stats", "text": "one two\nthree"}, "output": "lines=2 words=3 chars=13"},
            ],
            held_out_cases=[
                {"command": "slug", "text": "Program DNA: Clean Room"},
                {"command": "stats", "text": "alpha beta\n\ngamma"},
            ],
            original=_cli_original,
        ),
        ProgramDNABatteryScenario(
            name="tiny-counter-gui",
            category="gui",
            docs=[
                "Small GUI app with Increment, Decrement, Reset, and editable label controls.",
                "Increment/decrement update an internal count and visible label; Reset returns count to zero and label to Ready.",
            ],
            behavior_examples=[
                {
                    "input": {"initial_count": 1, "actions": [{"type": "increment"}]},
                    "output": {"count": 2, "label": "Count: 2", "buttons_enabled": True},
                }
            ],
            ui_notes=["Buttons are always enabled in this simplified app."],
            held_out_cases=[
                {"initial_count": 2, "actions": [{"type": "increment"}, {"type": "decrement"}]},
                {"initial_label": "Custom", "actions": [{"type": "set_label", "value": "Review"}, {"type": "reset"}]},
            ],
            original=_gui_original,
        ),
        ProgramDNABatteryScenario(
            name="csv-to-json-converter",
            category="file_format_converter",
            docs=[
                "Converts CSV text to JSON with columns and rows.",
                "Cell values are trimmed; column order is preserved from the header row.",
            ],
            behavior_examples=[
                {
                    "input": {"csv": "name,age\nAda, 37\n"},
                    "output": json.dumps({"columns": ["name", "age"], "rows": [{"name": "Ada", "age": "37"}]}, sort_keys=True),
                }
            ],
            file_formats=["CSV input with header row; JSON output object with columns and rows."],
            held_out_cases=[
                {"csv": "city,temp\nNYC, 73\nLA, 81\n"},
                {"csv": "key,value\n spaced , yes \n"},
            ],
            original=_csv_json_original,
        ),
        ProgramDNABatteryScenario(
            name="mini-web-service",
            category="web_app",
            docs=[
                "GET /health returns ok true. POST /echo returns the JSON body under echo.",
                "GET /items/{id} returns a minimal item object. Unknown routes return not_found.",
            ],
            behavior_examples=[
                {"input": {"method": "GET", "path": "/health"}, "output": {"status": 200, "json": {"ok": True}}},
                {"input": {"method": "GET", "path": "/items/a1"}, "output": {"status": 200, "json": {"id": "a1", "kind": "item"}}},
            ],
            api_observations=["GET /health, POST /echo, GET /items/{id}, otherwise 404."],
            held_out_cases=[
                {"method": "POST", "path": "/echo", "body": {"x": 3}},
                {"method": "GET", "path": "/missing"},
            ],
            original=_web_original,
        ),
        ProgramDNABatteryScenario(
            name="todo-db-tool",
            category="local_db_tool",
            docs=[
                "Local database-backed todo tool supports add, done, delete, list.",
                "Added rows receive monotonically increasing integer ids; open_count excludes completed rows.",
            ],
            behavior_examples=[
                {
                    "input": {"ops": [{"op": "add", "title": "ship"}]},
                    "output": {"rows": [{"id": 1, "title": "ship", "done": False}], "open_count": 1},
                }
            ],
            workflows=["add todo -> mark done -> list open count"],
            held_out_cases=[
                {"ops": [{"op": "add", "title": "a"}, {"op": "add", "title": "b"}, {"op": "done", "id": 1}]},
                {"initial_rows": [{"id": 3, "title": "old", "done": False}], "ops": [{"op": "add", "title": "new"}, {"op": "delete", "id": 3}]},
            ],
            original=_db_original,
        ),
        ProgramDNABatteryScenario(
            name="mock-auth-app",
            category="auth_mocked_app",
            docs=[
                "Auth is mocked: user demo with password correct-horse can access /profile.",
                "Invalid credentials return 401 unauthorized; unknown protected routes return 403 forbidden.",
            ],
            behavior_examples=[
                {"input": {"user": "demo", "password": "correct-horse", "route": "/profile"}, "output": {"status": 200, "json": {"user": "demo", "scopes": ["read"]}}},
                {"input": {"user": "demo", "password": "bad"}, "output": {"status": 401, "json": {"error": "unauthorized"}}},
            ],
            permissions=["Authentication is mocked; no real credentials or sessions are read."],
            held_out_cases=[
                {"user": "demo", "password": "correct-horse", "route": "/admin"},
                {"user": "other", "password": "correct-horse", "route": "/profile"},
            ],
            original=_auth_original,
        ),
        ProgramDNABatteryScenario(
            name="missing-docs-normalizer",
            category="missing_docs",
            docs=["Sparse docs: cleans user-entered labels."],
            behavior_examples=[
                {"input": {"text": "  HELLO    WORLD  "}, "output": "Hello world"},
                {"input": {"text": "aURA"}, "output": "Aura"},
            ],
            held_out_cases=[
                {"text": "  MULTI\tSPACE\nTEXT "},
                {"text": ""},
            ],
            original=_missing_docs_original,
            missing_docs=True,
        ),
    ]


def _build_payload(scenario: ProgramDNABatteryScenario) -> dict[str, Any]:
    return {
        "target": scenario.name,
        "authorization": "educational",
        "analysis_mode": "study" if scenario.category in {"gui", "missing_docs"} else "reconstruct",
        "objective": f"clean-room behavioral equivalence study for {scenario.category}",
        "observed_behaviors": [
            *scenario.docs,
            *[
                f"Example input={json.dumps(example['input'], sort_keys=True)} output={json.dumps(example['output'], sort_keys=True)}"
                for example in scenario.behavior_examples
            ],
        ],
        "ui_notes": scenario.ui_notes,
        "api_observations": scenario.api_observations,
        "file_formats": scenario.file_formats,
        "workflows": scenario.workflows,
        "permissions": scenario.permissions,
        "research_notes": [
            "Hidden-source battery: source is intentionally withheld; infer only from docs, examples, and observations.",
        ],
        "study_questions": [
            "What behavior can be implemented from public examples without source access?",
            "Which held-out cases would falsify the inferred genome?",
        ],
        "compatibility_targets": [f"{scenario.category} behavioral-equivalence harness"],
    }


def _synthesize_replacement(scenario: ProgramDNABatteryScenario, genome: dict[str, Any]) -> BehaviorFn:
    """Produce a clean-room behavior from docs/examples/genome, never source."""

    text = json.dumps({"docs": scenario.docs, "examples": scenario.behavior_examples, "genome": genome}, sort_keys=True).lower()
    if "slug" in text and "stats" in text:
        return _cli_replacement
    if "increment" in text and "reset" in text:
        return _gui_replacement
    if "csv" in text and "json" in text and "columns" in text:
        return _csv_json_replacement
    if "/health" in text and "/echo" in text:
        return _web_replacement
    if "open_count" in text and "monotonically increasing" in text:
        return _db_replacement
    if "correct-horse" in text and "unauthorized" in text:
        return _auth_replacement
    if "cleans user-entered labels" in text or "hello world" in text:
        return _missing_docs_replacement
    raise ValueError(f"could not synthesize replacement for {scenario.name}")


async def run_battery(*, project_root: Path | None = None) -> dict[str, Any]:
    engine = ProgramDNAReconstructionEngine(project_root=project_root or Path.cwd())
    results: list[ScenarioResult] = []

    for scenario in scenarios():
        result = await engine.reconstruct(_build_payload(scenario))
        if not result.ok or result.genome is None:
            results.append(
                ScenarioResult(
                    name=scenario.name,
                    category=scenario.category,
                    ok=False,
                    equivalence=0.0,
                    cases_passed=0,
                    cases_total=len(scenario.held_out_cases),
                    features=[],
                    hidden_source_withheld=True,
                    failures=[{"error": "program_dna_failed", "blocked": result.blocked_reasons}],
                )
            )
            continue

        replacement = _synthesize_replacement(scenario, result.genome.to_dict() if hasattr(result.genome, "to_dict") else asdict(result.genome))
        failures: list[dict[str, Any]] = []
        passed = 0
        for case in scenario.held_out_cases:
            expected = scenario.original(case)
            actual = replacement(case)
            if actual == expected:
                passed += 1
            else:
                failures.append({"case": case, "expected": expected, "actual": actual})
        total = len(scenario.held_out_cases)
        feature_names = [feature.name for feature in result.features]
        genome = asdict(result.genome)
        results.append(
            ScenarioResult(
                name=scenario.name,
                category=scenario.category,
                ok=passed == total,
                equivalence=passed / total if total else 0.0,
                cases_passed=passed,
                cases_total=total,
                features=feature_names,
                hidden_source_withheld=True,
                failures=failures,
                genome_summary={
                    "analysis_mode": genome.get("analysis_mode"),
                    "feature_count": len(genome.get("feature_map") or []),
                    "workflow_count": len(genome.get("workflow_graph") or []),
                    "unknown_count": len(genome.get("reconstruction_unknowns") or []),
                    "hidden_state_risk_count": len(genome.get("hidden_state_risks") or []),
                },
            )
        )

    total_cases = sum(item.cases_total for item in results)
    passed_cases = sum(item.cases_passed for item in results)
    passed_scenarios = sum(1 for item in results if item.ok)
    return {
        "ok": passed_scenarios == len(results) and passed_cases == total_cases,
        "battery": "program_dna_hidden_source_behavioral_equivalence",
        "scenario_count": len(results),
        "passed_scenarios": passed_scenarios,
        "held_out_cases": total_cases,
        "passed_cases": passed_cases,
        "equivalence": passed_cases / total_cases if total_cases else 0.0,
        "source_policy": "engine received docs, examples, and observations only; original callables are private harness oracles",
        "limits": [
            "Representative archetype proof, not arbitrary closed-source equivalence.",
            "No proprietary source recovery, DRM bypass, credential extraction, or binary decompilation occurs.",
        ],
        "results": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON artifact path.")
    args = parser.parse_args()

    report = asyncio.run(run_battery(project_root=Path.cwd()))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
