#!/usr/bin/env python3
"""Prove cognitive gates and executable skill discovery as one closure surface."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402


class Completed(Protocol):
    returncode: int
    stdout: str | None
    stderr: str | None


_COMPONENTS: tuple[tuple[str, tuple[str, ...], str | None, str], ...] = (
    (
        "cognitive_gate_inventory",
        ("tools/closeout/audit_cognitive_candidate_gates.py", "--json"),
        None,
        "passed",
    ),
    (
        "skill_catalog",
        ("tools/closeout/audit_skill_catalog.py", "--require-rust"),
        None,
        "ok",
    ),
    (
        "skill_runtime_route",
        ("tools/closeout/audit_skill_runtime_route.py",),
        "AURA_SKILL_RUNTIME_ROUTE_AUDIT=",
        "ok",
    ),
)


def _extract_report(stdout: str, *, prefix: str | None) -> dict[str, Any]:
    text = stdout.strip()
    if prefix is not None:
        matches = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {prefix.rstrip('=')} receipt")
        text = matches[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("child receipt was not a JSON object")
    return value


def _run_child(argv: Sequence[str]) -> Completed:
    return get_subprocess_gateway().run(
        [sys.executable, *argv],
        cwd=ROOT,
        timeout=300,
        offline_tooling=True,
        check=False,
        source="proof_tooling:audit_gate_skill_closure",
        accelerator_capability="none",
    )


def audit_gate_skill_closure(
    *,
    runner: Callable[[Sequence[str]], Completed] = _run_child,
) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name, argv, prefix, verdict_field in _COMPONENTS:
        completed = runner(argv)
        try:
            report = _extract_report(completed.stdout or "", prefix=prefix)
        except (json.JSONDecodeError, ValueError) as exc:
            report = {}
            failures.append(f"{name}:invalid_receipt:{exc}")
        if completed.returncode != 0:
            failures.append(f"{name}:exit_{completed.returncode}")
        if report.get(verdict_field) is not True:
            failures.append(f"{name}:verdict_not_pass")
        components[name] = {
            "receipt": report,
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-1000:],
        }

    cognitive = components["cognitive_gate_inventory"]["receipt"]
    catalog = components["skill_catalog"]["receipt"]
    route = components["skill_runtime_route"]["receipt"]
    invariants = {
        "all_gate_surfaces_classified": cognitive.get("declared_count")
        == cognitive.get("discovered_count"),
        "catalog_live_source_equal": catalog.get("accepted_count")
        == catalog.get("live_count"),
        "catalog_preflight_complete": catalog.get("preflight_complete") is True,
        "catalog_preflight_body_free": catalog.get("preflight_proves_skill_body_not_invoked")
        is True,
        "rust_python_parity": catalog.get("parity_status") == "matched",
        "runtime_route_executed": (route.get("execution") or {}).get("ok") is True,
        "runtime_authority_closed": (route.get("authority_closure") or {}).get("closed")
        is True,
    }
    failures.extend(name for name, passed in invariants.items() if not passed)
    return {
        "components": components,
        "failures": sorted(set(failures)),
        "invariants": invariants,
        "ok": not failures,
        "schema": "aura.gate_skill_closure_audit.v1",
    }


def main() -> int:
    report = audit_gate_skill_closure()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
