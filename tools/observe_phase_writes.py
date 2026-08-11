#!/usr/bin/env python3
"""Measure what each cognitive phase actually writes, so contracts can be real.

`cognitive_provenance.write_profile` is documented as "the productive end of
the ratchet" — the tool that grounds the next contract in observation rather
than in reading a phase and guessing. It needs a running system to read from.

This runs the phases directly against a real AuraState, with no model and no
full runtime, and prints the observed write set for each. What it produces is
a floor, not a ceiling: a phase whose interesting branch needs an LLM will
report only what its no-model path touched, and that is said plainly in the
output rather than presented as the whole truth.

    python tools/observe_phase_writes.py
    python tools/observe_phase_writes.py --json out.json

Contracts written from this are grounded in what a phase DID. Fields it can
write only on a path not exercised here still have to be read out of the
code — and the receipt check will catch the omission the first time that
branch runs, which is the point of declaring rather than documenting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AURA_TESTING", "1")


def _kernel_constructed_phases() -> dict[str, type]:
    """`self.<attr> = <PhaseClass>(self)` lines from the kernel's own source."""
    import ast
    import importlib
    import re

    source = (ROOT / "core" / "kernel" / "aura_kernel.py").read_text("utf-8")
    tree = ast.parse(source)
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module

    found: dict[str, type] = {}
    for attribute, class_name in re.findall(
        r"self\.(\w+)\s*=\s*(\w+Phase|\w+Engine|\w+Bridge|\w+Loop|\w+Tools?)\(self\)", source
    ):
        module_name = imported.get(class_name)
        if not module_name:
            continue
        try:
            found[attribute] = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
    return found


class _KernelStub:
    """Enough of a kernel for a constructor that only stores a reference."""

    def __init__(self) -> None:
        from core.container import ServiceContainer as _Container

        self.container = _Container
        self.state = None


async def _observe() -> dict[str, dict]:
    from core.container import ServiceContainer
    from core.runtime import pipeline_blueprint as blueprint
    from core.runtime.cognitive_provenance import begin_transformation
    from core.runtime.pipeline_blueprint import (
        kernel_phase_attribute_order,
        phase_class_for_attribute,
    )
    from core.state.aura_state import AuraState

    # The blueprint's own specs, so this probes the pipeline the runtime
    # builds rather than a second list that would drift from it.
    by_attribute = {
        spec.attribute_name: spec.phase_cls
        for spec in (*blueprint._LEGACY_PIPELINE_PREFIX, *blueprint._LEGACY_PIPELINE_SUFFIX)
    }
    # The kernel constructs the rest directly. Read the classes off its own
    # source rather than keeping a second table here — a table would be one
    # more copy to go stale, which is the failure this whole subsystem is
    # about.
    by_attribute.update(_kernel_constructed_phases())
    results: dict[str, dict] = {}

    for attribute in kernel_phase_attribute_order():
        phase_name = phase_class_for_attribute(attribute)
        phase_cls = by_attribute.get(attribute)
        if phase_cls is None:
            results[phase_name] = {"status": "unresolved", "observed_writes": []}
            continue

        # Phases take the container, the kernel, or nothing. Try each rather
        # than hard-coding a table that would drift from the constructors.
        phase = None
        construction_error = ""
        for args in ((), (ServiceContainer,), (_KernelStub(),)):
            try:
                phase = phase_cls(*args)
                break
            except Exception as exc:  # noqa: BLE001 — record, do not abort the sweep
                construction_error = f"{type(exc).__name__}: {exc}"
        if phase is None:
            results[phase_name] = {
                "status": "uninstantiable",
                "error": construction_error,
                "observed_writes": [],
            }
            continue

        # A fresh state per phase: sharing one would attribute an earlier
        # phase's writes to a later one, which is the exact confusion these
        # receipts exist to remove.
        state = AuraState()
        state.cognition.working_memory = [
            {"role": "user", "content": "what are you doing right now"}
        ]

        transformation = begin_transformation(phase_name, state)
        error = ""
        try:
            result = await phase.execute(state, objective="observation probe")
            state = result if result is not None else state
        except Exception as exc:  # noqa: BLE001 — a probe records failures, it does not raise
            error = f"{type(exc).__name__}: {exc}"

        receipt = transformation.complete(state, error=error)
        results[phase_name] = {
            "status": "error" if error else "ran",
            "error": error,
            "observed_writes": list(receipt.observed_writes) if receipt else [],
            "contracted": bool(receipt.contract_hash) if receipt else False,
        }

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args(argv)

    results = asyncio.run(_observe())

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    ran = sum(1 for r in results.values() if r["status"] == "ran")
    wrote = sum(1 for r in results.values() if r["observed_writes"])
    print(f"phases probed: {len(results)}  ran: {ran}  wrote something: {wrote}\n")
    for name in sorted(results):
        entry = results[name]
        mark = "C" if entry.get("contracted") else " "
        writes = entry["observed_writes"]
        print(f"[{mark}] {name}: {entry['status']}")
        if writes:
            for path in writes:
                print(f"        {path}")
        if entry.get("error"):
            print(f"        ! {entry['error'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
