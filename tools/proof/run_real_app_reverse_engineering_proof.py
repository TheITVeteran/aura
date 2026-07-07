#!/usr/bin/env python3
"""Live proof: Aura reverse-engineers a REAL program on this machine.

Unlike the synthetic behavioral-equivalence battery, this targets an ACTUAL
binary installed on the host (e.g. /usr/bin/base64) that Aura has NO source
for. The pipeline is lawful, behavior-only reverse engineering:

  1. OBSERVE   — run the real binary on training inputs, capture its outputs
  2. SPECIFY   — read the observable spec (its `man` page) — no source, no
                 decompilation
  3. RECONSTRUCT — ProgramDNAReconstructionEngine.reconstruct_executable_via_cognition
                 asks the live 32B to reimplement the behavior from the spec +
                 examples ONLY
  4. VERIFY    — a sandbox that genuinely fails wrong code differentially checks
                 the reconstruction against HELD-OUT real-binary outputs the
                 model never saw

The reported number is honest reconstruction coverage against the real program.

Two modes:
  --self-test   proves the PIPELINE is real with NO model: a known-correct
                clean-room reimpl passes the held-out differential and a
                deliberately-wrong one is caught. This is what makes the harness
                trustworthy and is runnable offline / in CI.
  (default)     runs the genuine capability; requires the live model. Skips the
                cognition step honestly (status=conjecture) if no model router
                is registered, so it never fabricates a result.

    python tools/proof/run_real_app_reverse_engineering_proof.py --self-test
    python tools/proof/run_real_app_reverse_engineering_proof.py --target base64 \
        --out artifacts/live_proof/real_reverse_engineering.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64 as _b64
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class RealTarget:
    """A real, user-owned host binary reverse-engineered from behavior only."""

    name: str
    binary: str                      # resolved via shutil.which
    man_topic: str
    argv: Callable[[str], list[str]]  # argv for a given stdin payload
    fn_name: str
    train_inputs: list[str]
    held_out_inputs: list[str]
    # a correct clean-room reimplementation, used ONLY by --self-test to prove
    # the differential can pass; the live proof never sees this.
    reference_impl: str
    # a deliberately-wrong impl, used ONLY by --self-test to prove the
    # differential can fail (a harness that cannot fail proves nothing).
    broken_impl: str
    case_key: str = "text"


def _b64_targets() -> RealTarget:
    return RealTarget(
        name="base64",
        binary="base64",
        man_topic="base64",
        argv=lambda _payload: [],  # reads stdin, encodes to stdout
        fn_name="reconstructed",
        train_inputs=["hello", "Aura", "", "a", "The quick brown fox."],
        held_out_inputs=["reverse-engineered", "1234567890", "\n", "unit test", "Zenith"],
        reference_impl=(
            "import base64\n"
            "def reconstructed(case):\n"
            "    return base64.b64encode(case['text'].encode()).decode() + '\\n'\n"
        ),
        broken_impl=(
            "import base64\n"
            "def reconstructed(case):\n"
            "    # wrong: forgets the trailing newline the real binary emits\n"
            "    return base64.b64encode(case['text'].encode()).decode()\n"
        ),
    )


def _rev_targets() -> RealTarget:
    return RealTarget(
        name="rev",
        binary="rev",
        man_topic="rev",
        argv=lambda _payload: [],  # reverses each line of stdin
        fn_name="reconstructed",
        train_inputs=["hello", "abc", "racecar", "Aura", "123"],
        held_out_inputs=["reverse", "xyz", "level", "Zenith", "9876"],
        reference_impl=(
            "def reconstructed(case):\n"
            "    return case['text'][::-1] + '\\n'\n"
        ),
        broken_impl=(
            "def reconstructed(case):\n"
            "    return case['text'] + '\\n'  # wrong: does not reverse\n"
        ),
    )


def _md5_targets() -> RealTarget:
    return RealTarget(
        name="md5",
        binary="md5",
        man_topic="md5",
        argv=lambda _payload: ["-q"],  # -q: quiet, hash of stdin only
        fn_name="reconstructed",
        train_inputs=["hello", "Aura", "abc", "The quick brown fox.", "12345"],
        held_out_inputs=["reverse-engineered", "Zenith", "held-out", "sandbox", "67890"],
        reference_impl=(
            "import hashlib\n"
            "def reconstructed(case):\n"
            "    return hashlib.md5(case['text'].encode()).hexdigest() + '\\n'\n"
        ),
        broken_impl=(
            "import hashlib\n"
            "def reconstructed(case):\n"
            "    return hashlib.sha1(case['text'].encode()).hexdigest() + '\\n'  # wrong algorithm\n"
        ),
    )


_TARGETS: dict[str, Callable[[], RealTarget]] = {
    "base64": _b64_targets,
    "rev": _rev_targets,
    "md5": _md5_targets,
}


def _observe(target: RealTarget, payload: str) -> str:
    """Run the REAL binary on a payload and capture its exact stdout."""
    binary = shutil.which(target.binary)
    if not binary:
        raise FileNotFoundError(f"real binary not found on host: {target.binary}")
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    completed = get_subprocess_gateway().run(
        [binary, *target.argv(payload)],
        input=payload,
        capture_output=True,
        timeout=10,
        check=False,
        offline_tooling=True,
        source="proof_tooling:real_app_reverse_engineering.observe",
    )
    return completed.stdout


def _read_man(topic: str, limit: int = 60) -> str:
    """The observable specification: the program's own man page (no source)."""
    man = shutil.which("man")
    if not man:
        return f"(man unavailable) reconstruct the observable behavior of `{topic}`."
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        out = get_subprocess_gateway().run(
            [man, topic],
            capture_output=True,
            timeout=10,
            check=False,
            env={"MANPAGER": "cat", "PAGER": "cat", "MANWIDTH": "80", "PATH": "/usr/bin:/bin"},
            offline_tooling=True,
            source="proof_tooling:real_app_reverse_engineering.read_man",
        )
        text = out.stdout
        # strip backspace-overstrike bolding man emits
        text = text.replace("\x08", "")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[:limit])[:4000]
    except (OSError, subprocess.SubprocessError):
        return f"reconstruct the observable behavior of `{topic}`."


def _held_out(target: RealTarget) -> list[dict[str, Any]]:
    return [
        {"input": {target.case_key: payload}, "expected": _observe(target, payload)}
        for payload in target.held_out_inputs
    ]


def _train_examples(target: RealTarget) -> list[dict[str, Any]]:
    return [
        {"input": {target.case_key: payload}, "output": _observe(target, payload)}
        for payload in target.train_inputs
    ]


def _spec_docs(target: RealTarget) -> list[str]:
    return [
        f"Reconstruct the observable stdout behavior of the `{target.name}` command.",
        f"The function receives one dict argument with key '{target.case_key}' (the stdin payload) "
        f"and must return the EXACT stdout the real program produces, including trailing newlines.",
        "Specification (the program's own man page — observable, not source):",
        _read_man(target.man_topic),
    ]


def _evaluate(code: str, fn_name: str, held_out: list[dict[str, Any]]) -> dict[str, Any]:
    """Differential check against held-out real-binary outputs using the same
    general-capability sandbox the engine uses for realistic reconstruction."""
    from core.discovery.reconstruction_sandbox import GeneralReconstructionEvaluator

    evaluator = GeneralReconstructionEvaluator(timeout_seconds=5.0)
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in held_out:
        expected = case.get("expected")
        inp = case.get("input", case)
        evaluation = evaluator.evaluate(code, fn_name, [((inp,), expected)])
        if evaluation.outcome == "passed" and evaluation.passed == 1:
            passed += 1
        else:
            failures.append({"input": inp, "expected": expected, "outcome": evaluation.outcome})
    total = len(held_out)
    return {
        "held_out_passed": passed,
        "held_out_total": total,
        "equivalence": (passed / total) if total else 0.0,
        "status": "supported" if (total and passed == total) else "refuted",
        "failures": failures[:10],
    }


def run_self_test(target: RealTarget) -> dict[str, Any]:
    """Prove the harness is real: correct reimpl passes, wrong reimpl fails —
    both against HELD-OUT outputs observed from the real binary. No model."""
    held_out = _held_out(target)
    correct = _evaluate(target.reference_impl, target.fn_name, held_out)
    broken = _evaluate(target.broken_impl, target.fn_name, held_out)
    ok = correct["status"] == "supported" and broken["status"] == "refuted"
    return {
        "mode": "self_test",
        "target": target.name,
        "binary": shutil.which(target.binary) or target.binary,
        "held_out_cases": len(held_out),
        "correct_reimpl": correct,
        "broken_reimpl": broken,
        "pipeline_proven": ok,
        "meaning": (
            "the reverse-engineering differential genuinely distinguishes a correct clean-room "
            "reimplementation from a wrong one, verified against the REAL binary's held-out outputs"
        ),
    }


def _router_registered() -> bool:
    try:
        from core.container import ServiceContainer
    except (ImportError, RuntimeError):
        return False
    for name in ("inference_gate", "llm_router", "cognitive_engine"):
        try:
            if ServiceContainer.get(name, default=None) is not None:
                return True
        except (AttributeError, RuntimeError):
            continue
    return False


async def run_live(target: RealTarget, *, out: Path | None = None) -> dict[str, Any]:
    from core.self_improvement.program_dna import ProgramDNAReconstructionEngine

    engine = ProgramDNAReconstructionEngine(project_root=REPO_ROOT)
    held_out = _held_out(target)
    train = _train_examples(target)
    router = _router_registered()

    if router:
        outcome = await engine.reconstruct_executable_via_cognition(
            target=f"real:{target.name}",
            spec_docs=_spec_docs(target),
            train_examples=train,
            held_out=held_out,
            fn_name=target.fn_name,
            authorization="host_observation",
            objective=f"clean-room reconstruction of the host command `{target.name}` from behavior",
        )
    else:
        outcome = {
            "status": "conjecture",
            "held_out_passed": 0,
            "held_out_total": len(held_out),
            "equivalence": 0.0,
            "reason": "no_llm_router_registered (run inside the live instance, or with the model attached)",
        }

    report = {
        "mode": "live_reverse_engineering",
        "target": f"real:{target.name}",
        "binary": shutil.which(target.binary) or target.binary,
        "policy": "behavior-only: observed I/O + man page; NO source, NO decompilation",
        "router_available": router,
        "status": outcome.get("status"),
        "reason": outcome.get("reason", ""),
        "held_out_passed": outcome.get("held_out_passed", 0),
        "held_out_total": outcome.get("held_out_total", len(held_out)),
        "equivalence": outcome.get("equivalence", 0.0),
        "failures": outcome.get("failures", []),
        "reconstructed_code": outcome.get("code", ""),
        "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="base64", choices=sorted(_TARGETS))
    parser.add_argument("--self-test", action="store_true", help="prove the pipeline offline (no model)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    target = _TARGETS[args.target]()
    if args.self_test:
        report = run_self_test(target)
        print(json.dumps(report, indent=2))
        return 0 if report["pipeline_proven"] else 1

    report = asyncio.run(run_live(target, out=args.out))
    print(json.dumps(report, indent=2))
    # coverage is a measurement, not a gate; the run succeeds if it produced
    # honest numbers.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
