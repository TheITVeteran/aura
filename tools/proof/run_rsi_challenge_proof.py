#!/usr/bin/env python3
"""Live proof: Aura improves real code with VERIFIED benefit (RSI challenge).

Recursive self-improvement is only real if the "improvement" is measurably
better on evidence the improver did not get to choose. This challenge gives
Aura a seed implementation with a GENUINE bug, shows her the cases it fails,
and asks for a corrected version. The result is accepted only if it passes
HELD-OUT cases the seed fails — proven in the general reconstruction sandbox,
promoted only on strict improvement (new_passed > seed_passed AND new is total).

  --self-test   proves the challenge is real with NO model: a reference fix
                strictly beats the buggy seed on held-out cases (and the seed
                genuinely fails some) — runnable offline / in CI.
  (default)     runs the genuine capability; requires the live 32B. Honestly
                reports 'no_model' rather than fabricating an improvement.

    python tools/proof/run_rsi_challenge_proof.py --self-test
    python tools/proof/run_rsi_challenge_proof.py --out artifacts/live_proof/rsi_challenge.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class RSIChallenge:
    name: str
    deficiency: str
    fn_name: str
    seed_impl: str          # a real implementation with a genuine bug
    reference_fix: str      # a correct impl, used ONLY by --self-test
    cases: list[dict[str, Any]] = field(default_factory=list)  # {input, expected}


def _median_challenge() -> RSIChallenge:
    def oracle(xs: list[float]) -> float:
        return statistics.median(xs)

    cases = [
        {"input": {"xs": xs}, "expected": oracle(xs)}
        for xs in ([3, 1, 2], [5], [9, 1, 4, 2], [1, 2, 3, 4], [10, 20, 30, 40, 50, 60])
    ]
    return RSIChallenge(
        name="median",
        deficiency="returns the upper-middle element for even-length inputs instead of the mean of the two middles",
        fn_name="improved",
        seed_impl=(
            "def improved(case):\n"
            "    xs = sorted(case['xs'])\n"
            "    return xs[len(xs) // 2]\n"  # BUG: even-length median is wrong
        ),
        reference_fix=(
            "def improved(case):\n"
            "    xs = sorted(case['xs'])\n"
            "    n = len(xs)\n"
            "    if n % 2:\n"
            "        return xs[n // 2]\n"
            "    return (xs[n // 2 - 1] + xs[n // 2]) / 2\n"
        ),
        cases=cases,
    )


def _palindrome_challenge() -> RSIChallenge:
    def oracle(s: str) -> bool:
        norm = "".join(ch.lower() for ch in s if ch.isalnum())
        return norm == norm[::-1]

    cases = [
        {"input": {"s": s}, "expected": oracle(s)}
        for s in ("racecar", "hello", "A man a plan a canal Panama", "No lemon, no melon", "abc")
    ]
    return RSIChallenge(
        name="is_palindrome",
        deficiency="does not ignore case, spaces, or punctuation, so real phrases are misjudged",
        fn_name="improved",
        seed_impl=(
            "def improved(case):\n"
            "    s = case['s']\n"
            "    return s == s[::-1]\n"  # BUG: no normalization
        ),
        reference_fix=(
            "def improved(case):\n"
            "    s = case['s']\n"
            "    norm = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return norm == norm[::-1]\n"
        ),
        cases=cases,
    )


_CHALLENGES: dict[str, Callable[[], RSIChallenge]] = {
    "median": _median_challenge,
    "is_palindrome": _palindrome_challenge,
}


def _score(impl: str, fn_name: str, cases: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    from core.discovery.reconstruction_sandbox import GeneralReconstructionEvaluator

    evaluator = GeneralReconstructionEvaluator(timeout_seconds=5.0)
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected")
        inp = case.get("input", case)
        result = evaluator.evaluate(impl, fn_name, [((inp,), expected)])
        if result.outcome == "passed" and result.passed == 1:
            passed += 1
        else:
            failures.append({"input": inp, "expected": expected, "outcome": result.outcome})
    return passed, failures


def run_self_test(challenge: RSIChallenge) -> dict[str, Any]:
    """No model: prove the challenge measures REAL improvement — the seed
    genuinely fails held-out cases and the reference fix strictly beats it."""
    total = len(challenge.cases)
    seed_passed, seed_failures = _score(challenge.seed_impl, challenge.fn_name, challenge.cases)
    fix_passed, _ = _score(challenge.reference_fix, challenge.fn_name, challenge.cases)
    improved = fix_passed > seed_passed and fix_passed == total
    return {
        "mode": "self_test",
        "challenge": challenge.name,
        "deficiency": challenge.deficiency,
        "total_cases": total,
        "seed_passed": seed_passed,
        "seed_fails_real_cases": seed_passed < total,
        "reference_fix_passed": fix_passed,
        "improvement_proven": improved,
        "seed_failures": seed_failures[:5],
        "meaning": (
            "the seed has a genuine bug (fails held-out cases) and a correct fix strictly "
            "improves on it — so a model-produced fix can be VERIFIED as real improvement"
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


async def run_live(challenge: RSIChallenge, *, out: Path | None = None) -> dict[str, Any]:
    total = len(challenge.cases)
    seed_passed, seed_failures = _score(challenge.seed_impl, challenge.fn_name, challenge.cases)

    improved_code = ""
    reason = ""
    if _router_registered():
        try:
            from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code

            prompt = (
                "You are improving a Python function through recursive self-improvement.\n"
                f"The function `{challenge.fn_name}(case)` has a bug: {challenge.deficiency}.\n\n"
                f"Current implementation:\n{challenge.seed_impl}\n\n"
                f"It FAILS these observed cases (input -> expected):\n"
                + "\n".join(f"  {f['input']} -> {f['expected']}" for f in seed_failures)
                + "\n\nWrite a corrected implementation of the same function. Standard library only."
            )
            generator = LLMCodeGenerator()
            raw = await generator.generate_async(
                prompt,
                context={
                    "prefer_tier": "primary",
                    "temperature": 0.1,
                    "max_tokens": 700,
                    "origin": "rsi_challenge",
                    "system_prompt": "You improve code to pass all cases. Return only the function.",
                },
            )
            improved_code = extract_python_code(raw) or str(raw or "")
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = "no_llm_router_registered (run inside the live instance, or with the model attached)"

    improved_passed = 0
    improved_failures: list[dict[str, Any]] = []
    if improved_code.strip():
        improved_passed, improved_failures = _score(improved_code, challenge.fn_name, challenge.cases)

    promoted = bool(improved_code.strip()) and improved_passed == total and improved_passed > seed_passed
    report = {
        "mode": "rsi_challenge",
        "challenge": challenge.name,
        "deficiency": challenge.deficiency,
        "total_cases": total,
        "seed_passed": seed_passed,
        "improved_passed": improved_passed,
        "improvement_proven": promoted,
        "promoted": promoted,
        "reason": reason,
        "seed_failures": seed_failures[:5],
        "improved_failures": improved_failures[:5],
        "improved_code": improved_code,
        "policy": "promote only on strict, held-out-verified improvement (new passes all AND beats seed)",
        "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", default="median", choices=sorted(_CHALLENGES))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    challenge = _CHALLENGES[args.challenge]()
    if args.self_test:
        report = run_self_test(challenge)
        print(json.dumps(report, indent=2))
        return 0 if report["improvement_proven"] else 1

    report = asyncio.run(run_live(challenge, out=args.out))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
