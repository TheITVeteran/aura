"""Reasoning-delta harness — does amplification make the small model punch up?

Measures the thesis directly: on verifiable hard problems, compare three
conditions and report the deltas:

    A. cortex_single   — the 32B cortex, one pass (the floor)
    B. cortex_amplified — the 32B cortex + reasoning amplifier (Φ-gated search +
                          verifiers + cache)   ← our system
    C. solver_single   — the 72B solver, one pass (the bar to beat)

The win condition is **B ≥ C**: amplification lets the small local model match or
beat a single pass of the much larger model. Scoring is objective — each answer is
graded by the same domain truth-engines the amplifier uses, so there is no judge to
game.

Safety: this module never loads a model on import and never auto-runs heavy
inference. ``--live`` is an explicit opt-in; everything else runs against injected
or deterministic generators so the wiring is testable without a GPU. The live run
is bounded (task cap, per-task timeout, wall-clock deadline) — see NO-UNBOUNDED.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ReasoningDelta")

GenerateFn = Callable[[str, float], Awaitable[str]]


@dataclass
class ReasoningTask:
    task_id: str
    prompt: str
    task_type: str  # "math" | "code" | "logic"
    gold: str | None = None  # optional exact-match reference


# A compact, verifiable suite. Truth-engines grade correctness; gold is a bonus
# exact-match signal. Extend freely — keep every task objectively checkable.
DEFAULT_TASKS: tuple[ReasoningTask, ...] = (
    ReasoningTask("math_mul", "Compute 137 * 248. Give only the number.", "math", "33976"),
    ReasoningTask("math_primes", "How many prime numbers are there below 100?", "math", "25"),
    ReasoningTask("math_word", "A train travels 60 km in 45 minutes. What is its speed in km/h?", "math", "80"),
    ReasoningTask(
        "code_fib",
        "Write a Python function fib(n) returning the n-th Fibonacci number (fib(0)=0, fib(1)=1).",
        "code",
    ),
    ReasoningTask(
        "code_palindrome",
        "Write a Python function is_palindrome(s) that ignores case and non-alphanumerics.",
        "code",
    ),
    ReasoningTask(
        "logic_knights",
        "On an island, knights always tell the truth and knaves always lie. A says 'B is a knave'. "
        "B says 'A and I are the same type'. What is A and what is B? Explain.",
        "logic",
    ),
)


@dataclass
class ConditionResult:
    name: str
    scores: list[float] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def mean_runtime(self) -> float:
        return sum(self.runtimes) / len(self.runtimes) if self.runtimes else 0.0


@dataclass
class DeltaReport:
    by_condition: dict[str, ConditionResult]

    @property
    def amplification_lift(self) -> float:
        a = self.by_condition.get("cortex_single")
        b = self.by_condition.get("cortex_amplified")
        return (b.mean_score - a.mean_score) if (a and b) else 0.0

    @property
    def small_vs_large(self) -> float:
        b = self.by_condition.get("cortex_amplified")
        c = self.by_condition.get("solver_single")
        return (b.mean_score - c.mean_score) if (b and c) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplification_lift": round(self.amplification_lift, 4),
            "small_amplified_vs_large_single": round(self.small_vs_large, 4),
            "by_condition": {
                name: {
                    "mean_score": round(r.mean_score, 4),
                    "mean_runtime_s": round(r.mean_runtime, 3),
                    "n": len(r.scores),
                }
                for name, r in self.by_condition.items()
            },
        }


async def _score_answer(answer: str, task: ReasoningTask) -> float:
    """Grade an answer with the same truth-engines the amplifier uses → [0,1]."""
    if not str(answer or "").strip():
        return 0.0
    score = 0.0
    try:
        from core.brain.verifiers import get_verifier_registry

        verdict = await get_verifier_registry().verify(
            answer, task_type=task.task_type, context={}
        )
        if getattr(verdict, "ok", False):
            score = 1.0
        else:
            score = float(getattr(verdict, "score", 0.0) or 0.0)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("verifier scoring failed for %s: %s", task.task_id, exc)
    # Gold exact-match is a hard ceiling-confirmation for closed-form answers.
    if task.gold and task.gold.strip() in str(answer):
        score = max(score, 1.0)
    return max(0.0, min(1.0, score))


async def _run_one(
    name: str,
    task: Any,
    runner: Callable[[Any], Awaitable[str]],
    *,
    per_task_timeout: float,
    score_fn: Callable[[str, Any], Awaitable[float]],
) -> tuple[float, float]:
    start = time.monotonic()
    try:
        answer = await asyncio.wait_for(runner(task), timeout=per_task_timeout)
    except (asyncio.TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("[%s] task %s failed/timed out: %s", name, task.task_id, exc)
        answer = ""
    score = await score_fn(answer, task)
    return score, time.monotonic() - start


async def run_delta(
    *,
    cortex_generate: GenerateFn,
    solver_generate: GenerateFn | None = None,
    tasks: tuple[Any, ...] = DEFAULT_TASKS,
    per_task_timeout: float = 60.0,
    wall_clock_deadline_s: float = 600.0,
    amplifier_time_budget_s: float = 45.0,
    grader: Callable[[Any, str], float] | None = None,
    skip_amplified: bool = False,
) -> DeltaReport:
    """Run the conditions over the suite and return a delta report.

    ``grader(task, answer) -> float`` overrides the default verifier-based scoring
    (use the hard-suite's external grader for an honest, independent grade).
    ``skip_amplified`` measures only single-pass (for a baseline run of a big model).
    Bounded: each task capped at ``per_task_timeout``; run stops at
    ``wall_clock_deadline_s`` (NO-UNBOUNDED).
    """
    from core.brain.reasoning_amplifier_v2 import amplify_turn, classify_task_type

    async def _score_fn(answer: str, task: Any) -> float:
        if grader is not None:
            return float(grader(task, answer))
        return await _score_answer(answer, task)

    start = time.monotonic()
    results: dict[str, ConditionResult] = {"cortex_single": ConditionResult("cortex_single")}
    if not skip_amplified:
        results["cortex_amplified"] = ConditionResult("cortex_amplified")
    if solver_generate is not None:
        results["solver_single"] = ConditionResult("solver_single")

    async def _cortex_single(task: ReasoningTask) -> str:
        return str(await cortex_generate(task.prompt, 0.3) or "").strip()

    async def _cortex_amplified(task: ReasoningTask) -> str:
        out = await amplify_turn(
            task.prompt,
            cortex_generate,
            task_type=task.task_type if task.task_type in {"code", "math"} else classify_task_type(task.prompt),
            time_budget_s=amplifier_time_budget_s,
            # Measurement must be read-only w.r.t. the live cache: never write (so a
            # bench run can't poison production) and never serve a prior cache hit.
            extra_context={"skip_cache": True},
        )
        return str(out.answer or "").strip()

    async def _solver_single(task: ReasoningTask) -> str:
        return str(await solver_generate(task.prompt, 0.3) or "").strip()  # type: ignore[misc]

    for task in tasks:
        if time.monotonic() - start > wall_clock_deadline_s:
            logger.warning("Wall-clock deadline hit — stopping after partial suite.")
            break
        for name, runner in (
            ("cortex_single", _cortex_single),
            ("cortex_amplified", None if skip_amplified else _cortex_amplified),
            ("solver_single", _solver_single if solver_generate is not None else None),
        ):
            if runner is None:
                continue
            # Code tasks (generate full functions + sandbox-repair) need more wall-clock
            # than numeric ones; give them a larger cap so they aren't false-zeroed.
            eff_timeout = per_task_timeout * (2.5 if getattr(task, "task_type", "") == "code" else 1.0)
            score, runtime = await _run_one(
                name, task, runner, per_task_timeout=eff_timeout, score_fn=_score_fn
            )
            results[name].scores.append(score)
            results[name].runtimes.append(runtime)
            logger.info("[%s] %s: score=%.2f (%.1fs)", name, task.task_id, score, runtime)

    return DeltaReport(by_condition=results)


# ── live wiring (explicit; never auto-runs on import) ───────────────────────
def make_mlx_lm_generator(model_path: str, *, max_tokens: int = 640) -> GenerateFn:
    """Build a standalone async generate fn over a local MLX model via mlx_lm.

    No app boot, no InferenceGate — loads the weights directly (the same pattern as
    benchmarks/reasoning). Only call this from an explicit --live run with verified
    memory headroom; each call loads one model copy into unified memory.
    """
    from mlx_lm import generate as _mlx_generate
    from mlx_lm import load as _mlx_load

    model, tokenizer = _mlx_load(model_path)

    def _sampler(temperature: float):
        try:
            from mlx_lm.sample_utils import make_sampler

            return make_sampler(temp=float(temperature))
        except (ImportError, TypeError, ValueError):
            return None

    async def _gen(prompt: str, temperature: float) -> str:
        def _run() -> str:
            try:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except (AttributeError, TypeError, ValueError):
                chat = prompt
            sampler = _sampler(temperature)
            kwargs: dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
            if sampler is not None:
                kwargs["sampler"] = sampler
            try:
                return str(_mlx_generate(model, tokenizer, prompt=chat, **kwargs) or "")
            except TypeError:
                # Older signature without sampler/verbose kwargs.
                return str(_mlx_generate(model, tokenizer, prompt=chat, max_tokens=max_tokens) or "")

        return await asyncio.to_thread(_run)

    return _gen


def _deterministic_generators() -> tuple[GenerateFn, GenerateFn]:
    """Stub generators for wiring/CI — no model. Returns canned, mostly-wrong drafts
    so the harness plumbing and scoring are exercised without a GPU."""

    async def _stub(prompt: str, temperature: float) -> str:
        return "I am not sure; here is a guess."

    return _stub, _stub


def main(argv: list[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(description="Aura reasoning-delta harness")
    parser.add_argument("--live", action="store_true", help="Load real MLX models via mlx_lm (explicit, heavy)")
    parser.add_argument("--model", type=str, default="models/Qwen2.5-7B-Instruct-4bit",
                        help="Cortex model path for --live (default: safe 7B)")
    parser.add_argument("--solver-model", type=str, default="",
                        help="Optional larger solver model path for the 72B comparison (omit to skip)")
    parser.add_argument("--suite", choices=["default", "hard"], default="default",
                        help="Task suite: 'hard' = base-failing tasks + sound external grader")
    parser.add_argument("--skip-amplified", action="store_true",
                        help="Only run single-pass (use for a big-model baseline run)")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of tasks (0 = all)")
    parser.add_argument("--per-task-timeout", type=float, default=120.0)
    parser.add_argument("--deadline", type=float, default=900.0, help="Wall-clock deadline (s)")
    parser.add_argument("--amp-budget", type=float, default=60.0, help="Amplifier per-turn time budget (s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    grader = None
    if args.suite == "hard":
        from aura_bench.hard_suite import HARD_TASKS, grade as hard_grade

        suite = HARD_TASKS
        grader = hard_grade
    else:
        suite = DEFAULT_TASKS
    tasks = suite[: args.limit] if args.limit > 0 else suite

    if args.live:
        logger.info("Loading cortex model: %s", args.model)
        cortex = make_mlx_lm_generator(args.model)
        solver = None
        if args.solver_model:
            logger.info("Loading solver model: %s", args.solver_model)
            solver = make_mlx_lm_generator(args.solver_model)
    else:
        logger.warning("Running with DETERMINISTIC STUB generators (no models). Use --live for real numbers.")
        cortex, solver = _deterministic_generators()

    report = asyncio.run(
        run_delta(
            cortex_generate=cortex,
            solver_generate=solver,
            tasks=tasks,
            per_task_timeout=args.per_task_timeout,
            wall_clock_deadline_s=args.deadline,
            amplifier_time_budget_s=args.amp_budget,
            grader=grader,
            skip_amplified=args.skip_amplified,
        )
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
