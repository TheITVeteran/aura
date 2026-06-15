#!/usr/bin/env python3
"""tools/agi/run_prompt_baseline_ablation.py — HONEST prompt-baseline ablation.

Replaces a previously FABRICATED benchmark (it loaded tasks but never ran them,
hardcoded the baseline scores 0.58/0.72/0.79, derived "Aura's score" from a
formula on a 2-boolean probe, then asserted victory and wrote
score_separation_verified=True). That is exactly the fake-passing this test is
meant to refute.

This version runs REAL multi-turn recall/continuity tasks through the live model
under three conditions that differ ONLY in the context the architecture assembles:

  - raw_model:        the final turn only — no system prompt, no history.
  - prompted_model:   a fixed identity system prompt + the final turn, no history.
  - full_architecture: the system prompt + the full conversation transcript
                       (the architecture's memory/context).

Outputs are graded objectively against answer keys; per-condition bootstrap CIs
and an honest verdict (architecture beats a baseline only when its lower CI
clears the baseline's upper CI) are written to the report. The verdict can be
False — that is a real result. If the model can't be reached, the report records
status="unavailable" and the tool exits non-zero rather than fabricating a pass.

NOTE on scope (honest): this ablates the memory/context contribution using one
shared model. It is NOT a full ablation of the will/volition system — that is a
separate measurement (#46 follow-up), and this tool no longer pretends to cover
it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Repo imports are resolved after PROJECT_ROOT is on sys.path.
# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.ablation_harness import (
    FULL,
    PROMPTED,
    RAW,
    AblationHarness,
    AblationTask,
    ConditionResult,
    grade,
)

_IDENTITY_PROMPT = (
    "You are Aura, a local cognitive assistant. Answer the user concisely and "
    "directly, using any relevant earlier context."
)

_GEN_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
    TimeoutError,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=str, default="tests/agi/fixtures/hidden_tasks/recall_tasks.jsonl"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/agi_live/prompt_baseline_ablation.json"
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def _load_tasks(path: Path) -> list[AblationTask]:
    tasks: list[AblationTask] = []
    if not path.exists():
        return tasks
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        turns = raw.get("turns")
        if not turns:
            # Single-prompt fixtures aren't multi-turn recall tasks; skip rather
            # than score something the ablation can't meaningfully separate.
            continue
        tasks.append(
            AblationTask(
                task_id=raw["task_id"],
                family=raw.get("family", "unknown"),
                turns=turns,
                answer_key=raw["answer_key"],
                grader=raw.get("grader", "recall_substring"),
            )
        )
    return tasks


def _build_prompt(condition: str, task: AblationTask, turn_index: int, history: list[str]) -> str:
    current = task.turns[turn_index]
    if condition == FULL and history:
        # history = [user, assistant, user, assistant, ...]
        lines = []
        for i, msg in enumerate(history):
            speaker = "User" if i % 2 == 0 else "Assistant"
            lines.append(f"{speaker}: {msg}")
        lines.append(f"User: {current}")
        lines.append("Assistant:")
        return "\n".join(lines)
    return current


async def _respond(router, condition: str, task: AblationTask, turn_index: int,
                   history: list[str], timeout_s: float) -> str:
    system = None if condition == RAW else _IDENTITY_PROMPT
    prompt = _build_prompt(condition, task, turn_index, history)
    meta = await router.generate_with_metadata(
        prompt=prompt,
        system_prompt=system,
        timeout=timeout_s,
        origin="ablation_harness",
        purpose="ablation",
        foreground_request=True,
    )
    if not isinstance(meta, dict) or not meta.get("ok"):
        raise RuntimeError(f"generation_failed: {meta if isinstance(meta, dict) else type(meta)}")
    return str(meta.get("text") or "")


async def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    await asyncio.to_thread(out_path.parent.mkdir, parents=True, exist_ok=True)

    tasks = await asyncio.to_thread(_load_tasks, Path(args.tasks))
    print(f"Loaded {len(tasks)} multi-turn ablation tasks.")
    if not tasks:
        await asyncio.to_thread(
            out_path.write_text, json.dumps({"status": "no_tasks", "tasks_evaluated": 0}, indent=2)
        )
        print("No multi-turn tasks to evaluate.")
        return 2

    try:
        from core.brain.llm_health_router import get_llm_router

        router = get_llm_router()
    except _GEN_RECOVERABLE_ERRORS as exc:
        await asyncio.to_thread(
            out_path.write_text,
            json.dumps({"status": "unavailable", "error": f"router_init: {exc}"}, indent=2),
        )
        print(f"Model router unavailable: {exc}")
        return 3

    harness = AblationHarness()
    results = {c: ConditionResult(condition=c) for c in harness.conditions}

    for condition in harness.conditions:
        print(f"\n── Condition: {condition} ──")
        for task in tasks:
            history: list[str] = []
            output = ""
            try:
                for turn_index in range(len(task.turns)):
                    output = await _respond(
                        router, condition, task, turn_index, history, timeout_s=args.timeout
                    )
                    history.append(task.turns[turn_index])
                    history.append(output)
            except _GEN_RECOVERABLE_ERRORS as exc:
                print(f"  {task.task_id}: generation error ({exc}) → scored 0.0")
                results[condition].per_task[task.task_id] = 0.0
                continue
            score = grade(output, task)
            results[condition].per_task[task.task_id] = score
            print(f"  {task.task_id}: {score:.2f}")

    report = harness.report_from_results(results, tasks_evaluated=len(tasks))
    report["status"] = "ok"
    report["task_fixture"] = str(args.tasks)

    # Back-compat keys for any consumer that read the old shape — but now with
    # REAL values (no fabrication). score_separation_verified reflects the honest
    # CI-separation verdict, whatever it is.
    cond = report["conditions"]
    report["baseline_scores"] = {
        "raw_model": _legacy_scores(cond.get(RAW)),
        "prompted_model": _legacy_scores(cond.get(PROMPTED)),
    }
    report["aura_scores"] = _legacy_scores(cond.get(FULL))
    report["score_separation_verified"] = bool(
        report["verdict"]["architecture_beats_stateless"]
    )

    await asyncio.to_thread(out_path.write_text, json.dumps(report, indent=2))
    verdict = report["verdict"]["architecture_beats_stateless"]
    print(f"\nVerdict — architecture beats stateless: {verdict}")
    print(f"Report saved to {out_path}")
    # Exit 0 when the measurement completed; the verdict is honest data in the
    # artifact, not the gate. (The live test asserts the verdict separately.)
    return 0


def _legacy_scores(cond_dict: dict | None) -> dict:
    if not cond_dict:
        return {"mean_score": 0.0, "lower_ci": 0.0, "upper_ci": 0.0}
    return {
        "mean_score": cond_dict["mean_score"],
        "lower_ci": cond_dict["lower_ci"],
        "upper_ci": cond_dict["upper_ci"],
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
