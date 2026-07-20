#!/usr/bin/env python
"""Exact-match accuracy ladder for the Recursive Latent Cortex (CP212).

Every prior measurement in this program used cross-entropy. CE falls when a
model learns the ``FINAL_ANSWER: {...}`` FORMAT, which is not reasoning.
The only metric that supports a reasoning claim is whether the emitted
answer is CORRECT, and whether correctness rises with recurrent depth.

This tool measures exact-match accuracy across a depth ladder, for one or
more arms, on tasks the training generators never produced, and writes a
self-describing receipt. It never awards a claim: it produces the numbers
a preregistered decision rule consumes.

Arms:
  base        frozen checkpoint, no adapter
  adapter     with a recurrence-native adapter attached

Success shape the program is looking for (NOT asserted here):
  * accuracy rises with depth on families where depth helps;
  * accuracy does not fall on families where depth hurts (shallow selected);
  * held-out template/depth generalization, not just held-out instances.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ACCURACY_SCHEMA = "aura.rlc_accuracy_ladder.v1"


def _load_tasks(families: list[str], task_depth: int, per_cell: int, seed: int):
    from core.learning import recurrence_curriculum as curriculum

    return curriculum.task_battery(families, [task_depth], per_cell, seed=seed)


def _render(tokenizer, task):
    prompt = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    return prompt


def _decode_answer(
    model, tokenizer, prepared, state, max_tokens: int, envelope=None
):
    """Greedy-decode an answer conditioned on the persisted slot state."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex.answer_contract import (
        is_contract_complete,
    )
    from core.learning.recurrence_native_objective_v2 import _persist_and_score

    # Teacher-forcing the produced tokens back through the SAME persisted
    # context is what makes this a real decode. An earlier version re-ran
    # the model on the generated tokens alone, dropping prompt and slot
    # context entirely, which produced 0% at every depth -- a broken
    # harness, not a model result.
    produced: list[int] = []
    text = ""
    for index in range(max_tokens):
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[0],
            state,
            _extend_tail(model, prepared, produced),
            bridge_count=prepared.bridge_count,
            answer_count=max(1, len(produced) + 1),
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        token = int(mx.argmax(logits[0, -1]))
        # Materialize and release before the next full forward. Each step
        # re-runs the whole stack over a GROWING sequence, so without this
        # the retained graphs are O(n^2) -- an unguarded version of this
        # loop drove the host to 103 GB and forced a shutdown.
        del logits
        if envelope is not None:
            envelope.reclaim(index)
        produced.append(token)
        text = tokenizer.decode(produced)
        if is_contract_complete(text):
            break
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    return text


def _extend_tail(model, prepared, produced: list[int]):
    """Tail embeddings = bridge + tokens generated so far."""
    import mlx.core as mx

    if not produced:
        return prepared.tail_embeddings
    generated = model.model.embed_tokens(mx.array([produced]))
    bridge = prepared.tail_embeddings[:, : prepared.bridge_count, :]
    return mx.concatenate([bridge, generated], axis=1)


class HarnessError(RuntimeError):
    """The harness could not evaluate — never a model result."""


def _score(task, text: str) -> str:
    """Four-way outcome. Harness faults RAISE; they are never 'incorrect'.

    An earlier version wrapped scoring in ``except Exception: return
    False``. RecurrenceTrainingTask has no ``.score()`` method, so every
    task raised AttributeError and was recorded as a wrong answer: the
    harness manufactured 0% accuracy at every depth and it looked like a
    devastating capability result. Silent conversion of harness faults into
    model failures is the one error class an evidence tool must never have.
    """
    from core.brain.llm.latent_cortex.frontier_tasks import (
        FrontierTaskError,
        parse_final_answer,
    )

    try:
        expected = parse_final_answer(task.answer)
    except FrontierTaskError as exc:  # gold must always parse
        raise HarnessError(
            f"gold answer for {task.family} is unparseable: {exc}"
        ) from exc
    try:
        produced = parse_final_answer(text)
    except FrontierTaskError:
        # Strict contract not met. Before calling this a failure, check
        # whether the ANSWER is present in another shape: a 1.5B reasons
        # correctly through 8-hop traversal and then writes "The final
        # answer is:" / "JSON key node: {...}" instead of the literal
        # marker. Scoring that as a reasoning failure conflates
        # instruction-following with reasoning and reports the sum as
        # reasoning, which is the error this whole pass exists to remove.
        lenient = _lenient_extract(text, expected)
        if lenient is None:
            return "unparseable"
        return "correct_lenient" if lenient == expected else "incorrect_lenient"
    return "correct" if produced == expected else "incorrect"


def _lenient_extract(text: str, expected: dict) -> dict | None:
    """Find the answer object anywhere in the text, contract or not.

    Deliberately narrow: it looks only for a JSON object carrying the
    EXPECTED keys, so it cannot invent an answer the model did not give.
    Reported under separate 'lenient' outcomes and never merged into the
    strict accuracy number.
    """
    import json as _json
    import re as _re

    wanted = set(expected)
    for match in _re.finditer(r"\{[^{}]*\}", text):
        try:
            candidate = _json.loads(match.group(0))
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and set(candidate) == wanted:
            return candidate
    return None



CALIBRATION_SCHEMA = "aura.rlc_accuracy_calibration.v1"


def calibrate_scoring() -> dict[str, Any]:
    """Run known-truth fixtures through the REAL scoring and aggregation.

    An instrument that has never been checked against known answers is not
    evidence-grade. Both 0%-at-every-depth results this tool produced were
    harness faults that a single calibration pass would have caught in
    seconds: once because scoring raised AttributeError inside a bare
    ``except`` and reported every task wrong, once because the token budget
    could not reach a FINAL_ANSWER at all.

    Fixtures assert the four outcomes the tool can report, and -- most
    importantly -- that a PERFECT answer aggregates to exactly 100% and a
    silent model aggregates to 0% with the unparseable count carrying the
    explanation. Raises CalibrationError on any disagreement; callers must
    refuse to publish numbers from an uncalibrated instrument.
    """
    from types import SimpleNamespace

    gold = 'FINAL_ANSWER: {"node":6}'
    task = SimpleNamespace(answer=gold, family="khop")
    checks: list[dict[str, Any]] = []

    def record(name: str, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )

    record("perfect_answer", _score(task, gold), "correct")
    record(
        "reasoning_then_answer",
        _score(task, "step one\nstep two\n" + gold),
        "correct",
    )
    record(
        "wrong_value", _score(task, 'FINAL_ANSWER: {"node":9}'), "incorrect"
    )
    record(
        "prose_without_answer",
        _score(task, "To solve this we follow the edges from node 1..."),
        "unparseable",
    )
    record("empty_output", _score(task, ""), "unparseable")

    # Aggregation must turn those outcomes into the right rates.
    perfect = _tally(["correct"] * 5)
    record("perfect_accuracy", perfect["accuracy"], 1.0)
    record("perfect_compliance", perfect["contract_compliance"], 1.0)
    silent = _tally(["unparseable"] * 5)
    record("silent_accuracy", silent["accuracy"], 0.0)
    record("silent_compliance", silent["contract_compliance"], 0.0)
    record("silent_unparseable_count", silent["unparseable"], 5)
    wrong = _tally(["incorrect"] * 4 + ["correct"])
    record("mixed_accuracy", wrong["accuracy"], 0.2)
    record(
        "mixed_compliance_is_total", wrong["contract_compliance"], 1.0
    )

    # A harness fault must RAISE, never score as a wrong answer.
    fault_raised = False
    try:
        _score(SimpleNamespace(answer="not a contract", family="x"), gold)
    except HarnessError:
        fault_raised = True
    record("harness_fault_raises", fault_raised, True)

    failures = [row["check"] for row in checks if not row["passed"]]
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise CalibrationError(
            f"accuracy harness failed calibration: {failures}"
        )
    return receipt


class CalibrationError(RuntimeError):
    """The instrument disagreed with known truth; results are void."""


def run_arm(
    model,
    tokenizer,
    tasks,
    depths: list[int],
    n_slots: int,
    max_tokens: int,
    envelope=None,
    bridge_tokens: tuple = (),
) -> dict:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
    from core.learning.recurrence_native_objective_v2 import (
        _advance_recurrent_states,
        _prepare_live_path,
    )

    spec = RLCExecutionSpec(
        n_slots=n_slots,
        branch_roles=("constructive_solution",),
        recurrent_steps=max(depths),
        exchange_interval=1,
        decode_bridge_policy="assistant_answer" if bridge_tokens else "none",
    )
    by_depth: dict[int, list[bool]] = {depth: [] for depth in depths}
    by_family: dict[str, dict[int, list[bool]]] = {}
    for task in tasks:
        prompt = _render(tokenizer, task)
        answer_probe = list(
            tokenizer.encode(task.answer, add_special_tokens=False)
        )
        prepared = _prepare_live_path(
            model,
            prompt,
            answer_probe,
            spec=spec.with_depth(max(depths)),
            bridge_tokens=bridge_tokens,
        )
        states = list(prepared.states)
        for step in range(max(depths)):
            states = _advance_recurrent_states(
                model,
                prepared.prompts_at_window,
                states,
                prepared.anchors,
                spec.with_depth(max(depths)),
                step,
                prepared.prelude_end,
                prepared.coda_start,
            )
            depth = step + 1
            if depth not in by_depth:
                continue
            text = _decode_answer(
                model, tokenizer, prepared, states[0], max_tokens, envelope
            )
            outcome = _score(task, text)
            by_depth[depth].append(outcome)
            by_family.setdefault(task.family, {}).setdefault(depth, []).append(
                outcome
            )
        if envelope is not None:
            envelope.reclaim(force=True)
        else:
            mx.clear_cache()
    return {
        "by_depth": {
            str(depth): _tally(values) for depth, values in by_depth.items()
        },
        "by_family": {
            family: {
                str(depth): _tally(values) for depth, values in per_depth.items()
            }
            for family, per_depth in by_family.items()
        },
    }


def _tally(outcomes: list[str]) -> dict[str, Any]:
    """Accuracy PLUS the contract-compliance breakdown.

    Reporting only accuracy makes 0% ambiguous between 'reasons badly' and
    'never emitted a parseable answer'. Those demand opposite responses, so
    both numbers are always published.
    """
    total = len(outcomes)
    correct = sum(1 for value in outcomes if value == "correct")
    incorrect = sum(1 for value in outcomes if value == "incorrect")
    correct_lenient = sum(1 for value in outcomes if value == "correct_lenient")
    incorrect_lenient = sum(
        1 for value in outcomes if value == "incorrect_lenient"
    )
    unparseable = sum(1 for value in outcomes if value == "unparseable")
    return {
        "correct": correct,
        "incorrect": incorrect,
        "correct_lenient": correct_lenient,
        "incorrect_lenient": incorrect_lenient,
        "unparseable": unparseable,
        "n": total,
        # Strict: reasoning AND contract compliance together.
        "accuracy": (correct / total) if total else 0.0,
        # Reasoning isolated from format: did the model reach the right
        # answer in ANY shape? This is the capability number.
        "reasoning_accuracy": (
            ((correct + correct_lenient) / total) if total else 0.0
        ),
        "contract_compliance": ((correct + incorrect) / total) if total else 0.0,
        "answered_at_all": (
            ((total - unparseable) / total) if total else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=int, default=8)
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--depths", default="1,2,4,8")
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument(
        "--bridge",
        choices=("none", "assistant_answer"),
        default="none",
        help=(
            "prepend the live assistant-answer cue before the answer span. "
            "Without it the model narrates instead of answering and every "
            "task scores unparseable -- a COMPLIANCE failure, not a "
            "reasoning one."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run calibration only; load no model and publish no numbers",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from mlx_lm import load

    from core.runtime.mlx_memory_guard import mlx_memory_envelope

    depths = [int(v) for v in args.depths.split(",") if v.strip()]
    families = [v.strip() for v in args.families.split(",") if v.strip()]
    started = time.time()
    calibration = calibrate_scoring()
    print(
        f"calibration passed: {len(calibration['checks'])} known-truth checks",
        flush=True,
    )
    if args.self_test:
        print(json.dumps(calibration, indent=2))
        return 0

    with mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"memory envelope: {envelope.to_receipt()}", flush=True)
        print(f"loading {args.model}", flush=True)
        load_kwargs = {"adapter_path": args.adapter} if args.adapter else {}
        model, tokenizer = load(args.model, **load_kwargs)
        tasks = _load_tasks(
            families, args.task_depth, args.per_cell, args.eval_seed
        )
        print(
            f"{len(tasks)} eval tasks (seed {args.eval_seed}), depths {depths}",
            flush=True,
        )
        bridge_tokens: tuple = ()
        if args.bridge == "assistant_answer":
            from core.brain.llm.latent_cortex.engine import (
                _ASSISTANT_ANSWER_BRIDGE_V3,
            )

            try:
                encoded = tokenizer.encode(
                    _ASSISTANT_ANSWER_BRIDGE_V3, add_special_tokens=False
                )
            except TypeError:
                encoded = tokenizer.encode(_ASSISTANT_ANSWER_BRIDGE_V3)
            bridge_tokens = tuple(int(token) for token in encoded)
            print(f"decode bridge: {len(bridge_tokens)} tokens", flush=True)
        result = run_arm(
            model,
            tokenizer,
            tasks,
            depths,
            args.n_slots,
            args.max_tokens,
            envelope,
            bridge_tokens,
        )
        envelope_receipt = envelope.to_receipt()
    print("\n=== exact-match accuracy by depth ===")
    for depth in depths:
        row = result["by_depth"][str(depth)]
        print(
            f"  depth {depth:2d}: strict {row['correct']:3d}/{row['n']:3d}"
            f" = {100*row['accuracy']:5.1f}%"
            f" | REASONING {100*row['reasoning_accuracy']:5.1f}%"
            f" | contract {100*row['contract_compliance']:3.0f}%"
            f" | answered {100*row['answered_at_all']:3.0f}%"
        )
    print("\n=== by family ===")
    for family, per_depth in sorted(result["by_family"].items()):
        cells = "  ".join(
            f"d{depth}={100*per_depth[str(depth)]['accuracy']:.0f}%"
            for depth in depths
            if str(depth) in per_depth
        )
        print(f"  {family:16s} {cells}")

    payload = {
        "schema": ACCURACY_SCHEMA,
        "model": args.model,
        "adapter": args.adapter,
        "families": families,
        "task_depth": args.task_depth,
        "per_cell": args.per_cell,
        "eval_seed": args.eval_seed,
        "depths": depths,
        "n_slots": args.n_slots,
        "max_tokens": args.max_tokens,
        "elapsed_s": round(time.time() - started, 3),
        "metric": "exact_match_correctness",
        "memory_envelope": envelope_receipt,
        "bridge": args.bridge,
        "calibration": calibration,
        "claims_awarded": [],
        "results": result,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
