"""THINKING or SPINNING: does each recurrent step make the ANSWER better?

Every prior depth measurement compared SEPARATE runs (a depth-4 run vs a
depth-8 run), which conflates "the run was configured for T steps" with
"the state at step T". The question that actually matters is within ONE
trajectory: does the answer decoded from z_5 beat the answer decoded from
z_4?

An open-loop recurrence has no error signal -- nothing tells step 5 what
is still wrong -- so there is no mechanism forcing per-step improvement.
If answer CE is flat or noisy across steps within a trajectory, the loop
is spinning, and extra depth is cost without cognition. That would mean
per-step improvement has to be TRAINED (and/or the loop closed with a
verifier), not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/bryan/.aura/live-source")
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm import load  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.learning import recurrence_curriculum as curriculum  # noqa: E402
import core.learning.recurrence_native_objective_v2 as objective  # noqa: E402

MODEL = str(REPO / "models/Qwen2.5-1.5B-Instruct-4bit")
DEPTH = 16
FAMILIES = ["khop", "modular"]


def trajectory_answer_ce(model, tokenizer, task, phase: float):
    """Answer CE decoded from EVERY intermediate state of one trajectory."""
    prompt = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    answer = list(tokenizer.encode(task.answer, add_special_tokens=False))
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=DEPTH,
        exchange_interval=1,
    ).with_depth(DEPTH)

    captured: list = []
    original = objective._window_pass

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    objective._window_pass = recording
    try:
        with objective.recurrent_phase(phase):
            prepared = objective._prepare_live_path(
                model, prompt, answer, spec=spec, bridge_tokens=()
            )
            states = list(prepared.states)
            for step in range(DEPTH):
                states = objective._advance_recurrent_states(
                    model,
                    prepared.prompts_at_window,
                    states,
                    prepared.anchors,
                    spec,
                    step,
                    prepared.prelude_end,
                    prepared.coda_start,
                )
                captured.append(states[0])
    finally:
        objective._window_pass = original

    # Score the answer from each committed step state.
    targets = mx.array(answer)[None, :]
    per_step = []
    seen = captured[-DEPTH:]
    for state in seen:
        logits = objective._persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[0],
            state,
            prepared.tail_embeddings,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        mx.eval(loss)
        per_step.append(float(loss))
    return per_step


def main() -> int:
    print(f"loading {MODEL}\n", flush=True)
    model, tokenizer = load(MODEL)
    results = {}
    for family in FAMILIES:
        tasks = curriculum.task_battery([family], [8], 2, seed=4242)
        for phase_label, phase in (("base", 0.0), ("phase", 0.08)):
            curves = [
                trajectory_answer_ce(model, tokenizer, task, phase)
                for task in tasks
            ]
            mean = [sum(c[i] for c in curves) / len(curves) for i in range(DEPTH)]
            results[f"{family}_{phase_label}"] = mean
            print(f"=== {family} / {phase_label}: answer CE per step ===")
            print(
                "  "
                + "  ".join(
                    f"s{i+1}={v:.3f}" for i, v in enumerate(mean) if i < 8
                )
            )
            print(
                "  "
                + "  ".join(
                    f"s{i+1}={v:.3f}" for i, v in enumerate(mean) if i >= 8
                )
            )
            best = min(range(DEPTH), key=lambda i: mean[i])
            improving = sum(
                1 for i in range(1, DEPTH) if mean[i] < mean[i - 1] - 1e-4
            )
            print(
                f"  best step = {best+1}/{DEPTH}   "
                f"steps that improved = {improving}/{DEPTH-1}   "
                f"first->best = {100*(mean[best]-mean[0])/mean[0]:+.1f}%   "
                f"best->last = {100*(mean[-1]-mean[best])/mean[best]:+.1f}%"
            )
            verdict = (
                "THINKING (most steps improve the answer)"
                if improving >= 0.6 * (DEPTH - 1)
                else "SPINNING (steps move state without improving answers)"
            )
            print(f"  VERDICT: {verdict}\n")
            results[f"{family}_{phase_label}_verdict"] = verdict
            results[f"{family}_{phase_label}_best_step"] = best + 1
            results[f"{family}_{phase_label}_improving_steps"] = improving

    out = Path(__file__).with_name("per_step_probe_result.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
