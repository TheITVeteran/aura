#!/usr/bin/env python3
"""Run a source-bound 1.5B decode canary over neural semantic state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.semantic_neural_decode_context import (  # noqa: E402
    SemanticNeuralDecodeState,
    execute_semantic_neural_decode_state,
    render_semantic_neural_decode_context,
)
from core.brain.llm.unified_recurrent_transfer_decode import (  # noqa: E402
    decode_base_greedy_tokens,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.rlc.semantic_neural_decode_canary.v1"
CLAIM_BOUNDARY: Final = (
    "bounded teacher-free multi-domain neural-state-to-free-decode transfer on "
    "the model bound in model_identity; not open-domain, resident-32B, broad "
    "reasoning, fusion, frontier performance, or WOW"
)
ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "coefficient_lesion",
    "matched_wrong_state",
)
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/semantic_neural_decode_context.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_semantic_neural_decode_canary.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prompt_tokens(tokenizer: Any, objective: str, context: str = "") -> tuple[int, ...]:
    content = objective if not context else f"{objective}\n\n{context}"
    return tuple(
        int(token)
        for token in tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def _complete(tokenizer: Any, token_ids: tuple[int, ...]) -> bool:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return contract_decode_disposition(text) in {
        ContractDecodeDisposition.COMPLETE,
        ContractDecodeDisposition.INVALID,
    }


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _wrong_state_index(states: list[SemanticNeuralDecodeState], index: int) -> int:
    own = states[index].semantic_result
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        if states[candidate].family == states[index].family and (
            states[candidate].semantic_result != own
        ):
            return candidate
    raise RuntimeError("semantic decode canary cannot construct a same-family derangement")


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    exact = sum(bool(row["correct"]) for row in selected)
    parsed = sum(bool(row["parsed"]) for row in selected)
    body = {
        "examples": len(selected),
        "exact": exact,
        "parsed": parsed,
        "exact_accuracy": round(exact / len(selected), 6),
        "parsed_accuracy": round(parsed / len(selected), 6),
        "mean_prompt_tokens": round(
            sum(int(row["prompt_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_latency_ms": round(
            sum(int(row["latency_ms"]) for row in selected) / len(selected), 3
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _lesion_machine() -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    tissue.raw_coefficients = tissue.raw_coefficients.at[1, 2].add(
        -tissue.raw_coefficients[1, 2]
    )
    return SemanticNeuralMachine(tissue)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_815_48)
    parser.add_argument("--tasks-per-difficulty", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    return parser


def _lane_kwargs(model_path: Path, output: Path) -> dict[str, Any]:
    return {
        "owner_id": f"semantic-neural-decode:{output.name}",
        "model_path": str(model_path),
        "purpose": "evaluation",
        "preemptible": False,
        "allow_owner_eviction": False,
        "metadata": {"tool": Path(__file__).name},
    }


def _run(args: argparse.Namespace, model_path: Path) -> int:
    if not 2 <= args.tasks_per_difficulty <= 20:
        raise ValueError("semantic decode task count is outside [2, 20]")
    if not 32 <= args.max_tokens <= 384:
        raise ValueError("semantic decode token budget is outside [32, 384]")
    source_commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("semantic decode canary requires clean measured source")

    from mlx_lm import load

    started = time.time()
    tasks = frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise"),
        (1, 2, 3),
        args.tasks_per_difficulty,
        seed=args.seed,
    )
    treatment_states = [
        execute_semantic_neural_decode_state(task.prompt, task.family) for task in tasks
    ]
    lesion = _lesion_machine()
    lesion_states: list[SemanticNeuralDecodeState | None] = []
    for task in tasks:
        try:
            lesion_states.append(
                execute_semantic_neural_decode_state(
                    task.prompt, task.family, machine=lesion
                )
            )
        except (RuntimeError, ValueError):
            lesion_states.append(None)

    model, tokenizer = load(str(model_path))
    wire_prefill = tuple(
        int(token)
        for token in tokenizer.encode("FINAL_ANSWER: ", add_special_tokens=False)
    )
    if not wire_prefill:
        raise RuntimeError("semantic decode wire prefill tokenization is empty")

    rows: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        wrong = treatment_states[_wrong_state_index(treatment_states, index)]
        state_by_arm = {
            "treatment": treatment_states[index],
            "coefficient_lesion": lesion_states[index],
            "matched_wrong_state": wrong,
        }
        for arm in _arm_order(task.task_id):
            selected = state_by_arm.get(arm)
            context = ""
            if selected is not None:
                context = render_semantic_neural_decode_context(selected)
            elif arm == "coefficient_lesion":
                context = (
                    "Internal recurrent semantic computation did not produce an "
                    "admissible terminal state after the declared tissue lesion."
                )
            active_prefill = () if arm == "ordinary_base" else wire_prefill
            prompt = _prompt_tokens(tokenizer, task.prompt, context)
            generated, stopped, latency_ms = decode_base_greedy_tokens(
                model,
                prompt,
                eos_token_id=tokenizer.eos_token_id,
                max_tokens=args.max_tokens,
                prefill_tokens=active_prefill,
                completion_check=lambda values: _complete(tokenizer, values),
            )
            text = tokenizer.decode(list(generated), skip_special_tokens=True)
            score = task.score(text)
            row = {
                "task_id": task.task_id,
                "family": task.family,
                "difficulty": task.depth,
                "arm": arm,
                "correct": bool(score.correct),
                "parsed": bool(score.parsed),
                "prompt_tokens": len(prompt),
                "generated_tokens": len(generated) - len(active_prefill),
                "stopped": stopped,
                "latency_ms": latency_ms,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "state_receipt_sha256": (
                    "" if selected is None else selected.receipt()["receipt_sha256"]
                ),
            }
            rows.append(row)
            raw_outputs.append({"task_id": task.task_id, "arm": arm, "response": text})
            print(
                json.dumps(
                    {
                        "event": "decode_complete",
                        "completed": len(rows),
                        "total": len(tasks) * len(ARMS),
                        "family": task.family,
                        "arm": arm,
                        "correct": bool(score.correct),
                        "latency_ms": latency_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:  # pragma: no cover
                pass

    arms = {arm: _summary(rows, arm) for arm in ARMS}
    base = {
        row["task_id"]: bool(row["correct"])
        for row in rows
        if row["arm"] == "ordinary_base"
    }
    treatment = {
        row["task_id"]: bool(row["correct"])
        for row in rows
        if row["arm"] == "treatment"
    }
    gain_set = sorted(key for key, value in treatment.items() if value and not base[key])
    regressions = sorted(key for key, value in treatment.items() if not value and base[key])
    treatment_accuracy = float(arms["treatment"]["exact_accuracy"])
    admitted = bool(
        treatment_accuracy == 1.0
        and gain_set
        and not regressions
        and all(
            float(arms[arm]["exact_accuracy"]) < treatment_accuracy
            for arm in ("matched_wire_base", "coefficient_lesion", "matched_wrong_state")
        )
    )
    payload = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_sha256s": {path: _file_sha(REPO_ROOT / path) for path in SOURCE_PATHS},
        "model_identity": {
            "path": str(model_path),
            "config_sha256": _file_sha(model_path / "config.json"),
            "weights_index_sha256": _file_sha(model_path / "model.safetensors.index.json"),
        },
        "seed": args.seed,
        "tasks_per_difficulty": args.tasks_per_difficulty,
        "task_count": len(tasks),
        "arms": arms,
        "gain_set_sha256": _sha(gain_set),
        "gain_count": len(gain_set),
        "regression_set_sha256": _sha(regressions),
        "regression_count": len(regressions),
        "treatment_state_receipt_sha256s": [
            state.receipt()["receipt_sha256"] for state in treatment_states
        ],
        "admitted": admitted,
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_seconds": round(time.time() - started, 3),
        "raw_outputs": raw_outputs,
    }
    payload["receipt_sha256"] = _sha(payload)
    atomic_write_text(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "canary_complete", "admitted": admitted}, sort_keys=True))
    return 0 if admitted else 2


def main() -> int:
    args = _parser().parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.out = output
    with standalone_model_lane(**_lane_kwargs(model_path, output)):
        return _run(args, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
