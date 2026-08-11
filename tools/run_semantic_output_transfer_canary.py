#!/usr/bin/env python
"""Bounded real-checkpoint test of task-disjoint semantic output transfer.

The calibration split may use exact generated answers as supervision.  Gain is
selected on a disjoint validation split.  The sealed test split receives only
its prompt and the frozen learned readout; neither teacher nor executable
producer is called while test answers are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_prompt(tokenizer: Any, task: Any) -> list[int]:
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    ]


def _encode_answer(tokenizer: Any, answer: str) -> list[int]:
    try:
        return [int(token) for token in tokenizer.encode(answer, add_special_tokens=False)]
    except TypeError:
        return [int(token) for token in tokenizer.encode(answer)]


def _generate(model: Any, tokenizer: Any, prompt: list[int], max_tokens: int):
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    pieces: list[str] = []
    tokens: list[int] = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        pieces.append(response.text)
        if response.finish_reason != "stop":
            tokens.append(int(response.token))
        if "}" in response.text and is_contract_complete("".join(pieces)):
            break
    text = tokenizer.decode(tokens)
    if text != "".join(pieces):
        raise RuntimeError("semantic transfer decode token/text round trip differs")
    return text, tokens


def _capture_corrections(model: Any, tokenizer: Any, proxy: Any, tasks: list[Any]):
    import mlx.core as mx

    key_rows: list[np.ndarray] = []
    targets: list[int] = []
    incumbents: list[int] = []
    task_ids: list[str] = []
    task_rows: list[dict[str, Any]] = []
    proxy.capture = True
    try:
        for task in tasks:
            prompt = _render_prompt(tokenizer, task)
            answer = _encode_answer(tokenizer, task.answer)
            if not prompt or not answer:
                raise RuntimeError("semantic transfer calibration tokenization is empty")
            sequence = prompt + answer[:-1]
            logits = model(mx.array([sequence]))
            hidden = proxy.last_hidden
            if hidden is None:
                raise RuntimeError("semantic transfer output boundary was not captured")
            start = len(prompt) - 1
            stop = start + len(answer)
            if stop > int(logits.shape[1]) or stop > int(hidden.shape[1]):
                raise RuntimeError("semantic transfer answer alignment exceeds sequence")
            aligned_logits = logits[0, start:stop]
            aligned_hidden = hidden[0, start:stop]
            predicted = np.asarray(mx.argmax(aligned_logits, axis=-1)).astype(np.int64)
            key_array = np.asarray(aligned_hidden, dtype=np.float32)
            corrected = 0
            for index, (target, incumbent) in enumerate(
                zip(answer, predicted.tolist(), strict=True)
            ):
                if target == incumbent:
                    continue
                key_rows.append(key_array[index])
                targets.append(target)
                incumbents.append(int(incumbent))
                task_ids.append(task.task_id)
                corrected += 1
            task_rows.append(
                {
                    "task_id_sha256": hashlib.sha256(task.task_id.encode()).hexdigest(),
                    "answer_token_count": len(answer),
                    "teacher_forced_correction_count": corrected,
                    "teacher_forced_accuracy": round(1.0 - corrected / len(answer), 6),
                }
            )
            del logits, hidden, aligned_logits, aligned_hidden
            mx.clear_cache()
    finally:
        proxy.capture = False
        proxy.last_hidden = None
    if len(key_rows) < 2 or len(set(task_ids)) < 2:
        raise RuntimeError("semantic transfer calibration produced insufficient corrections")
    return (
        np.stack(key_rows),
        tuple(targets),
        tuple(incumbents),
        tuple(task_ids),
        task_rows,
    )


def _score_split(
    model: Any,
    tokenizer: Any,
    proxy: Any,
    tasks: list[Any],
    *,
    treatment: Any,
    sham: Any,
    gain: float,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], float, float, float]:
    from core.brain.llm.latent_cortex.fast_weight_learning import token_sequence_sha256

    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = _render_prompt(tokenizer, task)
        baseline_text, baseline_tokens = _generate(model, tokenizer, prompt, max_tokens)
        baseline_score = float(task.grade(baseline_text)["correct"])

        treatment.reset(gain=gain)
        proxy.attach(treatment)
        try:
            treatment_text, treatment_tokens = _generate(model, tokenizer, prompt, max_tokens)
        finally:
            proxy.detach()
        treatment_score = float(task.grade(treatment_text)["correct"])

        sham.reset(gain=gain)
        proxy.attach(sham)
        try:
            sham_text, sham_tokens = _generate(model, tokenizer, prompt, max_tokens)
        finally:
            proxy.detach()
        sham_score = float(task.grade(sham_text)["correct"])
        rows.append(
            {
                "task_id_sha256": hashlib.sha256(task.task_id.encode()).hexdigest(),
                "baseline_score": baseline_score,
                "treatment_score": treatment_score,
                "sham_score": sham_score,
                "baseline_tokens_sha256": token_sequence_sha256(baseline_tokens),
                "treatment_tokens_sha256": token_sequence_sha256(treatment_tokens),
                "sham_tokens_sha256": token_sequence_sha256(sham_tokens),
            }
        )
    count = len(rows)
    return (
        rows,
        sum(row["baseline_score"] for row in rows) / count,
        sum(row["treatment_score"] for row in rows) / count,
        sum(row["sham_score"] for row in rows) / count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        default=("modular", "register_trace", "code_trace"),
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--train-per-cell", type=int, default=4)
    parser.add_argument("--validation-per-cell", type=int, default=1)
    parser.add_argument("--test-per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026081019)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    return parser


def _run_admitted(args: argparse.Namespace, model_path: Path) -> int:
    if min(args.train_per_cell, args.validation_per_cell, args.test_per_cell) < 1:
        raise ValueError("semantic transfer split sizes must be positive")
    if not 8 <= args.max_tokens <= 512:
        raise ValueError("semantic transfer max tokens must be inside [8, 512]")

    from mlx_lm import load

    from core.brain.llm.latent_cortex.semantic_output_adapter import (
        SEMANTIC_OUTPUT_GAIN_GRID,
        SemanticOutputAdapter,
        SemanticOutputEmbeddingProxy,
        build_semantic_output_transfer_receipt,
        deterministic_sham_tokens,
        validate_semantic_output_transfer_receipt,
    )
    from core.learning.recurrence_curriculum import task_battery
    from core.runtime.atomic_writer import atomic_write_text, ensure_private_directory

    started = time.time()
    model, tokenizer = load(str(model_path))
    inner = model.model
    original_embedding = inner.embed_tokens
    proxy = SemanticOutputEmbeddingProxy(original_embedding)
    inner.embed_tokens = proxy
    try:
        train = task_battery(
            args.families,
            [args.depth],
            args.train_per_cell,
            seed=args.seed,
        )
        excluded_prompts = [task.prompt for task in train]
        excluded_ids = [task.task_id for task in train]
        validation = task_battery(
            args.families,
            [args.depth],
            args.validation_per_cell,
            seed=args.seed + 1,
            excluded_prompts=excluded_prompts,
            excluded_task_ids=excluded_ids,
        )
        excluded_prompts.extend(task.prompt for task in validation)
        excluded_ids.extend(task.task_id for task in validation)
        test = task_battery(
            args.families,
            [args.depth],
            args.test_per_cell,
            seed=args.seed + 2,
            excluded_prompts=excluded_prompts,
            excluded_task_ids=excluded_ids,
        )

        keys, targets, incumbents, task_ids, calibration_rows = _capture_corrections(
            model, tokenizer, proxy, train
        )
        treatment = SemanticOutputAdapter.fit(
            keys,
            targets,
            incumbents,
            task_ids=task_ids,
            ridge=args.ridge,
            logit_scale=args.logit_scale,
        )
        sham_targets = deterministic_sham_tokens(
            targets,
            task_ids=tuple(f"{task_id}:{index}" for index, task_id in enumerate(task_ids)),
            incumbent_tokens=incumbents,
        )
        sham = SemanticOutputAdapter.fit(
            keys,
            sham_targets,
            incumbents,
            task_ids=task_ids,
            ridge=args.ridge,
            logit_scale=args.logit_scale,
        )
        treatment_identity = treatment.receipt()
        sham_identity = sham.receipt()

        validation_rows: list[dict[str, Any]] = []
        for gain in SEMANTIC_OUTPUT_GAIN_GRID:
            _, baseline_mean, treatment_mean, sham_mean = _score_split(
                model,
                tokenizer,
                proxy,
                validation,
                treatment=treatment,
                sham=sham,
                gain=gain,
                max_tokens=args.max_tokens,
            )
            validation_rows.append(
                {
                    "gain": gain,
                    "baseline_mean": baseline_mean,
                    "treatment_mean": treatment_mean,
                    "sham_mean": sham_mean,
                }
            )
        selected = max(
            validation_rows,
            key=lambda row: (
                row["treatment_mean"] - max(row["baseline_mean"], row["sham_mean"]),
                -row["gain"],
            ),
        )
        test_rows, _, _, _ = _score_split(
            model,
            tokenizer,
            proxy,
            test,
            treatment=treatment,
            sham=sham,
            gain=float(selected["gain"]),
            max_tokens=args.max_tokens,
        )
        treatment.erase()
        sham.erase()
        experiment = build_semantic_output_transfer_receipt(
            treatment_identity=treatment_identity,
            sham_identity=sham_identity,
            validation_task_ids=tuple(task.task_id for task in validation),
            test_task_ids=tuple(task.task_id for task in test),
            validation_rows=validation_rows,
            test_rows=test_rows,
            erase_proven=treatment.erased and sham.erased and proxy.adapter is None,
        )
        validate_semantic_output_transfer_receipt(experiment)
    finally:
        if proxy.adapter is not None:
            proxy.detach()
        inner.embed_tokens = original_embedding

    if inner.embed_tokens is not original_embedding:
        raise RuntimeError("semantic output proxy did not restore model identity")
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "core/brain/llm/latent_cortex/semantic_output_adapter.py",
        REPO_ROOT / "core/learning/recurrence_curriculum.py",
    )
    payload = {
        "schema": "aura.rlc.semantic_output_transfer_canary.v1",
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256_file(model_path / "config.json"),
            "tokenizer_config_sha256": _sha256_file(model_path / "tokenizer_config.json"),
        },
        "protocol": {
            "families": list(args.families),
            "depth": args.depth,
            "train_per_cell": args.train_per_cell,
            "validation_per_cell": args.validation_per_cell,
            "test_per_cell": args.test_per_cell,
            "seed": args.seed,
            "ridge": args.ridge,
            "logit_scale": args.logit_scale,
            "max_tokens": args.max_tokens,
        },
        "source_bindings": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
        "calibration": {
            "task_count": len(train),
            "correction_count": len(targets),
            "rows": calibration_rows,
        },
        "experiment": experiment,
        "model_object_restored": True,
        "checkpoint_parameter_mutation": False,
        "elapsed_s": round(time.time() - started, 3),
        "claims": {
            "mechanism_transfer": experiment["accepted"],
            "general_reasoning_gain": False,
            "resident_32b_gain": False,
            "frontier_gain": False,
            "fusion_eligible": False,
            "wow_signal": False,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out = args.out.expanduser().resolve()
    ensure_private_directory(out.parent)
    atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if experiment["accepted"] else 2


def main() -> int:
    args = _parser().parse_args()
    model_path = Path(args.model).expanduser().resolve(strict=True)
    with standalone_model_lane(
        owner_id=f"semantic-output-transfer:{args.out.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        allow_owner_eviction=False,
        metadata={"tool": Path(__file__).name},
    ):
        return _run_admitted(args, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
