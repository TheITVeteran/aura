"""Shared held-out behavioral probe for recurrent checkpoint training.

Training objectives may differ, but their promotion evidence must not. This
module runs the same source-task generations, random-stream coordinates,
semantic grader, and full episode-receipt capture for generated-rollin SFT and
on-policy recurrent GRPO canaries.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrent_checkpoint_admission import build_free_generation_report
from core.learning.recurrent_grpo import (
    RecurrentSamplingConfig,
    cortex_config_from_execution_spec,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def tokenize_task(
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    try:
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    except TypeError:
        answer_tokens = tokenizer.encode(answer)
    eos = getattr(tokenizer, "eos_token_id", None)
    normalized_answer = [int(token) for token in answer_tokens]
    if eos is not None and (not normalized_answer or normalized_answer[-1] != int(eos)):
        normalized_answer.append(int(eos))
    return [int(token) for token in prompt_tokens], normalized_answer


def paired_generation_seed(
    campaign_seed: int,
    task_ordinal: int,
    task_id: str,
    depth: int,
) -> int:
    """Use one random stream per task/depth coordinate across both arms."""

    material = f"{campaign_seed}:{task_ordinal}:{task_id}:{depth}"
    return int.from_bytes(
        hashlib.sha256(material.encode("ascii")).digest()[:4],
        "big",
    )


def free_generation_sampling_config() -> RecurrentSamplingConfig:
    """Use the exact categorical policy required by recurrent proof runs."""

    return RecurrentSamplingConfig(
        max_tokens=320,
        temperature=1.0,
        top_p=1.0,
    )


def build_behavioral_probe_report(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    spec: RLCExecutionSpec,
    arm: str,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Run exact held-out generations at shallow and full recurrent depth."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    depths = tuple(sorted({1, spec.recurrent_steps}))
    records: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(tasks):
        prompt_tokens, _answer_tokens = tokenize_task(
            tokenizer,
            task.prompt,
            task.answer,
        )
        for depth in depths:
            depth_spec = spec.with_depth(depth)
            config = cortex_config_from_execution_spec(
                depth_spec,
                sampling=free_generation_sampling_config(),
            )
            config.decode_contract = "none"
            config.decode_contract_grace_tokens = 0
            config.decode_incumbent_policy = "latent"
            engine = LatentCortexEngine(
                model,
                tokenizer=tokenizer,
                config=config,
                schedule_library=None,
            )
            generation_seed = paired_generation_seed(
                seed,
                task_ordinal,
                task.task_id,
                depth,
            )
            mx.random.seed(generation_seed)
            result = engine.reason(
                token_ids=prompt_tokens,
                decode_max_tokens=320,
                decode_sentence_grace_tokens=0,
                nonparametric_memory_enabled=False,
                sample_seed=generation_seed,
                episode_id=f"behavioral-probe-{seed}-{task_ordinal}-{depth}",
            )
            grade = dict(task.grade(result.text if result.ok else ""))
            grade["correct"] = bool(grade.get("correct"))
            receipt_payload = result.receipt.to_dict()
            records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                    "response_text": result.text,
                    "tokens_sha256": hashlib.sha256(
                        canonical_json_bytes(result.tokens)
                    ).hexdigest(),
                    "tokens": list(result.tokens),
                    "token_count": len(result.tokens),
                    "correct": bool(result.ok and grade["correct"]),
                    "grade_receipt": {
                        **grade,
                        "correct": bool(result.ok and grade["correct"]),
                    },
                    "episode_ok": bool(result.ok),
                    "episode_reason": str(result.reason or ""),
                    "decode_termination": str(result.receipt.decode_termination or "not_reached"),
                    "branch_selection_admitted": bool(result.receipt.branch_selection_admitted),
                    "decode_incumbent_policy": (result.receipt.decode_incumbent_policy),
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                    "episode_receipt": receipt_payload,
                }
            )
            del engine, result
            mx.synchronize()
            mx.clear_cache()
    return build_free_generation_report(
        arm=arm,
        adapter_sha256=adapter_sha256,
        execution_spec_sha256=spec.sha256,
        task_manifest_sha256=task_manifest_sha256,
        task_ids=[task.task_id for task in tasks],
        depths=depths,
        records=records,
    )


def build_ordinary_decode_probe_report(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    spec: RLCExecutionSpec,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """The vanilla control: the same weights answering without the RLC path.

    Reported at every depth coordinate the recurrent arms use so the two
    reports are directly comparable, even though ordinary decode has no
    recurrent depth -- the same generation is recorded at each coordinate
    because the ordinary path does not vary with it. Without this arm an
    admission can only compare a trained adapter to an untrained adapter on
    the same path, which is how a checkpoint scoring 3/28 passed review while
    ordinary decode on identical weights scored 13/28.
    """

    import mlx.core as mx
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    depths = tuple(sorted({1, spec.recurrent_steps}))
    records: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(tasks):
        prompt_tokens, _answer_tokens = tokenize_task(
            tokenizer,
            task.prompt,
            task.answer,
        )
        generation_seed = paired_generation_seed(seed, task_ordinal, task.task_id, 1)
        mx.random.seed(generation_seed)
        rendered = tokenizer.decode(prompt_tokens)
        pieces: list[str] = []
        for response in stream_generate(
            model, tokenizer, prompt=rendered, max_tokens=320
        ):
            pieces.append(response.text)
            if "}" in response.text and is_contract_complete("".join(pieces)):
                break
        text = "".join(pieces)
        tokens = list(tokenizer.encode(text, add_special_tokens=False))
        grade = dict(task.grade(text))
        grade["correct"] = bool(grade.get("correct"))
        receipt_payload = {
            "schema": "aura.rlc.ordinary_decode_probe.v1",
            "arm": "ordinary_decode",
            "generation_seed": generation_seed,
            "recurrent_steps": 0,
        }
        for depth in depths:
            records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "response_text": text,
                    "tokens_sha256": hashlib.sha256(
                        canonical_json_bytes(tokens)
                    ).hexdigest(),
                    "tokens": list(tokens),
                    "token_count": len(tokens),
                    "correct": bool(grade["correct"]),
                    "grade_receipt": dict(grade),
                    "episode_ok": True,
                    "episode_reason": "",
                    "decode_termination": "contract_complete",
                    "branch_selection_admitted": True,
                    "decode_incumbent_policy": "vanilla_incumbent",
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                    "episode_receipt": receipt_payload,
                }
            )
        mx.synchronize()
        mx.clear_cache()
    return build_free_generation_report(
        arm="ordinary_decode",
        adapter_sha256=adapter_sha256,
        execution_spec_sha256=spec.sha256,
        task_manifest_sha256=task_manifest_sha256,
        task_ids=[task.task_id for task in tasks],
        depths=depths,
        records=records,
    )


__all__ = [
    "build_behavioral_probe_report",
    "build_ordinary_decode_probe_report",
    "canonical_json_bytes",
    "free_generation_sampling_config",
    "paired_generation_seed",
    "tokenize_task",
]
