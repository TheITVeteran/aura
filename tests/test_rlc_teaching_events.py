from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.latent_cortex import frontier_tasks as ft
from core.brain.llm.latent_cortex.fast_weight_learning import empty_learning_state
from core.brain.llm.latent_cortex.objective_program_verifier import (
    solve_objective_program,
)
from core.brain.llm.latent_cortex.teaching_events import (
    build_exact_objective_teaching_event,
    validate_teaching_event,
)


class _ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))


def _task():
    return ft.generate_task(
        "misleading_premise",
        seed=7_100_001,
        difficulty=2,
        registry_version=ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
    )


def _structural() -> dict[str, object]:
    return {
        "certified": True,
        "receipt_sha256": hashlib.sha256(b"teaching-structural").hexdigest(),
    }


def test_verified_objective_teacher_emits_private_query_scoped_target() -> None:
    task = _task()
    source_state_sha = hashlib.sha256(b"wrong-neural-state").hexdigest()
    event, admission, target = build_exact_objective_teaching_event(
        objective=task.public.prompt,
        incumbent_candidate="FINAL_ANSWER: {}",
        source_state_sha256=source_state_sha,
        tokenizer=_ByteTokenizer(),
        structural_diversity=_structural(),
    )

    assert admission["admitted"] is True
    assert admission["critic_recalibration"]["verifier_family"] == (
        "exact_objective_program"
    )
    assert event["teacher_removed_before_causal_probe_required"] is True
    assert event["capability_claim_authority"] is False
    assert event["allowed_plasticity_scopes"] == [
        "activation_state",
        "episodic_fast_weights",
    ]
    assert bytes(target).decode("utf-8").startswith("FINAL_ANSWER: {")
    validate_teaching_event(
        event,
        admission=admission,
        expected_objective_sha256=hashlib.sha256(
            task.public.prompt.encode("utf-8")
        ).hexdigest(),
        expected_source_state_sha256=source_state_sha,
    )
    learning_state = empty_learning_state(
        episode_id="teacher-episode",
        input_tokens_sha256=hashlib.sha256(b"input").hexdigest(),
        selected_branch=0,
        winner_state_sha256=source_state_sha,
        admission=admission,
        teaching_event=event,
    )
    assert learning_state["teaching_event"] == event

    public_wire = json.dumps(event, sort_keys=True)
    assert bytes(target).decode("utf-8") not in public_wire
    assert task.reveal_for_verifier()["expected"] != {}
    assert json.dumps(task.reveal_for_verifier()["expected"], sort_keys=True) not in (
        public_wire
    )


def test_teaching_event_refuses_an_already_correct_incumbent() -> None:
    task = _task()
    solved = solve_objective_program(task.public.prompt)
    assert solved is not None
    correct = solved[0].rsplit("\n", 1)[-1]

    with pytest.raises(ValueError, match="already objective-verified"):
        build_exact_objective_teaching_event(
            objective=task.public.prompt,
            incumbent_candidate=correct,
            source_state_sha256=hashlib.sha256(b"correct-state").hexdigest(),
            tokenizer=_ByteTokenizer(),
            structural_diversity=_structural(),
        )


def test_teaching_event_rejects_policy_or_admission_tampering() -> None:
    task = _task()
    event, admission, _target = build_exact_objective_teaching_event(
        objective=task.public.prompt,
        incumbent_candidate="not a valid terminal answer",
        source_state_sha256=hashlib.sha256(b"state").hexdigest(),
        tokenizer=_ByteTokenizer(),
        structural_diversity=_structural(),
    )

    tampered = copy.deepcopy(event)
    tampered["capability_claim_authority"] = True
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_teaching_event(tampered, admission=admission)

    wrong_admission = copy.deepcopy(admission)
    wrong_admission["target_token_count"] += 1
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_teaching_event(event, admission=wrong_admission)
