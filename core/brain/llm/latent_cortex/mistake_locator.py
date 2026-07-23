"""Live, receipt-bearing localization of suspect recurrent transitions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.learning.mistake_locator import MistakeLocatorHead

MISTAKE_LOCATOR_RECEIPT_SCHEMA = "aura.rlc.mistake_locator_receipt.v1"
MISTAKE_OBSERVATION_SCHEMA = "aura.rlc.mistake_transition_observation.v1"
UNAVAILABLE = "unavailable"
LEARNED = "learned"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector_sha256(value: list[float]) -> str:
    return hashlib.sha256(
        ",".join(f"{item:.8f}" for item in value).encode("ascii")
    ).hexdigest()


def _pooled_hidden(state: Any, *, width: int) -> list[float]:
    import mlx.core as mx

    if state.ndim < 1 or int(state.shape[-1]) != width or int(state.size) < width:
        raise ValueError("mistake-locator hidden state has incompatible shape")
    if not bool(mx.all(mx.isfinite(state)).item()):
        raise ValueError("mistake-locator hidden state is non-finite")
    axes = tuple(range(state.ndim - 1))
    pooled = mx.mean(state.astype(mx.float32), axis=axes)
    mx.eval(pooled)
    values = [round(float(item), 8) for item in pooled.tolist()]
    if len(values) != width or any(not math.isfinite(item) for item in values):
        raise ValueError("mistake-locator pooled state is invalid")
    return values


class MistakeLocatorRuntime:
    """Pinned admitted locator; unavailable mode emits no localization."""

    def __init__(
        self,
        *,
        mode: str,
        head: MistakeLocatorHead | None = None,
        head_sha256: str = "",
    ) -> None:
        if mode not in {UNAVAILABLE, LEARNED}:
            raise ValueError("mistake-locator mode is invalid")
        if mode == UNAVAILABLE and (head is not None or head_sha256):
            raise ValueError("unavailable mistake locator cannot carry a head")
        if mode == LEARNED and (
            head is None or not head.admitted or not _is_sha256(head_sha256)
        ):
            raise ValueError("learned mistake locator requires an admitted pinned head")
        if head is not None:
            head.validate()
        self.mode = mode
        self.head = head
        self.head_sha256 = head_sha256

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, Any] | None,
    ) -> MistakeLocatorRuntime:
        config = dict(value or {})
        mode = str(config.get("mode", UNAVAILABLE))
        if mode == UNAVAILABLE:
            if set(config) - {"mode"}:
                raise ValueError("unavailable mistake locator carries a head")
            return cls(mode=UNAVAILABLE)
        if mode != LEARNED or set(config) != {
            "mode",
            "head_path",
            "head_sha256",
        }:
            raise ValueError("mistake-locator config fields differ")
        path = config.get("head_path")
        digest = config.get("head_sha256")
        if not isinstance(path, str) or not path.strip() or not _is_sha256(digest):
            raise ValueError("learned mistake-locator config is incomplete")
        try:
            head = MistakeLocatorHead.load(
                Path(path).expanduser(),
                expected_sha256=digest,
            )
        except OSError as exc:
            raise ValueError("mistake-locator artifact is unreadable") from exc
        return cls(mode=LEARNED, head=head, head_sha256=digest)

    @property
    def manifest(self) -> dict[str, Any]:
        return {} if self.head is None else self.head.manifest()

    @property
    def state_width(self) -> int:
        return 0 if self.head is None else self.head.state_width

    def observe(
        self,
        prior_state: Any,
        proposal_state: Any,
        *,
        branch_index: int,
        branch_step: int,
        prior_state_sha256: str,
        proposal_state_sha256: str,
        admitted_state_sha256: str,
        accepted: bool,
    ) -> dict[str, Any]:
        if self.head is None:
            raise ValueError("mistake localization is unavailable")
        if (
            type(branch_index) is not int
            or branch_index < 0
            or type(branch_step) is not int
            or branch_step < 0
            or not _is_sha256(prior_state_sha256)
            or not _is_sha256(proposal_state_sha256)
            or not _is_sha256(admitted_state_sha256)
            or type(accepted) is not bool
        ):
            raise ValueError("mistake observation identity is invalid")
        prior = _pooled_hidden(prior_state, width=self.head.state_width)
        proposal = _pooled_hidden(proposal_state, width=self.head.state_width)
        probability = self.head.probability(prior, proposal)
        payload = {
            "schema": MISTAKE_OBSERVATION_SCHEMA,
            "branch_index": branch_index,
            "branch_step": branch_step,
            "prior_reasoning_sha256": prior_state_sha256,
            "proposal_reasoning_sha256": proposal_state_sha256,
            "admitted_reasoning_sha256": admitted_state_sha256,
            "accepted": accepted,
            "prior_pooled_hidden": prior,
            "prior_pooled_hidden_sha256": _vector_sha256(prior),
            "proposal_pooled_hidden": proposal,
            "proposal_pooled_hidden_sha256": _vector_sha256(proposal),
            "head_sha256": self.head_sha256,
            "error_probability": round(probability, 10),
        }
        return {**payload, "observation_sha256": canonical_sha256(payload)}


def _validate_observation(
    value: Any,
    *,
    runtime: MistakeLocatorRuntime,
    branch_index: int,
    source_transition: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema",
        "branch_index",
        "branch_step",
        "prior_reasoning_sha256",
        "proposal_reasoning_sha256",
        "admitted_reasoning_sha256",
        "accepted",
        "prior_pooled_hidden",
        "prior_pooled_hidden_sha256",
        "proposal_pooled_hidden",
        "proposal_pooled_hidden_sha256",
        "head_sha256",
        "error_probability",
        "observation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("mistake observation fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"observation_sha256"}}
    if (
        runtime.head is None
        or row["schema"] != MISTAKE_OBSERVATION_SCHEMA
        or row["branch_index"] != branch_index
        or row["branch_step"] != source_transition.get("branch_step")
        or row["prior_reasoning_sha256"]
        != source_transition.get("prior_reasoning_sha256")
        or row["proposal_reasoning_sha256"]
        != source_transition.get("proposal_reasoning_sha256")
        or row["admitted_reasoning_sha256"]
        != source_transition.get("admitted_reasoning_sha256")
        or row["accepted"] is not source_transition.get("accepted")
        or row["head_sha256"] != runtime.head_sha256
        or row["observation_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("mistake observation identity is invalid")
    prior = row["prior_pooled_hidden"]
    proposal = row["proposal_pooled_hidden"]
    if (
        not isinstance(prior, list)
        or not isinstance(proposal, list)
        or len(prior) != runtime.head.state_width
        or len(proposal) != runtime.head.state_width
    ):
        raise ValueError("mistake observation vectors differ")
    prior_values = [round(float(item), 8) for item in prior]
    proposal_values = [round(float(item), 8) for item in proposal]
    expected_probability = round(
        runtime.head.probability(prior_values, proposal_values),
        10,
    )
    if (
        any(
            not math.isfinite(item)
            for item in prior_values + proposal_values
        )
        or prior_values != prior
        or proposal_values != proposal
        or row["prior_pooled_hidden_sha256"] != _vector_sha256(prior_values)
        or row["proposal_pooled_hidden_sha256"]
        != _vector_sha256(proposal_values)
        or row["error_probability"] != expected_probability
    ):
        raise ValueError("mistake observation reconstruction failed")
    return row


def _prediction(
    observations: list[Mapping[str, Any]],
    *,
    runtime: MistakeLocatorRuntime,
) -> tuple[int | None, float | None]:
    if not observations or runtime.head is None:
        return None, None
    probabilities = [
        float(observation["error_probability"]) for observation in observations
    ]
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    probability = probabilities[index]
    return (
        index if probability >= runtime.head.threshold else None,
        probability,
    )


def build_mistake_locator_receipt(
    *,
    branches: list[Any],
    runtime: MistakeLocatorRuntime,
    update_acceptance: Mapping[str, Any],
    selected_branch: int,
) -> dict[str, Any]:
    if (
        not isinstance(update_acceptance, Mapping)
        or not _is_sha256(update_acceptance.get("receipt_sha256"))
        or type(selected_branch) is not int
        or not 0 <= selected_branch < len(branches)
    ):
        raise ValueError("mistake-locator receipt source is invalid")
    rows = []
    for branch in branches:
        observations = [dict(row) for row in branch.mistake_locator_trace]
        predicted, probability = _prediction(observations, runtime=runtime)
        rows.append(
            {
                "branch_index": int(branch.index),
                "observations": observations,
                "predicted_error_transition": predicted,
                "predicted_error_probability": probability,
            }
        )
    payload = {
        "schema": MISTAKE_LOCATOR_RECEIPT_SCHEMA,
        "mode": runtime.mode,
        "head_sha256": runtime.head_sha256,
        "head_manifest": runtime.manifest,
        "update_acceptance_sha256": update_acceptance["receipt_sha256"],
        "branches": rows,
        "observation_count": sum(len(row["observations"]) for row in rows),
        "candidate_count": sum(
            row["predicted_error_transition"] is not None for row in rows
        ),
        "selected_branch": selected_branch,
        "selected_branch_candidate": rows[selected_branch][
            "predicted_error_transition"
        ],
        "localization_admitted": runtime.mode == LEARNED,
        # SPARK-029 measures. A later independently gated repair mechanism
        # must opt in; this artifact can never authorize mutation by itself.
        "repair_steering_authorized": False,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_mistake_locator_receipt(
        receipt,
        expected_runtime=runtime,
        update_acceptance=update_acceptance,
        expected_n_branches=len(branches),
    )


def validate_mistake_locator_receipt(
    value: Any,
    *,
    expected_runtime: MistakeLocatorRuntime,
    update_acceptance: Mapping[str, Any],
    expected_n_branches: int,
) -> dict[str, Any]:
    fields = {
        "schema",
        "mode",
        "head_sha256",
        "head_manifest",
        "update_acceptance_sha256",
        "branches",
        "observation_count",
        "candidate_count",
        "selected_branch",
        "selected_branch_candidate",
        "localization_admitted",
        "repair_steering_authorized",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(update_acceptance, Mapping)
        or not _is_sha256(update_acceptance.get("receipt_sha256"))
    ):
        raise ValueError("mistake-locator receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    branches = receipt["branches"]
    source_branches = update_acceptance.get("branches")
    if (
        receipt["schema"] != MISTAKE_LOCATOR_RECEIPT_SCHEMA
        or receipt["mode"] != expected_runtime.mode
        or receipt["head_sha256"] != expected_runtime.head_sha256
        or receipt["head_manifest"] != expected_runtime.manifest
        or receipt["update_acceptance_sha256"]
        != update_acceptance["receipt_sha256"]
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or type(expected_n_branches) is not int
        or expected_n_branches < 1
        or not isinstance(branches, list)
        or len(branches) != expected_n_branches
        or not isinstance(source_branches, list)
        or len(source_branches) != expected_n_branches
    ):
        raise ValueError("mistake-locator receipt identity is invalid")
    observation_count = candidate_count = 0
    predictions: list[int | None] = []
    for branch_index, branch in enumerate(branches):
        if (
            not isinstance(branch, Mapping)
            or set(branch)
            != {
                "branch_index",
                "observations",
                "predicted_error_transition",
                "predicted_error_probability",
            }
            or branch["branch_index"] != branch_index
            or not isinstance(branch["observations"], list)
        ):
            raise ValueError("mistake-locator branch evidence is invalid")
        source = source_branches[branch_index]
        transitions = (
            source.get("transitions") if isinstance(source, Mapping) else None
        )
        if not isinstance(transitions, list):
            raise ValueError("mistake-locator transitions are unavailable")
        observations = branch["observations"]
        if expected_runtime.mode == UNAVAILABLE and observations:
            raise ValueError("unavailable mistake locator emitted observations")
        if expected_runtime.mode == LEARNED and len(observations) != len(
            transitions
        ):
            raise ValueError("mistake-locator observation coverage differs")
        for ordinal, observation in enumerate(observations):
            _validate_observation(
                observation,
                runtime=expected_runtime,
                branch_index=branch_index,
                source_transition=transitions[ordinal],
            )
            observation_count += 1
        prediction, probability = _prediction(
            observations,
            runtime=expected_runtime,
        )
        if (
            branch["predicted_error_transition"] != prediction
            or branch["predicted_error_probability"] != probability
        ):
            raise ValueError("mistake-locator branch prediction differs")
        candidate_count += int(prediction is not None)
        predictions.append(prediction)
    selected = receipt["selected_branch"]
    if (
        receipt["observation_count"] != observation_count
        or receipt["candidate_count"] != candidate_count
        or type(selected) is not int
        or not 0 <= selected < expected_n_branches
        or receipt["selected_branch_candidate"] != predictions[selected]
        or receipt["localization_admitted"]
        is not (expected_runtime.mode == LEARNED)
        or receipt["repair_steering_authorized"] is not False
    ):
        raise ValueError("mistake-locator aggregate evidence differs")
    return receipt


__all__ = [
    "LEARNED",
    "MISTAKE_LOCATOR_RECEIPT_SCHEMA",
    "MISTAKE_OBSERVATION_SCHEMA",
    "UNAVAILABLE",
    "MistakeLocatorRuntime",
    "build_mistake_locator_receipt",
    "validate_mistake_locator_receipt",
]
