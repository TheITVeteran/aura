"""Live hidden-state correctness and entropy observations for RLC."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.learning.neural_uncertainty import NeuralUncertaintyHead

NEURAL_UNCERTAINTY_RECEIPT_SCHEMA = "aura.rlc.neural_uncertainty_receipt.v1"
NEURAL_UNCERTAINTY_OBSERVATION_SCHEMA = "aura.rlc.neural_uncertainty_observation.v1"
UNAVAILABLE = "unavailable"
LEARNED = "learned"
_DIRECT_SELECTION_BASES = frozenset(
    {
        "convergence",
        "neural_uncertainty",
        "process_verifier",
        "task_verifier",
    }
)
_ADMITTED_SELECTION_PIPELINES = _DIRECT_SELECTION_BASES | {
    "task_verifier_counterfactual_tiebreak",
    "task_verifier_generative_refutation_veto",
    "task_verifier_counterfactual_tiebreak_generative_refutation_veto",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector_sha256(value: list[float]) -> str:
    return hashlib.sha256(",".join(f"{item:.8f}" for item in value).encode("ascii")).hexdigest()


def _pooled_hidden(state: Any, *, width: int) -> list[float]:
    import mlx.core as mx

    if state.ndim < 1 or int(state.shape[-1]) != width or int(state.size) < width:
        raise ValueError("neural-uncertainty hidden state has incompatible shape")
    if not bool(mx.all(mx.isfinite(state)).item()):
        raise ValueError("neural-uncertainty hidden state is non-finite")
    axes = tuple(range(state.ndim - 1))
    pooled = mx.mean(state.astype(mx.float32), axis=axes)
    mx.eval(pooled)
    values = [round(float(item), 8) for item in pooled.tolist()]
    if len(values) != width or any(not math.isfinite(item) for item in values):
        raise ValueError("neural-uncertainty pooled state is invalid")
    return values


class NeuralUncertaintyRuntime:
    """Pinned learned head; unavailable mode never invents confidence."""

    def __init__(
        self,
        *,
        mode: str,
        head: NeuralUncertaintyHead | None = None,
        head_sha256: str = "",
    ) -> None:
        if mode not in {UNAVAILABLE, LEARNED}:
            raise ValueError("neural-uncertainty mode is invalid")
        if mode == UNAVAILABLE and (head is not None or head_sha256):
            raise ValueError("unavailable uncertainty cannot carry a head")
        if mode == LEARNED and (head is None or not head.calibrated or not _is_sha256(head_sha256)):
            raise ValueError("learned uncertainty requires a calibrated pinned head")
        if head is not None:
            head.validate()
        self.mode = mode
        self.head = head
        self.head_sha256 = head_sha256

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, Any] | None,
    ) -> NeuralUncertaintyRuntime:
        config = dict(value or {})
        mode = str(config.get("mode", UNAVAILABLE))
        if mode == UNAVAILABLE:
            if set(config) - {"mode"}:
                raise ValueError("unavailable uncertainty config carries a head")
            return cls(mode=UNAVAILABLE)
        if mode != LEARNED:
            raise ValueError("neural-uncertainty mode is invalid")
        if set(config) != {"mode", "head_path", "head_sha256"}:
            raise ValueError("learned neural-uncertainty config fields differ")
        path = config.get("head_path")
        digest = config.get("head_sha256")
        if not isinstance(path, str) or not path.strip() or not _is_sha256(digest):
            raise ValueError("learned neural-uncertainty config is incomplete")
        try:
            head = NeuralUncertaintyHead.load(
                Path(path).expanduser(),
                expected_sha256=digest,
            )
        except OSError as exc:
            raise ValueError("neural-uncertainty artifact is unreadable") from exc
        return cls(mode=LEARNED, head=head, head_sha256=digest)

    @property
    def manifest(self) -> dict[str, Any]:
        return {} if self.head is None else self.head.manifest()

    @property
    def input_width(self) -> int:
        return 0 if self.head is None else self.head.input_width

    def observe(
        self,
        state: Any,
        *,
        branch_index: int,
        branch_step: int,
        state_sha256: str,
    ) -> dict[str, Any]:
        if self.head is None:
            raise ValueError("neural-uncertainty observation is unavailable")
        if (
            type(branch_index) is not int
            or branch_index < 0
            or type(branch_step) is not int
            or branch_step < 0
            or not _is_sha256(state_sha256)
        ):
            raise ValueError("neural-uncertainty observation identity is invalid")
        vector = _pooled_hidden(state, width=self.head.input_width)
        estimate = self.head.estimate(vector)
        payload = {
            "schema": NEURAL_UNCERTAINTY_OBSERVATION_SCHEMA,
            "branch_index": branch_index,
            "branch_step": branch_step,
            "admitted_reasoning_sha256": state_sha256,
            "pooled_hidden": vector,
            "pooled_hidden_sha256": _vector_sha256(vector),
            "head_sha256": self.head_sha256,
            "estimate": estimate,
        }
        return {**payload, "observation_sha256": canonical_sha256(payload)}


def _validate_observation(
    value: Any,
    *,
    runtime: NeuralUncertaintyRuntime,
    branch_index: int,
    source_transition: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema",
        "branch_index",
        "branch_step",
        "admitted_reasoning_sha256",
        "pooled_hidden",
        "pooled_hidden_sha256",
        "head_sha256",
        "estimate",
        "observation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("neural-uncertainty observation fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"observation_sha256"}}
    if (
        runtime.head is None
        or row["schema"] != NEURAL_UNCERTAINTY_OBSERVATION_SCHEMA
        or row["branch_index"] != branch_index
        or row["branch_step"] != source_transition.get("branch_step")
        or row["admitted_reasoning_sha256"] != source_transition.get("admitted_reasoning_sha256")
        or row["head_sha256"] != runtime.head_sha256
        or row["observation_sha256"] != canonical_sha256(payload)
        or not isinstance(row["pooled_hidden"], list)
        or len(row["pooled_hidden"]) != runtime.head.input_width
    ):
        raise ValueError("neural-uncertainty observation identity is invalid")
    vector = [round(float(item), 8) for item in row["pooled_hidden"]]
    if (
        any(not math.isfinite(item) for item in vector)
        or vector != row["pooled_hidden"]
        or row["pooled_hidden_sha256"] != _vector_sha256(vector)
        or row["estimate"] != runtime.head.estimate(vector)
    ):
        raise ValueError("neural-uncertainty observation reconstruction failed")
    return row


def build_neural_uncertainty_receipt(
    *,
    branches: list[Any],
    runtime: NeuralUncertaintyRuntime,
    update_acceptance: Mapping[str, Any],
    selected_branch: int,
    selection_basis: str,
) -> dict[str, Any]:
    if not isinstance(update_acceptance, Mapping) or not _is_sha256(
        update_acceptance.get("receipt_sha256")
    ):
        raise ValueError("neural-uncertainty update source is invalid")
    rows = [
        {
            "branch_index": int(branch.index),
            "observations": [dict(row) for row in branch.uncertainty_trace],
        }
        for branch in branches
    ]
    if (
        type(selected_branch) is not int
        or not 0 <= selected_branch < len(branches)
        or selection_basis not in _ADMITTED_SELECTION_PIPELINES
    ):
        raise ValueError("neural-uncertainty selection identity is invalid")
    latest_scores = {
        str(row["branch_index"]): (
            row["observations"][-1]["estimate"]["correctness_probability"]
            if (row["observations"] and row["observations"][-1]["estimate"]["supported"])
            else None
        )
        for row in rows
    }
    selection_eligible = all(score is not None for score in latest_scores.values())
    payload = {
        "schema": NEURAL_UNCERTAINTY_RECEIPT_SCHEMA,
        "mode": runtime.mode,
        "head_sha256": runtime.head_sha256,
        "head_manifest": runtime.manifest,
        "update_acceptance_sha256": update_acceptance["receipt_sha256"],
        "branches": rows,
        "observation_count": sum(len(row["observations"]) for row in rows),
        "supported_count": sum(
            observation["estimate"]["supported"]
            for row in rows
            for observation in row["observations"]
        ),
        "latest_supported_scores": latest_scores,
        "selection_eligible": selection_eligible,
        "selected_branch": selected_branch,
        "selection_basis": selection_basis,
        "selection_causal": selection_basis == "neural_uncertainty",
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_neural_uncertainty_receipt(
        receipt,
        expected_runtime=runtime,
        update_acceptance=update_acceptance,
        expected_n_branches=len(branches),
    )


def validate_neural_uncertainty_receipt(
    value: Any,
    *,
    expected_runtime: NeuralUncertaintyRuntime,
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
        "supported_count",
        "latest_supported_scores",
        "selection_eligible",
        "selected_branch",
        "selection_basis",
        "selection_causal",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(update_acceptance, Mapping)
        or not _is_sha256(update_acceptance.get("receipt_sha256"))
    ):
        raise ValueError("neural-uncertainty receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    if (
        receipt["schema"] != NEURAL_UNCERTAINTY_RECEIPT_SCHEMA
        or receipt["mode"] != expected_runtime.mode
        or receipt["head_sha256"] != expected_runtime.head_sha256
        or receipt["head_manifest"] != expected_runtime.manifest
        or receipt["update_acceptance_sha256"] != update_acceptance["receipt_sha256"]
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or type(expected_n_branches) is not int
        or expected_n_branches < 1
        or not isinstance(receipt["branches"], list)
        or len(receipt["branches"]) != expected_n_branches
    ):
        raise ValueError("neural-uncertainty receipt identity is invalid")
    update_rows = update_acceptance.get("branches")
    if not isinstance(update_rows, list) or len(update_rows) != expected_n_branches:
        raise ValueError("neural-uncertainty transition source is invalid")
    observation_count = supported_count = 0
    latest_scores: dict[str, float | None] = {}
    for branch_index, branch in enumerate(receipt["branches"]):
        if (
            not isinstance(branch, Mapping)
            or set(branch) != {"branch_index", "observations"}
            or branch["branch_index"] != branch_index
            or not isinstance(branch["observations"], list)
        ):
            raise ValueError("neural-uncertainty branch evidence is invalid")
        source_branch = update_rows[branch_index]
        transitions = (
            source_branch.get("transitions") if isinstance(source_branch, Mapping) else None
        )
        if not isinstance(transitions, list):
            raise ValueError("neural-uncertainty transitions are unavailable")
        if expected_runtime.mode == UNAVAILABLE and branch["observations"]:
            raise ValueError("unavailable uncertainty emitted observations")
        if expected_runtime.mode == LEARNED and len(branch["observations"]) != len(transitions):
            raise ValueError("neural-uncertainty observation coverage differs")
        for ordinal, observation in enumerate(branch["observations"]):
            validated = _validate_observation(
                observation,
                runtime=expected_runtime,
                branch_index=branch_index,
                source_transition=transitions[ordinal],
            )
            observation_count += 1
            supported_count += int(validated["estimate"]["supported"])
        latest_scores[str(branch_index)] = (
            branch["observations"][-1]["estimate"]["correctness_probability"]
            if (branch["observations"] and branch["observations"][-1]["estimate"]["supported"])
            else None
        )
    selection_eligible = all(score is not None for score in latest_scores.values())
    selected_branch = receipt["selected_branch"]
    selection_basis = receipt["selection_basis"]
    if (
        receipt["observation_count"] != observation_count
        or receipt["supported_count"] != supported_count
        or receipt["latest_supported_scores"] != latest_scores
        or receipt["selection_eligible"] is not selection_eligible
        or type(selected_branch) is not int
        or not 0 <= selected_branch < expected_n_branches
        or selection_basis not in _ADMITTED_SELECTION_PIPELINES
        or receipt["selection_causal"] is not (selection_basis == "neural_uncertainty")
        or (
            selection_basis == "neural_uncertainty"
            and (
                not selection_eligible
                or selected_branch
                != max(
                    range(expected_n_branches),
                    key=lambda index: float(latest_scores[str(index)]),
                )
            )
        )
        or (selection_basis == "convergence" and selection_eligible)
    ):
        raise ValueError("neural-uncertainty aggregate evidence differs")
    return receipt


__all__ = [
    "LEARNED",
    "NEURAL_UNCERTAINTY_OBSERVATION_SCHEMA",
    "NEURAL_UNCERTAINTY_RECEIPT_SCHEMA",
    "UNAVAILABLE",
    "NeuralUncertaintyRuntime",
    "build_neural_uncertainty_receipt",
    "validate_neural_uncertainty_receipt",
]
