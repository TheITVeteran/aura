"""Task-disjoint verified correction learning at the output boundary.

Unlike :mod:`episodic_output_memory`, this adapter never receives the answer
for the query it is serving.  It fits one bounded sparse residual readout from
verified calibration corrections, freezes it, and can then be evaluated on
disjoint queries.  Only tokens observed in the calibration correction set are
addressable; all other vocabulary logits remain byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.fast_weight_learning import token_sequence_sha256
from core.brain.llm.latent_cortex.verified_best import tensor_sha256

SEMANTIC_OUTPUT_ADAPTER_SCHEMA = "aura.rlc.semantic_output_adapter.v1"
SEMANTIC_MARGIN_ADAPTER_SCHEMA = "aura.rlc.semantic_output_adapter.v2"
SEMANTIC_OUTPUT_TRANSFER_SCHEMA = "aura.rlc.semantic_output_transfer_experiment.v1"
SEMANTIC_OUTPUT_GAIN_GRID = (0.0, 0.5, 1.0, 2.0)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _task_inventory_sha256(task_ids: Sequence[str]) -> str:
    normalized = sorted(set(task_ids))
    if (
        not normalized
        or len(normalized) != len(task_ids)
        or any(not isinstance(task_id, str) or not task_id for task_id in normalized)
    ):
        raise ValueError("semantic adapter task identities must be unique non-empty text")
    return _canonical_sha256(
        sorted(hashlib.sha256(item.encode("utf-8")).hexdigest() for item in normalized)
    )


def _task_sha256s(task_ids: Sequence[str]) -> list[str]:
    _task_inventory_sha256(task_ids)
    return sorted(hashlib.sha256(item.encode("utf-8")).hexdigest() for item in task_ids)


def _as_normalized_rows(keys: Any) -> np.ndarray:
    rows = np.asarray(keys, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 1:
        raise ValueError("semantic adapter requires at least two hidden-state rows")
    if not np.isfinite(rows).all():
        raise ValueError("semantic adapter hidden states must be finite")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("semantic adapter hidden state contains a zero vector")
    return rows / norms


def _validate_adapter_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    base_fields = {
        "schema",
        "erased",
        "weights_sha256",
        "tokens_sha256",
        "task_inventory_sha256",
        "task_sha256s",
        "hidden_width",
        "token_count",
        "sample_count",
        "ridge",
        "logit_scale",
        "gain",
        "applications",
    }
    if not isinstance(value, Mapping):
        raise ValueError("semantic adapter identity fields differ")
    schema = value.get("schema")
    fields = set(base_fields)
    if schema == SEMANTIC_MARGIN_ADAPTER_SCHEMA:
        fields.update(
            {
                "fit_objective",
                "effective_rank",
                "target_values_sha256",
            }
        )
    if set(value) != fields:
        raise ValueError("semantic adapter identity fields differ")
    tasks = value["task_sha256s"]
    if (
        schema not in {SEMANTIC_OUTPUT_ADAPTER_SCHEMA, SEMANTIC_MARGIN_ADAPTER_SCHEMA}
        or value["erased"] is not False
        or any(
            not _is_sha256(value[name])
            for name in (
                "weights_sha256",
                "tokens_sha256",
                "task_inventory_sha256",
            )
        )
        or not isinstance(tasks, list)
        or not tasks
        or tasks != sorted(set(tasks))
        or any(not _is_sha256(item) for item in tasks)
        or value["task_inventory_sha256"] != _canonical_sha256(tasks)
        or type(value["hidden_width"]) is not int
        or value["hidden_width"] <= 0
        or type(value["token_count"]) is not int
        or value["token_count"] <= 1
        or type(value["sample_count"]) is not int
        or value["sample_count"] < len(tasks)
        or not math.isfinite(float(value["ridge"]))
        or not 1e-8 <= float(value["ridge"]) <= 1e4
        or not math.isfinite(float(value["logit_scale"]))
        or not 0.0 < float(value["logit_scale"]) <= 64.0
        or float(value["gain"]) != 0.0
        or value["applications"] != 0
    ):
        raise ValueError("semantic adapter identity is invalid")
    if schema == SEMANTIC_MARGIN_ADAPTER_SCHEMA and (
        value["fit_objective"] != "required_logit_margin_v1"
        or type(value["effective_rank"]) is not int
        or not 0 < value["effective_rank"] <= value["sample_count"]
        or not _is_sha256(value["target_values_sha256"])
        or float(value["logit_scale"]) != 1.0
    ):
        raise ValueError("semantic margin adapter identity is invalid")
    return dict(value)


class SemanticOutputAdapter:
    """Sparse low-rank readout learned from verified token corrections.

    The fitted matrix is the dual ridge solution to a signed objective.  Each
    calibration row gives +1 credit to its verified token and -1 credit to the
    incumbent token it corrects.  This makes the learned direction answer-
    semantic rather than a generic reconstruction or norm objective.
    """

    def __init__(
        self,
        *,
        token_ids: Sequence[int],
        weights: Any,
        task_ids: Sequence[str],
        sample_count: int,
        ridge: float,
        logit_scale: float,
        fit_objective: str = "signed_correction_v1",
        effective_rank: int | None = None,
        target_values_sha256: str = "",
    ) -> None:
        tokens = tuple(int(token) for token in token_ids)
        matrix = np.asarray(weights, dtype=np.float32)
        if (
            not tokens
            or tokens != tuple(sorted(set(tokens)))
            or any(token < 0 for token in tokens)
            or matrix.ndim != 2
            or matrix.shape[1] != len(tokens)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("semantic adapter sparse readout is invalid")
        if type(sample_count) is not int or sample_count < 2:
            raise ValueError("semantic adapter sample count is invalid")
        if (
            isinstance(ridge, bool)
            or not math.isfinite(float(ridge))
            or not 1e-8 <= float(ridge) <= 1e4
            or isinstance(logit_scale, bool)
            or not math.isfinite(float(logit_scale))
            or not 0.0 < float(logit_scale) <= 64.0
        ):
            raise ValueError("semantic adapter fit parameters are invalid")
        self.token_ids = tokens
        self.weights = matrix
        self.task_ids = tuple(sorted(task_ids))
        self.task_inventory_sha256 = _task_inventory_sha256(self.task_ids)
        self.sample_count = sample_count
        self.ridge = float(ridge)
        self.logit_scale = float(logit_scale)
        self.fit_objective = fit_objective
        self.effective_rank = effective_rank
        self.target_values_sha256 = target_values_sha256
        self.gain = 0.0
        self.applications = 0
        self.erased = False
        self._mlx_weights = None

    @classmethod
    def fit(
        cls,
        keys: Any,
        target_tokens: Sequence[int],
        incumbent_tokens: Sequence[int],
        *,
        task_ids: Sequence[str],
        ridge: float = 1.0,
        logit_scale: float = 8.0,
    ) -> SemanticOutputAdapter:
        rows = _as_normalized_rows(keys)
        targets = tuple(int(token) for token in target_tokens)
        incumbents = tuple(int(token) for token in incumbent_tokens)
        tasks = tuple(task_ids)
        if (
            len(targets) != rows.shape[0]
            or len(incumbents) != rows.shape[0]
            or len(tasks) != rows.shape[0]
            or any(token < 0 for token in (*targets, *incumbents))
            or any(
                target == incumbent for target, incumbent in zip(targets, incumbents, strict=True)
            )
        ):
            raise ValueError("semantic adapter corrections are not row-aligned")
        if len(set(tasks)) < 2:
            raise ValueError("semantic adapter requires corrections from multiple tasks")
        if (
            isinstance(ridge, bool)
            or not math.isfinite(float(ridge))
            or not 1e-8 <= float(ridge) <= 1e4
        ):
            raise ValueError("semantic adapter ridge is invalid")

        token_ids = tuple(sorted(set((*targets, *incumbents))))
        columns = {token: index for index, token in enumerate(token_ids)}
        labels = np.zeros((rows.shape[0], len(token_ids)), dtype=np.float32)
        for index, (target, incumbent) in enumerate(zip(targets, incumbents, strict=True)):
            labels[index, columns[target]] = 1.0
            labels[index, columns[incumbent]] = -1.0

        # The dual form solves an N x N system rather than hidden_width x
        # hidden_width.  Calibration sets are intentionally bounded, so this
        # remains cheap even on the resident checkpoint's wide coda.
        gram = rows @ rows.T
        gram.flat[:: gram.shape[0] + 1] += float(ridge)
        try:
            coefficients = np.linalg.solve(gram, labels)
        except np.linalg.LinAlgError as exc:
            raise ValueError("semantic adapter ridge system is singular") from exc
        weights = rows.T @ coefficients
        if not np.isfinite(weights).all() or not np.any(weights):
            raise FloatingPointError("semantic adapter fit produced no finite correction")
        return cls(
            token_ids=token_ids,
            weights=weights,
            task_ids=sorted(set(tasks)),
            sample_count=len(tasks),
            ridge=float(ridge),
            logit_scale=logit_scale,
        )

    @classmethod
    def fit_required_margins(
        cls,
        keys: Any,
        target_tokens: Sequence[int],
        required_margins: Sequence[float],
        *,
        task_ids: Sequence[str],
        max_rank: int = 32,
        ridge: float = 1e-4,
    ) -> SemanticOutputAdapter:
        """Fit a bounded-rank readout to measured target-logit deficits."""

        rows = _as_normalized_rows(keys)
        targets = tuple(int(token) for token in target_tokens)
        required = np.asarray(required_margins, dtype=np.float32)
        tasks = tuple(task_ids)
        if (
            len(targets) != rows.shape[0]
            or required.shape != (rows.shape[0],)
            or len(tasks) != rows.shape[0]
            or any(token < 0 for token in targets)
            or not np.isfinite(required).all()
            or np.any(required <= 0.0)
            or len(set(tasks)) < 2
        ):
            raise ValueError("semantic margin adapter rows are not aligned")
        if type(max_rank) is not int or max_rank <= 0:
            raise ValueError("semantic margin adapter rank bound must be positive")
        if (
            isinstance(ridge, bool)
            or not math.isfinite(float(ridge))
            or not 1e-8 <= float(ridge) <= 1e4
        ):
            raise ValueError("semantic margin adapter ridge is invalid")

        token_ids = tuple(sorted(set(targets)))
        columns = {token: index for index, token in enumerate(token_ids)}
        desired = np.zeros((len(targets), len(token_ids)), dtype=np.float32)
        for index, token in enumerate(targets):
            desired[index, columns[token]] = required[index]
        gram = rows @ rows.T
        gram.flat[:: gram.shape[0] + 1] += float(ridge)
        try:
            weights = rows.T @ np.linalg.solve(gram, desired)
        except np.linalg.LinAlgError as exc:
            raise ValueError("semantic margin adapter ridge system is singular") from exc
        left, singular, right = np.linalg.svd(weights, full_matrices=False)
        numerical_rank = int(np.count_nonzero(singular > 1e-7))
        effective_rank = min(max_rank, numerical_rank)
        if effective_rank <= 0:
            raise ValueError("semantic margin adapter fit collapsed")
        weights = (left[:, :effective_rank] * singular[np.newaxis, :effective_rank]) @ right[
            :effective_rank
        ]
        if not np.isfinite(weights).all() or not np.any(weights):
            raise FloatingPointError("semantic margin adapter fit produced no correction")
        return cls(
            token_ids=token_ids,
            weights=weights,
            task_ids=sorted(set(tasks)),
            sample_count=len(tasks),
            ridge=float(ridge),
            logit_scale=1.0,
            fit_objective="required_logit_margin_v1",
            effective_rank=effective_rank,
            target_values_sha256=tensor_sha256(required),
        )

    def reset(self, *, gain: float) -> None:
        if (
            isinstance(gain, bool)
            or not math.isfinite(float(gain))
            or not 0.0 <= float(gain) <= 2.0
        ):
            raise ValueError("semantic adapter gain is outside [0, 2]")
        if self.erased:
            raise RuntimeError("semantic adapter was erased")
        self.gain = float(gain)
        self.applications = 0

    def apply(self, hidden: Any, logits: Any):
        """Apply the frozen sparse residual to one decoder boundary."""

        if self.erased or self.gain <= 0.0:
            return logits
        if getattr(hidden, "ndim", 0) != 3 or getattr(logits, "ndim", 0) != 3:
            raise ValueError("semantic adapter requires batched sequence tensors")
        if int(hidden.shape[-1]) != int(self.weights.shape[0]):
            raise ValueError("semantic adapter hidden width differs")
        if self.token_ids[-1] >= int(logits.shape[-1]):
            raise ValueError("semantic adapter token exceeds vocabulary")

        import mlx.core as mx

        if self._mlx_weights is None:
            self._mlx_weights = mx.array(self.weights)
            mx.eval(self._mlx_weights)
        query = hidden[0, -1].astype(mx.float32)
        query = query / mx.maximum(mx.linalg.norm(query), 1e-8)
        signed = query @ self._mlx_weights
        if self.fit_objective == "required_logit_margin_v1":
            delta = mx.maximum(signed, 0.0) * self.gain
        else:
            delta = mx.tanh(signed) * (self.gain * self.logit_scale)
        updated = logits.at[0, -1, list(self.token_ids)].add(delta.astype(logits.dtype))
        self.applications += 1
        return updated

    def erase(self) -> None:
        self.weights = None
        self._mlx_weights = None
        self.token_ids = ()
        self.task_ids = ()
        self.gain = 0.0
        self.erased = True

    def receipt(self) -> dict[str, Any]:
        if self.erased:
            return {
                "schema": SEMANTIC_OUTPUT_ADAPTER_SCHEMA,
                "erased": True,
                "token_count": 0,
                "sample_count": self.sample_count,
            }
        receipt = {
            "schema": SEMANTIC_OUTPUT_ADAPTER_SCHEMA,
            "erased": False,
            "weights_sha256": tensor_sha256(self.weights),
            "tokens_sha256": token_sequence_sha256(self.token_ids),
            "task_inventory_sha256": self.task_inventory_sha256,
            "task_sha256s": _task_sha256s(self.task_ids),
            "hidden_width": int(self.weights.shape[0]),
            "token_count": len(self.token_ids),
            "sample_count": self.sample_count,
            "ridge": self.ridge,
            "logit_scale": self.logit_scale,
            "gain": self.gain,
            "applications": self.applications,
        }
        if self.fit_objective == "required_logit_margin_v1":
            receipt.update(
                {
                    "schema": SEMANTIC_MARGIN_ADAPTER_SCHEMA,
                    "fit_objective": self.fit_objective,
                    "effective_rank": self.effective_rank,
                    "target_values_sha256": self.target_values_sha256,
                }
            )
        return receipt


class SemanticOutputEmbeddingProxy:
    """Temporary tied-embedding seam for capture and adapted generation.

    Qwen ties its output projection to ``embed_tokens.as_linear``.  Replacing
    only that method is brittle because Python special-method lookup bypasses
    instance monkeypatches.  This narrow proxy delegates input embedding
    exactly and intercepts only the explicit output projection call.
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self.adapter: SemanticOutputAdapter | None = None
        self.capture = False
        self.last_hidden = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(self, value: Any) -> Any:
        return self.base(value)

    def as_linear(self, hidden: Any) -> Any:
        if self.capture:
            import mlx.core as mx

            self.last_hidden = mx.stop_gradient(hidden.astype(mx.float32))
            mx.eval(self.last_hidden)
        logits = self.base.as_linear(hidden)
        if self.adapter is not None:
            logits = self.adapter.apply(hidden, logits)
        return logits

    def attach(self, adapter: SemanticOutputAdapter) -> None:
        if self.adapter is not None:
            raise RuntimeError("semantic output proxy already has an adapter")
        if adapter.erased:
            raise RuntimeError("cannot attach an erased semantic adapter")
        self.adapter = adapter

    def detach(self) -> SemanticOutputAdapter:
        if self.adapter is None:
            raise RuntimeError("semantic output proxy has no adapter")
        adapter = self.adapter
        self.adapter = None
        return adapter


def deterministic_sham_tokens(
    target_tokens: Sequence[int],
    *,
    task_ids: Sequence[str],
    incumbent_tokens: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return a deterministic cross-task wrong-label control."""

    targets = tuple(int(token) for token in target_tokens)
    tasks = tuple(task_ids)
    incumbents = (
        tuple(int(token) for token in incumbent_tokens) if incumbent_tokens is not None else None
    )
    if (
        len(targets) < 2
        or len(tasks) != len(targets)
        or (incumbents is not None and len(incumbents) != len(targets))
    ):
        raise ValueError("sham labels require aligned target tasks")
    destinations = list(range(len(tasks)))
    candidates = {
        destination: [
            source
            for source in range(len(targets))
            if targets[source] != targets[destination]
            and (incumbents is None or targets[source] != incumbents[destination])
        ]
        for destination in destinations
    }
    if any(not sources for sources in candidates.values()):
        raise ValueError("no matched deterministic sham derangement exists")
    destinations.sort(
        key=lambda destination: (
            len(candidates[destination]),
            tasks[destination],
            destination,
        )
    )
    source_owner: dict[int, int] = {}

    def assign(destination: int, visited: set[int]) -> bool:
        for source in sorted(
            candidates[destination],
            key=lambda index: (targets[index], tasks[index], index),
        ):
            if source in visited:
                continue
            visited.add(source)
            owner = source_owner.get(source)
            if owner is None or assign(owner, visited):
                source_owner[source] = destination
                return True
        return False

    for destination in destinations:
        if not assign(destination, set()):
            raise ValueError("no matched deterministic sham derangement exists")
    source_by_destination = {destination: source for source, destination in source_owner.items()}
    if len(source_by_destination) != len(targets):
        raise RuntimeError("deterministic sham assignment is incomplete")
    return tuple(targets[source_by_destination[index]] for index in range(len(targets)))


def build_semantic_output_transfer_receipt(
    *,
    treatment_identity: Mapping[str, Any],
    sham_identity: Mapping[str, Any],
    validation_task_ids: Sequence[str],
    test_task_ids: Sequence[str],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    erase_proven: bool,
) -> dict[str, Any]:
    """Freeze gain on validation and adjudicate one teacher-free test split."""

    treatment_identity = _validate_adapter_identity(treatment_identity)
    sham_identity = _validate_adapter_identity(sham_identity)
    train_sha = treatment_identity["task_inventory_sha256"]
    train_tasks = treatment_identity["task_sha256s"]
    if (
        train_sha != sham_identity["task_inventory_sha256"]
        or train_tasks != sham_identity["task_sha256s"]
        or treatment_identity["hidden_width"] != sham_identity["hidden_width"]
        or treatment_identity["token_count"] != sham_identity["token_count"]
        or treatment_identity["sample_count"] != sham_identity["sample_count"]
        or treatment_identity["ridge"] != sham_identity["ridge"]
        or treatment_identity["logit_scale"] != sham_identity["logit_scale"]
        or treatment_identity["tokens_sha256"] != sham_identity["tokens_sha256"]
        or treatment_identity["weights_sha256"] == sham_identity["weights_sha256"]
    ):
        raise ValueError("semantic transfer adapters are not matched")
    if treatment_identity["schema"] != sham_identity["schema"]:
        raise ValueError("semantic transfer adapter schemas differ")
    if treatment_identity["schema"] == SEMANTIC_MARGIN_ADAPTER_SCHEMA and (
        treatment_identity["fit_objective"] != sham_identity["fit_objective"]
        or treatment_identity["effective_rank"] != sham_identity["effective_rank"]
    ):
        raise ValueError("semantic transfer margin adapters are not matched")
    validation = tuple(validation_task_ids)
    test = tuple(test_task_ids)
    validation_tasks = _task_sha256s(validation)
    test_tasks = _task_sha256s(test)
    validation_sha = _task_inventory_sha256(validation)
    test_sha = _task_inventory_sha256(test)
    if set(train_tasks) & (set(validation_tasks) | set(test_tasks)) or set(validation_tasks) & set(
        test_tasks
    ):
        raise ValueError("semantic transfer task splits overlap")

    expected_gains = list(SEMANTIC_OUTPUT_GAIN_GRID)
    normalized_validation = [dict(row) for row in validation_rows]
    if [row.get("gain") for row in normalized_validation] != expected_gains:
        raise ValueError("semantic transfer validation gain inventory differs")
    for row in normalized_validation:
        if set(row) != {"gain", "baseline_mean", "treatment_mean", "sham_mean"} or any(
            not math.isfinite(float(row[name])) or not 0.0 <= float(row[name]) <= 1.0
            for name in ("baseline_mean", "treatment_mean", "sham_mean")
        ):
            raise ValueError("semantic transfer validation row is invalid")
    selected = max(
        normalized_validation,
        key=lambda row: (
            float(row["treatment_mean"])
            - max(float(row["baseline_mean"]), float(row["sham_mean"])),
            -float(row["gain"]),
        ),
    )
    selected_gain = float(selected["gain"])

    normalized_test = [dict(row) for row in test_rows]
    if len(normalized_test) != len(test):
        raise ValueError("semantic transfer test rows differ from sealed inventory")
    for row in normalized_test:
        if (
            set(row)
            != {
                "task_id_sha256",
                "baseline_score",
                "treatment_score",
                "sham_score",
                "baseline_tokens_sha256",
                "treatment_tokens_sha256",
                "sham_tokens_sha256",
            }
            or not _is_sha256(row["task_id_sha256"])
            or any(
                not _is_sha256(row[name])
                for name in (
                    "baseline_tokens_sha256",
                    "treatment_tokens_sha256",
                    "sham_tokens_sha256",
                )
            )
            or any(
                not math.isfinite(float(row[name])) or not 0.0 <= float(row[name]) <= 1.0
                for name in ("baseline_score", "treatment_score", "sham_score")
            )
        ):
            raise ValueError("semantic transfer test row is invalid")
    baseline_mean = sum(float(row["baseline_score"]) for row in normalized_test) / len(test)
    treatment_mean = sum(float(row["treatment_score"]) for row in normalized_test) / len(test)
    sham_mean = sum(float(row["sham_score"]) for row in normalized_test) / len(test)
    regressions = sum(
        float(row["treatment_score"]) + 1e-9 < float(row["baseline_score"])
        for row in normalized_test
    )
    accepted = bool(
        selected_gain > 0.0
        and treatment_mean > baseline_mean + 1e-6
        and treatment_mean > sham_mean + 1e-6
        and regressions == 0
        and erase_proven is True
    )
    payload = {
        "schema": SEMANTIC_OUTPUT_TRANSFER_SCHEMA,
        "treatment_identity": dict(treatment_identity),
        "sham_identity": dict(sham_identity),
        "validation_task_inventory_sha256": validation_sha,
        "test_task_inventory_sha256": test_sha,
        "validation_task_sha256s": validation_tasks,
        "test_task_sha256s": test_tasks,
        "validation_task_count": len(validation),
        "test_task_count": len(test),
        "gain_grid": expected_gains,
        "validation": normalized_validation,
        "selected_gain": selected_gain,
        "test": normalized_test,
        "test_baseline_mean": baseline_mean,
        "test_treatment_mean": treatment_mean,
        "test_sham_mean": sham_mean,
        "test_regressions": regressions,
        "teacher_available_during_test": False,
        "producer_available_during_test": False,
        "split_disjointness_asserted": True,
        "matched_control": True,
        "erase_proven": bool(erase_proven),
        "accepted": accepted,
        "capability_claim_authority": False,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_semantic_output_transfer_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the sealed split, selected gain, scores, and verdict."""

    if not isinstance(value, Mapping):
        raise ValueError("semantic transfer receipt must be a mapping")
    expected = {
        "schema",
        "treatment_identity",
        "sham_identity",
        "validation_task_inventory_sha256",
        "test_task_inventory_sha256",
        "validation_task_sha256s",
        "test_task_sha256s",
        "validation_task_count",
        "test_task_count",
        "gain_grid",
        "validation",
        "selected_gain",
        "test",
        "test_baseline_mean",
        "test_treatment_mean",
        "test_sham_mean",
        "test_regressions",
        "teacher_available_during_test",
        "producer_available_during_test",
        "split_disjointness_asserted",
        "matched_control",
        "erase_proven",
        "accepted",
        "capability_claim_authority",
        "receipt_sha256",
    }
    if set(value) != expected or value.get("schema") != SEMANTIC_OUTPUT_TRANSFER_SCHEMA:
        raise ValueError("semantic transfer receipt fields differ")
    payload = {name: value[name] for name in expected - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("semantic transfer receipt commitment differs")
    treatment = _validate_adapter_identity(value["treatment_identity"])
    sham = _validate_adapter_identity(value["sham_identity"])
    train_tasks = treatment["task_sha256s"]
    if (
        train_tasks != sham["task_sha256s"]
        or treatment["task_inventory_sha256"] != sham["task_inventory_sha256"]
        or treatment["hidden_width"] != sham["hidden_width"]
        or treatment["token_count"] != sham["token_count"]
        or treatment["sample_count"] != sham["sample_count"]
        or treatment["ridge"] != sham["ridge"]
        or treatment["logit_scale"] != sham["logit_scale"]
        or treatment["tokens_sha256"] != sham["tokens_sha256"]
        or treatment["weights_sha256"] == sham["weights_sha256"]
    ):
        raise ValueError("semantic transfer training identities differ")
    if treatment["schema"] != sham["schema"]:
        raise ValueError("semantic transfer training schemas differ")
    if treatment["schema"] == SEMANTIC_MARGIN_ADAPTER_SCHEMA and (
        treatment["fit_objective"] != sham["fit_objective"]
        or treatment["effective_rank"] != sham["effective_rank"]
    ):
        raise ValueError("semantic transfer margin training identities differ")
    validation_tasks = value["validation_task_sha256s"]
    test_tasks = value["test_task_sha256s"]
    if any(
        not isinstance(items, list)
        or not items
        or items != sorted(set(items))
        or any(not _is_sha256(item) for item in items)
        for items in (validation_tasks, test_tasks)
    ):
        raise ValueError("semantic transfer split commitments are invalid")
    if (
        set(train_tasks) & (set(validation_tasks) | set(test_tasks))
        or set(validation_tasks) & set(test_tasks)
        or value["validation_task_inventory_sha256"] != _canonical_sha256(validation_tasks)
        or value["test_task_inventory_sha256"] != _canonical_sha256(test_tasks)
        or value["validation_task_count"] != len(validation_tasks)
        or value["test_task_count"] != len(test_tasks)
    ):
        raise ValueError("semantic transfer split disjointness differs")
    validation = value["validation"]
    if (
        value["gain_grid"] != list(SEMANTIC_OUTPUT_GAIN_GRID)
        or not isinstance(validation, list)
        or [row.get("gain") for row in validation if isinstance(row, Mapping)]
        != list(SEMANTIC_OUTPUT_GAIN_GRID)
    ):
        raise ValueError("semantic transfer validation grid differs")
    for row in validation:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"gain", "baseline_mean", "treatment_mean", "sham_mean"}
            or any(
                not math.isfinite(float(row[name])) or not 0.0 <= float(row[name]) <= 1.0
                for name in ("baseline_mean", "treatment_mean", "sham_mean")
            )
        ):
            raise ValueError("semantic transfer validation row is invalid")
    selected = max(
        validation,
        key=lambda row: (
            float(row["treatment_mean"])
            - max(float(row["baseline_mean"]), float(row["sham_mean"])),
            -float(row["gain"]),
        ),
    )
    test = value["test"]
    if not isinstance(test, list) or len(test) != len(test_tasks):
        raise ValueError("semantic transfer test inventory differs")
    for row in test:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "task_id_sha256",
                "baseline_score",
                "treatment_score",
                "sham_score",
                "baseline_tokens_sha256",
                "treatment_tokens_sha256",
                "sham_tokens_sha256",
            }
            or any(
                not _is_sha256(row[name])
                for name in (
                    "task_id_sha256",
                    "baseline_tokens_sha256",
                    "treatment_tokens_sha256",
                    "sham_tokens_sha256",
                )
            )
            or any(
                not math.isfinite(float(row[name])) or not 0.0 <= float(row[name]) <= 1.0
                for name in ("baseline_score", "treatment_score", "sham_score")
            )
        ):
            raise ValueError("semantic transfer test row is invalid")
    baseline = sum(float(row["baseline_score"]) for row in test) / len(test)
    treatment_mean = sum(float(row["treatment_score"]) for row in test) / len(test)
    sham_mean = sum(float(row["sham_score"]) for row in test) / len(test)
    regressions = sum(
        float(row["treatment_score"]) + 1e-9 < float(row["baseline_score"]) for row in test
    )
    accepted = bool(
        float(selected["gain"]) > 0.0
        and treatment_mean > baseline + 1e-6
        and treatment_mean > sham_mean + 1e-6
        and regressions == 0
        and value["erase_proven"] is True
    )
    if (
        value["selected_gain"] != float(selected["gain"])
        or not math.isclose(value["test_baseline_mean"], baseline)
        or not math.isclose(value["test_treatment_mean"], treatment_mean)
        or not math.isclose(value["test_sham_mean"], sham_mean)
        or value["test_regressions"] != regressions
        or value["teacher_available_during_test"] is not False
        or value["producer_available_during_test"] is not False
        or value["split_disjointness_asserted"] is not True
        or value["matched_control"] is not True
        or value["capability_claim_authority"] is not False
        or value["accepted"] is not accepted
    ):
        raise ValueError("semantic transfer verdict does not reconstruct")
    return dict(value)


__all__ = [
    "SEMANTIC_OUTPUT_ADAPTER_SCHEMA",
    "SEMANTIC_MARGIN_ADAPTER_SCHEMA",
    "SEMANTIC_OUTPUT_GAIN_GRID",
    "SEMANTIC_OUTPUT_TRANSFER_SCHEMA",
    "SemanticOutputAdapter",
    "SemanticOutputEmbeddingProxy",
    "build_semantic_output_transfer_receipt",
    "deterministic_sham_tokens",
    "validate_semantic_output_transfer_receipt",
]
