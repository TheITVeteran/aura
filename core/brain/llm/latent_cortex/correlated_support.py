"""Checked error-correlation estimates and causal branch-support discounting."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.cognitive_operators import operator_for_role

CORRELATION_EVIDENCE_SCHEMA = "aura.rlc.branch_error_correlation.v1"
CORRELATED_SUPPORT_SCHEMA = "aura.rlc.correlated_support.v1"
MIN_PAIRED_OUTCOMES = 12
_SHRINKAGE_PRIOR = 24.0
_MAX_OUTCOME_ROWS = 5000


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def _phi(n11: int, n10: int, n01: int, n00: int) -> float:
    numerator = n11 * n00 - n10 * n01
    denominator = math.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    return 0.0 if denominator <= 0.0 else max(-1.0, min(1.0, numerator / denominator))


def build_correlation_evidence(
    *, bucket: str, roles: list[str] | tuple[str, ...], checked_outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Estimate pairwise error correlation from independently checked tasks."""

    if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
        raise ValueError("correlation bucket is invalid")
    role_list = list(roles)
    if (
        not role_list
        or len(role_list) != len(set(role_list))
        or any(not isinstance(role, str) or not role for role in role_list)
    ):
        raise ValueError("correlation roles must be unique non-empty strings")
    for role in role_list:
        operator_for_role(role)
    seen_tasks: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in checked_outcomes:
        if not isinstance(raw, dict) or raw.get("checked") is not True:
            raise ValueError("branch outcomes must be independently checked")
        task = raw.get("task_sha256")
        outcomes = raw.get("correct_by_role")
        if (
            not isinstance(task, str)
            or len(task) != 64
            or any(character not in "0123456789abcdef" for character in task)
            or task in seen_tasks
            or not isinstance(outcomes, dict)
            or set(outcomes) != set(role_list)
            or any(type(value) is not bool for value in outcomes.values())
        ):
            raise ValueError("checked branch outcome row is invalid")
        seen_tasks.add(task)
        rows.append({"task_sha256": task, "correct_by_role": dict(outcomes)})

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(role_list, 2):
        n11 = n10 = n01 = n00 = 0
        for row in rows:
            left_error = not row["correct_by_role"][left]
            right_error = not row["correct_by_role"][right]
            if left_error and right_error:
                n11 += 1
            elif left_error:
                n10 += 1
            elif right_error:
                n01 += 1
            else:
                n00 += 1
        n = n11 + n10 + n01 + n00
        raw_phi = _phi(n11, n10, n01, n00)
        enough = n >= MIN_PAIRED_OUTCOMES
        shrunk = max(0.0, raw_phi) * n / (n + _SHRINKAGE_PRIOR) if enough else 0.0
        pairs.append(
            {
                "pair": _pair_key(left, right),
                "left": left,
                "right": right,
                "n": n,
                "error_table": {"both": n11, "left_only": n10, "right_only": n01, "neither": n00},
                "phi": round(raw_phi, 8),
                "positive_shrunk_correlation": round(shrunk, 8),
                "enough_evidence": enough,
            }
        )
    payload = {
        "schema": CORRELATION_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "roles": role_list,
        "checked_tasks": len(rows),
        "minimum_paired_outcomes": MIN_PAIRED_OUTCOMES,
        "pairs": pairs,
        "evidence_state": "measured" if len(rows) >= MIN_PAIRED_OUTCOMES else "bootstrap_unmeasured",
    }
    return {**payload, "snapshot_sha256": _sha(payload)}


def validate_correlation_evidence(value: Any, *, roles: list[str]) -> dict[str, Any]:
    if value is None:
        return build_correlation_evidence(
            bucket="runtime|unmeasured", roles=roles, checked_outcomes=[]
        )
    if not isinstance(value, dict):
        raise ValueError("correlation evidence must be a mapping")
    required = {
        "schema", "bucket", "roles", "checked_tasks", "minimum_paired_outcomes",
        "pairs", "evidence_state", "snapshot_sha256",
    }
    if set(value) != required or value.get("schema") != CORRELATION_EVIDENCE_SCHEMA:
        raise ValueError("correlation evidence schema is invalid")
    if value.get("roles") != roles:
        raise ValueError("correlation evidence roles differ from runtime roles")
    payload = {key: value[key] for key in required - {"snapshot_sha256"}}
    if value.get("snapshot_sha256") != _sha(payload):
        raise ValueError("correlation evidence digest differs")
    expected_pairs = {_pair_key(left, right) for left, right in combinations(roles, 2)}
    rows = value.get("pairs")
    if not isinstance(rows, list) or {row.get("pair") for row in rows if isinstance(row, dict)} != expected_pairs:
        raise ValueError("correlation evidence pair coverage is invalid")
    for row in rows:
        correlation = row.get("positive_shrunk_correlation")
        n = row.get("n")
        enough = row.get("enough_evidence")
        if (
            type(n) is not int
            or n < 0
            or type(enough) is not bool
            or isinstance(correlation, bool)
            or not isinstance(correlation, (int, float))
            or not math.isfinite(float(correlation))
            or not 0.0 <= float(correlation) <= 1.0
            or enough != (n >= MIN_PAIRED_OUTCOMES)
            or (not enough and float(correlation) != 0.0)
        ):
            raise ValueError("correlation evidence estimate is invalid")
    return dict(value)


def initial_exchange_weights(
    *, roles: list[str], correlation_evidence: Any
) -> dict[int, float]:
    """Weights available before the first cross-branch exchange.

    Exact duplicate programs are collapsed immediately. Empirical penalties
    enter only after the minimum independently checked paired-task floor.
    """

    evidence = validate_correlation_evidence(correlation_evidence, roles=roles)
    empirical = {
        row["pair"]: float(row["positive_shrunk_correlation"])
        for row in evidence["pairs"]
        if row["enough_evidence"] is True
    }
    weights: dict[int, float] = {}
    for index, role in enumerate(roles):
        burden = 0.0
        for other_index, other_role in enumerate(roles):
            if index == other_index:
                continue
            duplicate_program = operator_for_role(role) is operator_for_role(other_role)
            burden += max(
                1.0 if duplicate_program else 0.0,
                empirical.get(_pair_key(role, other_role), 0.0),
            )
        weights[index] = 1.0 / (1.0 + burden)
    return weights


def _dependence_by_pair(structural: dict[str, Any], evidence: dict[str, Any]) -> dict[str, float]:
    empirical = {
        row["pair"]: float(row["positive_shrunk_correlation"])
        for row in evidence["pairs"]
        if row["enough_evidence"] is True
    }
    branches = {row["index"]: row for row in structural["branches"]}
    result: dict[str, float] = {}
    for row in structural["pairwise"]:
        left = branches[row["left"]]
        right = branches[row["right"]]
        role_pair = _pair_key(left["role_path"][0], right["role_path"][0])
        structural_similarity = max(0.0, min(1.0, 1.0 - float(row["distance"])))
        if left["structural_sha256"] == right["structural_sha256"]:
            structural_similarity = 1.0
        result[f"{row['left']}|{row['right']}"] = max(
            structural_similarity,
            empirical.get(role_pair, 0.0),
        )
    return result


def build_correlated_support_receipt(
    *, structural_diversity: dict[str, Any], correlation_evidence: Any
) -> dict[str, Any]:
    """Collapse duplicate votes and compute effective independent support."""

    if not isinstance(structural_diversity, dict) or not isinstance(
        structural_diversity.get("branches"), list
    ):
        raise ValueError("structural diversity evidence is missing")
    branches = structural_diversity["branches"]
    roles = [row["role_path"][0] for row in branches]
    evidence = validate_correlation_evidence(correlation_evidence, roles=roles)
    exchange_weights = initial_exchange_weights(
        roles=roles,
        correlation_evidence=evidence,
    )
    dependence = _dependence_by_pair(structural_diversity, evidence)
    weights: dict[int, float] = {}
    for branch in branches:
        index = int(branch["index"])
        burden = sum(
            value
            for pair, value in dependence.items()
            if str(index) in pair.split("|")
        )
        weights[index] = 1.0 / (1.0 + burden)
    raw_count = len(branches)
    effective = sum(weights.values())
    payload = {
        "schema": CORRELATED_SUPPORT_SCHEMA,
        "correlation_snapshot_sha256": evidence["snapshot_sha256"],
        "evidence_state": evidence["evidence_state"],
        "raw_support_count": raw_count,
        "effective_support_count": round(effective, 8),
        "confidence_multiplier": round(min(1.0, effective / max(1, raw_count)), 8),
        "branch_weights": [
            {"branch": index, "weight": round(weights[index], 8)}
            for index in sorted(weights)
        ],
        "exchange_weights_applied": [
            {"branch": index, "weight": round(exchange_weights[index], 8)}
            for index in sorted(exchange_weights)
        ],
        "pairwise_dependence": [
            {"pair": pair, "dependence": round(value, 8)}
            for pair, value in sorted(dependence.items())
        ],
        "duplicate_votes_collapsed": bool(structural_diversity.get("duplicate_groups")),
        "empirical_correlation_applied": any(
            row["enough_evidence"] is True
            and float(row["positive_shrunk_correlation"]) > 0.0
            for row in evidence["pairs"]
        ),
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_correlated_support_receipt(
    value: Any, *, structural_diversity: dict[str, Any], correlation_evidence: Any
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("correlated support receipt is missing")
    expected = build_correlated_support_receipt(
        structural_diversity=structural_diversity,
        correlation_evidence=correlation_evidence,
    )
    if value != expected:
        raise ValueError("correlated support receipt differs from reconstruction")
    return dict(value)


class BranchCorrelationLedger:
    """Governed durable source for independently checked branch outcomes."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            try:
                from core.config import DATA_DIR

                path = Path(DATA_DIR) / "latent_cortex" / "correlated_support" / "checked_outcomes.jsonl"
            except (ImportError, AttributeError, RuntimeError, TypeError):
                path = Path("data/latent_cortex/correlated_support/checked_outcomes.jsonl")
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._task_keys: set[tuple[str, str]] = set()
        self.restore_errors = 0
        self._restore()

    @staticmethod
    def _validate_row(row: Any) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError("branch correlation ledger row is invalid")
        required = {"schema", "bucket", "roles", "checked", "task_sha256", "correct_by_role"}
        if set(row) != required or row.get("schema") != "aura.rlc.checked_branch_outcome.v1":
            raise ValueError("branch correlation ledger schema is invalid")
        roles = row.get("roles")
        build_correlation_evidence(
            bucket=row.get("bucket"),
            roles=roles,
            checked_outcomes=[row],
        )
        return dict(row)

    def _restore(self) -> None:
        self._rows = []
        self._task_keys = set()
        try:
            if not self.path.exists():
                return
            with open(self.path, encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        row = self._validate_row(json.loads(raw))
                        key = (row["bucket"], row["task_sha256"])
                        if key in self._task_keys:
                            raise ValueError("duplicate checked task")
                        self._task_keys.add(key)
                        self._rows.append(row)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        self.restore_errors += 1
        except OSError:
            self.restore_errors += 1
        if len(self._rows) > _MAX_OUTCOME_ROWS:
            self._rows = self._rows[-_MAX_OUTCOME_ROWS:]
            self._task_keys = {
                (row["bucket"], row["task_sha256"]) for row in self._rows
            }

    def record_checked(
        self,
        *,
        bucket: str,
        task_sha256: str,
        correct_by_role: dict[str, bool],
    ) -> bool:
        roles = list(correct_by_role)
        row = {
            "schema": "aura.rlc.checked_branch_outcome.v1",
            "bucket": bucket,
            "roles": roles,
            "checked": True,
            "task_sha256": task_sha256,
            "correct_by_role": dict(correct_by_role),
        }
        row = self._validate_row(row)
        key = (bucket, task_sha256)
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.atomic_writer import interprocess_file_lock
            from core.runtime.file_write_gateway import get_file_write_gateway

            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            with interprocess_file_lock(lock_path):
                self._restore()
                if key in self._task_keys:
                    raise ValueError("checked branch task is already recorded")
                with local_internal_governed_scope(
                    "latent_branch_correlation", domain="state_mutation"
                ):
                    gateway = get_file_write_gateway()
                    gateway.append_text(
                        self.path,
                        line,
                        source="latent_branch_correlation",
                    )
                    rows = [*self._rows, row]
                    if len(rows) > _MAX_OUTCOME_ROWS:
                        rows = rows[-_MAX_OUTCOME_ROWS:]
                        gateway.write_text(
                            self.path,
                            "".join(
                                json.dumps(item, sort_keys=True, separators=(",", ":"))
                                + "\n"
                                for item in rows
                            ),
                            source="latent_branch_correlation.compact",
                        )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if key in self._task_keys:
                raise ValueError("checked branch task is already recorded") from exc
            return False
        self._task_keys.add(key)
        self._rows.append(row)
        if len(self._rows) > _MAX_OUTCOME_ROWS:
            self._rows = self._rows[-_MAX_OUTCOME_ROWS:]
        return True

    def evidence(self, *, bucket: str, roles: list[str]) -> dict[str, Any]:
        self._restore()
        role_set = set(roles)
        rows = [
            row
            for row in self._rows
            if row["bucket"] == bucket and set(row["roles"]) == role_set
        ]
        return build_correlation_evidence(
            bucket=bucket,
            roles=roles,
            checked_outcomes=rows,
        )

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "checked_outcomes": len(self._rows),
            "restore_errors": self.restore_errors,
        }


_LEDGER: BranchCorrelationLedger | None = None


def get_branch_correlation_ledger() -> BranchCorrelationLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = BranchCorrelationLedger()
    return _LEDGER


__all__ = [
    "CORRELATED_SUPPORT_SCHEMA",
    "CORRELATION_EVIDENCE_SCHEMA",
    "MIN_PAIRED_OUTCOMES",
    "BranchCorrelationLedger",
    "build_correlated_support_receipt",
    "build_correlation_evidence",
    "initial_exchange_weights",
    "get_branch_correlation_ledger",
    "validate_correlated_support_receipt",
    "validate_correlation_evidence",
]
