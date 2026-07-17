"""Layer-schedule programs: each transformer block becomes an instruction.

The ordinary schedule (0,1,…,63, once each) is one point in an enormous
family of virtual architectures the same frozen tensors can execute. A
``LayerSchedule`` is a validated neural program over the recurrent region:

    [(start, end, repeats, alpha_override), ...]

Guarantees enforced here, not hoped for:
- Programs are validated against the model's actual topology and the
  episode's compute budget BEFORE execution — an invalid program never
  touches the model.
- Canonical serialization + content hash so every receipt names exactly
  which program ran.
- The library tracks per-domain reliability with Wilson lower bounds (the
  same statistics the Verifier Foundry uses) — a schedule is only ever
  preferred on evidence, never on vibes.
- Search is evolutionary, seeded, and budgeted; candidates are scored ONLY
  by a caller-supplied evaluator (which must itself be verifier-backed) and
  every promotion carries provenance.

Whatever shape a program takes, the engine's final clean persist pass
guarantees coherent slot KV — a pathological program can waste compute but
cannot corrupt attention state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.brain.verifiers.foundry import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger("Aura.LatentCortex.Schedules")

# A schedule may apply at most this many window-layer applications per slot
# token, regardless of budget — programs beyond this are degenerate.
MAX_TOTAL_LAYER_REPEATS = 4096
SCHEDULE_LIBRARY_SCHEMA_VERSION = 2
COMPUTE_MATCH_RELATIVE_TOLERANCE = 0.05
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ORDERS = frozenset({"candidate_first", "default_first"})
_MAX_LAYER_APPS = (1 << 63) - 1


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: Any, *, field_name: str, limit: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > limit or any(ord(char) < 32 for char in normalized):
        raise ValueError(f"{field_name} must be a non-empty printable identifier")
    return normalized


def _safe_display(value: Any, *, limit: int = 120) -> str:
    try:
        rendered = repr(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return f"<{type(value).__name__}:unprintable>"
    if len(rendered) > limit:
        return f"{rendered[:limit]}..."
    return rendered


@dataclass(frozen=True)
class StageOp:
    """One typed instruction of the neural bytecode.

    Kinds:
      window        — run layers [start, end) ``repeats`` times (the
                      original schedule instruction; the only kind that
                      spends layer compute directly).
      exchange      — force a branch-communication exchange NOW, instead
                      of waiting for the step-count interval.
      savepoint     — snapshot every branch's latent state (one slot;
                      later savepoints overwrite).
      verify_probe  — decode a probe from the current leading branch and
                      score it with the episode verifier; with
                      ``revert_on_drop`` every branch backtracks to its
                      savepoint when the verified score fell — explicit,
                      receipted, verifier-guided backtracking.

    Every kind is validated, canonically serialized, and covered by the
    schedule hash. ``window`` payloads serialize exactly as before, so
    existing library hashes and provenance receipts remain valid.
    """

    start: int = 0
    end: int = 0
    repeats: int = 1
    alpha: float | None = None  # override the recurrence alpha for this stage
    kind: str = "window"
    revert_on_drop: bool = False

    def to_dict(self) -> dict[str, Any]:
        if self.kind != "window":
            out: dict[str, Any] = {"kind": self.kind}
            if self.revert_on_drop:
                out["revert_on_drop"] = True
            return out
        out = {"start": self.start, "end": self.end, "repeats": self.repeats}
        if self.alpha is not None:
            try:
                alpha = float(self.alpha)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("stage alpha must be a finite number") from exc
            if not math.isfinite(alpha):
                raise ValueError("stage alpha must be a finite number")
            out["alpha"] = round(alpha, 6)
        return out


STAGE_OP_KINDS: frozenset[str] = frozenset(
    {"window", "exchange", "savepoint", "verify_probe"}
)
_MAX_VERIFY_PROBES = 4


@dataclass
class LayerSchedule:
    """A validated program over the recurrent region [prelude_end, coda_start)."""

    ops: tuple[StageOp, ...]
    name: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LayerSchedule:
        if not isinstance(payload, dict):
            raise ValueError("schedule payload must be a mapping")
        unknown = sorted(set(payload) - {"name", "ops"})
        if unknown:
            raise ValueError(f"schedule contains unknown keys: {unknown}")
        raw_ops = payload.get("ops")
        if not isinstance(raw_ops, list):
            raise ValueError("schedule ops must be a list")
        if len(raw_ops) > 256:
            raise ValueError("schedule contains more than 256 ops")
        parsed: list[StageOp] = []
        for index, op in enumerate(raw_ops):
            if not isinstance(op, dict):
                raise ValueError(f"schedule op{index} must be a mapping")
            kind = op.get("kind", "window")
            if kind not in STAGE_OP_KINDS:
                raise ValueError(
                    f"schedule op{index}.kind must be one of {sorted(STAGE_OP_KINDS)}"
                )
            if kind != "window":
                op_unknown = sorted(set(op) - {"kind", "revert_on_drop"})
                if op_unknown:
                    raise ValueError(
                        f"schedule op{index} ({kind}) contains unknown keys: {op_unknown}"
                    )
                revert_on_drop = op.get("revert_on_drop", False)
                if type(revert_on_drop) is not bool:
                    raise ValueError(
                        f"schedule op{index}.revert_on_drop must be boolean"
                    )
                if revert_on_drop and kind != "verify_probe":
                    raise ValueError(
                        f"schedule op{index}.revert_on_drop only applies to verify_probe"
                    )
                parsed.append(StageOp(kind=kind, revert_on_drop=revert_on_drop))
                continue
            op_unknown = sorted(set(op) - {"start", "end", "repeats", "alpha", "kind"})
            if op_unknown:
                raise ValueError(f"schedule op{index} contains unknown keys: {op_unknown}")
            for key in ("start", "end"):
                if type(op.get(key)) is not int:
                    raise ValueError(f"schedule op{index}.{key} must be an integer")
            repeats = op.get("repeats", 1)
            if type(repeats) is not int:
                raise ValueError(f"schedule op{index}.repeats must be an integer")
            alpha = op.get("alpha")
            alpha_float: float | None = None
            if alpha is not None:
                if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                    raise ValueError(f"schedule op{index}.alpha must be a finite number")
                try:
                    alpha_float = float(alpha)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"schedule op{index}.alpha must be a finite number"
                    ) from exc
                if not math.isfinite(alpha_float):
                    raise ValueError(f"schedule op{index}.alpha must be a finite number")
            parsed.append(
                StageOp(
                    start=op["start"],
                    end=op["end"],
                    repeats=repeats,
                    alpha=alpha_float,
                )
            )
        name = payload.get("name", "")
        if not isinstance(name, str):
            raise ValueError("schedule name must be a string")
        return cls(ops=tuple(parsed), name=name[:200])

    @classmethod
    def single_window(cls, prelude_end: int, coda_start: int, repeats: int) -> LayerSchedule:
        return cls(
            ops=(StageOp(prelude_end, coda_start, repeats),),
            name=f"window[{prelude_end}:{coda_start}]x{repeats}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ops": [op.to_dict() for op in self.ops]}

    def canonical_json(self) -> str:
        # name excluded: identity is the program, not its label.
        return json.dumps([op.to_dict() for op in self.ops], sort_keys=True, separators=(",", ":"))

    @property
    def schedule_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def total_layer_repeats(self) -> int:
        return sum((op.end - op.start) * op.repeats for op in self.ops)

    def validate(self, *, prelude_end: int, coda_start: int) -> list[str]:
        """Human-readable violations; empty ⇒ the program may execute."""
        problems: list[str] = []
        if not self.ops:
            problems.append("schedule has no ops")
        total_layer_repeats = 0
        window_ops = 0
        verify_probes = 0
        savepoint_seen = False
        for i, op in enumerate(self.ops):
            kind = getattr(op, "kind", "window")
            if kind not in STAGE_OP_KINDS:
                problems.append(f"op{i} unknown kind {_safe_display(kind)}")
                continue
            if kind != "window":
                if kind == "savepoint":
                    savepoint_seen = True
                elif kind == "verify_probe":
                    verify_probes += 1
                    if op.revert_on_drop and not savepoint_seen:
                        problems.append(
                            f"op{i} verify_probe revert_on_drop needs a "
                            "preceding savepoint op"
                        )
                elif op.revert_on_drop:
                    problems.append(
                        f"op{i} revert_on_drop only applies to verify_probe"
                    )
                continue
            window_ops += 1
            if any(type(value) is not int for value in (op.start, op.end, op.repeats)):
                problems.append(f"op{i} start/end/repeats must be integers")
                continue
            if op.start < prelude_end or op.end > coda_start:
                problems.append(
                    f"op{i} [{_safe_display(op.start)}:{_safe_display(op.end)}) "
                    "escapes recurrent region "
                    f"[{prelude_end}:{coda_start})"
                )
            if op.start >= op.end:
                problems.append(
                    f"op{i} empty window "
                    f"[{_safe_display(op.start)}:{_safe_display(op.end)})"
                )
            if op.repeats < 1:
                problems.append(f"op{i} repeats {_safe_display(op.repeats)} < 1")
            if op.alpha is not None:
                try:
                    alpha = float(op.alpha)
                except (TypeError, ValueError, OverflowError):
                    alpha = math.nan
                if (
                    isinstance(op.alpha, bool)
                    or not isinstance(op.alpha, (int, float))
                    or not math.isfinite(alpha)
                    or not 0.0 < alpha <= 1.0
                ):
                    problems.append(
                        f"op{i} alpha {_safe_display(op.alpha)} outside finite (0, 1]"
                    )
            if op.start < op.end and op.repeats >= 1:
                total_layer_repeats += (op.end - op.start) * op.repeats
        if total_layer_repeats > MAX_TOTAL_LAYER_REPEATS:
            problems.append(
                f"total layer repeats {_safe_display(total_layer_repeats)} exceeds "
                f"{MAX_TOTAL_LAYER_REPEATS}"
            )
        if self.ops and window_ops == 0:
            problems.append("schedule needs at least one window op")
        if verify_probes > _MAX_VERIFY_PROBES:
            problems.append(
                f"{verify_probes} verify_probe ops exceed the "
                f"{_MAX_VERIFY_PROBES}-probe budget"
            )
        return problems


# ── Reliability-tracked library ─────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleComputeReceipt:
    """Comparable measured work for one arm of a paired schedule trial."""

    layer_apps: int
    estimator_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.layer_apps) is not int
            or self.layer_apps <= 0
            or self.layer_apps > _MAX_LAYER_APPS
        ):
            raise ValueError("schedule compute layer_apps must be a bounded positive integer")
        _require_sha256(self.estimator_sha256, field_name="estimator_sha256")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleComputeReceipt:
        if not isinstance(payload, dict) or set(payload) != {"layer_apps", "estimator_sha256"}:
            raise ValueError("schedule compute receipt has an invalid schema")
        return cls(
            layer_apps=payload["layer_apps"],
            estimator_sha256=payload["estimator_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_apps": self.layer_apps,
            "estimator_sha256": self.estimator_sha256,
        }


@dataclass(frozen=True)
class PairedScheduleOutcome:
    """One held-out candidate/default comparison with tamper-evident provenance."""

    task_id: str
    task_commitment_sha256: str
    candidate_success: bool
    default_success: bool
    candidate_compute: ScheduleComputeReceipt
    default_compute: ScheduleComputeReceipt
    run_order: str
    held_out: bool
    contamination_scan_passed: bool
    scorer_receipt_sha256: str
    verifier_receipt_sha256: str
    evaluation_run_id: str
    evaluator_build_sha256: str
    model_checkpoint_sha256: str
    evidence_protocol_sha256: str
    evidence_binding_sha256: str

    @classmethod
    def create(
        cls,
        *,
        schedule_hash: str,
        domain: str,
        task_id: str,
        task_commitment_sha256: str,
        candidate_success: bool,
        default_success: bool,
        candidate_compute: ScheduleComputeReceipt,
        default_compute: ScheduleComputeReceipt,
        run_order: str,
        held_out: bool,
        contamination_scan_passed: bool,
        scorer_receipt_sha256: str,
        verifier_receipt_sha256: str,
        evaluation_run_id: str,
        evaluator_build_sha256: str,
        model_checkpoint_sha256: str,
        evidence_protocol_sha256: str,
    ) -> PairedScheduleOutcome:
        if not isinstance(candidate_compute, ScheduleComputeReceipt) or not isinstance(
            default_compute, ScheduleComputeReceipt
        ):
            raise ValueError("paired schedule compute receipts are required")
        values: dict[str, Any] = {
            "task_id": task_id,
            "task_commitment_sha256": task_commitment_sha256,
            "candidate_success": candidate_success,
            "default_success": default_success,
            "candidate_compute": candidate_compute.to_dict(),
            "default_compute": default_compute.to_dict(),
            "run_order": run_order,
            "held_out": held_out,
            "contamination_scan_passed": contamination_scan_passed,
            "scorer_receipt_sha256": scorer_receipt_sha256,
            "verifier_receipt_sha256": verifier_receipt_sha256,
            "evaluation_run_id": evaluation_run_id,
            "evaluator_build_sha256": evaluator_build_sha256,
            "model_checkpoint_sha256": model_checkpoint_sha256,
            "evidence_protocol_sha256": evidence_protocol_sha256,
        }
        binding = cls.binding_sha256(
            schedule_hash=schedule_hash,
            domain=domain,
            values=values,
        )
        outcome = cls(
            task_id=task_id,
            task_commitment_sha256=task_commitment_sha256,
            candidate_success=candidate_success,
            default_success=default_success,
            candidate_compute=candidate_compute,
            default_compute=default_compute,
            run_order=run_order,
            held_out=held_out,
            contamination_scan_passed=contamination_scan_passed,
            scorer_receipt_sha256=scorer_receipt_sha256,
            verifier_receipt_sha256=verifier_receipt_sha256,
            evaluation_run_id=evaluation_run_id,
            evaluator_build_sha256=evaluator_build_sha256,
            model_checkpoint_sha256=model_checkpoint_sha256,
            evidence_protocol_sha256=evidence_protocol_sha256,
            evidence_binding_sha256=binding,
        )
        outcome.validate(schedule_hash=schedule_hash, domain=domain)
        return outcome

    @staticmethod
    def binding_sha256(*, schedule_hash: str, domain: str, values: dict[str, Any]) -> str:
        _require_sha256(schedule_hash, field_name="schedule_hash")
        normalized_domain = _require_identifier(domain, field_name="domain").lower()
        return _canonical_sha256(
            {
                "domain": normalized_domain,
                "schedule_hash": schedule_hash,
                "outcome": values,
            }
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        schedule_hash: str,
        domain: str,
    ) -> PairedScheduleOutcome:
        expected = {
            "task_id",
            "task_commitment_sha256",
            "candidate_success",
            "default_success",
            "candidate_compute",
            "default_compute",
            "run_order",
            "held_out",
            "contamination_scan_passed",
            "scorer_receipt_sha256",
            "verifier_receipt_sha256",
            "evaluation_run_id",
            "evaluator_build_sha256",
            "model_checkpoint_sha256",
            "evidence_protocol_sha256",
            "evidence_binding_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("paired schedule outcome has an invalid schema")
        outcome = cls(
            task_id=payload["task_id"],
            task_commitment_sha256=payload["task_commitment_sha256"],
            candidate_success=payload["candidate_success"],
            default_success=payload["default_success"],
            candidate_compute=ScheduleComputeReceipt.from_dict(payload["candidate_compute"]),
            default_compute=ScheduleComputeReceipt.from_dict(payload["default_compute"]),
            run_order=payload["run_order"],
            held_out=payload["held_out"],
            contamination_scan_passed=payload["contamination_scan_passed"],
            scorer_receipt_sha256=payload["scorer_receipt_sha256"],
            verifier_receipt_sha256=payload["verifier_receipt_sha256"],
            evaluation_run_id=payload["evaluation_run_id"],
            evaluator_build_sha256=payload["evaluator_build_sha256"],
            model_checkpoint_sha256=payload["model_checkpoint_sha256"],
            evidence_protocol_sha256=payload["evidence_protocol_sha256"],
            evidence_binding_sha256=payload["evidence_binding_sha256"],
        )
        outcome.validate(schedule_hash=schedule_hash, domain=domain)
        return outcome

    def _binding_values(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_binding_sha256")
        return payload

    def validate(self, *, schedule_hash: str, domain: str) -> None:
        _require_identifier(self.task_id, field_name="task_id")
        _require_sha256(self.task_commitment_sha256, field_name="task_commitment_sha256")
        if type(self.candidate_success) is not bool or type(self.default_success) is not bool:
            raise ValueError("paired schedule outcomes must contain boolean arm results")
        if self.run_order not in _RUN_ORDERS:
            raise ValueError("run_order must be candidate_first or default_first")
        if self.held_out is not True:
            raise ValueError("schedule promotion evidence must be held out")
        if self.contamination_scan_passed is not True:
            raise ValueError("schedule promotion evidence failed contamination screening")
        for field_name in (
            "scorer_receipt_sha256",
            "verifier_receipt_sha256",
            "evaluator_build_sha256",
            "model_checkpoint_sha256",
            "evidence_protocol_sha256",
            "evidence_binding_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        _require_identifier(self.evaluation_run_id, field_name="evaluation_run_id")
        if self.candidate_compute.estimator_sha256 != self.default_compute.estimator_sha256:
            raise ValueError("paired schedule arms used different compute estimators")
        larger = max(self.candidate_compute.layer_apps, self.default_compute.layer_apps)
        allowed_delta = max(1, math.ceil(larger * COMPUTE_MATCH_RELATIVE_TOLERANCE))
        if abs(self.candidate_compute.layer_apps - self.default_compute.layer_apps) > allowed_delta:
            raise ValueError("paired schedule arms exceeded the compute matching tolerance")
        expected_binding = self.binding_sha256(
            schedule_hash=schedule_hash,
            domain=domain,
            values=self._binding_values(),
        )
        if self.evidence_binding_sha256 != expected_binding:
            raise ValueError("paired schedule evidence binding does not match its contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_commitment_sha256": self.task_commitment_sha256,
            "candidate_success": self.candidate_success,
            "default_success": self.default_success,
            "candidate_compute": self.candidate_compute.to_dict(),
            "default_compute": self.default_compute.to_dict(),
            "run_order": self.run_order,
            "held_out": self.held_out,
            "contamination_scan_passed": self.contamination_scan_passed,
            "scorer_receipt_sha256": self.scorer_receipt_sha256,
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluator_build_sha256": self.evaluator_build_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "evidence_protocol_sha256": self.evidence_protocol_sha256,
            "evidence_binding_sha256": self.evidence_binding_sha256,
        }


@dataclass
class ScheduleRecord:
    schedule: LayerSchedule
    domain: str
    outcomes: dict[str, PairedScheduleOutcome] = field(default_factory=dict)

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(int(outcome.candidate_success) for outcome in self.outcomes.values())

    @property
    def default_successes(self) -> int:
        return sum(int(outcome.default_success) for outcome in self.outcomes.values())

    @property
    def reliability_lb(self) -> float:
        return wilson_lower_bound(self.successes, self.trials)

    @property
    def default_reliability_ub(self) -> float:
        return wilson_upper_bound(self.default_successes, self.trials)

    def _profile(self) -> tuple[str, str, str] | None:
        profiles = {
            (
                outcome.evaluator_build_sha256,
                outcome.model_checkpoint_sha256,
                outcome.evidence_protocol_sha256,
            )
            for outcome in self.outcomes.values()
        }
        if len(profiles) != 1:
            return None
        return next(iter(profiles))

    def promotion_ready(self, *, min_trials: int) -> bool:
        if self.trials < min_trials or self._profile() is None:
            return False
        candidate_first = sum(
            outcome.run_order == "candidate_first" for outcome in self.outcomes.values()
        )
        default_first = self.trials - candidate_first
        return (
            abs(candidate_first - default_first) <= 1
            and self.reliability_lb > self.default_reliability_ub
        )

    def add_outcome(self, outcome: PairedScheduleOutcome, *, allow_identical: bool = False) -> bool:
        outcome.validate(schedule_hash=self.schedule.schedule_hash, domain=self.domain)
        existing = self.outcomes.get(outcome.task_id)
        if existing is not None:
            if allow_identical and existing == outcome:
                return False
            raise ValueError(f"duplicate or conflicting schedule task_id: {outcome.task_id}")
        for prior in self.outcomes.values():
            if prior.task_commitment_sha256 == outcome.task_commitment_sha256:
                raise ValueError("replayed schedule task commitment")
            if prior.scorer_receipt_sha256 == outcome.scorer_receipt_sha256:
                raise ValueError("replayed schedule scorer receipt")
            if prior.verifier_receipt_sha256 == outcome.verifier_receipt_sha256:
                raise ValueError("replayed schedule verifier receipt")
        current_profile = self._profile()
        new_profile = (
            outcome.evaluator_build_sha256,
            outcome.model_checkpoint_sha256,
            outcome.evidence_protocol_sha256,
        )
        if current_profile is not None and current_profile != new_profile:
            raise ValueError("schedule evidence profile changed within one candidate record")
        self.outcomes[outcome.task_id] = outcome
        return True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleRecord:
        if not isinstance(payload, dict) or set(payload) != {"schedule", "domain", "outcomes"}:
            raise ValueError("schedule record has an invalid schema")
        schedule = LayerSchedule.from_dict(payload["schedule"])
        domain = _require_identifier(payload["domain"], field_name="domain").lower()
        raw_outcomes = payload["outcomes"]
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) > 100_000:
            raise ValueError("schedule outcomes must be a bounded list")
        record = cls(schedule=schedule, domain=domain)
        for raw_outcome in raw_outcomes:
            record.add_outcome(
                PairedScheduleOutcome.from_dict(
                    raw_outcome,
                    schedule_hash=schedule.schedule_hash,
                    domain=domain,
                )
            )
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict(),
            "domain": self.domain,
            "outcomes": [self.outcomes[key].to_dict() for key in sorted(self.outcomes)],
        }


class ScheduleLibrary:
    """Per-domain schedule ledger gated by matched held-out evidence."""

    MIN_TRIALS = 20
    MAX_SAVE_RETRIES = 4

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[tuple[str, str], ScheduleRecord] = {}
        self._revision = 0
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            self._load()

    # ── Persistence (governed writes; loads are plain reads) ───────────
    @staticmethod
    def _parse_store(payload: Any) -> tuple[int, dict[tuple[str, str], ScheduleRecord]]:
        if not isinstance(payload, dict) or set(payload) != {"version", "revision", "records"}:
            raise ValueError("schedule library root has an invalid schema")
        if payload["version"] != SCHEDULE_LIBRARY_SCHEMA_VERSION:
            raise ValueError("unsupported schedule library schema version")
        revision = payload["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("schedule library revision must be a non-negative integer")
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or len(raw_records) > 10_000:
            raise ValueError("schedule library records must be a bounded list")
        records: dict[tuple[str, str], ScheduleRecord] = {}
        for raw_record in raw_records:
            record = ScheduleRecord.from_dict(raw_record)
            key = (record.domain, record.schedule.schedule_hash)
            if key in records:
                raise ValueError("duplicate schedule record in persisted library")
            records[key] = record
        return revision, records

    def _read_store(self) -> tuple[int, dict[tuple[str, str], ScheduleRecord]]:
        if self._path is None:
            return 0, {}
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return self._parse_store(payload)

    def _load(self) -> None:
        try:
            revision, records = self._read_store()
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Schedule library unreadable at %s: %s - starting empty",
                self._path,
                exc,
            )
            return
        with self._lock:
            self._revision = revision
            self._records = records

    def _merge_records(self, incoming: dict[tuple[str, str], ScheduleRecord]) -> None:
        for key, incoming_record in incoming.items():
            current = self._records.get(key)
            if current is None:
                self._records[key] = incoming_record
                continue
            for task_id in sorted(incoming_record.outcomes):
                current.add_outcome(
                    incoming_record.outcomes[task_id],
                    allow_identical=True,
                )

    def _serialized_payload(self, revision: int) -> bytes:
        records = [self._records[key].to_dict() for key in sorted(self._records)]
        return json.dumps(
            {
                "version": SCHEDULE_LIBRARY_SCHEMA_VERSION,
                "revision": revision,
                "records": records,
            },
            indent=1,
            sort_keys=True,
        ).encode("utf-8")

    def save(self) -> bool:
        if self._path is None:
            return False
        try:
            from core.brain.llm.latent_cortex.persistence import (
                StaleScheduleLibraryError,
                get_latent_cortex_persistence,
            )

            with self._lock:
                for _attempt in range(self.MAX_SAVE_RETRIES):
                    expected_revision = self._revision
                    next_revision = expected_revision + 1
                    try:
                        get_latent_cortex_persistence().save_schedule_library(
                            self._path,
                            self._serialized_payload(next_revision),
                            expected_revision=expected_revision,
                        )
                    except StaleScheduleLibraryError:
                        disk_revision, disk_records = self._read_store()
                        self._merge_records(disk_records)
                        self._revision = disk_revision
                        continue
                    self._revision = next_revision
                    return True
                raise RuntimeError("schedule library remained stale after bounded CAS retries")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                exc,
                action="kept schedule library in memory after persist failed",
            )
            return False

    # ── Evidence ────────────────────────────────────────────────────────
    def record_paired_outcome(
        self,
        schedule: LayerSchedule,
        domain: str,
        outcome: PairedScheduleOutcome,
    ) -> ScheduleRecord:
        if not isinstance(schedule, LayerSchedule):
            raise ValueError("schedule outcome requires a LayerSchedule")
        normalized_domain = _require_identifier(domain, field_name="domain").lower()
        if not isinstance(outcome, PairedScheduleOutcome):
            raise ValueError("schedule outcome requires a PairedScheduleOutcome")
        key = (normalized_domain, schedule.schedule_hash)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = ScheduleRecord(schedule=schedule, domain=normalized_domain)
                self._records[key] = record
            record.add_outcome(outcome)
            return record

    def best_for_domain(
        self, domain: str, *, prelude_end: int, coda_start: int, default_repeats: int
    ) -> LayerSchedule:
        default = LayerSchedule.single_window(prelude_end, coda_start, default_repeats)
        normalized_domain = str(domain or "").strip().lower()
        with self._lock:
            records = list(self._records.items())

        best, best_lb = default, 0.0
        for (record_domain, _), record in records:
            if record_domain != normalized_domain:
                continue
            if record.schedule.schedule_hash == default.schedule_hash:
                continue
            if record.schedule.validate(prelude_end=prelude_end, coda_start=coda_start):
                continue
            if not record.promotion_ready(min_trials=self.MIN_TRIALS):
                continue
            if record.reliability_lb > best_lb:
                best, best_lb = record.schedule, record.reliability_lb
        return best

    def status(self) -> dict[str, Any]:
        domains: dict[str, int] = {}
        with self._lock:
            records = list(self._records.items())
            revision = self._revision
        for (domain, _schedule_hash), _record in records:
            domains[domain] = domains.get(domain, 0) + 1
        return {
            "records": len(records),
            "observations": sum(record.trials for _key, record in records),
            "domains": domains,
            "revision": revision,
        }


# ── Evolutionary schedule search ────────────────────────────────────────


@dataclass
class SearchResult:
    best: LayerSchedule
    best_score: float
    evaluated: int
    history: list[dict[str, Any]] = field(default_factory=list)


class ScheduleSearch:
    """Budgeted, seeded evolutionary search over schedule programs.

    The evaluator is the honesty boundary: it must return a score derived
    from VERIFIED task outcomes (the experiments harness provides one). The
    search itself never inspects model internals — it only proposes programs
    and listens to evidence.
    """

    def __init__(
        self,
        *,
        prelude_end: int,
        coda_start: int,
        max_repeats: int = 8,
        seed: int = 0,
    ) -> None:
        if coda_start - prelude_end < 2:
            raise ValueError("recurrent region too small to search")
        self._p = prelude_end
        self._c = coda_start
        self._max_repeats = max_repeats
        self._rng = random.Random(seed)

    # ── Mutation operators ──────────────────────────────────────────────
    def _mutate(self, schedule: LayerSchedule) -> LayerSchedule:
        ops = list(schedule.ops)
        choice = self._rng.random()
        idx = self._rng.randrange(len(ops))
        op = ops[idx]
        if choice < 0.30:  # adjust repeats
            delta = self._rng.choice((-2, -1, 1, 2))
            ops[idx] = StageOp(op.start, op.end, max(1, min(self._max_repeats, op.repeats + delta)), op.alpha)
        elif choice < 0.55:  # shift window
            shift = self._rng.choice((-2, -1, 1, 2))
            start = max(self._p, min(op.start + shift, self._c - 1))
            end = max(start + 1, min(op.end + shift, self._c))
            ops[idx] = StageOp(start, end, op.repeats, op.alpha)
        elif choice < 0.75 and op.end - op.start >= 2:  # split stage
            mid = self._rng.randrange(op.start + 1, op.end)
            ops[idx : idx + 1] = [
                StageOp(op.start, mid, op.repeats, op.alpha),
                StageOp(mid, op.end, op.repeats, op.alpha),
            ]
        elif choice < 0.90 and len(ops) >= 2:  # merge adjacent stages
            j = min(idx + 1, len(ops) - 1)
            if j != idx:
                a, b = ops[idx], ops[j]
                merged = StageOp(min(a.start, b.start), max(a.end, b.end), max(a.repeats, b.repeats), a.alpha)
                ops[idx : j + 1] = [merged]
        else:  # adjust alpha
            alpha = op.alpha if op.alpha is not None else 0.5
            alpha = min(1.0, max(0.05, alpha + self._rng.choice((-0.15, -0.05, 0.05, 0.15))))
            ops[idx] = StageOp(op.start, op.end, op.repeats, round(alpha, 4))
        mutated = LayerSchedule(ops=tuple(ops), name="searched")
        if mutated.validate(prelude_end=self._p, coda_start=self._c):
            return schedule  # invalid mutation ⇒ keep parent
        return mutated

    def run(
        self,
        evaluator: Callable[[LayerSchedule], float],
        *,
        population: int = 6,
        generations: int = 4,
        seed_schedule: LayerSchedule | None = None,
    ) -> SearchResult:
        if population < 2 or population > 128:
            raise ValueError("population outside [2, 128]")
        if generations < 1 or generations > 256:
            raise ValueError("generations outside [1, 256]")
        base = seed_schedule or LayerSchedule.single_window(self._p, self._c, 4)
        violations = base.validate(prelude_end=self._p, coda_start=self._c)
        if violations:
            raise ValueError(f"seed schedule invalid: {violations}")

        scored: dict[str, tuple[LayerSchedule, float]] = {}

        def score(s: LayerSchedule) -> float:
            key = s.schedule_hash
            if key not in scored:
                value = float(evaluator(s))
                if not math.isfinite(value):
                    raise ValueError("schedule evaluator returned a non-finite score")
                scored[key] = (s, value)
            return scored[key][1]

        pool = [base] + [self._mutate(base) for _ in range(population - 1)]
        history: list[dict[str, Any]] = []
        for gen in range(generations):
            ranked = sorted(pool, key=score, reverse=True)
            survivors = ranked[: max(2, population // 2)]
            history.append(
                {
                    "generation": gen,
                    "best_hash": survivors[0].schedule_hash,
                    "best_score": score(survivors[0]),
                    "pool": [s.schedule_hash for s in ranked],
                }
            )
            children = [self._mutate(self._rng.choice(survivors)) for _ in range(population - len(survivors))]
            pool = survivors + children

        best_hash = max(scored, key=lambda k: scored[k][1])
        best, best_score = scored[best_hash]
        return SearchResult(best=best, best_score=best_score, evaluated=len(scored), history=history)


__all__ = [
    "COMPUTE_MATCH_RELATIVE_TOLERANCE",
    "LayerSchedule",
    "MAX_TOTAL_LAYER_REPEATS",
    "PairedScheduleOutcome",
    "SCHEDULE_LIBRARY_SCHEMA_VERSION",
    "ScheduleComputeReceipt",
    "ScheduleLibrary",
    "ScheduleRecord",
    "ScheduleSearch",
    "SearchResult",
    "StageOp",
]
