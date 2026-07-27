"""SPARK-070 falsification matrix: one registry over every required arm.

The ledger names twelve falsification rows the post-training run must
execute on fresh held-out tasks.  This module makes the matrix itself a
typed, fail-closed object: each row is either ``runnable`` today (bound
to a concrete executor over the existing experiment harness),
``enforced`` (the property is structural — enforced by construction and
proven by named suite tests rather than an A/B arm, e.g. review
blindness, for which no unblinded mode exists by design), or ``blocked``
(its machinery belongs to a later SPARK item, named explicitly).  The
matrix receipt binds every runnable row's result payload by digest, the
enforced rows to the threat-model registry, and the blocked rows to
their blockers — so "the matrix ran" can never silently mean "the rows
that were convenient ran".

``replay_falsification_matrix_receipt`` regrades a receipt against the
CURRENT registry, which makes registry drift (a row quietly reclassified
or dropped) detectable from the receipt alone.

The pre-training dry run on the untrained baseline proves the harness
end to end; the acceptance event remains the post-training run on fresh
held-out tasks against the SPARK-069 treatment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

FALSIFICATION_MATRIX_SCHEMA = "aura.latent_cortex.falsification_matrix.v1"

ROW_RUNNABLE = "runnable"
ROW_ENFORCED = "enforced"
ROW_BLOCKED = "blocked"
ROW_STATUSES = (ROW_RUNNABLE, ROW_ENFORCED, ROW_BLOCKED)

REQUIRED_ROW_IDS = (
    "recurrence_depth_curves",
    "wrong_right_transition_matrix",
    "blind_review_arms",
    "structural_diversity_arms",
    "verifier_arms",
    "latent_ablate_perturb_transplant",
    "sham_noise_noop_controls",
    "adversarial_ood_variants",
    "compute_generalization",
    "lesions_restorations",
    "fast_weight_controls",
    "non_reasoning_regressions",
)


class FalsificationMatrixError(ValueError):
    """Stable fail-closed falsification-matrix error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise FalsificationMatrixError(code)


# Why a row cannot run. The distinction matters because the three reasons
# have completely different remedies, and collapsing them into an integer list
# of SPARK ids is what let two rows drift into naming machinery that had
# already landed — hiding what they were actually waiting on.
BLOCKED_OPEN_SPARK_ITEMS = "open_spark_items"
BLOCKED_PRODUCER_ABSENT = "producer_absent"
BLOCKED_ACCEPTANCE_RUN_ONLY = "acceptance_run_only"
BLOCKED_REASONS = (
    BLOCKED_OPEN_SPARK_ITEMS,
    BLOCKED_PRODUCER_ABSENT,
    BLOCKED_ACCEPTANCE_RUN_ONLY,
)


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """One required falsification arm and how it is produced today."""

    row_id: str
    ledger_clause: str
    status: str
    producer: str
    blockers: tuple[int, ...]
    notes: str
    blocked_reason: str = ""

    def __post_init__(self) -> None:
        if (
            self.row_id not in REQUIRED_ROW_IDS
            or not isinstance(self.ledger_clause, str)
            or len(self.ledger_clause) < 12
            or self.status not in ROW_STATUSES
            or not isinstance(self.producer, str)
            or not isinstance(self.blockers, tuple)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 1 <= item <= 72
                for item in self.blockers
            )
            or len(set(self.blockers)) != len(self.blockers)
            or not isinstance(self.notes, str)
            or len(self.notes) < 12
            or not isinstance(self.blocked_reason, str)
        ):
            _fail("falsification_matrix_row_invalid")
        if self.status == ROW_RUNNABLE and (not self.producer or self.blockers):
            _fail("falsification_matrix_row_invalid")
        if self.status == ROW_ENFORCED and not self.producer:
            _fail("falsification_matrix_row_invalid")
        if self.status != ROW_BLOCKED and (
            self.blocked_reason or self.blockers
        ):
            _fail("falsification_matrix_row_invalid")
        if self.status == ROW_BLOCKED:
            if self.blocked_reason not in BLOCKED_REASONS:
                _fail("falsification_matrix_row_invalid")
            # Only the "waiting on an unlanded item" reason may carry ids, and
            # under that reason it MUST carry them: a row blocked on nothing
            # nameable is a row nobody can unblock.
            if self.blocked_reason == BLOCKED_OPEN_SPARK_ITEMS:
                if not self.blockers:
                    _fail("falsification_matrix_row_invalid")
            elif self.blocked_reason == BLOCKED_PRODUCER_ABSENT and self.blockers:
                _fail("falsification_matrix_row_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "ledger_clause": self.ledger_clause,
            "status": self.status,
            "producer": self.producer,
            "blockers": list(self.blockers),
            "blocked_reason": self.blocked_reason,
            "notes": self.notes,
        }


MATRIX_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow(
        row_id="recurrence_depth_curves",
        ledger_clause="recurrence-depth curves",
        status=ROW_RUNNABLE,
        producer="experiments.run_recurrence_sweep",
        blockers=(),
        notes=(
            "Accuracy as a function of forced recurrence depth over the "
            "task battery; monotone-gain and paired claims graded."
        ),
    ),
    MatrixRow(
        row_id="wrong_right_transition_matrix",
        ledger_clause="wrong/right transition matrix",
        status=ROW_RUNNABLE,
        producer="falsification_matrix.run_transition_matrix",
        blockers=(),
        notes=(
            "Per-task paired outcomes at shallow versus deep recurrence; "
            "the 2x2 table separates repair from damage instead of hiding "
            "both inside a net accuracy."
        ),
    ),
    MatrixRow(
        row_id="blind_review_arms",
        ledger_clause="blind-review arms",
        status=ROW_ENFORCED,
        producer="threat_model:anchoring,verifier_collusion",
        blockers=(),
        notes=(
            "Review blindness is structural: no unblinded review mode "
            "exists to A/B by design. The property is proven by the bound "
            "threat-model checks (origin-free candidates, decoy balance, "
            "tamper-evident preflight)."
        ),
    ),
    MatrixRow(
        row_id="structural_diversity_arms",
        ledger_clause="structural-diversity arms",
        status=ROW_RUNNABLE,
        producer="experiments.run_role_lesion",
        blockers=(),
        notes=(
            "Distinct versus lesioned-uniform versus swapped versus "
            "restored role anchors; behavioral, divergence, swap-parity, "
            "and restoration claims."
        ),
    ),
    MatrixRow(
        row_id="verifier_arms",
        ledger_clause="verifier arms",
        status=ROW_BLOCKED,
        producer="",
        blockers=(),
        blocked_reason=BLOCKED_PRODUCER_ABSENT,
        notes=(
            "SPARK-039..046 have all landed, so the verifier-mesh machinery "
            "exists; what is missing is an experiment producer that drives "
            "the arms end to end. Marking this runnable while only the "
            "exact-checker arms execute would restate the defect the mesh "
            "was built to remove, because the generative and counterfactual "
            "arms cannot run model-free."
        ),
    ),
    MatrixRow(
        row_id="latent_ablate_perturb_transplant",
        ledger_clause="latent ablate/perturb/transplant",
        status=ROW_RUNNABLE,
        producer="experiments.run_slot_causality",
        blockers=(),
        notes=(
            "Slot ablation with restoration is runnable now; bounded "
            "contradiction-guided perturbation carries its own controls "
            "(CP346); cross-problem state transplant is designed at the "
            "acceptance run, noted here so its absence is visible."
        ),
    ),
    MatrixRow(
        row_id="sham_noise_noop_controls",
        ledger_clause="sham/noise/no-op controls",
        status=ROW_RUNNABLE,
        producer="experiments.run_latent_opt_control",
        blockers=(),
        notes=(
            "Latent-opt versus matched random-control arm plus the "
            "state-causality sham/inert identities; every intervention "
            "must beat its matched sham, not just baseline."
        ),
    ),
    MatrixRow(
        row_id="adversarial_ood_variants",
        ledger_clause="adversarial and OOD variants",
        status=ROW_BLOCKED,
        producer="",
        blockers=(70,),
        blocked_reason=BLOCKED_ACCEPTANCE_RUN_ONLY,
        notes=(
            "Adversarial renaming/reordering and genuinely out-of-"
            "distribution variants are generated fresh at the acceptance "
            "run; pre-registering them now would let the dry run see them."
        ),
    ),
    MatrixRow(
        row_id="compute_generalization",
        ledger_clause="d+1/2d/4d compute generalization",
        status=ROW_RUNNABLE,
        producer="experiments.run_depth_extrapolation",
        blockers=(),
        notes=(
            "T_required as a function of task depth, including depths "
            "beyond the training grid; genuine recurrence extrapolates "
            "before saturating."
        ),
    ),
    MatrixRow(
        row_id="lesions_restorations",
        ledger_clause="lesions/restorations",
        status=ROW_RUNNABLE,
        producer="state_causality.run_state_causality_experiment",
        blockers=(),
        notes=(
            "Typed-state information lesions with width-matched fillers, "
            "sham and inert controls, byte-exact restoration, and the "
            "prose-shadow prohibition; replayable row-level receipt."
        ),
    ),
    MatrixRow(
        row_id="fast_weight_controls",
        ledger_clause="fast-weight controls",
        status=ROW_BLOCKED,
        producer="",
        blockers=(),
        blocked_reason=BLOCKED_PRODUCER_ABSENT,
        notes=(
            "SPARK-055/056 have landed, so query-scoped fast weights and "
            "their integrity proofs exist; what is missing is a producer "
            "driving the on/off/sham arms as one comparison. "
            "run_latent_opt_control covers the optimization-control half."
        ),
    ),
    MatrixRow(
        row_id="non_reasoning_regressions",
        ledger_clause="broad non-reasoning regressions",
        status=ROW_RUNNABLE,
        producer="capability_canaries.compare_canaries",
        blockers=(),
        notes=(
            "The protected canary battery's teacher-forced logprobs under "
            "the treatment versus the frozen baseline; any drop beyond "
            "threshold is a named regression."
        ),
    ),
)


def validate_falsification_matrix() -> dict[str, Any]:
    """Fail closed unless the registry covers every required row exactly."""

    seen: set[str] = set()
    for row in MATRIX_ROWS:
        if row.row_id in seen:
            _fail("falsification_matrix_duplicate_row")
        seen.add(row.row_id)
    if seen != set(REQUIRED_ROW_IDS):
        _fail("falsification_matrix_coverage_incomplete")
    body = {
        "schema": FALSIFICATION_MATRIX_SCHEMA,
        "row_count": len(MATRIX_ROWS),
        "rows": [row.to_dict() for row in MATRIX_ROWS],
    }
    return {**body, "registry_sha256": canonical_sha256(body)}


def run_transition_matrix(
    solve: Callable[[Any, int], tuple[bool, int]],
    tasks: list[Any],
    *,
    shallow_steps: int,
    deep_steps: int,
) -> dict[str, Any]:
    """Per-task wrong/right transitions between shallow and deep recurrence.

    ``solve(task, steps)`` returns (verified_success, layer_apps).  The 2x2
    table separates repaired tasks (wrong→right) from damaged ones
    (right→wrong) — the two flows a net accuracy quietly nets against
    each other.  The paired claim reuses the house grading discipline.
    """

    if (
        isinstance(shallow_steps, bool)
        or isinstance(deep_steps, bool)
        or not isinstance(shallow_steps, int)
        or not isinstance(deep_steps, int)
        or not 1 <= shallow_steps < deep_steps
    ):
        _fail("falsification_matrix_transition_steps_invalid")
    if not isinstance(tasks, list) or not tasks:
        _fail("falsification_matrix_transition_tasks_invalid")

    from core.brain.llm.latent_cortex.experiments import (
        PairedObservation,
        grade_paired_treatment_vs_control,
    )

    table = {
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "unchanged_right": 0,
        "unchanged_wrong": 0,
    }
    rows: list[dict[str, Any]] = []
    paired: dict[str, list[PairedObservation]] = {}
    for index, task in enumerate(tasks):
        shallow_ok, shallow_cost = solve(task, shallow_steps)
        deep_ok, deep_cost = solve(task, deep_steps)
        shallow_ok = bool(shallow_ok)
        deep_ok = bool(deep_ok)
        if deep_ok and not shallow_ok:
            table["wrong_to_right"] += 1
        elif shallow_ok and not deep_ok:
            table["right_to_wrong"] += 1
        elif shallow_ok:
            table["unchanged_right"] += 1
        else:
            table["unchanged_wrong"] += 1
        family = str(getattr(task, "family", "all"))
        depth = getattr(task, "depth", 0)
        seed = getattr(task, "seed", 0)
        rows.append(
            {
                "task_id": f"{family}:{depth}:{seed}:{index}",
                "family": family,
                "shallow_success": shallow_ok,
                "deep_success": deep_ok,
                "shallow_layer_apps": int(shallow_cost or 0),
                "deep_layer_apps": int(deep_cost or 0),
            }
        )
        paired.setdefault(family, []).append(
            PairedObservation(
                task_id=f"{family}:{depth}:{seed}:{index}",
                family=family,
                treatment_success=deep_ok,
                control_success=shallow_ok,
                treatment_layer_apps=int(deep_cost or 0),
                control_layer_apps=int(shallow_cost or 0),
            )
        )
    claim = grade_paired_treatment_vs_control(
        "mx_transition_matrix",
        (
            f"recurrence at {deep_steps} steps repairs more tasks than it "
            f"damages relative to {shallow_steps} steps"
        ),
        paired,
    )
    return {
        "shallow_steps": shallow_steps,
        "deep_steps": deep_steps,
        "n_tasks": len(tasks),
        "table": table,
        "rows": rows,
        "claim": claim.to_dict(),
    }


def _claim_tiers(payload: Any) -> list[dict[str, str]]:
    """Collect (experiment, tier) pairs from any harness result payload."""

    tiers: list[dict[str, str]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            experiment = node.get("experiment")
            tier = node.get("tier")
            if isinstance(experiment, str) and isinstance(tier, str):
                tiers.append({"experiment": experiment, "tier": tier})
                return
            for key in sorted(node, key=str):
                _walk(node[key])
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return tiers


def assemble_falsification_matrix_receipt(
    *,
    row_results: Mapping[str, Mapping[str, Any]],
    runner_identity: Mapping[str, Any],
    threat_model_registry_sha256: str,
) -> dict[str, Any]:
    """Bind every registry row to its evidence, enforcement, or blockers."""

    registry = validate_falsification_matrix()
    if not isinstance(row_results, Mapping):
        _fail("falsification_matrix_results_invalid")
    if (
        not isinstance(threat_model_registry_sha256, str)
        or len(threat_model_registry_sha256) != 64
    ):
        _fail("falsification_matrix_threat_registry_invalid")
    runnable_ids = {
        row.row_id for row in MATRIX_ROWS if row.status == ROW_RUNNABLE
    }
    unknown = set(row_results) - runnable_ids
    if unknown:
        _fail("falsification_matrix_unknown_result_row")
    missing = runnable_ids - set(row_results)
    if missing:
        _fail("falsification_matrix_missing_result_row")

    receipt_rows: list[dict[str, Any]] = []
    for row in MATRIX_ROWS:
        entry = row.to_dict()
        if row.status == ROW_RUNNABLE:
            payload = row_results[row.row_id]
            if not isinstance(payload, Mapping) or not payload:
                _fail("falsification_matrix_result_payload_invalid")
            entry["result_sha256"] = canonical_sha256(dict(payload))
            entry["claim_tiers"] = _claim_tiers(payload)
        elif row.status == ROW_ENFORCED:
            entry["result_sha256"] = threat_model_registry_sha256
            entry["claim_tiers"] = []
        else:
            entry["result_sha256"] = ""
            entry["claim_tiers"] = []
        receipt_rows.append(entry)

    body = {
        "schema": FALSIFICATION_MATRIX_SCHEMA,
        "registry_sha256": registry["registry_sha256"],
        "threat_model_registry_sha256": threat_model_registry_sha256,
        "runner_identity": dict(runner_identity),
        "rows": receipt_rows,
        "runnable_rows": sorted(runnable_ids),
        "blocked_rows": sorted(
            row.row_id for row in MATRIX_ROWS if row.status == ROW_BLOCKED
        ),
        "enforced_rows": sorted(
            row.row_id for row in MATRIX_ROWS if row.status == ROW_ENFORCED
        ),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def replay_falsification_matrix_receipt(
    receipt: Mapping[str, Any],
    *,
    row_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-verify a matrix receipt against the CURRENT registry.

    Registry drift — a row reclassified, dropped, or reassigned since the
    receipt was produced — fails the replay, because a matrix receipt is
    only meaningful relative to the registry that defined its rows.  When
    ``row_payloads`` is supplied, each runnable row's payload is re-hashed
    against its binding.
    """

    registry = validate_falsification_matrix()
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "registry_sha256",
        "threat_model_registry_sha256",
        "runner_identity",
        "rows",
        "runnable_rows",
        "blocked_rows",
        "enforced_rows",
        "receipt_sha256",
    }:
        _fail("falsification_matrix_receipt_invalid")
    if receipt.get("schema") != FALSIFICATION_MATRIX_SCHEMA:
        _fail("falsification_matrix_receipt_invalid")
    body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(body):
        _fail("falsification_matrix_receipt_digest_mismatch")
    if receipt.get("registry_sha256") != registry["registry_sha256"]:
        _fail("falsification_matrix_registry_drift")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != len(MATRIX_ROWS):
        _fail("falsification_matrix_receipt_invalid")
    by_id = {row.row_id: row for row in MATRIX_ROWS}
    for entry in rows:
        if not isinstance(entry, Mapping):
            _fail("falsification_matrix_receipt_invalid")
        registry_row = by_id.get(str(entry.get("row_id")))
        if registry_row is None:
            _fail("falsification_matrix_registry_drift")
        for field in ("ledger_clause", "status", "producer", "notes"):
            if entry.get(field) != getattr(registry_row, field):
                _fail("falsification_matrix_registry_drift")
        if list(entry.get("blockers") or []) != list(registry_row.blockers):
            _fail("falsification_matrix_registry_drift")
        if row_payloads is not None and registry_row.status == ROW_RUNNABLE:
            payload = row_payloads.get(registry_row.row_id)
            if payload is None:
                _fail("falsification_matrix_replay_payload_missing")
            if canonical_sha256(dict(payload)) != entry.get("result_sha256"):
                _fail("falsification_matrix_replay_payload_mismatch")
    return {
        "schema": FALSIFICATION_MATRIX_SCHEMA,
        "receipt_sha256": receipt["receipt_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "replayed": True,
    }


def open_ledger_items(ledger_path: Path | None = None) -> frozenset[int]:
    """The SPARK ids still unchecked in the execution ledger.

    Parsed from the ledger rather than mirrored in code, because a mirrored
    copy is exactly what drifts. An unparseable or empty ledger raises: an
    empty open set would silently make every blocker look stale.
    """

    path = (
        Path(ledger_path)
        if ledger_path is not None
        else Path(__file__).resolve().parents[4]
        / "docs"
        / "RLC_SPARK_EXECUTION_LEDGER.md"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        _fail("falsification_matrix_ledger_unreadable")
    open_items: set[int] = set()
    closed_items: set[int] = set()
    for line in text.splitlines():
        stripped = line.strip()
        for marker, sink in (("- [ ] **SPARK-", open_items), ("- [x] **SPARK-", closed_items)):
            if stripped.startswith(marker):
                digits = stripped[len(marker) : len(marker) + 3]
                if digits.isdigit():
                    sink.add(int(digits))
    if not open_items or not closed_items:
        _fail("falsification_matrix_ledger_unparsed")
    return frozenset(open_items)


def validate_blockers_against_ledger(
    *,
    open_items: frozenset[int] | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Refuse any row blocked on a SPARK item that has already closed.

    The drift this prevents was real and survived several checkpoints: two
    rows named SPARK-039..046 and SPARK-055/056 long after all ten of those
    items closed. The effect was not cosmetic — it made two of twelve rows
    look blocked on machinery that had landed, which hid what they were
    actually waiting on (a producer, and the acceptance run).

    Correcting a stale list by hand fixes one instance. This makes the class
    of defect detectable, which is why it is wired into the pre-training
    preflight rather than left as a convention.
    """

    resolved = (
        open_items
        if open_items is not None
        else open_ledger_items(ledger_path)
    )
    stale: list[dict[str, Any]] = []
    for row in MATRIX_ROWS:
        closed = sorted(item for item in row.blockers if item not in resolved)
        if closed:
            stale.append({"row_id": row.row_id, "closed_blockers": closed})
    if stale:
        _fail(
            "falsification_matrix_blocker_closed:"
            + ",".join(
                f"{row['row_id']}={row['closed_blockers']}" for row in stale
            )
        )
    return {
        "checked_rows": len(MATRIX_ROWS),
        "blocked_rows": sorted(
            row.row_id for row in MATRIX_ROWS if row.status == ROW_BLOCKED
        ),
        "open_items_considered": sorted(resolved),
        "stale_blockers": [],
    }


__all__ = [
    "BLOCKED_ACCEPTANCE_RUN_ONLY",
    "BLOCKED_OPEN_SPARK_ITEMS",
    "BLOCKED_PRODUCER_ABSENT",
    "BLOCKED_REASONS",
    "FALSIFICATION_MATRIX_SCHEMA",
    "MATRIX_ROWS",
    "REQUIRED_ROW_IDS",
    "ROW_BLOCKED",
    "ROW_ENFORCED",
    "ROW_RUNNABLE",
    "ROW_STATUSES",
    "FalsificationMatrixError",
    "MatrixRow",
    "assemble_falsification_matrix_receipt",
    "open_ledger_items",
    "replay_falsification_matrix_receipt",
    "run_transition_matrix",
    "validate_blockers_against_ledger",
    "validate_falsification_matrix",
]
