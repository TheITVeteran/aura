"""Bounded, evidence-gated control over the recurrent architecture itself.

SPARK-065 asks for something that sounds dangerous and is only safe if every
step is nailed down: Aura measuring where her own expert/router/depth structure
fails, proposing changes to it, and rolling them out without a human approving
each one. The safety is not in refusing to do it; it is in what each step is
allowed to be.

- **A finding must have measured something.** Every observation declares the
  episode count behind it. Below the floor the surface reports
  `insufficient_evidence` and produces no proposal at all. A quiet surface is
  not a healthy surface.
- **A proposal is one knob, inside a declared bound.** The finding→knob map is
  explicit and deterministic; a proposal that names an unknown knob, moves more
  than one, or leaves the knob's declared bound is refused. There is no
  free-form architecture edit.
- **A candidate runs somewhere else.** Trials carry an isolation attestation
  naming a runtime identity distinct from the live one. A trial that reports
  the live runtime's own identity is refused rather than counted.
- **Invariants are complete by declaration.** All six must be present and hold
  with evidence; a missing invariant is an invalid trial, not a passing one.
- **Rollout is a ladder with an exit.** canary → expanded → full, each stage
  carrying its own health verdict, and any regression forces rollback to the
  previous revision. The rollback target is checked before the rollout starts,
  so there is never a stage with no way back.
- **Approval is independent, and some changes are not automatable.** The
  approver must be a different role than the proposer -- self-approval is
  refused. Automated approval is allowed exactly because the class of change is
  bounded; a proposal touching a knob marked `requires_human` cannot be
  auto-approved no matter how good its evidence is.

Like the rest of the Spark surface this module is a strict state machine over
data with no Aura runtime imports, so an independent verifier can replay an
architecture decision without booting cognition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

OBSERVATION_SCHEMA: Final = "aura.rlc.architecture_control.observation.v1"
FINDING_REPORT_SCHEMA: Final = "aura.rlc.architecture_control.findings.v1"
PROPOSAL_SCHEMA: Final = "aura.rlc.architecture_control.proposal.v1"
TRIAL_SCHEMA: Final = "aura.rlc.architecture_control.trial.v1"
APPROVAL_SCHEMA: Final = "aura.rlc.architecture_control.approval.v1"
ROLLOUT_SCHEMA: Final = "aura.rlc.architecture_control.rollout.v1"

# The three architecture surfaces SPARK-065 names.
EXPERT: Final = "expert"
ROUTER: Final = "router"
DEPTH: Final = "depth"
SURFACES: Final = (EXPERT, ROUTER, DEPTH)

# What each surface can be found to be doing wrong. Each failure maps to
# exactly one knob; that map is the whole proposal vocabulary.
FAILURE_MODES: Final = {
    "dead_expert": EXPERT,
    "overloaded_expert": EXPERT,
    "router_collapse": ROUTER,
    "router_misroute": ROUTER,
    "depth_saturation": DEPTH,
    "depth_starvation": DEPTH,
}

# One knob per failure mode, each with a hard inclusive bound and a step cap.
# `requires_human` marks the changes an automated approver may never admit.
KNOB_BOUNDS: Final = {
    "expert_capacity_factor": {
        "minimum": 0.5,
        "maximum": 2.0,
        "max_step": 0.25,
        "requires_human": False,
    },
    "expert_dropout": {
        "minimum": 0.0,
        "maximum": 0.3,
        "max_step": 0.05,
        "requires_human": False,
    },
    "router_temperature": {
        "minimum": 0.25,
        "maximum": 4.0,
        "max_step": 0.5,
        "requires_human": False,
    },
    "router_top_k": {
        "minimum": 1.0,
        "maximum": 4.0,
        "max_step": 1.0,
        "requires_human": False,
    },
    "recurrence_max_depth": {
        "minimum": 1.0,
        "maximum": 16.0,
        "max_step": 2.0,
        "requires_human": False,
    },
    "recurrence_min_depth": {
        "minimum": 1.0,
        "maximum": 8.0,
        "max_step": 1.0,
        "requires_human": False,
    },
}

_FAILURE_KNOB: Final = {
    "dead_expert": "expert_dropout",
    "overloaded_expert": "expert_capacity_factor",
    "router_collapse": "router_temperature",
    "router_misroute": "router_top_k",
    "depth_saturation": "recurrence_max_depth",
    "depth_starvation": "recurrence_min_depth",
}

# A surface measured over fewer episodes than this has not been measured.
_MINIMUM_EPISODES: Final = 64

# Every trial must carry all six, holding, with evidence.
REQUIRED_INVARIANTS: Final = (
    "bounded_depth",
    "bounded_width",
    "no_authority_widening",
    "determinism_preserved",
    "equal_compute_honored",
    "rollback_available",
)

CANARY: Final = "canary"
EXPANDED: Final = "expanded"
FULL: Final = "full"
ROLLOUT_STAGES: Final = (CANARY, EXPANDED, FULL)
_STAGE_CEILING: Final = {CANARY: 0.05, EXPANDED: 0.25, FULL: 1.0}

HEALTHY: Final = "healthy"
REGRESSED: Final = "regressed"
STAGE_VERDICTS: Final = (HEALTHY, REGRESSED)

ADMIT: Final = "ADMIT"
REFUSE: Final = "REFUSE"
ROLLED_BACK: Final = "ROLLED_BACK"

PROPOSER: Final = "architecture_proposer"
APPROVER: Final = "architecture_approver"
HUMAN_APPROVER: Final = "human_approver"
_INDEPENDENT_APPROVERS: Final = (APPROVER, HUMAN_APPROVER)

_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_IDENTITY_PATTERN: Final = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:@/-]{0,191}\Z")
_OBSERVATION_FIELDS: Final = frozenset(
    {"surface", "failure_mode", "episodes", "statistic", "threshold", "evidence_sha256"}
)
_INVARIANT_FIELDS: Final = frozenset({"invariant", "holds", "evidence_sha256"})
_STAGE_FIELDS: Final = frozenset(
    {"stage", "traffic_fraction", "episodes", "verdict", "evidence_sha256"}
)


class ArchitectureControlError(ValueError):
    """An architecture observation, proposal, trial, or rollout is invalid."""


class ArchitectureControlRefusalError(ArchitectureControlError):
    """A change was blocked; the decision names what blocked it."""

    def __init__(self, decision: Mapping[str, Any]) -> None:
        super().__init__("architecture_control_change_refused")
        self.decision: dict[str, Any] = dict(decision)


def _fail(code: str) -> Never:
    raise ArchitectureControlError(str(code or "architecture_control_invalid"))


def _sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ArchitectureControlError(
            "architecture_control_noncanonical_value"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _identity(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_PATTERN.match(value):
        _fail(code)
    return value


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        _fail(code)
    return round(number, 9)


def _count(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        _fail(code)
    return value


# ---------------------------------------------------------------------------
# 1. Measure what the architecture is doing wrong
# ---------------------------------------------------------------------------


def architecture_observation(
    *,
    failure_mode: str,
    episodes: int,
    statistic: float,
    threshold: float,
    evidence_sha256: str,
) -> dict[str, Any]:
    """One measured claim about one architecture surface."""

    if failure_mode not in FAILURE_MODES:
        _fail("architecture_control_failure_mode_unknown")
    if not _is_sha256(evidence_sha256):
        _fail("architecture_control_observation_evidence_invalid")
    return {
        "surface": FAILURE_MODES[failure_mode],
        "failure_mode": failure_mode,
        "episodes": _count(episodes, "architecture_control_observation_episodes_invalid"),
        "statistic": _finite(
            statistic, "architecture_control_observation_statistic_invalid"
        ),
        "threshold": _finite(
            threshold, "architecture_control_observation_threshold_invalid"
        ),
        "evidence_sha256": evidence_sha256,
    }


def architecture_findings(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate measured findings from surfaces that were not measured enough.

    A surface below the episode floor lands in `insufficient_evidence`, never in
    `findings` -- so a proposal can never be justified by a statistic nobody had
    the episodes to trust.
    """

    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        _fail("architecture_control_findings_invalid")

    findings: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    measured: set[str] = set()
    seen: set[str] = set()
    for raw in observations:
        if not isinstance(raw, Mapping) or set(raw) != _OBSERVATION_FIELDS:
            _fail("architecture_control_observation_fields_differ")
        row = architecture_observation(
            failure_mode=raw["failure_mode"],
            episodes=raw["episodes"],
            statistic=raw["statistic"],
            threshold=raw["threshold"],
            evidence_sha256=raw["evidence_sha256"],
        )
        if row["surface"] != raw["surface"]:
            _fail("architecture_control_observation_surface_differs")
        if row["failure_mode"] in seen:
            _fail("architecture_control_observation_duplicate")
        seen.add(row["failure_mode"])
        if row["episodes"] < _MINIMUM_EPISODES:
            insufficient.append(
                {
                    **row,
                    "reason": "insufficient_evidence",
                    "episodes_required": _MINIMUM_EPISODES,
                }
            )
        elif row["statistic"] > row["threshold"]:
            measured.add(row["surface"])
            findings.append(row)
        else:
            # A surface that came back clean says so out loud. Letting it drop
            # out of the report would make "nobody looked" and "we looked and
            # it was fine" the same silence.
            measured.add(row["surface"])
            clean.append(row)

    findings.sort(key=lambda row: row["failure_mode"])
    insufficient.sort(key=lambda row: row["failure_mode"])
    clean.sort(key=lambda row: row["failure_mode"])
    body = {
        "schema": FINDING_REPORT_SCHEMA,
        "findings": findings,
        "clean": clean,
        "insufficient_evidence": insufficient,
        "surfaces_measured": sorted(measured),
        "surfaces_unmeasured": sorted(set(SURFACES) - measured),
    }
    return {**body, "findings_sha256": _sha256(body)}


# ---------------------------------------------------------------------------
# 2. Propose exactly one bounded change
# ---------------------------------------------------------------------------


def propose_architecture_change(
    *,
    findings: Mapping[str, Any],
    failure_mode: str,
    current_value: float,
    proposed_value: float,
    proposer_identity: str,
) -> dict[str, Any]:
    """Turn one measured finding into one bounded knob change."""

    if not isinstance(findings, Mapping) or findings.get("schema") != FINDING_REPORT_SCHEMA:
        _fail("architecture_control_findings_invalid")
    rows = findings.get("findings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        _fail("architecture_control_findings_invalid")
    finding = next(
        (
            dict(row)
            for row in rows
            if isinstance(row, Mapping) and row.get("failure_mode") == failure_mode
        ),
        None,
    )
    if finding is None:
        _fail("architecture_control_proposal_without_finding")

    knob = _FAILURE_KNOB[failure_mode]
    bound = KNOB_BOUNDS[knob]
    current = _finite(current_value, "architecture_control_proposal_value_invalid")
    proposed = _finite(proposed_value, "architecture_control_proposal_value_invalid")
    if proposed == current:
        _fail("architecture_control_proposal_is_a_no_op")
    if not bound["minimum"] <= proposed <= bound["maximum"]:
        _fail("architecture_control_proposal_out_of_bounds")
    if not bound["minimum"] <= current <= bound["maximum"]:
        _fail("architecture_control_proposal_incumbent_out_of_bounds")
    if abs(proposed - current) > bound["max_step"] + 1e-9:
        _fail("architecture_control_proposal_step_too_large")

    body = {
        "schema": PROPOSAL_SCHEMA,
        "surface": FAILURE_MODES[failure_mode],
        "failure_mode": failure_mode,
        "knob": knob,
        "current_value": current,
        "proposed_value": proposed,
        "bound": dict(bound),
        "findings_sha256": findings["findings_sha256"],
        "finding_evidence_sha256": finding["evidence_sha256"],
        "proposer_identity": _identity(
            proposer_identity, "architecture_control_proposer_invalid"
        ),
    }
    return {**body, "proposal_sha256": _sha256(body)}


# ---------------------------------------------------------------------------
# 3-4. Isolated trial with complete machine-checkable invariants
# ---------------------------------------------------------------------------


def invariant_result(
    *, invariant: str, holds: bool, evidence_sha256: str
) -> dict[str, Any]:
    if invariant not in REQUIRED_INVARIANTS:
        _fail("architecture_control_invariant_unknown")
    if type(holds) is not bool:
        _fail("architecture_control_invariant_verdict_invalid")
    if not _is_sha256(evidence_sha256):
        _fail("architecture_control_invariant_evidence_invalid")
    return {
        "invariant": invariant,
        "holds": holds,
        "evidence_sha256": evidence_sha256,
    }


def candidate_trial(
    *,
    proposal: Mapping[str, Any],
    live_runtime_identity: str,
    candidate_runtime_identity: str,
    incumbent_score: float,
    candidate_score: float,
    incumbent_compute_units: int,
    candidate_compute_units: int,
    episodes: int,
    invariants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record one isolated evaluation of a proposed architecture change.

    The trial refuses to exist if it ran inside the live runtime, if it did not
    hold compute equal, or if any declared invariant is missing.
    """

    if not isinstance(proposal, Mapping) or proposal.get("schema") != PROPOSAL_SCHEMA:
        _fail("architecture_control_proposal_invalid")
    live = _identity(live_runtime_identity, "architecture_control_runtime_invalid")
    candidate = _identity(
        candidate_runtime_identity, "architecture_control_runtime_invalid"
    )
    if live == candidate:
        _fail("architecture_control_trial_not_isolated")

    incumbent_units = _count(
        incumbent_compute_units, "architecture_control_trial_compute_invalid"
    )
    candidate_units = _count(
        candidate_compute_units, "architecture_control_trial_compute_invalid"
    )
    if incumbent_units == 0 or candidate_units == 0:
        _fail("architecture_control_trial_compute_invalid")
    graded = _count(episodes, "architecture_control_trial_episodes_invalid")
    if graded < _MINIMUM_EPISODES:
        _fail("architecture_control_trial_underpowered")

    if not isinstance(invariants, Sequence) or isinstance(invariants, (str, bytes)):
        _fail("architecture_control_invariants_invalid")
    rows: dict[str, dict[str, Any]] = {}
    for raw in invariants:
        if not isinstance(raw, Mapping) or set(raw) != _INVARIANT_FIELDS:
            _fail("architecture_control_invariant_fields_differ")
        row = invariant_result(
            invariant=raw["invariant"],
            holds=raw["holds"],
            evidence_sha256=raw["evidence_sha256"],
        )
        if row["invariant"] in rows:
            _fail("architecture_control_invariant_duplicate")
        rows[row["invariant"]] = row
    if set(rows) != set(REQUIRED_INVARIANTS):
        _fail("architecture_control_invariant_set_incomplete")

    # Equal compute is a precondition of the comparison, not a nice-to-have: a
    # candidate that simply spent more is not a better architecture.
    equal_compute = (
        abs(candidate_units - incumbent_units) <= max(1, incumbent_units // 100)
    )
    body = {
        "schema": TRIAL_SCHEMA,
        "proposal_sha256": proposal["proposal_sha256"],
        "live_runtime_identity": live,
        "candidate_runtime_identity": candidate,
        "incumbent_score": _finite(
            incumbent_score, "architecture_control_trial_score_invalid"
        ),
        "candidate_score": _finite(
            candidate_score, "architecture_control_trial_score_invalid"
        ),
        "incumbent_compute_units": incumbent_units,
        "candidate_compute_units": candidate_units,
        "equal_compute": equal_compute,
        "episodes": graded,
        "invariants": [rows[name] for name in REQUIRED_INVARIANTS],
    }
    return {**body, "trial_sha256": _sha256(body)}


# ---------------------------------------------------------------------------
# 5-6. Independent approval, then a rollout ladder with a way back
# ---------------------------------------------------------------------------


def approve_architecture_change(
    *,
    proposal: Mapping[str, Any],
    trial: Mapping[str, Any],
    approver_role: str,
    approver_identity: str,
    minimum_improvement: float = 0.01,
) -> dict[str, Any]:
    """Decide whether an evaluated change may enter a canary rollout."""

    if not isinstance(proposal, Mapping) or proposal.get("schema") != PROPOSAL_SCHEMA:
        _fail("architecture_control_proposal_invalid")
    if not isinstance(trial, Mapping) or trial.get("schema") != TRIAL_SCHEMA:
        _fail("architecture_control_trial_invalid")
    if trial.get("proposal_sha256") != proposal["proposal_sha256"]:
        _fail("architecture_control_trial_binds_other_proposal")
    if approver_role not in _INDEPENDENT_APPROVERS:
        _fail("architecture_control_approver_role_invalid")
    approver = _identity(approver_identity, "architecture_control_approver_invalid")

    floor = _finite(
        minimum_improvement, "architecture_control_improvement_floor_invalid"
    )
    refusals: list[dict[str, Any]] = []

    if approver == proposal["proposer_identity"]:
        refusals.append({"reason": "self_approval", "identity": approver})
    if (
        KNOB_BOUNDS[proposal["knob"]]["requires_human"]
        and approver_role != HUMAN_APPROVER
    ):
        refusals.append({"reason": "requires_human_approver", "knob": proposal["knob"]})
    failed = [row["invariant"] for row in trial["invariants"] if not row["holds"]]
    if failed:
        refusals.append({"reason": "invariant_violated", "invariants": failed})
    if not trial["equal_compute"]:
        refusals.append(
            {
                "reason": "unequal_compute",
                "incumbent_compute_units": trial["incumbent_compute_units"],
                "candidate_compute_units": trial["candidate_compute_units"],
            }
        )
    improvement = round(trial["candidate_score"] - trial["incumbent_score"], 9)
    if improvement < floor:
        refusals.append(
            {
                "reason": "improvement_below_floor",
                "improvement": improvement,
                "minimum_improvement": floor,
            }
        )

    body = {
        "schema": APPROVAL_SCHEMA,
        "decision": REFUSE if refusals else ADMIT,
        "proposal_sha256": proposal["proposal_sha256"],
        "trial_sha256": trial["trial_sha256"],
        "approver_role": approver_role,
        "approver_identity": approver,
        "improvement": improvement,
        "minimum_improvement": floor,
        "refusals": refusals,
    }
    return {**body, "approval_sha256": _sha256(body)}


def rollout_stage(
    *,
    stage: str,
    traffic_fraction: float,
    episodes: int,
    verdict: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    if stage not in ROLLOUT_STAGES:
        _fail("architecture_control_rollout_stage_unknown")
    if verdict not in STAGE_VERDICTS:
        _fail("architecture_control_rollout_verdict_invalid")
    if not _is_sha256(evidence_sha256):
        _fail("architecture_control_rollout_evidence_invalid")
    fraction = _finite(
        traffic_fraction, "architecture_control_rollout_fraction_invalid"
    )
    if not 0.0 < fraction <= _STAGE_CEILING[stage]:
        _fail("architecture_control_rollout_fraction_out_of_stage")
    return {
        "stage": stage,
        "traffic_fraction": fraction,
        "episodes": _count(episodes, "architecture_control_rollout_episodes_invalid"),
        "verdict": verdict,
        "evidence_sha256": evidence_sha256,
    }


def evaluate_rollout(
    *,
    approval: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    rollback_revision: str,
) -> dict[str, Any]:
    """Walk the canary ladder and say where the change actually ended up.

    The rollback target is required up front. A regression at any stage stops
    the ladder there and returns ROLLED_BACK naming the revision to restore --
    the change never silently continues past a bad stage.
    """

    if not isinstance(approval, Mapping) or approval.get("schema") != APPROVAL_SCHEMA:
        _fail("architecture_control_approval_invalid")
    if approval.get("decision") != ADMIT:
        _fail("architecture_control_rollout_without_approval")
    target = _identity(rollback_revision, "architecture_control_rollback_invalid")

    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
        _fail("architecture_control_rollout_stages_invalid")

    walked: list[dict[str, Any]] = []
    outcome = ROLLED_BACK
    regressed_at: str | None = None
    for index, raw in enumerate(stages):
        if not isinstance(raw, Mapping) or set(raw) != _STAGE_FIELDS:
            _fail("architecture_control_rollout_stage_fields_differ")
        row = rollout_stage(
            stage=raw["stage"],
            traffic_fraction=raw["traffic_fraction"],
            episodes=raw["episodes"],
            verdict=raw["verdict"],
            evidence_sha256=raw["evidence_sha256"],
        )
        if row["stage"] != ROLLOUT_STAGES[index]:
            _fail("architecture_control_rollout_stage_out_of_order")
        walked.append(row)
        if row["verdict"] == REGRESSED:
            regressed_at = row["stage"]
            break
    else:
        outcome = ADMIT if walked[-1]["stage"] == FULL else ROLLED_BACK
        if outcome == ROLLED_BACK:
            regressed_at = "incomplete_ladder"

    body = {
        "schema": ROLLOUT_SCHEMA,
        "outcome": outcome,
        "approval_sha256": approval["approval_sha256"],
        "proposal_sha256": approval["proposal_sha256"],
        "stages": walked,
        "regressed_at": regressed_at,
        "rollback_revision": target,
        "restored_revision": None if outcome == ADMIT else target,
    }
    return {**body, "rollout_sha256": _sha256(body)}


__all__ = [
    "ADMIT",
    "APPROVAL_SCHEMA",
    "APPROVER",
    "CANARY",
    "DEPTH",
    "EXPANDED",
    "EXPERT",
    "FAILURE_MODES",
    "FINDING_REPORT_SCHEMA",
    "FULL",
    "HEALTHY",
    "HUMAN_APPROVER",
    "KNOB_BOUNDS",
    "OBSERVATION_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PROPOSER",
    "REFUSE",
    "REGRESSED",
    "REQUIRED_INVARIANTS",
    "ROLLED_BACK",
    "ROLLOUT_SCHEMA",
    "ROLLOUT_STAGES",
    "ROUTER",
    "SURFACES",
    "TRIAL_SCHEMA",
    "ArchitectureControlError",
    "ArchitectureControlRefusalError",
    "approve_architecture_change",
    "architecture_findings",
    "architecture_observation",
    "candidate_trial",
    "evaluate_rollout",
    "invariant_result",
    "propose_architecture_change",
    "rollout_stage",
]
