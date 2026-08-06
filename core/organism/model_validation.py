"""core/organism/model_validation.py — claims must carry their tests.

Clean-room adoption of OpenWorm's validation discipline and the sciunit
model it is built on.

OpenWorm is a decade-long effort to simulate an organism, which means it
faces in an acute form the problem every ambitious system has: how do you
know the model resembles the thing? Their answer is a discipline rather
than a technique. **Every capability the model claims carries a validation
test, and every validation test is scored against a recorded observation
of the real system.** A model does not "support locomotion"; it passes
`SwimmingBehaviourTest` against observation data from a specific paper,
with a specific score, on a specific date. When the model changes, the
suite re-runs, and the claim either survives or does not.

Aura is full of claims about itself — that is unavoidable for a system
whose whole purpose is self-modelling — and the repository already carries
CLAIMS_SUPPORTED.md and CLAIMS_NOT_SUPPORTED.md, which is the right
instinct with no machinery behind it. Documents drift; suites do not.

The discipline, enforced structurally:

* A **claim** without a test cannot be registered. The registration call
  requires the test.
* A **test** without an observation cannot be constructed. There is no
  "expected value" that came from nowhere; an observation carries its
  source and when it was taken.
* A **score** carries its own interpretation. A raw number invites the
  reader to decide what counts as good after seeing it, which is the
  oldest way to fool yourself.
* A test whose **required capability is missing** yields ``N/A``, not a
  failure. Not applicable and failed are different facts, and a suite that
  conflates them makes an incapable model look broken and a broken one
  look incapable.

The suite runs against the live runtime, so the claims are checked against
what Aura is actually doing rather than what it did the day someone wrote
the document.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

logger = logging.getLogger("Aura.Validation")


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    #: The model does not have the capability the test requires. Not the
    #: same as failing it.
    NOT_APPLICABLE = "n/a"
    #: The test could not run. Also not the same as failing.
    ERROR = "error"


@dataclass(frozen=True)
class Observation:
    """Recorded ground truth, with where it came from.

    An expected value with no provenance is a number somebody remembered.
    """

    name: str
    value: Any
    source: str
    recorded_at: float = field(default_factory=time.time)
    units: str = ""
    tolerance: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "recorded_at": self.recorded_at,
            "units": self.units,
            "tolerance": self.tolerance,
            "note": self.note,
        }


@dataclass(frozen=True)
class Score:
    """A number that carries what it means."""

    kind: str
    value: float
    outcome: Outcome
    interpretation: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": round(self.value, 6) if isinstance(self.value, float) else self.value,
            "outcome": str(self.outcome),
            "interpretation": self.interpretation,
            "detail": dict(self.detail),
        }


def boolean_score(observed: bool, *, expected: bool = True, subject: str = "") -> Score:
    passed = bool(observed) == bool(expected)
    return Score(
        kind="boolean",
        value=1.0 if passed else 0.0,
        outcome=Outcome.PASS if passed else Outcome.FAIL,
        interpretation=(
            f"{subject or 'condition'} is {observed}, expected {expected}"
        ),
    )


def ratio_score(prediction: float, observation: float, *, tolerance: float = 0.1) -> Score:
    """Prediction over observation. 1.0 is exact; tolerance is fractional."""
    if observation == 0:
        return Score(
            kind="ratio",
            value=float("nan"),
            outcome=Outcome.ERROR,
            interpretation="observation is zero; a ratio against it means nothing",
        )
    ratio = prediction / observation
    passed = abs(ratio - 1.0) <= tolerance
    return Score(
        kind="ratio",
        value=ratio,
        outcome=Outcome.PASS if passed else Outcome.FAIL,
        interpretation=(
            f"predicted {prediction:.4g} against observed {observation:.4g} "
            f"= {ratio:.3f}× (tolerance ±{tolerance:.0%})"
        ),
        detail={"prediction": prediction, "observation": observation, "tolerance": tolerance},
    )


def threshold_score(
    prediction: float, threshold: float, *, direction: str = "at_most", units: str = ""
) -> Score:
    passed = prediction <= threshold if direction == "at_most" else prediction >= threshold
    comparator = "≤" if direction == "at_most" else "≥"
    return Score(
        kind="threshold",
        value=prediction,
        outcome=Outcome.PASS if passed else Outcome.FAIL,
        interpretation=f"{prediction:.4g}{units} {comparator} {threshold:.4g}{units}",
        detail={"threshold": threshold, "direction": direction},
    )


class Model(Protocol):
    """Anything under test. Capabilities are declared, not guessed."""

    name: str

    def capabilities(self) -> set[str]: ...


@dataclass
class RuntimeModel:
    """The live runtime as a model of itself."""

    name: str = "aura_runtime"
    _capabilities: set[str] = field(default_factory=set)

    def capabilities(self) -> set[str]:
        return set(self._capabilities)

    def declare(self, *capabilities: str) -> RuntimeModel:
        self._capabilities.update(capabilities)
        return self


@dataclass(frozen=True)
class ValidationTest:
    """One falsifiable check of one claim against one observation."""

    name: str
    description: str
    required_capability: str
    observation: Observation
    #: Produce the model's prediction. Raising is an ERROR, not a FAIL.
    predict: Callable[[Model], Any]
    #: Compare prediction to observation and interpret the result.
    score: Callable[[Any, Observation], Score]
    owner: str = "unknown"

    def run(self, model: Model) -> TestResult:
        started = time.perf_counter()
        if self.required_capability and self.required_capability not in model.capabilities():
            return TestResult(
                test=self.name,
                model=model.name,
                score=Score(
                    kind="n/a",
                    value=0.0,
                    outcome=Outcome.NOT_APPLICABLE,
                    interpretation=(
                        f"{model.name} does not declare {self.required_capability!r}; "
                        "not applicable is not the same as failed"
                    ),
                ),
                duration_s=time.perf_counter() - started,
            )
        try:
            prediction = self.predict(model)
        except Exception as exc:  # noqa: BLE001 — could not run is not failed
            logger.warning(
                "Validation prediction %s failed for model %s",
                self.name,
                model.name,
                exc_info=True,
            )
            return TestResult(
                test=self.name,
                model=model.name,
                score=Score(
                    kind="error",
                    value=0.0,
                    outcome=Outcome.ERROR,
                    interpretation=f"prediction raised {type(exc).__name__}: {exc}",
                ),
                duration_s=time.perf_counter() - started,
            )
        try:
            score = self.score(prediction, self.observation)
        except Exception as exc:  # noqa: BLE001
            score = Score(
                kind="error",
                value=0.0,
                outcome=Outcome.ERROR,
                interpretation=f"scoring raised {type(exc).__name__}: {exc}",
            )
        return TestResult(
            test=self.name,
            model=model.name,
            score=score,
            prediction=prediction,
            duration_s=time.perf_counter() - started,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required_capability": self.required_capability,
            "observation": self.observation.to_dict(),
            "owner": self.owner,
        }


@dataclass
class TestResult:
    test: str
    model: str
    score: Score
    prediction: Any = None
    duration_s: float = 0.0
    at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.score.outcome is Outcome.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "model": self.model,
            "score": self.score.to_dict(),
            "prediction": _summarize(self.prediction),
            "duration_ms": round(self.duration_s * 1000, 3),
            "at": self.at,
        }


def _summarize(value: Any, limit: int = 200) -> Any:
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return repr(value)[:limit]


@dataclass(frozen=True)
class Evidence(StrEnum):
    """What KIND of evidence a claim's test actually provides.

    Binding a claim to a test was never enough. A test can pass while
    establishing much less than the claim implies, and the registry had no way
    to say so — which is how two numbers came to stand as evidence they had not
    earned:

    * Φ. The old estimator assigned substantial integration to a system built
      to be memoryless, and at reachable history lengths could rank it ABOVE a
      genuinely coupled ring. Prior values (φ_s mean 0.253 among them) cannot
      be presented as quantitative evidence of integration. The corrected
      null-subtracted estimator separates cleanly — 0.000 / 0.049 / 0.563 — but
      only on SYNTHETIC systems. No live null-corrected result exists yet.

    * CAA steering. RETRACTED, both alphas. Every A/B behind those numbers ran
      through a statistic scoring d(steered, control) − d(steered, baseline)
      over a runner that gave steered and baseline the same prompt and the same
      seed. Steering with no effect makes them identical, zeroes the subtracted
      term, and leaves the control distance — positive by construction. The
      null hypothesis passed decisively, and the α=0.35 artifact records it
      doing so: identical steered/baseline samples, zero affect words, d=2.502.
      "Steering was injected" (supported, 41,450 injections) is not "steering
      changed the answer" (unmeasured).

    In both cases the honest position is not zero and not proven. It is
    UNMEASURED, and a registry that cannot say that will keep implying proof.
    """

    #: Measured on the live system, with provenance.
    MEASURED_LIVE = "measured_live"
    #: Measured, but on constructed systems with known answers. Establishes the
    #: estimator, not the subject.
    MEASURED_SYNTHETIC = "measured_synthetic"
    #: Instrumented and reported, with no measurement that settles it.
    UNMEASURED = "unmeasured"
    #: Previously asserted, now withdrawn because the measurement behind it
    #: did not support it.
    RETRACTED = "retracted"


@dataclass(frozen=True)
class Claim:
    """A statement about the system, bound to the test that checks it."""

    statement: str
    test: str
    owner: str
    #: Where the claim is asserted publicly, so a failing suite points at
    #: the document that has to change.
    asserted_in: str = ""
    #: What the bound test actually establishes. Defaults to the strongest
    #: reading ONLY because every pre-existing claim was written under it;
    #: anything weaker must say so explicitly.
    evidence: Evidence = Evidence.MEASURED_LIVE
    #: Required when evidence is not MEASURED_LIVE: what is missing, in a
    #: sentence someone reading the claim can act on.
    evidence_note: str = ""

    def __post_init__(self) -> None:
        if self.evidence is not Evidence.MEASURED_LIVE and not self.evidence_note.strip():
            raise ValueError(
                f"claim {self.statement!r} is {self.evidence.value} and says nothing "
                "about what is missing; an unmeasured claim with no note reads as a "
                "measured one"
            )

    @property
    def is_evidence_for_the_system(self) -> bool:
        """Whether this claim may be cited as evidence ABOUT AURA."""
        return self.evidence is Evidence.MEASURED_LIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "test": self.test,
            "owner": self.owner,
            "asserted_in": self.asserted_in,
            "evidence": self.evidence.value,
            "evidence_note": self.evidence_note,
            "citable_as_evidence": self.is_evidence_for_the_system,
        }


class ValidationSuite:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tests: dict[str, ValidationTest] = {}
        self._claims: dict[str, Claim] = {}
        self._models: dict[str, Model] = {}
        self._last: dict[tuple[str, str], TestResult] = {}
        self.runs = 0

    # ── registration ──────────────────────────────────────────────────
    def add_test(self, test: ValidationTest) -> ValidationTest:
        if not test.observation.source.strip():
            raise ValueError(
                f"test {test.name!r} has an observation with no source; an expected "
                "value with no provenance is a number somebody remembered"
            )
        with self._lock:
            self._tests[test.name] = test
        return test

    def add_claim(self, claim: Claim) -> Claim:
        """Register a claim. It must name a test that exists."""
        with self._lock:
            if claim.test not in self._tests:
                raise ValueError(
                    f"claim {claim.statement!r} names test {claim.test!r}, which is not "
                    "registered. A claim without a test is a document, not a fact"
                )
            self._claims[claim.statement] = claim
        return claim

    def add_model(self, model: Model) -> Model:
        with self._lock:
            self._models[model.name] = model
        return model

    def tests(self) -> list[ValidationTest]:
        with self._lock:
            return sorted(self._tests.values(), key=lambda t: t.name)

    def claims(self) -> list[Claim]:
        with self._lock:
            return sorted(self._claims.values(), key=lambda c: c.statement)

    # ── running ───────────────────────────────────────────────────────
    def run(self, model: Model | None = None) -> dict[str, Any]:
        with self._lock:
            models = [model] if model is not None else list(self._models.values())
            tests = list(self._tests.values())

        results: list[TestResult] = []
        for target in models:
            for test in tests:
                result = test.run(target)
                results.append(result)
                with self._lock:
                    self._last[(test.name, target.name)] = result
        self.runs += 1

        by_outcome: dict[str, int] = {}
        for result in results:
            key = str(result.score.outcome)
            by_outcome[key] = by_outcome.get(key, 0) + 1

        failures = [r for r in results if r.score.outcome is Outcome.FAIL]
        errors = [r for r in results if r.score.outcome is Outcome.ERROR]
        return {
            "at": time.time(),
            "models": [m.name for m in models],
            "tests": len(tests),
            "results": [r.to_dict() for r in results],
            "by_outcome": by_outcome,
            "passed": by_outcome.get(str(Outcome.PASS), 0),
            "failed": len(failures),
            "errored": len(errors),
            # A suite where everything is N/A passes vacuously; say so.
            "applicable": len(results) - by_outcome.get(str(Outcome.NOT_APPLICABLE), 0),
            "failures": [r.to_dict() for r in failures],
            "errors": [r.to_dict() for r in errors],
        }

    def unsupported_claims(self) -> list[dict[str, Any]]:
        """Claims whose test last failed or could not run.

        This is the machine-checked version of CLAIMS_NOT_SUPPORTED.md.
        """
        out: list[dict[str, Any]] = []
        with self._lock:
            claims = list(self._claims.values())
            last = dict(self._last)
        for claim in claims:
            relevant = [r for (test, _model), r in last.items() if test == claim.test]
            if not relevant:
                out.append({**claim.to_dict(), "reason": "never run"})
                continue
            if any(r.score.outcome in (Outcome.FAIL, Outcome.ERROR) for r in relevant):
                worst = next(
                    r for r in relevant if r.score.outcome in (Outcome.FAIL, Outcome.ERROR)
                )
                out.append(
                    {
                        **claim.to_dict(),
                        "reason": worst.score.interpretation,
                        "outcome": str(worst.score.outcome),
                    }
                )
        return out

    def report(self) -> dict[str, Any]:
        with self._lock:
            tests = [t.to_dict() for t in self._tests.values()]
            claims = [c.to_dict() for c in self._claims.values()]
            last = {f"{t}/{m}": r.to_dict() for (t, m), r in self._last.items()}
        return {
            "tests": tests,
            "claims": claims,
            "models": sorted(self._models),
            "runs": self.runs,
            "last_results": last,
            "unsupported_claims": self.unsupported_claims(),
            "tests_without_claims": sorted(
                {t["name"] for t in tests} - {c["test"] for c in claims}
            ),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._tests.clear()
            self._claims.clear()
            self._models.clear()
            self._last.clear()
            self.runs = 0


_SUITE = ValidationSuite()


def get_suite() -> ValidationSuite:
    return _SUITE


def install_runtime_validation() -> dict[str, Any]:
    """Bind Aura's claims about its own runtime to tests over live telemetry.

    Each of these is a statement the runtime makes somewhere — in a
    docstring, a design document, or an architecture claim — turned into
    something that fails visibly when it stops being true.
    """
    suite = _SUITE
    model = RuntimeModel().declare(
        "lock_ordering",
        "memory_attribution",
        "periodic_scheduling",
        "structural_verification",
        "active_health",
        "integrity_reporting",
        "semantic_autonomous_action",
    )
    suite.add_model(model)

    suite.add_test(
        ValidationTest(
            name="lockdep_reports_no_order_violations",
            description="the runtime takes its locks in a consistent global order",
            required_capability="lock_ordering",
            observation=Observation(
                name="expected_splats",
                value=0,
                source="core/runtime/lockdep.py — a clean process has no splats",
                units="violations",
            ),
            predict=lambda _m: len(_lockdep()["splats"]),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" splats"),
            owner="core/runtime/lockdep.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="rate_group_period_is_a_period",
            description=(
                "a 1Hz rate group's median cycle is well under its period, so the "
                "declared rate is the actual rate"
            ),
            required_capability="periodic_scheduling",
            observation=Observation(
                name="max_median_cycle_fraction",
                value=0.5,
                source="core/fsw/rate_groups.py — members budget 20-40% of the period",
                units="fraction of period",
            ),
            predict=lambda _m: _slowest_group_fraction(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" of period"),
            owner="core/fsw/rate_groups.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="memory_growth_is_attributable_to_a_component",
            description=(
                "a diff between two dumps names the component that grew — checked by "
                "growing one and seeing whether the diff says so"
            ),
            required_capability="memory_attribution",
            observation=Observation(
                name="growth_is_named",
                value=True,
                source=(
                    "core/runtime/memory_infra.py — the whole purpose of attribution "
                    "is that a diff answers 'which component'"
                ),
                note=(
                    "Deliberately behavioural rather than a fraction-of-RSS threshold: "
                    "the claim is that growth can be NAMED, and a static share of a "
                    "mostly-interpreter process measures something else."
                ),
            ),
            predict=lambda _m: _growth_is_attributable(),
            score=lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="growth attribution"
            ),
            owner="core/runtime/memory_infra.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="structural_invariants_hold",
            description="the verifier finds no ERROR-severity structural violations",
            required_capability="structural_verification",
            observation=Observation(
                name="expected_errors",
                value=0,
                source="core/verify/runtime_invariants.py — the declared invariant set",
                units="errors",
            ),
            predict=lambda _m: _verifier_errors(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" errors"),
            owner="core/verify/invariants.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="no_critical_component_is_wedged",
            description="every component declared critical answers active health pings",
            required_capability="active_health",
            observation=Observation(
                name="expected_unresponsive",
                value=0,
                source="core/fsw/health_checker.py — critical components must answer",
                units="components",
            ),
            predict=lambda _m: len(_health()["critical_unresponsive"]),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" components"),
            owner="core/fsw/health_checker.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="health_verdicts_are_not_reported_over_hidden_damage",
            description=(
                "no credibility-affecting taint is set without the health surface "
                "carrying the caveat"
            ),
            required_capability="integrity_reporting",
            observation=Observation(
                name="caveat_present_when_tainted",
                value=True,
                source="core/runtime/taint.py — a tainted runtime must say so",
            ),
            predict=lambda _m: _taint_caveat_consistent(),
            score=lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="taint caveat consistency"
            ),
            owner="core/runtime/taint.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="semantic_action_routing_preserves_speech_act",
            description=(
                "indirect self-chosen objectives reach semantic planning while "
                "hypothetical or negated tool language remains non-executing"
            ),
            required_capability="semantic_autonomous_action",
            observation=Observation(
                name="speech_act_preserved",
                value=True,
                source=(
                    "core/conversation/request_mood.py and "
                    "core/runtime/overt_action_loop.py contract tests"
                ),
            ),
            predict=lambda _m: _semantic_autonomy_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="semantic autonomous-action routing",
            ),
            owner="core/runtime/turn_analysis.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="reality_metrology_contract_separates_sources",
            description=(
                "measurement contracts require explicit live/simulated roles for HIL "
                "and reject simulated evidence presented as a live acquisition"
            ),
            required_capability="reality_metrology",
            observation=Observation(
                name="source_partition_enforced",
                value=True,
                source="core/reality_reach/metrology.py contract tests",
            ),
            predict=lambda _m: _metrology_source_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="Reality Reach metrology source separation",
            ),
            owner="core/reality_reach/metrology.py",
        )
    )

    for statement, test_name, asserted_in in (
        (
            "The runtime takes its locks in a consistent order and has no latent ABBA deadlock.",
            "lockdep_reports_no_order_violations",
            "core/runtime/lockdep.py",
        ),
        (
            "Periodic work runs at its declared rate rather than at rate-plus-work-time.",
            "rate_group_period_is_a_period",
            "core/fsw/rate_groups.py",
        ),
        (
            "Memory growth can be attributed to a named component.",
            "memory_growth_is_attributable_to_a_component",
            "core/runtime/memory_infra.py",
        ),
        (
            "The runtime's structural invariants are enforced, not merely documented.",
            "structural_invariants_hold",
            "core/verify/runtime_invariants.py",
        ),
        (
            "A wedged critical component is detected rather than inferred from silence.",
            "no_critical_component_is_wedged",
            "core/fsw/health_checker.py",
        ),
        (
            "A green health verdict is never reported over known, hidden damage.",
            "health_verdicts_are_not_reported_over_hidden_damage",
            "core/runtime/taint.py",
        ),
        (
            "Action routing follows the turn's semantic speech act rather than requiring a trigger phrase.",
            "semantic_action_routing_preserves_speech_act",
            "core/runtime/turn_analysis.py",
        ),
        (
            "Physical measurement keeps live, simulated, and hardware-in-loop evidence causally distinct.",
            "reality_metrology_contract_separates_sources",
            "core/reality_reach/metrology.py",
        ),
    ):
        suite.add_claim(
            Claim(statement=statement, test=test_name, owner=asserted_in, asserted_in=asserted_in)
        )

    return {
        "model": model.name,
        "tests": [t.name for t in suite.tests()],
        "claims": len(suite.claims()),
    }


# ── prediction helpers, kept small and failure-tolerant ───────────────

def _lockdep() -> dict[str, Any]:
    from core.runtime.lockdep import lockdep_report

    return lockdep_report()


def _semantic_autonomy_contract_holds() -> bool:
    from core.conversation.request_mood import assess_request_mood
    from core.runtime.overt_action_loop import OvertActionLoop

    indirect = assess_request_mood(
        "It would help if you compared the current evidence and saved the result."
    )
    hypothetical = assess_request_mood(
        "If I asked you to open Notes, how would you decide whether to do it?"
    )
    selection = OvertActionLoop()._choose_skill_and_params(
        {
            "goal": "Compare the current evidence and preserve a verified result.",
            "source": "cognitive_loop",
        },
        {},
    )
    return bool(
        indirect.asks_for_action
        and hypothetical.is_about_rather_than_asking
        and selection.actionable
        and selection.execution_mode == "planned_goal"
        and selection.provenance == "semantic_plan:live_capability_catalog"
    )


def _metrology_source_contract_holds() -> bool:
    from core.reality_reach.metrology import (
        AcquisitionChannel,
        AcquisitionMode,
        AcquisitionTask,
        EvidenceSource,
    )

    hil = AcquisitionTask(
        task_id="validation.hil",
        channels=(
            AcquisitionChannel("validation.live", EvidenceSource.LIVE),
            AcquisitionChannel("validation.simulated", EvidenceSource.SIMULATED),
        ),
        mode=AcquisitionMode.HARDWARE_IN_LOOP,
        scenario_id="validation.scenario",
    )
    try:
        AcquisitionTask(
            task_id="validation.invalid-live",
            channels=(
                AcquisitionChannel("validation.simulated", EvidenceSource.SIMULATED),
            ),
            mode=AcquisitionMode.LIVE,
        )
    except ValueError:
        refused = True
    else:
        refused = False
    return bool(hil.mode is AcquisitionMode.HARDWARE_IN_LOOP and refused)


def _health() -> dict[str, Any]:
    from core.fsw.health_checker import health_checker_report

    return health_checker_report()


def _slowest_group_fraction() -> float:
    from core.fsw.rate_groups import rate_group_report

    groups = rate_group_report()["groups"]
    fractions = [
        g["p50_ms"] / g["period_ms"] for g in groups if g["period_ms"] and g["cycles"]
    ]
    return max(fractions) if fractions else 0.0


#: Name of the probe container the attribution test grows. Registered once
#: and reused so repeated runs do not accumulate providers.
_ATTRIBUTION_PROBE = "validation.attribution_probe"
_PROBE_CONTAINER: dict[str, int] = {}


def _growth_is_attributable() -> bool:
    """Grow a registered component and see whether the diff names it.

    This is a behavioural check rather than a threshold on a static share
    of RSS. The claim being validated is that memory GROWTH can be
    attributed to a component; measuring a fraction of a mostly-interpreter
    process measures something else and would pass or fail for reasons
    unrelated to the claim.
    """
    from core.runtime.memory_infra import get_memory_infra, register_sized_container

    infra = get_memory_infra()
    if _ATTRIBUTION_PROBE not in infra.providers():
        register_sized_container(
            _ATTRIBUTION_PROBE,
            _PROBE_CONTAINER,
            owner="core/organism/model_validation.py",
            bytes_per_entry=4096,
        )

    before = infra.dump()
    base = len(_PROBE_CONTAINER)
    for index in range(64):
        _PROBE_CONTAINER[f"probe-{base + index}"] = index
    after = infra.dump()
    diff = infra.diff(before, after)
    named = [name for name, delta in diff.top_growers(3) if delta > 0]
    return _ATTRIBUTION_PROBE in named


#: Scopes this test covers. Deliberately excludes "cognition": that scope
#: contains the claim invariant that reads THIS test's result, so
#: including it would make the test's outcome depend on its own previous
#: outcome — a feedback loop that oscillates instead of measuring.
_STRUCTURAL_SCOPES = (
    "container",
    "locks",
    "memory",
    "pressure",
    "flags",
    "integrity",
    "orchestration",
    "middleware",
    "observability",
    "flight_software",
)


def _verifier_errors() -> int:
    from core.verify.invariants import verify

    return len(verify(*_STRUCTURAL_SCOPES, record=False).errors)


def _taint_caveat_consistent() -> bool:
    from core.runtime.health_contract import _runtime_integrity_block
    from core.runtime.taint import credibility_caveat

    caveat = credibility_caveat()
    block = _runtime_integrity_block()
    if caveat is None:
        return "credibility_caveat" not in block
    return block.get("credibility_caveat") == caveat


def validation_report() -> dict[str, Any]:
    return _SUITE.report()


def run_validation() -> dict[str, Any]:
    return _SUITE.run()


def reset_validation_for_test() -> None:
    _SUITE.reset_for_test()


__all__ = [
    "Claim",
    "Model",
    "Observation",
    "Outcome",
    "RuntimeModel",
    "Score",
    "TestResult",
    "ValidationSuite",
    "ValidationTest",
    "boolean_score",
    "get_suite",
    "install_runtime_validation",
    "ratio_score",
    "reset_validation_for_test",
    "run_validation",
    "threshold_score",
    "validation_report",
]
