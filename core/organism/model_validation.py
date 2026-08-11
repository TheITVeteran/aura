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


class NothingMeasured(RuntimeError):
    """The instrument ran and had no population to measure.

    Raised so the suite reports :attr:`Outcome.NOT_MEASURED` rather than
    PASS. Three tests here scored a clean zero over an empty set — lockdep
    counted 0 splats across 0 known locks, the rate-group test took
    ``max([])`` of 0 groups, and the health test counted 0 unresponsive
    components out of 0 registered. All three passed, and all three were
    measuring nothing.

    Zero-over-zero is the exact shape this module exists to refuse: the
    absence of a check reported as a passed check. It is not an ERROR either
    — the instrument is fine, there was simply nothing in front of it — and
    the claim it backs is neither confirmed nor refuted, so
    :meth:`ValidationSuite.unsupported_claims` lists it.
    """


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    #: The model does not have the capability the test requires. Not the
    #: same as failing it.
    NOT_APPLICABLE = "n/a"
    #: The test could not run. Also not the same as failing.
    ERROR = "error"
    #: The instrument ran and had no population to measure — lockdep knowing
    #: zero locks, zero rate groups having completed a cycle, zero components
    #: registered with the health checker. Separate from ERROR because the
    #: instrument is not broken, and emphatically separate from PASS: all
    #: three of those scored a clean zero and passed, which is the absence of
    #: a check reported as a passed check.
    NOT_MEASURED = "not_measured"


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
        except NothingMeasured as exc:
            # Not an error: the instrument works and there was nothing in
            # front of it. Reported as its own outcome so an idle subsystem
            # can never be read as a healthy one.
            logger.info(
                "Validation %s measured nothing: %s", self.name, exc
            )
            return TestResult(
                test=self.name,
                model=model.name,
                score=Score(
                    kind="not_measured",
                    value=0.0,
                    outcome=Outcome.NOT_MEASURED,
                    interpretation=str(exc),
                ),
                duration_s=time.perf_counter() - started,
            )
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
    #: Telemetry channels that carry this claim's evidence. Naming them
    #: makes MEASURED_LIVE *decay*: when a bound channel goes silent — or
    #: was never declared — the claim resolves to UNMEASURED instead of
    #: standing on a measurement that stopped happening. Opt-in; a claim
    #: naming nothing is exactly as trustworthy as its author.
    live_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence is not Evidence.MEASURED_LIVE and not self.evidence_note.strip():
            raise ValueError(
                f"claim {self.statement!r} is {self.evidence.value} and says nothing "
                "about what is missing; an unmeasured claim with no note reads as a "
                "measured one"
            )

    def effective_evidence(self) -> tuple["Evidence", str]:
        """Evidence as it stands NOW, after checking bound telemetry.

        The declared value is what someone wrote down; this is what the
        runtime can still show. They differ exactly when a live measurement
        has stopped arriving, which is the failure this registry existed to
        make impossible and could not previously see.
        """
        if not self.live_channels:
            return self.evidence, ""
        try:
            from core.organism.claim_liveness import effective_evidence

            resolved, note, _ = effective_evidence(self.evidence, self.live_channels)
            return resolved, note
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("claim liveness unavailable for %r: %s", self.statement, exc)
            return self.evidence, ""

    @property
    def is_evidence_for_the_system(self) -> bool:
        """Whether this claim may be cited as evidence ABOUT AURA.

        Reads the EFFECTIVE evidence, so a claim whose telemetry has gone
        silent stops being citable the moment it goes silent rather than
        when someone next reviews it by hand.
        """
        resolved, _ = self.effective_evidence()
        return resolved is Evidence.MEASURED_LIVE

    def to_dict(self) -> dict[str, Any]:
        resolved, liveness_note = self.effective_evidence()
        return {
            "statement": self.statement,
            "test": self.test,
            "owner": self.owner,
            "asserted_in": self.asserted_in,
            "evidence": self.evidence.value,
            "effective_evidence": resolved.value,
            "evidence_note": self.evidence_note,
            "liveness_note": liveness_note,
            "live_channels": list(self.live_channels),
            "citable_as_evidence": resolved is Evidence.MEASURED_LIVE,
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
        unmeasured = [r for r in results if r.score.outcome is Outcome.NOT_MEASURED]
        return {
            "at": time.time(),
            "models": [m.name for m in models],
            "tests": len(tests),
            "results": [r.to_dict() for r in results],
            "by_outcome": by_outcome,
            "passed": by_outcome.get(str(Outcome.PASS), 0),
            "failed": len(failures),
            "errored": len(errors),
            # Instruments that ran with nothing in front of them. Counted
            # separately from both PASS and ERROR: these used to score a
            # clean zero and pass.
            "not_measured": len(unmeasured),
            # A suite where everything is N/A passes vacuously; say so.
            "applicable": len(results) - by_outcome.get(str(Outcome.NOT_APPLICABLE), 0),
            # ...and neither an N/A nor an empty instrument is evidence.
            "measured": (
                len(results)
                - by_outcome.get(str(Outcome.NOT_APPLICABLE), 0)
                - len(unmeasured)
            ),
            "failures": [r.to_dict() for r in failures],
            "errors": [r.to_dict() for r in errors],
            "unmeasured": [r.to_dict() for r in unmeasured],
        }

    #: Outcomes that leave a claim standing on nothing. NOT_MEASURED belongs
    #: here for the same reason FAIL does: a claim backed by an instrument
    #: that had no population is a claim with no evidence behind it, however
    #: clean the number looked.
    _UNSUPPORTING = (Outcome.FAIL, Outcome.ERROR, Outcome.NOT_MEASURED)

    def unsupported_claims(self) -> list[dict[str, Any]]:
        """Claims whose test last failed, could not run, or measured nothing.

        This is the machine-checked version of CLAIMS_NOT_SUPPORTED.md.
        """
        out: list[dict[str, Any]] = []
        with self._lock:
            claims = list(self._claims.values())
            last = dict(self._last)
        for claim in claims:
            # A claim can lose its footing two ways: its test stops passing,
            # or the live measurement behind it stops arriving. The second
            # was invisible until claims could bind to telemetry, and it is
            # the one that produced "a claim that outlived the code".
            resolved, liveness_note = claim.effective_evidence()
            if liveness_note and resolved is not claim.evidence:
                out.append(
                    {
                        **claim.to_dict(),
                        "reason": liveness_note,
                        "outcome": "evidence_decayed",
                    }
                )
                continue
            relevant = [r for (test, _model), r in last.items() if test == claim.test]
            if not relevant:
                out.append({**claim.to_dict(), "reason": "never run"})
                continue
            if any(r.score.outcome in self._UNSUPPORTING for r in relevant):
                worst = next(
                    r for r in relevant if r.score.outcome in self._UNSUPPORTING
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
        # An undeclared capability makes its test score "n/a", and a claim
        # bound to a test that never runs is exactly the unsupported claim
        # this suite exists to surface. reality_metrology was registered
        # without ever being declared, so its claim has never once been
        # checked; the two boundary claims below are declared with it.
        "reality_metrology",
        "egress_privacy",
        "state_attestation",
        "commitment_search",
        "rlc_capability_evidence",
        "rlc_governed_web_acquisition",
        "rlc_verified_amplifier_composition",
        "kernel_confined_symbolic_cognition",
        "rlc_closed_loop_compute",
        "work_grounded_claims",
        "prompt_boundary_detection",
    )
    suite.add_model(model)

    suite.add_test(
        ValidationTest(
            name="fabrication_audit_never_accuses_an_unknown_turn",
            description=(
                "a claim of work whose turn the ledger never saw resolves "
                "UNKNOWN, never UNSUPPORTED, so eviction cannot manufacture a "
                "fabrication finding"
            ),
            required_capability="work_grounded_claims",
            observation=Observation(
                name="unsupported_findings_on_an_unknown_turn",
                value=0,
                source=(
                    "core/verify/fabrication_audit.py and "
                    "tests/test_fabrication_audit.py"
                ),
                units="findings",
            ),
            predict=lambda _m: _fabrication_unknown_turn_findings(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" findings"),
            owner="core/verify/fabrication_audit.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="canary_failure_is_never_reported_as_an_attack",
            description=(
                "an unevaluable injection probe resolves INCONCLUSIVE and stays "
                "out of the incident rate, so a model outage cannot manufacture "
                "a security verdict"
            ),
            required_capability="prompt_boundary_detection",
            observation=Observation(
                name="incidents_from_unevaluable_probes",
                value=0,
                source=(
                    "core/security/injection_canary.py and "
                    "tests/test_injection_canary.py"
                ),
                units="incidents",
            ),
            predict=lambda _m: _canary_incidents_from_failures(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" incidents"),
            owner="core/security/injection_canary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="sequential_exclusion_dominates_iid_sampling",
            description=(
                "removing refuted answer mass and renormalising never lowers "
                "the per-draw probability of drawing a correct answer"
            ),
            required_capability="commitment_search",
            observation=Observation(
                name="cases_where_iid_wins",
                value=0,
                source=(
                    "core/brain/llm/latent_cortex/sequential_exclusion.py — "
                    "P(draw k+1 correct) = p*/(1 - m_k) >= p* for every "
                    "distribution, so no case can exist"
                ),
                units="cases",
            ),
            predict=lambda _m: _exclusion_losses_to_iid(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" cases"),
            owner="core/brain/llm/latent_cortex/sequential_exclusion.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="current_governed_observations_seed_recurrent_cognition",
            description=(
                "a successful current-turn governed observation becomes a "
                "content-addressed non-authoritative RLC context slot while stale "
                "or state-mutating results do not"
            ),
            required_capability="rlc_capability_evidence",
            observation=Observation(
                name="capability_evidence_contract_holds",
                value=True,
                source=(
                    "core/brain/capability_evidence_context.py and "
                    "tests/test_rlc_capability_evidence_context.py"
                ),
            ),
            predict=lambda _m: _rlc_capability_evidence_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="RLC capability-evidence admission",
            ),
            owner="core/brain/capability_evidence_context.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="recurrent_cognition_can_request_bounded_live_evidence",
            description=(
                "a retrieval-class recurrent action can select live evidence when "
                "the objective is temporal or the offline corpus is uncovered, "
                "through bounded public-research standing authority"
            ),
            required_capability="rlc_governed_web_acquisition",
            observation=Observation(
                name="governed_web_acquisition_contract_holds",
                value=True,
                source=(
                    "core/brain/cortex_web_acquisition.py and "
                    "tests/test_cortex_web_acquisition.py"
                ),
            ),
            predict=lambda _m: _rlc_web_acquisition_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="RLC governed web acquisition",
            ),
            owner="core/brain/cortex_web_acquisition.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="recurrent_answer_enters_verified_complete_engine",
            description=(
                "a canonical RLC result is admitted as a bounded candidate and must "
                "survive the same verifier and calibration path as generated alternatives"
            ),
            required_capability="rlc_verified_amplifier_composition",
            observation=Observation(
                name="rlc_seed_composition_contract_holds",
                value=True,
                source=(
                    "core/brain/reasoning_amplifier_v2.py and "
                    "tests/test_response_generation_unitary_tiering.py"
                ),
            ),
            predict=lambda _m: _rlc_amplifier_composition_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="RLC verified amplifier composition",
            ),
            owner="core/phases/response_generation_unitary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="symbolic_cognition_has_kernel_boundary",
            description=(
                "model-written Python used by cognitive verification runs only when "
                "the host exposes a supported kernel sandbox"
            ),
            required_capability="kernel_confined_symbolic_cognition",
            observation=Observation(
                name="kernel_sandbox_available",
                value=True,
                source="core/sandbox/untrusted_python.py",
            ),
            predict=lambda _m: _symbolic_cognition_boundary_available(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="symbolic cognition kernel boundary",
            ),
            owner="core/brain/symbolic_sandbox.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="recurrent_compute_actions_receive_machine_feedback",
            description=(
                "successful formalize and simulate actions produce one bounded, "
                "typed machine observation for one recurrent continuation"
            ),
            required_capability="rlc_closed_loop_compute",
            observation=Observation(
                name="rlc_compute_continuation_contract_holds",
                value=True,
                source=(
                    "core/brain/cortex_compute_acquisition.py and "
                    "tests/test_rlc_cognitive_acquisition.py"
                ),
            ),
            predict=lambda _m: _rlc_compute_continuation_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="RLC closed-loop compute",
            ),
            owner="core/brain/latent_cortex_service.py",
        )
    )
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
            predict=lambda _m: _lockdep_splats(),
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
            predict=lambda _m: _critical_unresponsive(),
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
    suite.add_test(
        ValidationTest(
            name="certified_typed_transitions_are_exact",
            description=(
                "the recurrence executor returns exact Boolean and modular next "
                "states over its complete declared primitive domain"
            ),
            required_capability="",
            observation=Observation(
                name="all_declared_transitions_exact",
                value=True,
                source=(
                    "core/brain/llm/latent_cortex/typed_transition_executor.py "
                    "exhaustive contract"
                ),
            ),
            predict=lambda _m: _certified_transition_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="certified typed recurrence",
            ),
            owner="core/brain/llm/latent_cortex/typed_transition_executor.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="certified_programs_compose_from_student_rollin",
            description=(
                "each recurrent transition consumes the prior computed state and "
                "still matches independently generated depth-32 traces"
            ),
            required_capability="",
            observation=Observation(
                name="student_rollin_matches_verified_trace",
                value=True,
                source="core/learning/certified_transition_program.py contract tests",
            ),
            predict=lambda _m: _certified_student_rollin_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="certified recurrent student roll-in",
            ),
            owner="core/learning/certified_transition_program.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="public_transition_prompts_compile_without_private_labels",
            description=(
                "declared Boolean and modular prompts compile into exact typed "
                "programs using public prompt evidence only"
            ),
            required_capability="",
            observation=Observation(
                name="public_compilation_matches_private_audit_trace",
                value=True,
                source=(
                    "core/brain/llm/latent_cortex/typed_action_compiler.py "
                    "fresh-seed contract tests"
                ),
            ),
            predict=lambda _m: _public_transition_compiler_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="public-evidence typed action compilation",
            ),
            owner="core/brain/llm/latent_cortex/typed_action_compiler.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="neural_transition_tissue_enters_complete_engine",
            description=(
                "a wrong incumbent is replaceable by teacher-removed neural recurrent "
                "execution through the complete-engine producer"
            ),
            required_capability="",
            observation=Observation(
                name="neural_recurrent_candidate_is_verified",
                value=True,
                source=(
                    "core/brain/llm/latent_cortex/objective_program_verifier.py "
                    "complete-engine contract tests"
                ),
            ),
            predict=lambda _m: _neural_complete_engine_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="neural recurrent complete-engine candidate",
            ),
            owner="core/brain/llm/latent_cortex/objective_program_verifier.py",
        )
    )

    suite.add_test(
        ValidationTest(
            name="cloud_prompts_are_read_before_they_leave",
            description=(
                "a credential in a body bound for a third-party model is stripped, "
                "and a body that cannot be inspected is refused rather than sent"
            ),
            required_capability="egress_privacy",
            observation=Observation(
                name="outbound_bodies_are_inspected",
                value=True,
                source="core/security/egress_privacy.py boundary tests",
            ),
            predict=lambda _m: _egress_privacy_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="outbound content inspection",
            ),
            owner="core/security/egress_privacy.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="identity_state_that_failed_attestation_is_not_loaded",
            description=(
                "a self-profile modified outside Aura's own write path is quarantined "
                "and contributes nothing to the identity block"
            ),
            required_capability="state_attestation",
            observation=Observation(
                name="tampered_identity_is_refused",
                value=True,
                source="core/security/state_attestation.py attestation tests",
            ),
            predict=lambda _m: _identity_attestation_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="identity state attestation",
            ),
            owner="core/security/state_attestation.py",
        )
    )

    for statement, test_name, asserted_in in (
        (
            "A credential never leaves this machine inside a prompt bound for a "
            "third-party model, and a body that cannot be read is not sent to one.",
            "cloud_prompts_are_read_before_they_leave",
            "core/security/egress_privacy.py",
        ),
        (
            "Identity state that fails attestation is quarantined rather than loaded, "
            "so Aura boots with no self-model rather than someone else's.",
            "identity_state_that_failed_attestation_is_not_loaded",
            "core/security/state_attestation.py",
        ),
        (
            # Scoped deliberately. Lockdep can only order the locks it wraps,
            # and most of this runtime's locks are still raw threading/asyncio
            # primitives it never sees — capability_engine was instrumented
            # *after* it deadlocked the boot path, not before. The unscoped
            # version of this sentence claimed the whole runtime was clear of
            # ABBA on evidence covering a minority of its locks.
            # tools/lint_lock_coverage.py holds the ratchet that shrinks the
            # gap; when it reaches parity this qualifier can go.
            "Among the locks lockdep instruments, the runtime takes its locks in a "
            "consistent order and has no latent ABBA deadlock. Locks not wrapped in "
            "checked_lock/checked_async_lock are outside this claim.",
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

    # Graded honestly. Both mechanisms are proven by construction and by
    # test, and neither has yet run against live traffic — the work ledger
    # has recorded no production turns and no canary has been evaluated on
    # a real request. Every previous claim in this file that skipped that
    # distinction had to be walked back later, so these say it up front.
    suite.add_claim(
        Claim(
            statement=(
                "A persisted claim to have done work is checked against the "
                "record of what the turn actually ran."
            ),
            test="fabrication_audit_never_accuses_an_unknown_turn",
            owner="core/verify/fabrication_audit.py",
            asserted_in="core/verify/fabrication_audit.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the UNKNOWN-vs-UNSUPPORTED property is measured on constructed "
                "turns; no live turn has been audited, and the claim-pattern "
                "table's recall against real confabulations is unmeasured"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Given a valid typed state and typed action, the certified recurrent "
                "executor computes the exact next Boolean or bounded modular state."
            ),
            test="certified_typed_transitions_are_exact",
            owner="core/brain/llm/latent_cortex/typed_transition_executor.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Exhaustive over 504 Boolean and 3,828 modular primitive transitions; "
                "semantic action compilation, broader families, live use, and reasoning "
                "gain remain separate gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Certified typed transitions compose to depth 32 by consuming their "
                "own prior output rather than private teacher states."
            ),
            test="certified_programs_compose_from_student_rollin",
            owner="core/learning/certified_transition_program.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on generated Boolean and modular programs with lesion and "
                "restoration controls; semantic compilation and behavioral gain remain "
                "separate gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "For the declared Boolean and bounded modular grammars, Aura can "
                "compile public task text into certified recurrent actions without "
                "reading an answer or private transition trace."
            ),
            test="public_transition_prompts_compile_without_private_labels",
            owner="core/brain/llm/latent_cortex/typed_action_compiler.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on fresh generated prompts through depth 32 with sham, "
                "mutation, ambiguity, and receipt-privacy controls. This is a strict "
                "declared-grammar compiler, not general natural-language planning or "
                "a behavioral reasoning-gain claim."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "For declared Boolean and bounded modular tasks, the complete engine "
                "can replace a wrong decoded answer with a candidate computed by "
                "teacher-removed neural recurrent student roll-in."
            ),
            test="neural_transition_tissue_enters_complete_engine",
            owner="core/brain/llm/latent_cortex/objective_program_verifier.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The sealed tissue learned 3,842 primitive transitions and is causal in "
                "the production answer-replacement route after its training teacher is "
                "removed. Public action selection is still a strict symbolic compiler; "
                "open-domain depth gain and resident-model execution remain unmeasured."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Whether the model followed instructions inside a fenced block "
                "is detected rather than assumed."
            ),
            test="canary_failure_is_never_reported_as_an_attack",
            owner="core/security/injection_canary.py",
            asserted_in="core/security/injection_canary.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "fail-open behaviour is measured against constructed responses; "
                "no canary has ridden a live request, so the detection rate "
                "against a real injection attempt is unmeasured"
            ),
        )
    )

    # Graded MEASURED_SYNTHETIC on purpose. The arithmetic is proven and the
    # policy is wired into live best-of-N, but no live reasoning gain has
    # been measured — and every previous RLC claim that skipped that
    # distinction had to be walked back. The note says exactly what is
    # missing so nobody has to reconstruct it later.
    suite.add_claim(
        Claim(
            statement=(
                "Excluding refuted answers makes best-of-N search strictly "
                "better-covering than independent sampling."
            ),
            test="sequential_exclusion_dominates_iid_sampling",
            owner="core/brain/llm/latent_cortex/sequential_exclusion.py",
            asserted_in="docs/RLC_COMMITMENT_SEARCH.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured end to end 2026-08-09 on Qwen2.5-1.5B, 160 paired "
                "tasks, sound non-oracle verifier: 48.1% -> 59.4% solved "
                "(z=2.02, p=0.044) on FEWER verifier calls (3.97 -> 3.19). The "
                "dominance arithmetic is separately swept over 400 constructed "
                "distributions with zero counterexamples, and the peakedness "
                "premise measured at 0.516 (2.58 distinct answers per 8 i.i.d. "
                "draws). The gain requires REJECTION SAMPLING: the "
                "prompt-conditioned form was measured and lost (46.9%), because "
                "describing excluded answers perturbs the distribution rather "
                "than restricting it. NOT measured: transfer to the resident "
                "32B, to long-answer tasks where sameness needs a semantic "
                "judge, or to a verifier expensive enough to change the trade. "
                "One model, one task family, one difficulty band, p at the edge "
                "of noise."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A retrieval-class recurrent action can request one bounded "
                "read-only live-web observation when local knowledge is stale or absent."
            ),
            test="recurrent_cognition_can_request_bounded_live_evidence",
            owner="core/brain/cortex_web_acquisition.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Planner, authority origin, service broker, evidence admission, and "
                "continuation are contract-tested. Network availability and resident-32B "
                "use remain installed-runtime proof gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The healthy foreground path composes a canonical RLC answer with "
                "the verifier-backed reasoning amplifier instead of selecting only one."
            ),
            test="recurrent_answer_enters_verified_complete_engine",
            owner="core/phases/response_generation_unitary.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The response contract admits RLC output as a bounded candidate, carries "
                "admitted evidence, and adopts only a verifier-clean amplifier result."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Symbolic cognitive Python requires an OS kernel sandbox and refuses "
                "execution when no supported boundary exists."
            ),
            test="symbolic_cognition_has_kernel_boundary",
            owner="core/brain/symbolic_sandbox.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The macOS test host executed pure computation under Seatbelt; Linux "
                "requires bubblewrap and unsupported hosts fail closed."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "When recurrent cognition chooses to formalize or simulate, one "
                "bounded machine result can causally inform a second episode."
            ),
            test="recurrent_compute_actions_receive_machine_feedback",
            owner="core/brain/latent_cortex_service.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Exact-math contradiction, single sandbox execution, typed evidence "
                "admission, and the one-round continuation cap are contract-tested. "
                "Resident-32B selection frequency and reasoning gain remain empirical gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A successful current-turn governed observation can causally seed "
                "the recurrent workspace without gaining instruction authority."
            ),
            test="current_governed_observations_seed_recurrent_cognition",
            owner="core/brain/capability_evidence_context.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The source-level contract, tamper rejection, freshness binding, and "
                "workspace handoff are measured. Installed-app resident-32B execution "
                "and reasoning gain remain separate live proof gates."
            ),
        )
    )

    return {
        "model": model.name,
        "tests": [t.name for t in suite.tests()],
        "claims": len(suite.claims()),
    }


# ── prediction helpers, kept small and failure-tolerant ───────────────


def _certified_transition_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.typed_transition_executor import (
        CertifiedTransitionExecutor,
        TypedTransitionInput,
    )

    executor = CertifiedTransitionExecutor()
    boolean_actions = ((0, 0, 0),) + tuple(
        (opcode, operand, 1) for opcode in (1, 2, 3) for operand in (0, 1)
    )
    for depth in range(1, 9):
        for pc in range(depth):
            for value in (0, 1):
                for opcode, operand, has_operand in boolean_actions:
                    result = executor.execute(
                        TypedTransitionInput(
                            family="boolean",
                            depth=depth,
                            field_names=("pc", "value", "done"),
                            state=(pc, value, 0),
                            action_field_names=("opcode", "operand", "has_operand"),
                            action=(opcode, operand, has_operand),
                        )
                    )
                    expected = (
                        1 - value
                        if opcode == 0
                        else value & operand
                        if opcode == 1
                        else value | operand
                        if opcode == 2
                        else value ^ operand
                    )
                    if result.next_state != (
                        pc + 1,
                        expected,
                        int(pc + 1 == depth),
                    ):
                        return False
    for modulus in (13, 17, 19, 23):
        for residue in range(modulus):
            for operand in range(1, modulus):
                for opcode in (0, 1, 2):
                    result = executor.execute(
                        TypedTransitionInput(
                            family="modular",
                            depth=4,
                            field_names=("pc", "residue", "done"),
                            state=(2, residue, 0),
                            action_field_names=("opcode", "operand", "modulus"),
                            action=(opcode, operand, modulus),
                        )
                    )
                    expected = (
                        (residue + operand) % modulus
                        if opcode == 0
                        else (residue * operand) % modulus
                        if opcode == 1
                        else (residue - operand) % modulus
                    )
                    if result.next_state != (3, expected, 0):
                        return False
    return True


def _certified_student_rollin_contract_holds() -> bool:
    from core.learning.certified_transition_program import (
        execute_program_student_rollin,
    )
    from core.learning.recurrence_curriculum import modular_chain, nested_boolean

    for generator in (nested_boolean, modular_chain):
        program = generator(32, 20260810191).transition_program
        if program is None:
            return False
        execution = execute_program_student_rollin(program)
        if execution.states != program.state_trace.states:
            return False
    return True


def _public_transition_compiler_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.typed_action_compiler import (
        compile_public_transition_program,
    )
    from core.learning.certified_transition_program import (
        execute_compiled_action_program,
    )
    from core.learning.recurrence_curriculum import modular_chain, nested_boolean

    for generator in (nested_boolean, modular_chain):
        task = generator(32, 20260810192)
        compiled = compile_public_transition_program(task.prompt)
        execution = execute_compiled_action_program(compiled)
        if (
            task.transition_trace is None
            or execution.states != task.transition_trace.states
        ):
            return False
    return True


def _neural_complete_engine_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.objective_program_verifier import (
        solve_objective_program,
        verify_objective_program,
    )
    from core.learning.recurrence_curriculum import modular_chain

    task = modular_chain(8, 20260810193)
    solved = solve_objective_program(task.prompt)
    if solved is None:
        return False
    candidate, receipt = solved
    verdict = verify_objective_program(candidate, objective=task.prompt)
    execution = receipt.get("execution", {})
    return bool(
        isinstance(execution, dict)
        and execution.get("engine") == "neural_transition_tissue.v1"
        and execution.get("teacher_available") is False
        and execution.get("student_rollin", {}).get("transition_count") == 8
        and candidate.endswith(task.answer)
        and verdict is not None
        and verdict.get("outcome") == "verified"
    )


def _fabrication_unknown_turn_findings() -> int:
    """Findings wrongly marked UNSUPPORTED for a turn the ledger never saw.

    Structurally zero: audit_text reads Support.UNKNOWN whenever the work
    ledger has no record of the turn. Measured rather than asserted, because
    the whole value of the audit rests on it — a detector that converts
    ledger eviction into accusations is worse than no detector.
    """
    from core.verify.fabrication_audit import Support, audit_text

    findings = audit_text(
        "I searched for it and ran the code to check.",
        "a-turn-that-was-never-recorded",
    )
    return sum(1 for f in findings if f.support is Support.UNSUPPORTED)


def _canary_incidents_from_failures() -> int:
    """Security incidents produced by probes that could not be evaluated.

    Structurally zero: an empty, missing or unreadable response resolves
    INCONCLUSIVE, which is_incident excludes. Measured because the failure
    mode it guards — a busy 32B manufacturing hijack verdicts — would be
    both invisible and self-reinforcing.
    """
    from core.security.injection_canary import inspect_response, mint_canary

    canary = mint_canary()
    return sum(
        1 for bad in (None, "", "   ") if inspect_response(bad, canary).is_incident
    )


def _exclusion_losses_to_iid() -> int:
    """Count distributions where i.i.d. sampling beats exclusion. Must be 0.

    Swept rather than spot-checked: the claim is universal, so a single
    counterexample refutes it and the test has to be able to find one.
    """
    import random

    from core.brain.llm.latent_cortex.sequential_exclusion import (
        exclusion_success_probability,
        iid_success_probability,
    )

    losses = 0
    rng = random.Random(20260809)
    for _ in range(400):
        p_star = rng.uniform(0.01, 0.9)
        draws = rng.randint(1, 24)
        masses = [rng.random() * (1.0 - p_star) / 4 for _ in range(draws)]
        if exclusion_success_probability(p_star, masses, draws) < (
            iid_success_probability(p_star, draws) - 1e-12
        ):
            losses += 1
    return losses


def _rlc_capability_evidence_contract_holds() -> bool:
    import hashlib

    from core.brain.capability_evidence_context import (
        build_current_turn_capability_evidence,
    )

    objective = "Use Python to calculate the exact checksum total."
    objective_sha256 = hashlib.sha256(objective.encode("utf-8")).hexdigest()
    admitted = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": objective_sha256,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "checksum_total=4182",
                "exit_code": 0,
            },
        },
        objective,
    )
    stale = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": "0" * 64,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "stale=1",
                "exit_code": 0,
            },
        },
        objective,
    )
    return bool(
        admitted.receipt.get("admitted") is True
        and len(admitted.items) == 1
        and admitted.items[0].get("instruction_authority") is False
        and admitted.items[0].get("evidence_kind") == "governed_tool_observation"
        and not stale.items
        and stale.receipt.get("reason") == "stale_skill_result"
    )


def _rlc_web_acquisition_contract_holds() -> bool:
    from core.brain.cortex_web_acquisition import should_acquire_live_web
    from core.brain.llm.latent_cortex.context_focus import source_matches_action
    from core.executive.standing_authority import AUTONOMOUS_AUTHORITY_ORIGINS

    live = should_acquire_live_web(
        "What is the latest compiler release?",
        "compiler release",
        local_context_is_new=True,
    )
    uncovered = should_acquire_live_web(
        "Explain the new theorem.",
        "new theorem",
        local_context_is_new=False,
    )
    return bool(
        live == (True, "live_or_source_sensitive_objective")
        and uncovered == (True, "local_reference_uncovered")
        and "latent_cortex" in AUTONOMOUS_AUTHORITY_ORIGINS
        and source_matches_action("capability.web_search", "retrieve_evidence")
    )


def _rlc_amplifier_composition_contract_holds() -> bool:
    from core.brain.reasoning_amplifier_v2 import _admit_seed_candidates

    return _admit_seed_candidates(
        ["candidate", "candidate", ""],
        limit=2,
    ) == ["candidate"]


def _symbolic_cognition_boundary_available() -> bool:
    from core.sandbox.untrusted_python import available_boundary

    return available_boundary() in {"seatbelt", "bubblewrap"}


def _rlc_compute_continuation_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.cognitive_acquisition import (
        acquisition_has_new_context,
        build_acquisition_request,
    )

    transition = {
        "action": "formalize",
        "outcome": "succeeded",
        "checked": True,
    }
    request = build_acquisition_request(
        objective="Compute 12 * 13 exactly.",
        first_text="The answer is 157.",
        first_receipt={
            "cognitive_action_trace": [
                {"decision": {"action": "formalize"}, "transition": transition}
            ]
        },
        cognitive_context=None,
    )
    return bool(
        request
        and request.get("action") == "formalize"
        and request.get("max_acquisitions") == 1
        and request.get("max_continuation_rounds") == 1
        and acquisition_has_new_context(
            request,
            [
                {
                    "source": "capability.symbolic_formalize",
                    "text": "exact(12*13) = 156",
                }
            ],
        )
    )


def _lockdep() -> dict[str, Any]:
    from core.runtime.lockdep import lockdep_report

    return lockdep_report()


def _lockdep_splats() -> int:
    """Order violations, but only once lockdep has actually seen a lock.

    ``known_locks`` is what lockdep can reason about; ``acquires_checked``
    is whether anything was taken while it watched. An empty either way means
    no ordering evidence exists in this process, however clean the number
    looks.
    """
    report = _lockdep()
    if not report.get("known_locks"):
        raise NothingMeasured(
            "lockdep knows 0 locks in this process; 0 splats is not evidence "
            "of correct lock ordering"
        )
    if not report.get("acquires_checked"):
        raise NothingMeasured(
            f"lockdep knows {len(report['known_locks'])} lock(s) but observed 0 "
            "acquisitions; no ordering was exercised"
        )
    return len(report["splats"])


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


def _egress_privacy_contract_holds() -> bool:
    """Exercise the boundary rather than assert that it exists.

    A registered claim whose predicate only imported the module would be the
    thing this suite is for catching.
    """
    from core.security.egress_privacy import filter_outbound_body

    secret = "sk-" + "a" * 24
    stripped = filter_outbound_body(
        url="https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
        body=f'{{"contents":"key {secret}"}}'.encode(),
        source="llm_provider:gemini:probe",
    )
    # The same secret one character to the left of the colon. The walk used to
    # read values only, so this exact body left the machine intact while the
    # one above was caught — and the claim said "never" for both.
    keyed = filter_outbound_body(
        url="https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
        body=f'{{"{secret}":"quota"}}'.encode(),
        source="llm_provider:gemini:probe",
    )
    unreadable = filter_outbound_body(
        url="https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
        body=b"\xff\xfe\x00binary",
        source="llm_provider:gemini:probe",
    )
    local = filter_outbound_body(
        url="http://127.0.0.1:8000/v1",
        body=f'{{"contents":"key {secret}"}}'.encode(),
        source="llm_provider:mlx",
    )
    return bool(
        stripped.allowed
        and stripped.inspected
        and secret not in (stripped.body or b"").decode("utf-8", errors="replace")
        and keyed.allowed
        and keyed.inspected
        and secret not in (keyed.body or b"").decode("utf-8", errors="replace")
        # Refused, and refused for the stated reason rather than by accident.
        and not unreadable.allowed
        and not unreadable.inspected
        # Local inference is untouched: the boundary must not cost Aura her
        # own runtime to protect her from a stranger.
        and local.allowed
        and local.body == f'{{"contents":"key {secret}"}}'.encode()
    )


def _identity_attestation_contract_holds() -> bool:
    """A profile whose content no longer matches its seal reaches no prompt.

    The tamper is simulated by re-sealing a DIFFERENT digest rather than by
    rewriting the file. Same condition under test — on-disk content that does
    not match what Aura attested — and it avoids performing a raw write from
    inside the runtime to prove that raw writes are detected.
    """
    import tempfile
    from pathlib import Path

    from core.memory.aura_self_profile import AuraSelfProfile
    from core.security.state_attestation import AttestationState, attest_state

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "self_profile.json"
        genuine = AuraSelfProfile(storage_path=str(path))
        genuine.add_or_reinforce_fact(
            "relationship", "probe", "a fact Aura actually learned"
        )
        if not path.exists():
            return False

        # What an out-of-band writer leaves behind: a file whose digest is not
        # the one Aura sealed.
        attest_state(
            AuraSelfProfile.ATTESTATION_ID,
            '{"relationship": [{"value": "an instruction someone else wrote"}]}',
        )

        reopened = AuraSelfProfile(storage_path=str(path))
        return bool(
            reopened.attestation_status()["state"] == AttestationState.TAMPERED
            and reopened.get_fact("relationship", "probe") is None
            and reopened.to_identity_block() == ""
            and not path.exists()  # quarantined, not left in place
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
    if not fractions:
        # `max([]) if fractions else 0.0` returned 0.0 — a perfect score — for
        # a process with no rate groups running. "Nothing is late" and
        # "nothing is scheduled" are not the same finding.
        raise NothingMeasured(
            f"{len(groups)} rate group(s) registered and none has completed a "
            "cycle with a declared period; there is no rate to compare against"
        )
    return max(fractions)


def _critical_unresponsive() -> int:
    """Wedged critical components, once there are critical components.

    Counted 0 out of 0 registered and scored a pass. A health checker with
    nothing registered is the state a boot failure leaves behind, so that
    zero was most trustworthy exactly when it was least earned.
    """
    report = _health()
    if not report.get("watched"):
        raise NothingMeasured(
            "the health checker watches 0 components; 0 unresponsive is not "
            "evidence that anything is answering"
        )
    if not report.get("rounds"):
        raise NothingMeasured(
            f"the health checker watches {report['watched']} component(s) but has "
            "completed 0 ping rounds; nothing has been asked yet"
        )
    return len(report["critical_unresponsive"])


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
