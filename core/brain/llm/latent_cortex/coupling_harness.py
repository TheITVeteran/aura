"""Run a coupling seam three ways and let the results decide what it is.

`coupling_matrix` refuses a seam whose evidence is metadata, one-directional,
or survives its own lesion. It cannot enforce the first of those on its own:
`kind` arrives already labelled, and a caller who believes their coupling is
behavioral will write `behavioral`. Believing it is the normal case — the field
*is* being passed, the receipt *does* show it — and being wrong about it is
exactly the failure SPARK-067 names.

So this harness does not accept the label. It runs the behavior with the seam
open and with the seam closed, and compares the **observable outcome identity**
of each trial. If every trial produced an identical outcome while the seam was
open and closed, the seam moved a field and nothing else, and the evidence is
recorded as `metadata` no matter what any statistic says about it. That
classification is a measurement here, not a declaration.

The lesion works the same way and answers the other question: an effect that
survives cutting the seam was not flowing through the seam.

Everything is deterministic per trial index, so a seam measurement replays.
The harness runs whatever callables it is given — it has no opinion about what
a subsystem is — which is what lets one honest procedure cover all nine seams.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.coupling_matrix import (
    BEHAVIORAL,
    FORWARD,
    METADATA,
    REVERSE,
    coupling_effect,
    coupling_seam,
    lesion_result,
)

COUPLING_HARNESS_SCHEMA: Final = "aura.rlc.coupling_harness.v1"

# The matrix refuses a direction measured over fewer than 32 observations, so
# refusing here too surfaces the problem where it can still be fixed -- while
# the harness is running, rather than after the matrix rejects the seam.
_MINIMUM_TRIALS: Final = 32


class CouplingHarnessError(ValueError):
    """A seam measurement cannot be run or cannot be trusted."""


def _fail(code: str) -> Never:
    raise CouplingHarnessError(str(code or "coupling_harness_invalid"))


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        _fail(code)
    return number


def _run(
    trials: int,
    behavior: Callable[[int], Any],
    metric: Callable[[Any], float],
    identity: Callable[[Any], str],
    *,
    label: str,
) -> tuple[float, list[str]]:
    values: list[float] = []
    identities: list[str] = []
    for index in range(trials):
        try:
            outcome = behavior(index)
        except Exception as exc:  # noqa: BLE001 - the seam under test may fail
            # A seam that raises is a finding, not a measurement. Averaging
            # over the trials that happened to survive would report a broken
            # seam as a weak one.
            raise CouplingHarnessError(
                f"coupling_harness_{label}_trial_failed"
            ) from exc
        values.append(_finite(metric(outcome), f"coupling_harness_{label}_metric_invalid"))
        marker = identity(outcome)
        if not isinstance(marker, str):
            _fail(f"coupling_harness_{label}_identity_invalid")
        identities.append(marker)
    return sum(values) / trials, identities


def measure_direction(
    *,
    direction: str,
    trials: int,
    seam_closed: Callable[[int], Any],
    seam_open: Callable[[int], Any],
    outcome_metric: Callable[[Any], float],
    outcome_identity: Callable[[Any], str],
) -> dict[str, Any]:
    """Measure one direction, and classify what kind of evidence it produced.

    ``seam_closed`` is the same behavior with the coupling unavailable — the
    baseline. ``seam_open`` is that behavior with the coupling available.

    The returned `kind` is **determined here**: if every trial's outcome
    identity is unchanged between closed and open, the seam moved a field and
    changed nothing an observer could act on, and the evidence is `metadata`.
    """

    if direction not in (FORWARD, REVERSE):
        _fail("coupling_harness_direction_unknown")
    if type(trials) is not int or trials < _MINIMUM_TRIALS:
        _fail("coupling_harness_trials_below_floor")

    baseline, closed_ids = _run(
        trials, seam_closed, outcome_metric, outcome_identity, label="baseline"
    )
    observed, open_ids = _run(
        trials, seam_open, outcome_metric, outcome_identity, label="observed"
    )

    changed = sum(1 for left, right in zip(closed_ids, open_ids, strict=True) if left != right)
    kind = BEHAVIORAL if changed else METADATA

    return {
        **coupling_effect(
            direction=direction,
            kind=kind,
            baseline_statistic=baseline,
            observed_statistic=observed,
            observations=trials,
            evidence_sha256=_evidence(
                direction=direction,
                trials=trials,
                baseline=baseline,
                observed=observed,
                closed_ids=closed_ids,
                open_ids=open_ids,
            ),
        ),
    }


def measure_lesion(
    *,
    trials: int,
    seam_closed: Callable[[int], Any],
    seam_open: Callable[[int], Any],
    seam_lesioned: Callable[[int], Any],
    outcome_metric: Callable[[Any], float],
    outcome_identity: Callable[[Any], str],
) -> dict[str, Any]:
    """Measure baseline, intact, and lesioned behavior over the same trials."""

    if type(trials) is not int or trials < _MINIMUM_TRIALS:
        _fail("coupling_harness_trials_below_floor")

    baseline, baseline_ids = _run(
        trials, seam_closed, outcome_metric, outcome_identity, label="baseline"
    )
    intact, intact_ids = _run(
        trials, seam_open, outcome_metric, outcome_identity, label="intact"
    )
    lesioned, lesioned_ids = _run(
        trials, seam_lesioned, outcome_metric, outcome_identity, label="lesioned"
    )

    return lesion_result(
        baseline_statistic=baseline,
        intact_statistic=intact,
        lesioned_statistic=lesioned,
        observations=trials,
        evidence_sha256=_evidence(
            direction="lesion",
            trials=trials,
            baseline=baseline,
            observed=intact,
            closed_ids=baseline_ids,
            open_ids=intact_ids,
            extra_ids=lesioned_ids,
            extra_statistic=lesioned,
        ),
    )


def measure_coupling_seam(
    *,
    subsystem: str,
    trials: int,
    forward_closed: Callable[[int], Any],
    forward_open: Callable[[int], Any],
    forward_metric: Callable[[Any], float],
    forward_identity: Callable[[Any], str],
    reverse_closed: Callable[[int], Any],
    reverse_open: Callable[[int], Any],
    reverse_metric: Callable[[Any], float],
    reverse_identity: Callable[[Any], str],
    lesioned: Callable[[int], Any],
) -> dict[str, Any]:
    """Measure one seam end to end, ready for `coupling_matrix`.

    The lesion is measured on the forward behavior, because that is the
    direction whose effect a cut is supposed to remove. Both directions still
    have to show a behavioral effect on their own.
    """

    forward = measure_direction(
        direction=FORWARD,
        trials=trials,
        seam_closed=forward_closed,
        seam_open=forward_open,
        outcome_metric=forward_metric,
        outcome_identity=forward_identity,
    )
    reverse = measure_direction(
        direction=REVERSE,
        trials=trials,
        seam_closed=reverse_closed,
        seam_open=reverse_open,
        outcome_metric=reverse_metric,
        outcome_identity=reverse_identity,
    )
    lesion = measure_lesion(
        trials=trials,
        seam_closed=forward_closed,
        seam_open=forward_open,
        seam_lesioned=lesioned,
        outcome_metric=forward_metric,
        outcome_identity=forward_identity,
    )
    return coupling_seam(
        subsystem=subsystem,
        forward=forward,
        reverse=reverse,
        lesion=lesion,
    )


def _evidence(
    *,
    direction: str,
    trials: int,
    baseline: float,
    observed: float,
    closed_ids: Sequence[str],
    open_ids: Sequence[str],
    extra_ids: Sequence[str] | None = None,
    extra_statistic: float | None = None,
) -> str:
    import hashlib
    import json

    body: dict[str, Any] = {
        "schema": COUPLING_HARNESS_SCHEMA,
        "direction": direction,
        "trials": trials,
        "baseline_statistic": round(baseline, 9),
        "observed_statistic": round(observed, 9),
        # Outcome identities are hashed rather than embedded: they can be long
        # and they can carry content that has no business in a receipt.
        "closed_identity_sha256": _digest_all(closed_ids),
        "open_identity_sha256": _digest_all(open_ids),
    }
    if extra_ids is not None:
        body["lesioned_identity_sha256"] = _digest_all(extra_ids)
    if extra_statistic is not None:
        body["lesioned_statistic"] = round(extra_statistic, 9)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _digest_all(identities: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for marker in identities:
        digest.update(marker.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = [
    "COUPLING_HARNESS_SCHEMA",
    "CouplingHarnessError",
    "measure_coupling_seam",
    "measure_direction",
    "measure_lesion",
]
