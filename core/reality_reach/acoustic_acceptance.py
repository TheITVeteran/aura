"""Preregistered A1 acoustic calibration and held-out physical scoring."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.scalar_adapter import ScalarSample
from core.runtime.audit_chain import canonical_json, sha256_hex

ACOUSTIC_A1_RECEIPT_SCHEMA = "aura.reality_reach.acoustic_a1_acceptance.v1"


class AcousticAcceptanceError(RuntimeError):
    """Stable fail-closed A1 experiment error."""


class AcousticTrialArm(StrEnum):
    CALIBRATION = "calibration"
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"
    RESTORATION = "restoration"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@runtime_checkable
class AcousticTrialDriver(Protocol):
    async def measure_stimulus(
        self,
        amplitude: float,
        *,
        trial_id: str,
    ) -> ScalarSample: ...


@dataclass(frozen=True, slots=True)
class AcousticAcceptanceConfig:
    campaign_id: str
    calibration_amplitudes: tuple[float, ...] = (
        0.0,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.04,
        0.08,
    )
    calibration_repeats: int = 2
    heldout_repeats: int = 3
    heldout_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    minimum_signal_span_db: float = 12.0
    baseline_margin_db: float = 6.0
    ceiling_margin_db: float = 3.0
    required_error_reduction: float = 0.5
    restoration_tolerance_db: float = 3.0
    schedule_seed: str = "aura.reality-reach.a1.v1"

    def __post_init__(self) -> None:
        if not self.campaign_id or len(self.campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        amplitudes = tuple(
            _finite(value, name="calibration_amplitude")
            for value in self.calibration_amplitudes
        )
        if (
            len(amplitudes) < 4
            or amplitudes != tuple(sorted(set(amplitudes)))
            or amplitudes[0] != 0.0
            or amplitudes[-1] > 0.2
        ):
            raise ValueError("calibration amplitudes must be unique, sorted, and bounded")
        object.__setattr__(self, "calibration_amplitudes", amplitudes)
        if not 1 <= self.calibration_repeats <= 8 or not 1 <= self.heldout_repeats <= 8:
            raise ValueError("trial repeats must lie inside [1, 8]")
        quantiles = tuple(_finite(value, name="heldout_quantile") for value in self.heldout_quantiles)
        if (
            not quantiles
            or quantiles != tuple(sorted(set(quantiles)))
            or any(not 0.0 < value < 1.0 for value in quantiles)
        ):
            raise ValueError("heldout quantiles must be unique, sorted, and interior")
        object.__setattr__(self, "heldout_quantiles", quantiles)
        for name in (
            "minimum_signal_span_db",
            "baseline_margin_db",
            "ceiling_margin_db",
            "restoration_tolerance_db",
        ):
            value = _finite(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        reduction = _finite(self.required_error_reduction, name="required_error_reduction")
        if not 0.0 < reduction < 1.0:
            raise ValueError("required_error_reduction must lie inside (0, 1)")
        object.__setattr__(self, "required_error_reduction", reduction)
        if not self.schedule_seed or len(self.schedule_seed) > 256:
            raise ValueError("schedule_seed must be bounded and non-empty")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "campaign_id": self.campaign_id,
                "calibration_amplitudes": list(self.calibration_amplitudes),
                "calibration_repeats": self.calibration_repeats,
                "heldout_repeats": self.heldout_repeats,
                "heldout_quantiles": list(self.heldout_quantiles),
                "minimum_signal_span_db": self.minimum_signal_span_db,
                "baseline_margin_db": self.baseline_margin_db,
                "ceiling_margin_db": self.ceiling_margin_db,
                "required_error_reduction": self.required_error_reduction,
                "restoration_tolerance_db": self.restoration_tolerance_db,
                "schedule_seed": self.schedule_seed,
            }
        )


@dataclass(frozen=True, slots=True)
class AcousticTrial:
    trial_id: str
    arm: AcousticTrialArm
    order: int
    repeat: int
    amplitude: float
    target_dbfs: float | None
    observed_dbfs: float
    source_event_id: str
    captured_at_ns: int

    @property
    def absolute_error_db(self) -> float | None:
        return (
            abs(self.observed_dbfs - self.target_dbfs)
            if self.target_dbfs is not None
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "arm": self.arm.value,
            "order": self.order,
            "repeat": self.repeat,
            "amplitude": self.amplitude,
            "target_dbfs": self.target_dbfs,
            "observed_dbfs": self.observed_dbfs,
            "absolute_error_db": self.absolute_error_db,
            "source_event_id": self.source_event_id,
            "captured_at_ns": self.captured_at_ns,
        }


@dataclass(frozen=True, slots=True)
class AcousticA1AcceptanceReceipt:
    campaign_id: str
    config_sha256: str
    heldout_targets_dbfs: tuple[float, ...]
    trials: tuple[AcousticTrial, ...]
    open_loop_mae_db: float
    closed_loop_mae_db: float
    error_reduction: float
    restoration_error_db: float
    calibration_span_db: float
    blockers: tuple[str, ...]
    completed_at_ns: int

    def __post_init__(self) -> None:
        if not self.campaign_id or len(self.campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        digest = self.config_sha256.removeprefix("sha256:")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("config_sha256 must be a canonical sha256 digest")
        targets = tuple(
            _finite(value, name="heldout_target_dbfs")
            for value in self.heldout_targets_dbfs
        )
        if not targets or targets != tuple(sorted(set(targets))):
            raise ValueError("heldout targets must be unique and sorted")
        object.__setattr__(self, "heldout_targets_dbfs", targets)
        if not self.trials:
            raise ValueError("receipt requires measured trials")
        if tuple(trial.order for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ):
            raise ValueError("trial order must be contiguous and one-indexed")
        if self.trials[-1].arm is not AcousticTrialArm.RESTORATION:
            raise ValueError("receipt must end with a restoration trial")
        for name in (
            "open_loop_mae_db",
            "closed_loop_mae_db",
            "error_reduction",
            "restoration_error_db",
            "calibration_span_db",
        ):
            value = _finite(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.open_loop_mae_db < 0.0 or self.closed_loop_mae_db < 0.0:
            raise ValueError("mean absolute errors must be non-negative")
        if self.restoration_error_db < 0.0 or self.calibration_span_db < 0.0:
            raise ValueError("restoration error and calibration span must be non-negative")
        blockers = tuple(str(blocker).strip() for blocker in self.blockers)
        if any(not blocker for blocker in blockers) or len(set(blockers)) != len(blockers):
            raise ValueError("blockers must be unique non-empty strings")
        object.__setattr__(self, "blockers", blockers)
        if isinstance(self.completed_at_ns, bool) or self.completed_at_ns <= 0:
            raise ValueError("completed_at_ns must be positive")

    @property
    def accepted(self) -> bool:
        return not self.blockers

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACOUSTIC_A1_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "config_sha256": self.config_sha256,
            "heldout_targets_dbfs": list(self.heldout_targets_dbfs),
            "trials": [trial.to_dict() for trial in self.trials],
            "open_loop_mae_db": self.open_loop_mae_db,
            "closed_loop_mae_db": self.closed_loop_mae_db,
            "error_reduction": self.error_reduction,
            "restoration_error_db": self.restoration_error_db,
            "calibration_span_db": self.calibration_span_db,
            "blockers": list(self.blockers),
            "completed_at_ns": self.completed_at_ns,
            "raw_audio_retained": False,
            "accepted": self.accepted,
        }
        if include_digest:
            document["receipt_sha256"] = self.sha256
        return document


def _pava(values: Sequence[float]) -> tuple[float, ...]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((_finite(value, name="pava_value"), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_count = blocks.pop()
            left_mean, left_count = blocks.pop()
            count = left_count + right_count
            blocks.append(
                (
                    (left_mean * left_count + right_mean * right_count) / count,
                    count,
                )
            )
    fitted: list[float] = []
    for mean, count in blocks:
        fitted.extend([mean] * count)
    return tuple(fitted)


def _inverse_monotone(
    amplitudes: Sequence[float],
    observed: Sequence[float],
    target: float,
) -> float:
    if target <= observed[0]:
        return amplitudes[0]
    if target >= observed[-1]:
        return amplitudes[-1]
    for index in range(1, len(observed)):
        if target <= observed[index]:
            low_y = observed[index - 1]
            high_y = observed[index]
            if high_y <= low_y:
                return amplitudes[index]
            fraction = (target - low_y) / (high_y - low_y)
            return amplitudes[index - 1] + fraction * (
                amplitudes[index] - amplitudes[index - 1]
            )
    return amplitudes[-1]


def _nominal_open_loop_amplitude(target_dbfs: float, maximum: float) -> float:
    return float(
        min(maximum, max(0.0, math.sqrt(2.0) * 10.0 ** (target_dbfs / 20.0)))
    )


def _trial_order(seed: str, labels: Sequence[tuple[str, float, int]]) -> list[tuple[str, float, int]]:
    return sorted(
        labels,
        key=lambda item: hashlib.sha256(
            f"{seed}|{item[0]}|{item[1]:.9f}|{item[2]}".encode()
        ).digest(),
    )


async def run_acoustic_a1_acceptance(
    driver: AcousticTrialDriver,
    config: AcousticAcceptanceConfig,
) -> AcousticA1AcceptanceReceipt:
    """Run equal-work calibration/control arms and restore silence in ``finally``."""

    if not isinstance(driver, AcousticTrialDriver):
        raise TypeError("driver must satisfy AcousticTrialDriver")
    if not isinstance(config, AcousticAcceptanceConfig):
        raise TypeError("config must be an AcousticAcceptanceConfig")
    trials: list[AcousticTrial] = []
    order = 0

    async def capture(
        arm: AcousticTrialArm,
        amplitude: float,
        *,
        target: float | None,
        repeat: int,
    ) -> AcousticTrial:
        nonlocal order
        order += 1
        trial_id = f"{config.campaign_id}.{arm.value}.{order:04d}"
        sample = await driver.measure_stimulus(amplitude, trial_id=trial_id)
        trial = AcousticTrial(
            trial_id=trial_id,
            arm=arm,
            order=order,
            repeat=repeat,
            amplitude=amplitude,
            target_dbfs=target,
            observed_dbfs=sample.value,
            source_event_id=sample.source_event_id,
            captured_at_ns=sample.captured_at_ns,
        )
        trials.append(trial)
        return trial

    baseline = math.nan
    restoration_error = math.inf
    experiment_error: BaseException | None = None
    try:
        calibration_groups: list[list[float]] = []
        for amplitude in config.calibration_amplitudes:
            group: list[float] = []
            for repeat in range(config.calibration_repeats):
                trial = await capture(
                    AcousticTrialArm.CALIBRATION,
                    amplitude,
                    target=None,
                    repeat=repeat,
                )
                group.append(trial.observed_dbfs)
            calibration_groups.append(group)
        medians = tuple(statistics.median(group) for group in calibration_groups)
        fitted = _pava(medians)
        baseline = fitted[0]
        calibration_span = fitted[-1] - baseline
        lower = baseline + config.baseline_margin_db
        upper = fitted[-1] - config.ceiling_margin_db
        if upper <= lower:
            raise AcousticAcceptanceError("acoustic_a1_calibration_span_insufficient")
        targets = tuple(
            lower + quantile * (upper - lower)
            for quantile in config.heldout_quantiles
        )
        schedule = _trial_order(
            f"{config.schedule_seed}|{config.sha256}",
            tuple(
                (arm.value, target, repeat)
                for target in targets
                for repeat in range(config.heldout_repeats)
                for arm in (AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP)
            ),
        )
        maximum = config.calibration_amplitudes[-1]
        for arm_value, target, repeat in schedule:
            arm = AcousticTrialArm(arm_value)
            amplitude = (
                _nominal_open_loop_amplitude(target, maximum)
                if arm is AcousticTrialArm.OPEN_LOOP
                else _inverse_monotone(config.calibration_amplitudes, fitted, target)
            )
            await capture(arm, amplitude, target=target, repeat=repeat)
    except BaseException as exc:
        experiment_error = exc

    try:
        restored = await capture(
            AcousticTrialArm.RESTORATION,
            0.0,
            target=baseline if math.isfinite(baseline) else None,
            repeat=0,
        )
        if math.isfinite(baseline):
            restoration_error = abs(restored.observed_dbfs - baseline)
    except BaseException as restoration_exc:
        if experiment_error is not None:
            experiment_error.add_note(
                "acoustic silence restoration also failed: "
                f"{type(restoration_exc).__name__}: {restoration_exc}"
            )
            raise experiment_error from restoration_exc
        raise AcousticAcceptanceError("acoustic_a1_restoration_failed") from restoration_exc

    if experiment_error is not None:
        raise experiment_error

    open_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.OPEN_LOOP
        and trial.absolute_error_db is not None
    )
    closed_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.CLOSED_LOOP
        and trial.absolute_error_db is not None
    )
    if not open_errors or len(open_errors) != len(closed_errors):
        raise AcousticAcceptanceError("acoustic_a1_equal_work_contract_failed")
    open_mae = statistics.fmean(open_errors)
    closed_mae = statistics.fmean(closed_errors)
    reduction = 1.0 - closed_mae / max(open_mae, 1e-9)
    calibration_trials = tuple(
        trial for trial in trials if trial.arm is AcousticTrialArm.CALIBRATION
    )
    calibration_values = tuple(trial.observed_dbfs for trial in calibration_trials)
    calibration_span = max(calibration_values) - min(calibration_values)
    blockers: list[str] = []
    if calibration_span < config.minimum_signal_span_db:
        blockers.append("acoustic_a1_signal_span_below_threshold")
    if reduction < config.required_error_reduction:
        blockers.append("acoustic_a1_error_reduction_below_threshold")
    if restoration_error > config.restoration_tolerance_db:
        blockers.append("acoustic_a1_restoration_out_of_tolerance")
    heldout_targets = tuple(
        sorted(
            {
                float(trial.target_dbfs)
                for trial in trials
                if trial.target_dbfs is not None
                and trial.arm in {AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP}
            }
        )
    )
    return AcousticA1AcceptanceReceipt(
        campaign_id=config.campaign_id,
        config_sha256=config.sha256,
        heldout_targets_dbfs=heldout_targets,
        trials=tuple(trials),
        open_loop_mae_db=open_mae,
        closed_loop_mae_db=closed_mae,
        error_reduction=reduction,
        restoration_error_db=restoration_error,
        calibration_span_db=calibration_span,
        blockers=tuple(blockers),
        completed_at_ns=time.time_ns(),
    )


__all__ = [
    "ACOUSTIC_A1_RECEIPT_SCHEMA",
    "AcousticA1AcceptanceReceipt",
    "AcousticAcceptanceConfig",
    "AcousticAcceptanceError",
    "AcousticTrial",
    "AcousticTrialArm",
    "AcousticTrialDriver",
    "run_acoustic_a1_acceptance",
]
