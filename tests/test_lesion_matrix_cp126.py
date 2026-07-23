"""CP126 contract tests for the lesion (ablation) study."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.architect.lesion_matrix import (
    LesionableComponent,
    LesionStudy,
)


class _Region:
    """A tiny stateful component with an observable value."""

    def __init__(self, value: float = 1.0):
        self.state = np.array([value], dtype=np.float64)

    def component(self, name: str) -> LesionableComponent:
        return LesionableComponent(
            name=name,
            get_state=lambda: self.state,
            set_state=lambda s: setattr(self, "state", np.array(s, dtype=np.float64)),
            zero_fn=lambda: self.state.fill(0.0),
        )


def _study(tmp_path, **kwargs) -> LesionStudy:
    return LesionStudy(n_probe_steps=2, data_dir=tmp_path, **kwargs)


# --- 751ee90e: lesion/restore is transactional ----------------------------


def test_a_probe_exception_does_not_leave_a_component_lesioned(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))

    boom = {"count": 0}

    def exploding_probe():
        boom["count"] += 1
        if boom["count"] > 2:
            raise RuntimeError("probe blew up mid-lesion")
        return float(region.state[0])

    study.register_probe("p", exploding_probe)
    study.run()

    assert float(region.state[0]) == 1.0


def test_a_step_exception_does_not_leave_a_component_lesioned(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    calls = {"n": 0}

    def exploding_step():
        calls["n"] += 1
        if calls["n"] > 2:
            raise ValueError("step failed")

    study.set_step_function(exploding_step)
    matrix = study.run()

    assert float(region.state[0]) == 1.0
    assert matrix.step_failures >= 1


def test_restore_is_verified():
    region = _Region()
    component = region.component("a")
    component.save_and_lesion()

    assert float(region.state[0]) == 0.0
    assert component.restore() is True
    assert float(region.state[0]) == 1.0
    assert component.is_lesioned is False


def test_a_failed_restore_is_reported():
    region = _Region()
    component = LesionableComponent(
        name="stubborn",
        get_state=lambda: region.state,
        set_state=lambda s: None,  # refuses to restore
        zero_fn=lambda: region.state.fill(0.0),
    )
    component.save_and_lesion()

    assert component.restore() is False


# --- d857b24f: baselines are contemporaneous ------------------------------


def _drift_impacts(tmp_path, *, interleaved: bool):
    """Relative impact of a metric that drifts with elapsed time only."""
    drift = {"t": 0.0}
    regions = [_Region(), _Region(), _Region()]
    study = _study(tmp_path, interleaved_baseline=interleaved)
    for index, region in enumerate(regions):
        study.register_component(region.component(f"c{index}"))
    study.set_step_function(lambda: drift.__setitem__("t", drift["t"] + 1.0))
    study.register_probe("drift", lambda: drift["t"])

    matrix = study.run()
    return [result.relative_impact["drift"] for result in matrix.results], matrix


def test_a_stale_baseline_credits_elapsed_drift_to_later_components(tmp_path):
    impacts, matrix = _drift_impacts(tmp_path, interleaved=False)

    assert matrix.baseline_policy == "single"
    # The pure-drift metric grows against a fixed baseline, so the LAST
    # component looks far more "critical" than the first — the confound
    # CP126 d857b24f describes.
    assert impacts[-1] > impacts[0]


def test_interleaved_baselines_remove_the_drift_confound(tmp_path):
    impacts, matrix = _drift_impacts(tmp_path, interleaved=True)

    assert matrix.baseline_policy == "interleaved"
    # Measured next to its own baseline, a later component is not credited
    # with the elapsed dynamics of everything before it.
    assert impacts[-1] <= impacts[0]


def test_a_single_baseline_can_be_requested(tmp_path):
    region = _Region()
    study = _study(tmp_path, interleaved_baseline=False)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    assert study.run().baseline_policy == "single"


# --- 43446d39: every component is restored --------------------------------


def test_collateral_damage_to_another_component_is_restored(tmp_path):
    victim = _Region(5.0)
    target = _Region(1.0)
    study = _study(tmp_path)
    study.register_component(target.component("target"))
    study.register_component(victim.component("victim"))
    study.register_probe("p", lambda: float(target.state[0] + victim.state[0]))
    # The step function scribbles on a component that was never lesioned.
    study.set_step_function(lambda: victim.state.fill(99.0))

    study.run()

    assert float(victim.state[0]) == 5.0
    assert float(target.state[0]) == 1.0


def test_snapshot_does_not_lesion():
    region = _Region(3.0)
    component = region.component("a")

    component.snapshot()

    assert float(region.state[0]) == 3.0
    assert component.is_lesioned is True


# --- ec3611ea: failed probes are unavailable, not zero --------------------


def test_a_failing_probe_is_unavailable_not_zero_impact(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("broken", lambda: (_ for _ in ()).throw(RuntimeError("nope")))

    matrix = study.run()

    result = matrix.results[0]
    assert result.relative_impact["broken"] is None
    assert "broken" in result.unavailable_metrics
    assert math.isnan(result.criticality_score)
    assert matrix.probe_failures["broken"] >= 1


def test_an_unmeasured_component_is_not_called_redundant(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("broken", lambda: (_ for _ in ()).throw(RuntimeError("nope")))

    matrix = study.run()

    assert matrix.get_redundant_components() == []
    assert matrix.get_critical_components() == []
    assert matrix.get_unmeasured_components() == ["a"]


def test_a_non_finite_probe_value_is_unavailable(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("nan", lambda: float("nan"))

    matrix = study.run()

    assert matrix.results[0].relative_impact["nan"] is None


def test_a_working_probe_still_measures_impact(tmp_path):
    region = _Region(4.0)
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("value", lambda: float(region.state[0]))

    matrix = study.run()

    assert matrix.results[0].relative_impact["value"] == pytest.approx(1.0)
    assert matrix.get_critical_components() == ["a"]


def test_the_serialized_matrix_is_valid_json(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("broken", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    payload = study.run().to_dict()
    text = json.dumps(payload)

    assert "NaN" not in text
    assert payload["matrix"][0][0] is None
    assert payload["results"][0]["criticality_score"] is None


# --- 320b8881: live mutation is guarded -----------------------------------


def test_a_non_quiescent_runtime_refuses_the_study(tmp_path):
    region = _Region()
    study = _study(tmp_path, quiescence_check=lambda: False)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    with pytest.raises(RuntimeError, match="quiescent"):
        study.run()

    assert float(region.state[0]) == 1.0


def test_a_quiescent_runtime_permits_the_study(tmp_path):
    region = _Region()
    study = _study(tmp_path, quiescence_check=lambda: True)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    assert study.run().components == ["a"]


def test_a_per_run_quiescence_check_overrides_the_default(tmp_path):
    region = _Region()
    study = _study(tmp_path, quiescence_check=lambda: True)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    with pytest.raises(RuntimeError):
        study.run(quiescence_check=lambda: False)


def test_the_study_takes_an_interprocess_lock(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    study.run()

    assert (tmp_path / "lesion_study.lock").exists()


# --- 1c536bf5: history is durable and longitudinal ------------------------


def test_each_study_appends_to_a_durable_record(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))

    study.run()
    study.run()

    lines = (tmp_path / "lesion_study_history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["components"] == ["a"]


def test_history_is_reloaded_by_a_new_study_object(tmp_path):
    region = _Region()
    first = _study(tmp_path)
    first.register_component(region.component("a"))
    first.register_probe("p", lambda: float(region.state[0]))
    first.run()

    second = _study(tmp_path)

    assert len(second.history()) == 1
    assert second.get_status()["n_studies"] == 1


def test_the_latest_snapshot_is_still_written(tmp_path):
    region = _Region()
    study = _study(tmp_path)
    study.register_component(region.component("a"))
    study.register_probe("p", lambda: float(region.state[0]))
    study.run()

    payload = json.loads((tmp_path / "latest_lesion_matrix.json").read_text())

    assert payload["components"] == ["a"]
    assert payload["baseline_policy"] == "interleaved"


def test_corrupt_history_lines_are_skipped(tmp_path):
    (tmp_path / "lesion_study_history.jsonl").write_text('{"components": ["a"]}\nnot json\n')

    study = _study(tmp_path)

    assert len(study.history()) == 1
