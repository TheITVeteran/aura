"""#38 regression: intrinsic valence is load-bearing (measurably changes behavior).

Pins that the affect substrate's valence causally changes emotional-salience
memory weighting (active vs lesioned, CI-separated), and — the honesty guard —
that ABLATING the substrate removes the effect entirely (so the verdict is
genuinely falsifiable, not a foregone conclusion).
"""

import tools.agi.run_valence_load_bearing_proof as tool


def test_valence_is_load_bearing_with_ci_separation():
    bundle = tool.run_proof(ci_iterations=2000)
    assert bundle["passed"] is True
    assert bundle["valence_is_load_bearing"] is True
    active = bundle["active_salience_advantage"]
    lesioned = bundle["lesioned_salience_advantage"]
    assert active["ci_lower"] > lesioned["ci_upper"], "active advantage must CI-clear the lesioned advantage"
    assert active["mean"] > 0.05


def test_ablation_removes_the_effect():
    # If lesioning the affect substrate did NOT flatten the salience advantage,
    # the "load-bearing" claim would be false — this guard makes that falsifiable.
    lesioned = tool._advantages(lesioned=True)
    assert lesioned, "expected per-pair advantages"
    assert all(abs(x) < 1e-9 for x in lesioned), "lesioned substrate must remove the salience advantage"
    active = tool._advantages(lesioned=False)
    assert sum(active) / len(active) > 0.05, "active substrate must genuinely separate"
