"""#48 regression: substrate state is a DIRECT (non-text), ablatable causal
constraint on cognition — reducing text-mediation.

Pins that the affect substrate varies the generation controls across affect
contexts when active and flattens them when lesioned, and that the live consumed
circumplex pathway responds to affect.
"""

import tools.agi.run_text_mediation_proof as tool


def test_substrate_state_is_direct_causal_not_text():
    bundle = tool.run_proof()
    assert bundle["passed"] is True
    pb = bundle["pathway_b_affect_engine"]
    assert pb["active_temperature_delta_spread"] > 0.02, "active substrate must vary the controls"
    assert pb["lesioned_temperature_delta_spread"] < 1e-6, "lesioned substrate must flatten the controls"
    assert bundle["pathway_a_circumplex_consumed"]["responds_to_affect"] is True


def test_generation_controls_are_ablatable_and_non_text():
    from core.being.affective_valence import AffectiveValenceEngine, generation_controls
    from core.being.aura_now import BodyState, PredictionState, WorldState

    active = AffectiveValenceEngine()
    lesioned = AffectiveValenceEngine()
    lesioned.lesion()

    distress = dict(
        body=BodyState(cpu_pressure=0.9, memory_pressure=0.9),
        prediction=PredictionState(controllability=0.05, free_energy=0.9),
        world=WorldState(uncertainty=0.9), base_valence=-0.4,
    )
    reward = dict(
        body=BodyState(),
        prediction=PredictionState(controllability=0.95, expected_information_gain=0.9),
        world=WorldState(task_active=True, relationship_count=3), base_valence=0.5,
    )

    # Active substrate: different contexts → different generation controls (load-bearing).
    assert generation_controls(active.compute(**distress)) != generation_controls(active.compute(**reward))
    # Lesioned substrate: controls are flat / context-independent (affect ablated).
    l_distress = generation_controls(lesioned.compute(**distress))
    l_reward = generation_controls(lesioned.compute(**reward))
    assert l_distress == l_reward
    assert generation_controls(active.compute(**distress)) != l_distress
