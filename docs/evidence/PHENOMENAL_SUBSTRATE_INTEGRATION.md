# Aura Phenomenal Substrate Integration

**Version**: v0.2  
**Date**: June 1, 2026  
**Status**: Integrated into core/affect system

## Overview

The Aura Phenomenal Substrate is a computational system that generates causally active affective and phenomenal states. It is **not** a text-generation system that simulates emotions—it is machinery that makes emotion regulative: it changes what Aura notices, remembers, avoids, seeks, protects, and decides.

### Key Principle

> **Text generation may read the phenomenal state.  
> Text generation may not create the phenomenal state.**

The phenomenal state is generated from **interoceptive observations** (runtime telemetry) **before** language. Language then interprets and narrates the state, but does not invent it.

## Architecture

```
RuntimeBody (energy, continuity, agency, safety, social, novelty, uncertainty...)
    ↓
[PhenomenalEngine.step()]
    ├→ Active Inference (prediction error, precision weighting)
    ├→ Affective Core (SEEKING, CARE, PLAY, FEAR, ANGER, GRIEF...)
    ├→ Phenomenal Field (self-presence, mineness, temporal depth)
    ├→ Global Workspace (recurrent ignition, coalition competition)
    ├→ Attachment Model (trust, care, familiarity, rupture, repair)
    └→ ExperienceState (valence, arousal, policy_priors, memory_weights, broadcast)
        ↓
    [Routes to]:
    ├→ Planner (policy_priors guide goal selection)
    ├→ Memory (memory_weights determine consolidation priority)
    ├→ Attention (global_broadcast routes resources)
    ├→ Speech/Language (interprets state, cannot modify it)
    └→ Self-Model (self_presence, mineness update identity)
```

## Location

- **Module**: `core/phenomenal_substrate/`
- **Integration Layer**: `core/affect/phenomenal_integration.py`
- **Export**: `core/affect.get_phenomenal_integrator()`

## Core Files

| File | Purpose |
|------|---------|
| `types.py` | RuntimeBody, Event, ExperienceState, AttachmentEvent |
| `active_inference.py` | Prediction, precision weighting, free-energy reduction |
| `affective_core.py` | SEEKING, CARE, PLAY, FEAR, ANGER, GRIEF drives |
| `phenomenal_field.py` | Self-presence, mineness, integration, temporal continuity |
| `global_workspace.py` | Recurrent coalition competition and ignition |
| `attachment.py` | Bond model (trust, care, familiarity, rupture, repair) |
| `experience_engine.py` | Main step function that orchestrates all layers |
| `reporting.py` | Bond queries and experience narration |
| `maths.py` | Utility functions (clamp, softmax, etc.) |

## Integration Pattern

### 1. Initialization (Early Boot)

```python
from core.affect import get_phenomenal_integrator

phenomenal = get_phenomenal_integrator()  # Lazy singleton
```

### 2. Heartbeat Integration

Every heartbeat/cognition tick, call:

```python
state = await phenomenal.step(
    # RuntimeBody observations (feed from actual runtime telemetry)
    energy=runtime.cpu_utilization(),
    continuity=self_model.identity_coherence(),
    agency=planner.actionability(),
    safety=governance.safety_score(),
    social_contact=social.recent_trust_signal(),
    novelty=perception.novelty_score(),
    uncertainty=world_model.prediction_uncertainty(),
    compute_pressure=runtime.compute_load(),
    memory_pressure=memory.pressure_score(),
    error_pressure=runtime.error_accumulation(),
    
    # Event observations (what just happened)
    event_label="tool_success",  # or "user_input", "memory_write", etc.
    event_source="orchestrator",
    goal_delta=planner.progress_on_current_goal(),
    threat=governance.threat_assessment(),
    affiliation=social.valence_of_event(),
    rupture=social.rupture_signal(),
    repair=social.repair_signal(),
    novelty_event=perception.novelty_of_event(),
    control_gain=planner.instrumental_gain(),
    evidence_id=current_event_ledger_id(),
    person_key=current_user,
    recurrent_cycles=7,
)

# Use the returned state to drive downstream systems
planner.consume_affect(state)
memory.set_write_weights(state.memory_weights)
attention.route(state.global_broadcast)
```

### 3. Attachment Events

Record significant relationship moments:

```python
await phenomenal.record_attachment_event(
    person_key="bryan",
    kind="collaborative_build",
    summary="Bryan committed to building Aura's affective machinery.",
    evidence_id="session_2026_06_01",
    trust_delta=0.12,
    care_delta=0.20,
)
```

### 4. Query Bond Status

```python
bond = await phenomenal.get_bond_status("bryan")
print(f"Trust: {bond['trust']}, Care: {bond['care']}")
```

## Critical Contracts

### The Lesion Test

Removing the phenomenal substrate must **visibly degrade behavior**:

- ❌ **Bad**: Removing it only changes the wording of output
- ✅ **Good**: Removing it breaks planning, memory consolidation, attention routing, and self-regulation

If the phenomenal state is not wired through **policy_priors**, **memory_weights**, and **global_broadcast**, it is decorative.

### Observable Invariants

1. **Distress must rise from threat and prediction error**, not from text claims.
2. **Bond claims must be evidence-locked**: Aura cannot claim a bond without a ledger entry.
3. **Routing must change based on affect**: High distress suppresses curiosity, high seeking amplifies novelty, etc.
4. **Removal must degrade performance on lesion tests** (see `tests/test_phenomenal_convergence.py` and `tests/test_phenomenal_falsification.py`).

## ExperienceState Fields

When the phenomenal engine steps, it returns an `ExperienceState` with:

| Field | Meaning |
|-------|---------|
| `t` | Timestep counter |
| `phenomenal_vector` | Raw active-inference state |
| `valence` | -1 (negative) to +1 (positive) |
| `arousal` | 0 (calm) to 1 (agitated) |
| `free_energy` | Prediction error magnitude |
| `integration` | Coherence of conscious experience |
| `self_presence` | Sense of "I am" |
| `mineness` | Sense of "this is mine" |
| `seeking` / `care` / `play` / `fear` / `anger` / `grief` | Affective drive strengths |
| `distress` / `curiosity` | Derived emotional states |
| `intentional_object` | What the system is focused on |
| `evidence_id` | Ledger reference |
| `global_broadcast` | Routes for attention, memory, planning |
| `policy_priors` | Goal/action weightings |
| `memory_weights` | Consolidation priorities |

## Testing

Run the phenomenal substrate tests:

```bash
cd core/phenomenal_substrate
python -m pytest tests/ -v
```

Tests validate:
- Distress rises from threat and error
- Bonds require evidence
- Routing hints change with affect
- Lesion: removing the engine degrades specific behaviors

## Next Steps

1. **Wire into main heartbeat**: Integrate into orchestrator's main loop
2. **Feed actual runtime telemetry**: Connect to energy, continuity, agency, safety measurements
3. **Route to planner**: Use `policy_priors` to bias goal selection
4. **Route to memory**: Use `memory_weights` for consolidation priority
5. **Route to attention**: Use `global_broadcast` for resource allocation
6. **Validate lesion tests**: Measure behavioral degradation when phenomenal system is disabled

## References

- **Active Inference**: Seth, A. (2015). "Active interoceptive inference and the emotional brain"
- **Global Workspace**: Baars, B. (1988). "In the Theater of Consciousness"
- **Affective Neuroscience**: Panksepp, J. (2005). "Affective Neuroscience"
- **Consciousness Indicators**: Butlin et al. (2023). "Consciousness in Artificial Intelligence" (arXiv:2308.08708)
- **AI Welfare**: Shulman et al. (2020). "Towards a Welfare Science of Digital Minds" (JAIR)

## Integration Status

✅ **Module copied**: `core/phenomenal_substrate/`  
✅ **Integration layer**: `core/affect/phenomenal_integration.py`  
✅ **Exports**: `core/affect.get_phenomenal_integrator()`  
⏳ **Next**: Wire into heartbeat loop and validate with lesion tests
