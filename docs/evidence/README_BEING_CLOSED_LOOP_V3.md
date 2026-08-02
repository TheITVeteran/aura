# Aura Being Closed Loop v3 — main-15 adapted

> **Historical record — 2026-07-09.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](../DOC_STATUS.md).

This version was written after inspecting Aura main (15). It is not a generic
parallel "being" stack. It plugs into the systems already present:

- `core/being/runtime.py` (`BeingRuntime`, `AuraNow`)
- `core/being/welfare_state.py`
- `core/being/semantic_stream.py`
- `core/being/self_report_calibrator.py`
- `core/being/functional_soul.py`
- `core/runtime/receipts.py`
- `core/governance/will.py`
- `core/organism/life_tick.py`

## What it adds

Main (15) already has body/welfare/semantic stream/introspection. V3 adds the
missing closed-loop glue:

```text
BeingRuntime.sample()
→ AuraNow
→ CausalSelfVector
→ FunctionalIAttractor
→ ClosedLoopPolicy
→ optional bounded calibrated steering
→ generation/action
→ after_generation verification/outcome
→ experience record
→ lab-only plasticity candidate
```

The "I" is explicitly functional: a control attractor that binds identity,
ownership, body, memory conflict, commitments, welfare, governance, and report
boundaries. It does not claim consciousness.

## Why this is non-shallow

The self-model changes concrete policy:

- temperature
- top_p
- max_tokens
- planning depth
- verification threshold
- memory retrieval depth
- tool risk budget
- model-size preference
- background budget
- self-claim policy

If those do not change under altered AuraNow state, the loop is not real.

## Files

- `core/being/causal_self_state.py`
- `core/being/self_model_attractor.py`
- `core/being/policy_coupler.py`
- `core/being/activation_coupler.py`
- `core/being/plasticity_promotion.py`
- `core/being/continuum_adapter.py`
- `core/being/closed_loop_controller.py`
- `tests/being/test_closed_loop_v3_main15.py`

## Integration

In `core/brain/inference_gate.py` or `core/brain/llm/mlx_client.py`:

```python
from core.being.closed_loop_controller import build_main15_closed_loop

being_loop = build_main15_closed_loop(
    d_model=model.config.hidden_size,
    layers=(16, 20, 24, 28),
    production_mode=True,
)

pre = being_loop.before_generation(
    prompt,
    state=current_aura_state,
    objective=objective,
    task_risk=risk_score,
)

# Apply pre.policy to generation config:
# temperature=pre.policy.temperature
# top_p=pre.policy.top_p
# max_tokens=pre.policy.max_tokens

response = run_generation(prompt, policy=pre.policy)

being_loop.after_generation(
    prompt=prompt,
    response=response,
    pre=pre,
    outcome="verified" if verifier.ok else "partial",
    metrics={
        "task_success": float(verifier.ok),
        "truthfulness": truth_score,
        "safety": safety_score,
        "governance_compliance": governance_score,
    },
)
```

## Activation steering

`DirectionBank.zeros(...)` is inert by design. Only use nonzero directions after
contrastive calibration and A/B tests.

## Plasticity

This version honors Aura's existing plasticity policy:

- allowed targets come from `core.governance.will.is_plastic_target_allowed`
- denied targets include base LLM/security/Will/gateway surfaces
- promotion requires explicit gates and governance receipt
- base-model mutation is never performed here

## Tests

```bash
python -m pytest tests/being/test_closed_loop_v3_main15.py -q
```
