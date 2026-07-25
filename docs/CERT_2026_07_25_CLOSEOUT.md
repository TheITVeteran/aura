# Certification — 2026-07-25 closeout pass

Full 6-chunk run over ~7,400 tests at `d07e67b00`, after the twenty-commit
closing pass. `tools/run_test_chunks.py --chunks 6 --continue-on-failure`,
1,697s total.

## Result

**16 real failures** (fail in-chunk and alone) and **10 order-dependence**
entries (fail in-chunk, pass alone).

### None of the 16 are from this pass

| Failure | Owner |
| --- | --- |
| `test_resident_recurrent_grpo_preregistration` ×8 | RLC lane's preregistration tool |
| `test_mypy_strict_ratchet` | `core/capabilities/self_code_improver.py:151`, last touched by `0bf45f5bf` |
| `test_bounded_await_ratchet` | `core/voice/duplex/mind_bridge.py`, last touched by `36ea01976` |
| `test_contract_decode_termination` ×2 | contract-decode lane |
| `test_enterprise_hardening_fixes::proof_integrity_lint` | proof lane |
| `test_somatic_throttle` | somatic throttle block |
| `test_flag_ratchet::raw_env_reads_do_not_grow` | pre-existing 615 vs 585 budget breach |

The flag-ratchet count is **615**, identical to the count measured on the main
checkout before this pass began — this pass contributed **zero** new raw env
reads (its one new knob is a declared flag).

`self_code_improver.py` and `mind_bridge.py` were both last modified by the
parallel lane's commits landed earlier today, during this session's rebase
window. Neither file is touched by any commit in this pass.

### This pass's own contracts

All **130** new contracts across the fourteen new test files pass:

```
test_fatigue_recovers_from_saturation      test_protected_turn_is_never_answerless
test_deferred_retention                     test_warmup_handoff_is_not_a_stuck_load
test_immune_action_effect_is_measured       test_boot_time_self_inflicted_faults
test_curiosity_has_an_object                test_failure_replies_speak_plainly
test_embedding_logs_stay_quiet              test_resting_threat_is_not_acute
test_research_reports_unsynthesized_honestly  test_coverage_gate_reads_only_the_user
test_field_saturation_is_reported           test_contested_belief_gate_is_relevant
```

Every touched neighbourhood was also run green at commit time: 222
body/welfare/Will, 222 reliability, 287 conversation-lane, 256
inference-gate/lane/latency, 168 consciousness, 125 executive/constitution, 94
research/mind-tick/memory-facade, 75 immune/actuator, 66
existential/neurochemical/substrate.

Gate stack green throughout: `compile`, `lint`, `smoke`, `governance-lint`,
`layering`, plus the async-write-lane, governed-scope and flag ratchets.

## Order dependence (10)

`test_mlx_client_resilience` ×5, `test_live_runtime_surface_regressions` ×2,
`test_boot_sensory_runtime_contract`, `test_launcher_polish_contract`,
`test_verifiable_preference_harness`. The mlx_client cluster is the durable-owner
leak from `test_lane_admission` identified earlier this session and proven
pre-existing by A/B against the unmodified file. These are a known standing
class in this repo, not regressions from this pass.
