# Open Findings — 2026-06-09 closeout pass

Honest remainder after the live-runtime reliability waves. Each entry
lists isolation status and suspected cause. These are open, not hidden:
the chunked suite (`make test`) reports them until they are closed.

Fixed this pass (for context): 11 tiering/cortex failures, webview
collection killer, local-model setup `text=` kwarg production bug,
governance-strictness leakage between tests, tombstone lint finding,
foreground-guard pollution, enterprise-gate ratchet regression.

## Still failing in isolation (real, un-triaged)

| Test | Symptom | Suspected cause |
|---|---|---|
| unity/test_memory_unity_commits::test_memory_metadata_carries_unity_fields | add_memory returns True but injected `_vector` recorder never called | facade write routing changed (governed gateway/dual-store path) after test was written; needs routing archaeology |
| test_capability_engine_policy_regressions::test_high_cost_tool_blocks_when_self_preservation_check_fails | reason string `...guard_unavailable` vs expected `...n_unavailable` | block-reason rename drift; verify which string is the contract |
| test_cognitive_engine_2026::test_engine_think_no_response | confidence 0.0 vs expected 0.5 | no-response default confidence changed |
| test_core_affect_models::test_narrative_thread_start_seeds... | IndexError: no tracked task recorded | task-tracker fallback path changed |
| test_enterprise_static_contracts (2 tests) | gate parity assertions fail under pytest while `make enterprise-gate` passes | test invokes gate with different args/baseline than the Makefile; align them |
| test_feedback_audit_fixes::test_unitary_response_exact_format_turn_gets_format_priority | format priority not granted | unitary-response priority rules drifted |
| test_kernel_phase_timeouts (2 tests) | 180s/210s actual vs 300s/360s expected | phase budgets deliberately tightened? confirm intent, then pin |
| test_long_run_model::test_build_registry_extracts_runtime_hardening_contracts | extraction returns False | registry contract extraction misses a new module shape |
| test_personality_kernel_runtime_contract::test_..._identity_key_cannot_persist | lockdown asserted flag False | TypeError path enters lockdown but flag/contract mismatch |
| test_rsi_validation_gauntlet::test_rsi_gauntlet_runs_machine_checkable_suite | gauntlet suite result mismatch (28s runtime) | needs full run inspection |
| test_sensory_gate_runtime_hardening::test_search_handles_mismatched_response... | crash-handling contract mismatch | response-shape drift |
| test_skill_task_bridge::test_task_engine_planning_tool_specs... | KeyError 'results' | planning tool-spec schema drift |

## Environment-dependent (should carry capability markers)

| Test | Requirement |
|---|---|
| test_gemini_adapter_recovery (2), test_gemini_connectivity (1) | network + Gemini API surface — mark `network`/`external` so the default suite skips them honestly |

## Pollution victims (pass alone; fixed by conftest guards, verify in next chunk run)

nethack_audit_comprehensive, terminal_grid_live_canary, conscience,
governance_lint, environment_general_integration, forensic_audit,
internal_sandbox_runtime, planning_depth, react_loop_integration,
sandbox_runner_hardening, stem_cell_hardening.
