# Order-dependence register — triage and roots

Snapshot of the chunked-runner register (`tools/run_test_chunks.py` isolated-
retry pass) as of 2026-07-14, with root causes. The register only shrinks:
fix a root, delete its rows. Every entry here **fails in-chunk, passes
alone** — the defect is shared process state, not the test's assertion.

The recurring disease has one shape: **a process-global singleton captures
state some earlier test mutated, and nothing re-heals on access.** Fixes that
worked: re-register/rebuild on access (autonomy_latitude, personality_engine),
expire *all* leaked tokens instead of one (desktop posture), stub ambient
governance out of unit tests whose subject is something else (FTS ranking).

## Fixed

| Victim(s) | Root | Fix |
|---|---|---|
| `test_autonomy_latitude::test_singleton_and_registration` | accessor registered only at first construction; container reset desynced it forever | re-heal registration on every access (8d7656bb) |
| `test_desktop_agency::test_automatic_posture_reversion` | expired only the first presence token in the shared authority gateway | expire all active tokens (8d7656bb) |
| `test_live_runtime_surface_regressions` ×4 | committed importer, uncommitted module (`core.runtime.resource_observation`) | module committed (46b997c4) |
| `test_local_agent_client_recovery` ×3 | `get_personality_engine()` promoted a container-registered test stub (SimpleNamespace) into the permanent module global | container is read-through, never cached; module singleton only caches what it constructs |
| `test_memory_retrieval_backbone::TestKnowledgeSearchIsRanked` ×2 | `add_knowledge` consults the live constitutional core; an earlier test left AuraNow present-state policy in `aura_now_defer` ("stabilization first"), so unit-test writes were denied and searches scanned an empty table | ranking tests stub `_approve_memory_write` (governance has its own tests) |
| `tests/test_gateway_enforcement.py` ×7 | see current investigation below | — |

## Open (documented, deliberately not half-fixed)

These live in subsystems under active parallel development (shutdown
lifecycle, tool orchestration); their 1-victim fixes need the owning
context, not a drive-by patch.

- `test_shutdown_lifecycle_hardening::test_shutdown_verdict_blocks_on_unfinished_non_owner_task`
  — expected `verdict.clean is False` after planting an unfinished non-owner
  task; in-chunk the verdict came back clean, i.e. ambient task-tracker state
  (an earlier test's tracker config or reaping) swallowed the planted task.
- `test_sota_hardeners::test_extract_and_validate_args_resilience`
  — in-chunk failure without a clean assertion signature in captured output;
  needs a run with `-vv` inside the chunk composition to characterize.
- `test_tool_orchestrator_runtime_contract::test_python_sandbox_launch_failure_fails_closed`
  — the planted `DependencyUnavailable` escaped instead of converting to an
  explicit tool failure; some earlier test's orchestrator state changes the
  failure-conversion path.

## Receipt-store family (7 victims) — investigation state

`tests/test_gateway_enforcement.py` fails in-chunk with
`RuntimeError: receipt store is closed` (`core/runtime/receipts.py` `emit()`).
Facts established:
- `get_receipt_store()` hands back the closed global; only
  `reset_receipt_store()` nulls it (heals), `close_receipt_store()` closes
  and deliberately keeps it (terminal shutdown semantics — f1c228e5,
  13b93fba).
- No core or test caller of `close_receipt_store()` outside receipts.py.
- Collection leaves the store `None` (no import-time construction).
- Running chunk-3 files 0..105 (through gateway_enforcement): passes.
- Full collection + `-k gateway_enforcement`: passes.
- ⇒ the poisoner requires executed tests from files *after* index 105 or an
  interaction not yet reproduced; a per-test probe plugin
  (`store_probe.py`, prints the exact poisoning test) is the instrument.

Any fix must preserve: in-flight runtime shutdown keeps failing loud after
close (terminal, monotonic); only a *fresh accessor call in a new logical
run* may rebuild.
