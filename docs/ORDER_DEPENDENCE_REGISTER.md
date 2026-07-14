# Order-dependence register — triage and roots

Snapshot of the chunked-runner register (`tools/run_test_chunks.py` isolated-
retry pass) as of 2026-07-14, with root causes. Every entry here **fails
in-chunk, passes alone** — the defect is shared process state, not the
test's assertion.

**Count roots, not victims.** Two consecutive full runs produced different
victim sets from the same disease pool (run 1: gateway_enforcement ×7 +
local_agent_client ×3 + retrieval ×2 + 3 singles; run 2 after those roots
were fixed: llm_routing_tiering ×5, logging_integrity ×3, and others —
previously green tests). The register samples the pool per composition;
progress is measured by roots eliminated and by reproducible victims
staying gone across runs, not by one run's victim count.

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
| `tests/test_gateway_enforcement.py` ×7 | same root as the personality family: the captured SimpleNamespace persona corrupted a downstream runtime path that ended in the global receipt store being closed mid-chunk (`emit()` → "receipt store is closed") | healed by the personality read-through fix — with it, chunk 3 runs 2115/2115 green and the store is never closed (probe plugin, zero transitions across two prior deterministic failures) |

## Open pool (sampled 2026-07-14 certification run)

Confirmed-fixed families stayed gone; the run surfaced a fresh sample:
`test_llm_routing_tiering` ×5, `test_logging_integrity::TestQueueHandlerOverflow`
×3, `test_live_runtime_surface_regressions::test_file_operation_write_creates_nested_live_runtime_directory`,
`test_memory_facade_runtime::test_memory_ops_core_append_writes_to_block`,
and the recurrent `test_sota_hardeners::test_extract_and_validate_args_resilience`
(the only victim present in both runs — highest-value next target).
Cluster shapes suggest shared roots per file (routing: lane/endpoint state;
logging: handler/alarm throttle globals). These live in subsystems under
active parallel development; fixes need the owning context, not drive-by
patches.

## Receipt-store family (7 victims) — how it was pinned

`tests/test_gateway_enforcement.py` failed in-chunk with
`RuntimeError: receipt store is closed` (`core/runtime/receipts.py` `emit()`)
in two consecutive full-chunk runs, while every partial reproduction passed:
files 0..105 executed alone, and full collection with only gateway selected.
A per-test probe plugin (report the exact test after which the global store
turns up closed) came back with ZERO transitions once the personality
read-through fix was in — same files, same order, 2115/2115 green.

Standing notes for any future receipt-store work:
- `get_receipt_store()` hands back the global even when closed; only
  `reset_receipt_store()` nulls it (heals). `close_receipt_store()` closes
  and deliberately keeps it — terminal shutdown semantics (f1c228e5,
  13b93fba). Preserve that: in-flight runtime shutdown keeps failing loud
  after close; only a fresh accessor call in a new logical run may rebuild.
- Collection leaves the store `None` (no import-time construction).
- Lesson: a "who closed X" symptom can have a root in an unrelated
  singleton — chase the earliest corrupted state, not the loudest error.
