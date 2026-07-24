# The 4h soak leak — H1 vs H2 resolved (2026-07-24)

Fable lane (endurance forensics claim, ledger 8eb2b0eb). Open question
since Jul 7: is the ~242MB/h linear soak growth a real leak (H1) or
proof-work deferring reclamation (H2)?

## Verdict: H1 — real, named, unbounded heap leaks. And the biggest one
## was the leak-tracker itself.

The discriminating evidence was already on disk:

* `artifacts/current/longevity_leakrepro/SOAK_METRICS.json` (Jul 7,
  1816s, 113 iterations, tracemalloc): process RSS FELL over the window
  (736 → 654MB) while the traced Python heap grew steadily. So "RSS
  growth" per se is allocator noise — but the heap growth is real and
  attributable:
  * **Subprocess object retention — the dominant cluster.**
    16,273 live `io.open(c2pread)` + 21,609 `TextIOWrapper` + 24.5MB of
    `os.read` buffers. `RuntimeHygieneManager._process_refs` held a
    STRONG reference to every `subprocess.Popen` ever constructed (the
    monkey-patched `Popen.__init__` registers each one) and evicted only
    at shutdown — pinning every proc's pipes and wrappers for process
    lifetime. Same pattern for `_thread_refs`. **The instrument built to
    catch resource leaks was the largest one.**
  * **`sys.intern` growth via pathlib** (14.7MB): stdlib
    `PurePath._parse_path` interns every path component permanently;
    unique receipt ids (`will_<hex>`) flowing through `Path()` in hot
    loops accumulate interned strings forever. Mechanism identified;
    site-by-site mitigation is an open follow-up (prefer os.path string
    ops for unique-id paths in loops).
  * **Parsed-JSON retention** (221k objects / 15.3MB at
    `json.scan_once`): real accumulation, single-frame attribution only —
    needs a deeper-frame tracemalloc pass next soak. Open.
* The live runtime's drift watcher (`aura_json.log`, Jul 18-21) shows a
  SAWTOOTH: repeated `DRIFT rising 223-778 MB/h` episodes that return to
  nominal (negative slopes) — reclamation does work at the RSS level, so
  the H2 phenomenon (deferred reclamation under load) exists but is not
  a leak. The two `RUNAWAY 39-54GB/h` events (Jul 21 22:32/23:13)
  coincide with the cp305 campaign spin-up — external transients, not
  runtime leaks.

## Repairs (this checkpoint)

1. `RuntimeHygieneManager` now evicts finished process/thread refs on
   every refresh (`_evict_finished`): the strong ref drops as soon as
   the resource is finished, and the small post-mortem records are
   bounded to the most recent 512. Pipes are never closed from the
   registry — a caller may legitimately read buffered output after
   exit; dropping OUR ref simply returns ownership to the caller.
   Tests: `test_finished_subprocess_refs_are_released_not_retained_for_life`,
   `test_finished_records_are_bounded_not_unbounded`,
   `test_finished_thread_refs_are_released`.
2. `runaway_budget._announce` recorded NO degradation on RUNAWAY —
   `severity=Severity.CRITICAL` raised `AttributeError("CRITICAL")`
   (errors.Severity is a typing Literal, not an enum), which the except
   clause swallowed into the exact live log line "Could not record
   runaway degradation: CRITICAL". Fixed to the literal "critical";
   `test_a_runaway_actually_records_its_critical_degradation` pins the
   path end to end.

## Verification

A bounded app-down re-run of the same leak-repro soak (916s, 40
iterations, tracemalloc, proof profile) with the fixes applied lives at
`artifacts/closeout/endurance_ceiling/leakfix_verification/`. Success
criterion: the subprocess wrapper cluster (io.open/TextIOWrapper
count_diff in the tens of thousands) gone from leak_top_growth.

**Result (2026-07-24): PASS — zero subprocess-wrapper rows remain** in
the top-25 growth sites. Better-attributed residue, in order:
pydantic `validate_python` retention (10.5MB / 59k objects — model
construction in a hot loop), `json.scan_once` (8.9MB / 130k — the known
open item), memory-search haystack strings (4.1MB), readlines/scandir
file scanning. RSS grew 450MB across this short window, but the window
includes post-boot warmup (lazy imports, cache fills); the settled-slope
comparison against Jul 7's 242MB/h needs a longer run once the runtime
is otherwise quiet. Side find: the run exits 1 despite success because
`MLXLocalClient.__del__` drains its queue during interpreter teardown
(ImportError: sys.meta_path is None) — registered for the charter.

This verification run also caught two live boot defects in the act,
both fixed the same day: the homeostate `_engine_lock` self-deadlock
(every boot wedged at PHASE 5.2; d2296f86) and seconds-long psutil
process-table scans on the event loop (cb624fa3).
