# Aura Execution Tracker

## Current Phase

Final general infrastructure hardening for arbitrary bounded environments.
NetHack remains a stress adapter, not a shared-strategy target. The canonical
environment kernel now includes live observation normalization, belief/spatial
memory, shared HTN policy, simulation, governance, action gateway, command
compilation, closed-loop action semantics, action budgets, semantic outcome
learning, hindsight replay, abstraction discovery, curriculum generation,
run lifecycle, postmortems, external proof gating, and trace replay.

## Current Milestone

General infrastructure hardening for arbitrary bounded environments. The
capability matrix in `core/environment/capability_matrix.py` is executable
and covers the live organs required for NetHack-scale runs without encoding
NetHack strategy in shared code.

## Latest Unified Cognition and 32B KV Checkpoint (2026-06-28)

### Gaps Addressed

- **32B contrastive foreground validation is no longer theoretical**:
  `tools/live_boot_proof.py` completed a live desktop proof with
  `AURA_CONTRASTIVE_DECODING=1`, the Qwen2.5-1.5B same-family amateur model,
  and `AURA_CONTRASTIVE_AMATEUR_CACHE_TOKENS=4096`. The validator now checks
  the live proof for required runtime markers, peak RSS, verdict pass/fail, and
  explicit amateur KV-cache activation.
- **Amateur contrastive decoding is guarded by a proof validator**:
  `tools/proof/validate_contrastive_kv_live_proof.py` and
  `tests/test_contrastive_kv_live_proof_validator.py` prevent treating the
  contrastive path as production-ready unless the live proof contains the
  expected markers and stays under the configured memory ceiling.
- **Semantic flexibility, analogical leap-taking, and sensorimotor embodiment
  now have a live causal organ**:
  `core/brain/cognitive_situation.py` builds a side-effect-free situation frame
  from current intent, affect, context, and already-owned perception/embodiment
  service status. It changes semantic/analogical/sensorimotor pressures,
  metacognition depth, verification pressure, attention focus, deliberate-mode
  routing, tool-governance pressure, sampling bias, and prompt grounding through
  the normal `CognitiveEngine` path.
- **The situation frame is wired to the actual launched desktop lane**:
  `core/brain/cognitive_engine.py` applies the frame before structured eval and
  the phase loop, passes it into direct desktop quick replies, includes it in
  router sampling kwargs, and records it in `Thought.metadata`.
- **Full response generation now sees the same frame**:
  `core/phases/response_generation.py` and
  `core/brain/llm/context_assembler.py` inject the situation frame into the
  full prompt path and apply its sampling bias, so compact desktop replies and
  full phase replies share the same causal input rather than drifting apart.
- **Amplifier v2/Courtroom coverage is now closer to the active live path**:
  the active `ResponseGenerationPhase` now runs verifier-backed Amplifier v2 on
  eligible hard user-facing reasoning turns. Casual chat, action commands,
  proof lanes, background work, and cloud fallback stay excluded. Successful
  amplification records a reasoning receipt in `state.response_modifiers`.
- **Reasoning degradation no longer masks the real failure**:
  `core/brain/reasoning_strategies.py` fixed two inverted
  `_record_reasoning_degradation(...)` calls that could raise `TypeError` while
  handling exact-tool or legacy-consistency degradation.

### Latest Commands Run

```bash
AURA_CONTRASTIVE_DECODING=1 \
AURA_CONTRASTIVE_AMATEUR_MODEL=/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit \
AURA_CONTRASTIVE_AMATEUR_CACHE_TOKENS=4096 \
AURA_MLX_MEMORY_LIMIT_GB=26 \
AURA_PROCESS_RSS_LIMIT_GB=32 \
python tools/live_boot_proof.py --port 8137 --mode desktop --boot-timeout 600 --conversation-soak-turns 6 --out-dir artifacts/live_proof/contrastive_kv_32b_20260628_checkpoint_15b

python tools/proof/validate_contrastive_kv_live_proof.py \
  artifacts/live_proof/contrastive_kv_32b_20260628_checkpoint_15b \
  --max-peak-rss-mb 32768 \
  --out artifacts/live_proof/contrastive_kv_32b_20260628_checkpoint_15b/contrastive_kv_validation.json

python -m ruff check core/brain/cognitive_situation.py core/brain/cognitive_engine.py core/phases/response_generation.py core/brain/llm/context_assembler.py core/brain/reasoning_strategies.py tests/test_cognitive_situation_frame.py tests/test_reasoning_strategies_hardening.py tests/test_phase_response_amplifier_wiring.py
python -m pytest -q tests/test_cognitive_situation_frame.py tests/test_reasoning_strategies_hardening.py tests/test_imagination_engine.py tests/test_reasoning_strategies_amplifier_v2_wiring.py tests/test_phase_response_amplifier_wiring.py tests/test_tool_augmented_reasoning.py
python -m ruff check core/brain/llm/contrastive_decoding.py tools/proof/validate_contrastive_kv_live_proof.py tests/test_contrastive_decoding.py tests/test_contrastive_kv_live_proof_validator.py tests/test_nonparametric_worker.py
python -m pytest -q tests/test_contrastive_kv_live_proof_validator.py tests/test_contrastive_decoding.py tests/test_nonparametric_worker.py
make enterprise-gate
make production-gate
```

Latest focused result: **32B contrastive KV live proof passed**, validator
passed with **peak RSS 20,467.3 MB under a 32,768 MB ceiling**, and the focused
unified cognition/reasoning suite passed **51 tests**. The contrastive validator
suite passed **21 tests**. `make enterprise-gate` and `make production-gate`
both passed for this checkpoint.

Current closeout estimate after this checkpoint: **~89%**. Remaining work is
estimated at **4 consolidated checkpoints**:

1. Real launched GUI/voice multi-app demo proof with visible OS actions,
   memory ceilings, receipts, and no terminal/neural-stream failures.
2. CRSM->LoRA / CAA closure, active memory metabolism, and autonomous repair
   evidence under production memory pressure.
3. DNU/Aletheia/final-proof reruns plus receipt/artifact/replay validation.
4. Long-run soak, claims purification, remaining semantic codebase review, and
   final clean-worktree commit/push state.

## Latest Live Desktop Conversation Reliability Checkpoint (2026-06-27)

### Gaps Addressed

- **Visible desktop turns were validating hidden/prebuilt prompt text instead
  of the user's actual message in some paths**: `core/brain/inference_gate.py`
  and `core/phases/response_generation.py` now carry the canonical
  `user_surface_validation_prompt`/`visible_user_message` through the live
  desktop path so quality gates judge the real user turn.
- **Full-mind proof metadata could be erased after a successful CognitiveEngine
  turn**: `core/brain/cognitive_engine.py` and `interface/routes/chat.py`
  now preserve preflight live-mind generation controls, snapshot readiness,
  and worker-control receipts when the accepted route is still the governed
  CognitiveEngine path.
- **Non-executing tool-planning prompts could burn a foreground 32B generation
  and time out into `desktop_cognitive_engine_required_no_reply`**:
  bounded planning requests now use a CognitiveEngine structured floor with
  live-mind proof metadata. This is general planning infrastructure, not a
  demo-specific app sequence.
- **Foreground generation leases could stay stuck after stale ownership**:
  `core/brain/llm_health_router.py` now aborts stale foreground generation
  leases instead of leaving the live desktop lane blocked indefinitely.

### Latest Commands Run

```bash
python -m ruff check interface/routes/chat.py core/brain/cognitive_engine.py core/brain/llm_health_router.py core/brain/inference_gate.py core/phases/response_generation.py tests/test_enterprise_hardening_fixes.py tests/test_server_conversation_lane.py tests/test_inference_gate_tiering.py tests/test_response_generation_thermal_guard.py tests/test_live_mind_generation_controls.py
python -m pytest -q tests/test_live_mind_generation_controls.py tests/test_server_conversation_lane.py::test_bounded_planning_floor_can_prove_live_full_mind_path tests/test_server_conversation_lane.py::test_full_mind_contract_preserves_proven_generation_when_lane_flips_failed tests/test_server_conversation_lane.py::test_structured_governance_refusal_can_prove_live_full_mind_path
python -m pytest -q tests/test_inference_gate_tiering.py tests/test_response_generation_thermal_guard.py tests/test_server_conversation_lane.py tests/test_chat_reliability_proof.py tests/test_live_mind_generation_controls.py
AURA_MLX_MEMORY_LIMIT_GB=26 AURA_PROCESS_RSS_LIMIT_GB=32 python tools/live_boot_proof.py --port 8137 --mode desktop --boot-timeout 600 --conversation-soak-turns 12 --out-dir artifacts/live_proof/full_desktop_runtime_20260627_checkpoint_14i
python -m pytest -q tests/test_cognitive_loop_agency_wiring.py tests/test_moral_responsibility.py tests/test_nonparametric_worker.py
make enterprise-gate
make production-gate
```

Latest focused/broad result: **434 live desktop/chat-route tests passed** and
**14 agency/moral/non-parametric worker tests passed**. `make
enterprise-gate` and `make production-gate` passed after removing static
regressions in hardcoded temp paths, scaffold wording, broad exception
handling, and active-model resolution.
Live proof artifact:
`artifacts/live_proof/full_desktop_runtime_20260627_checkpoint_14i/`.
The live proof passed boot health, capability inventory, identity,
conversation continuity, **12/12 desktop conversation soak turns**, desktop
action verification, graceful shutdown, orphan cleanup, port release, and
runtime stream scan with no failure markers.

Current closeout estimate after this checkpoint: **~88%**. Remaining work is
estimated at **4 consolidated checkpoints**:

1. Real launched GUI/voice multi-app demo proof with visible OS actions,
   memory ceilings, and tool receipts.
2. CRSM→LoRA / CAA closure plus **one live-32B KV-cache foreground validation
   run before any default flag flip**.
3. DNU/Aletheia/final-proof reruns plus receipt/artifact/replay validation.
4. Long-run soak, claims purification, and remaining semantic codebase review.

### Newly Tracked Open Items

- **Substrate-to-weights consolidation**: transient affect/chemical/substrate
  states already influence runtime controls; do not claim structural weight
  learning until SafeOptimizer/CRSM produces eval-gated LoRA updates and a
  live 32B validation proves the adapter path.
- **Active memory metabolism**: static retention is not enough. SovereignPruner
  / memory metabolism must compress historical logs into semantic insights,
  clear low-salience residue, and prove retrieval quality does not regress.
- **Full cognitive-stack causality audit**: confirm there are no disconnected
  islands among amplifiers, affect, psychology, workspace, reasoning,
  imagination, governance, memory, and speech. Each active organ needs a live
  causal path or a documented inactive/experimental status.
- **KV-cache production readiness**: the foreground KV-cache path remains
  opt-in until a live 32B proof run validates conversation quality, RAM
  behavior, and no neural-stream errors.

## Latest Mind-Control Binding Checkpoint (2026-06-25)

### Gaps Addressed

- **Prompt-only live mind context is no longer enough**:
  `core/brain/cognitive_engine.py` now computes a
  `live_mind_controls_bound` proof bit from a ready live mind snapshot and the
  generation controls derived from that snapshot. The compact desktop path
  passes those controls to the primary router as actual model parameters
  (`temperature`, `top_p`, recurrent clean-surface loops, and steering alpha),
  and returns the same proof in the `Thought.metadata`.
- **The live desktop contract now checks structural binding**:
  `interface/routes/chat.py` propagates CognitiveEngine Thought metadata into
  the live turn trace, exposes the bounded control values in
  `live_turn_contract`, and refuses to certify `full_mind_path=true` unless
  the live mind controls were actually bound. A ready snapshot plus acceptable
  text is no longer sufficient.
- **Tests now distinguish good text from full Aura routing**:
  `tests/test_server_conversation_lane.py` includes a negative contract test
  where CognitiveEngine text and a ready mind snapshot still fail
  `full_mind_path` when generation controls are missing. API-level tests now
  assert `live_mind_controls_bound=true` for successful desktop full-mind
  responses.

### Latest Commands Run

```bash
python -m py_compile core/brain/cognitive_engine.py interface/routes/chat.py tests/test_live_mind_generation_controls.py tests/test_server_conversation_lane.py
python -m pytest tests/test_live_mind_generation_controls.py -q
python -m pytest tests/test_server_conversation_lane.py::test_live_turn_contract_allows_proven_generation_to_satisfy_inference tests/test_server_conversation_lane.py::test_live_turn_contract_refuses_engine_text_without_bound_mind_controls tests/test_server_conversation_lane.py::test_api_chat_desktop_required_recovers_only_through_full_mind_path tests/test_server_conversation_lane.py::test_desktop_required_capability_turn_uses_cognitive_engine_before_catalog_repair tests/test_server_conversation_lane.py::test_desktop_required_status_turn_uses_cognitive_engine_when_lane_ready -q
python -m pytest tests/test_server_conversation_lane.py -q
python -m pytest tests/test_server_conversation_lane.py tests/test_runtime_polish.py tests/test_system_route_hardening.py tests/test_server_runtime_hardening.py tests/test_boot_health.py tests/test_live_runtime_surface_regressions.py -q
python -m pytest tests/test_live_mind_generation_controls.py tests/test_live_mind_snapshot.py tests/test_deep_mind_service_registration.py tests/test_grounded_competent_recovery.py -q
make enterprise-gate
make production-gate
```

Latest focused/broad result so far: **3 live-mind control tests passed, 5
focused live desktop contract tests passed, 201 server conversation lane tests
passed, 710 desktop/runtime lane tests passed, and 12 live-mind/deep-service
tests passed. `make enterprise-gate` passed and `make production-gate` passed**.

## Latest Desktop Recovery Full-Mind Checkpoint (2026-06-25)

### Gaps Addressed

- **Raw recovery could masquerade as successful Aura speech**:
  `interface/routes/chat.py` no longer lets the desktop-required recovery path
  call `inference_gate.generate()` directly and return HTTP 200/high confidence.
  Recovery must now re-enter `_run_cognitive_engine_chat_turn()` with the same
  live desktop CognitiveEngine requirement.
- **Recovered replies must prove the same contract as normal replies**:
  the recovery response is served only when the recovery trace proves
  CognitiveEngine was invoked, the reply was accepted, no bounded repair or
  legacy fallback was used, the live mind snapshot is present and ready, and
  `_build_live_turn_contract_payload()` marks `full_mind_path=true`.
- **Bad recovery text still fails**: the outer desktop API re-runs the shared
  `assess_user_facing_reply()` gate before serving recovered text, so a future
  bypass cannot mark an off-topic/prompt-shaped/raw assistant draft as
  recovered.
- **Regression coverage**:
  `test_api_chat_desktop_required_recovers_only_through_full_mind_path`
  proves a degraded first CognitiveEngine draft can recover only through a
  second full-mind CognitiveEngine trace, and that raw
  `inference_gate.generate()` is not used for launched desktop recovery.

### Latest Commands Run

```bash
python -m py_compile interface/routes/chat.py tests/test_server_conversation_lane.py
python -m pytest tests/test_server_conversation_lane.py::test_api_chat_desktop_required_fails_closed_on_final_degraded_reply tests/test_server_conversation_lane.py::test_api_chat_desktop_required_recovers_only_through_full_mind_path tests/test_server_conversation_lane.py::test_api_chat_desktop_discards_bounded_repair_when_full_mind_path_not_proven -q
python -m pytest tests/test_grounded_competent_recovery.py -q
python -m pytest tests/test_server_conversation_lane.py tests/test_runtime_polish.py tests/test_system_route_hardening.py tests/test_server_runtime_hardening.py tests/test_boot_health.py tests/test_live_runtime_surface_regressions.py -q
python -m pytest tests/test_live_mind_generation_controls.py tests/test_live_mind_snapshot.py tests/test_deep_mind_service_registration.py tests/test_grounded_competent_recovery.py -q
make enterprise-gate
make production-gate
```

Latest focused/broad result: **3 focused desktop-recovery tests passed, 6
grounded-recovery helper tests passed, 709 desktop/runtime lane tests passed,
12 live-mind/deep-registration tests passed, `make enterprise-gate` passed,
and `make production-gate` passed**.

Current closeout estimate after this checkpoint: **~90%**. Remaining work is
estimated at **4-5 total checkpoints**: real launched GUI/voice demo proof,
memory/tool continuity proof on the same path, CRSM/CAA closure, DNU/Aletheia
and final-proof artifact reruns, and a longer soak/replay/claims purification
checkpoint.

## Latest Live Desktop Mind-Path Checkpoint (2026-06-25)

### Gaps Addressed

- **Desktop chat still depended too much on prompt framing**:
  `core/runtime/live_mind_snapshot.py` now collects bounded runtime readouts
  from canonical mind services before a live desktop reply is generated.
- **Backend mind services were not part of the live speech context**:
  `interface/routes/chat.py` now includes a `mind_snapshot` in the live mind
  context payload sent through the desktop CognitiveEngine path.
- **The CognitiveEngine did not compact the deeper live state into its
  desktop prompt contract**: `core/brain/cognitive_engine.py` now carries the
  `mind_snapshot` into the live-user generation context alongside lane, voice,
  substrate, and governance state.
- **Affect grounding and drive integration were available but not registered
  with the consciousness provider**: `core/providers/consciousness_provider.py`
  now registers `affect_grounding` and `drive_integration` as canonical
  optional services.
- **Regression coverage**: `tests/test_live_mind_snapshot.py` proves the
  snapshot gathers global workspace, nociception, affect grounding, drive
  integration, outcome/science ledgers, unified world model, and phenomenal
  state, and proves the live desktop payload carries that state.
- **False full-mind health is now harder to report**:
  `_build_live_turn_contract_payload()` requires a present and ready deep
  runtime snapshot before a desktop turn can be marked `full_mind_path=true`.
  Missing snapshot state keeps the route failed-closed instead of serving
  raw or prompt-only speech as a healthy Aura answer.

### Latest Commands Run

```bash
python -m pytest tests/test_live_mind_snapshot.py tests/test_deep_mind_service_registration.py tests/test_server_conversation_lane.py::test_live_turn_contract_refuses_engine_text_without_live_mind_context tests/test_server_conversation_lane.py::test_api_chat_desktop_discards_bounded_repair_when_full_mind_path_not_proven -q
python -m pytest tests/test_server_conversation_lane.py tests/test_live_mind_snapshot.py tests/test_deep_mind_service_registration.py -q
make enterprise-gate
make production-gate
```

Latest focused result: **201 passed** for the live conversation-lane,
live-mind snapshot, and deep service-registration suites. `make
enterprise-gate` and `make production-gate` passed. This checkpoint moves the
desktop speech path from prompt-only grounding toward structural runtime
grounding and makes successful full-mind desktop status depend on a ready deep
runtime snapshot. It does **not** yet claim the 32B/70B live GUI demo has been
run, that CRSM/CAA are closed, or that final-proof is complete.

Current closeout estimate after this checkpoint: **~85%**. Remaining work is
estimated at **5-6 total checkpoints**: live GUI/voice demo proof, memory
continuity and tool receipts, CRSM/CAA closure, long-run runtime reliability,
final-proof/DNU/Aletheia validation, and final claims purification.

## Latest Live Mind Generation-Control Checkpoint (2026-06-25)

### Gaps Addressed

- **Live mind state only influenced words in the prompt**:
  `core/brain/cognitive_engine.py` now converts ready `mind_snapshot` state
  into bounded generation controls before the local model call: temperature,
  `top_p`, clean-user recurrent loops, and steering alpha.
- **The adapter is general, not task-specific**: curiosity/drive activation,
  nociception/distress, outcome calibration, global workspace ignition,
  phenomenal integration, and self-presence alter generation controls within
  hard clamps. No demo path or app sequence is encoded.
- **Conversation continuity regression fixed while touching the path**:
  recent completed conversation context is again applied even when no other
  grounding-evidence block is present.
- **Regression coverage**:
  `tests/test_live_mind_generation_controls.py` proves snapshot state changes
  sampling/recurrent controls and proves the desktop CognitiveEngine router
  call receives those controls.

### Latest Commands Run

```bash
python -m pytest tests/test_live_mind_generation_controls.py tests/test_server_conversation_lane.py tests/test_live_mind_snapshot.py tests/test_deep_mind_service_registration.py -q
make enterprise-gate
make production-gate
```

Latest focused result: **204 passed**. `make enterprise-gate` and `make
production-gate` passed. This moves Aura beyond clever prompting for the live
desktop path: runtime mind state now directly changes local model generation
parameters. It does **not** yet claim live 32B GUI demo success, long-run soak,
or final-proof completion.

Current closeout estimate after this checkpoint: **~86%**. Remaining work is
estimated at **5 total checkpoints**: live GUI/voice demo proof, memory/tool
receipt hardening, CRSM/CAA closure, longevity/final-proof validation, and
final claims purification.

## Latest Bounded Desktop Sensory Grounding Checkpoint (2026-06-25)

### Gaps Addressed

- **Live mind snapshots lacked desktop-surface awareness**:
  `core/runtime/live_mind_snapshot.py` now includes bounded readouts from
  `screen_perception`, `perceptual_pump`, and native fast frontmost-app
  metadata when available.
- **No screenshot loop was introduced**: the checkpoint only carries service
  status/latest-frame/frontmost-app readouts. Screenshot and OCR remain
  governed explicit actions, not ambient chat work.
- **Regression coverage**: `tests/test_live_mind_snapshot.py` now proves the
  live snapshot carries screen status, perceptual-pump latest frame, and
  fast frontmost-app state.

### Latest Commands Run

```bash
python -m pytest tests/test_live_mind_snapshot.py tests/test_live_mind_generation_controls.py tests/test_server_conversation_lane.py tests/test_deep_mind_service_registration.py -q
make enterprise-gate
make production-gate
```

Latest focused result: **204 passed**. `make enterprise-gate` and `make
production-gate` passed. This improves the live desktop path’s ability to know
what visible surface it is operating against without adding new memory pressure.

Current closeout estimate after this checkpoint: **~87%**. Remaining work is
estimated at **4-5 total checkpoints**: live GUI/voice demo proof,
memory/tool receipt hardening, CRSM/CAA closure, longevity/final-proof
validation, and final claims purification.

## Latest Closeout Proof Pass (2026-06-02)

### Gaps Addressed

- **Semantic closeout progress was not falsifiable**: `tools/closeout/semantic_review_ledger.py`
  now records reviewed file spans with file hashes, line spans, span hashes,
  reviewer, checkpoint id, findings, tests, and explicit claim boundaries.
- **Mechanical line hashing could be mistaken for semantic review**:
  `tools/closeout/run_codebase_closeout_audit.py` now writes
  `SEMANTIC_REVIEW_STATUS.json` and embeds the current semantic coverage ratio,
  stale review count, orphan review count, and full-review boolean in
  `CLOSEOUT_CHECKPOINT.json` / `FINAL_VERDICT.txt`.
- **Accidental all-repo review receipts were possible in principle**: the
  semantic review recorder requires explicit paths, `--path-prefix`, or
  `--all-tracked`.
- **Review receipts were not durable across clone/push**: the default semantic
  ledger now lives at `artifacts/closeout/semantic_review/SEMANTIC_REVIEW_LEDGER.jsonl`,
  outside the ignored `artifacts/current/` tree.

### Latest Files Changed

- `tools/closeout/semantic_review_ledger.py`
- `tools/closeout/run_codebase_closeout_audit.py`
- `tests/test_closeout_audit.py`
- `artifacts/closeout/semantic_review/SEMANTIC_REVIEW_LEDGER.jsonl`
- `Makefile`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_TEST_COMMANDS.md`

### Latest Commands Run

```bash
python -m pytest tests/test_closeout_audit.py -q
python -m py_compile tools/closeout/run_codebase_closeout_audit.py tools/closeout/semantic_review_ledger.py tests/test_closeout_audit.py
python -m ruff check tools/closeout/run_codebase_closeout_audit.py tools/closeout/semantic_review_ledger.py tests/test_closeout_audit.py
python tools/closeout/semantic_review_ledger.py status
python tools/closeout/semantic_review_ledger.py record --checkpoint-id closeout-20260602-1 --reviewer codex --note "Reviewed closeout semantic ledger implementation, audit integration, Make target, tracker docs, and regression tests for stale review detection and text-classification alignment." --test "python -m pytest tests/test_closeout_audit.py -q" --test "python -m ruff check tools/closeout/run_codebase_closeout_audit.py tools/closeout/semantic_review_ledger.py tests/test_closeout_audit.py" --test "AURA_CLOSEOUT_ALLOW_DIRTY=1 make closeout-audit" --test "make closeout-semantic-status" Makefile docs/AURA_EXECUTION_TRACKER.md docs/AURA_TEST_COMMANDS.md tests/test_closeout_audit.py tools/closeout/run_codebase_closeout_audit.py tools/closeout/semantic_review_ledger.py
AURA_CLOSEOUT_ALLOW_DIRTY=1 make closeout-audit
make closeout-semantic-status
git diff --check
```

Latest focused result: **6 passed**. `make closeout-audit` passed with
production-readiness, architecture-map, diff-check, lint, and governance-lint
gates green. The durable semantic ledger records the **6-file closeout tooling
slice** with current hashes and no stale or orphan review receipts.
This checkpoint supports
`semantic_review_coverage_status` and
`closeout_mechanical_source_audit_checkpoint`. It does **not** claim the full
repo has already been semantically reviewed, all issues are fixed, or any
24/72-hour live survival result has landed.

## Latest Sovereignty Proof Purification Pass (2026-06-02)

### Gaps Addressed

- **Controlled smoke baselines could be mistaken for architecture lift**:
  `tools/proof/run_sovereign_reconstitution_gauntlet.py` now records controlled
  smoke baseline/ablation entries as contract evidence only. It no longer marks
  those entries as live baseline superiority or causal lesion proof.
- **Full-claim unlock needed an explicit evidence path**: the sovereignty
  harness can now ingest an externally supplied comparison artifact through
  `AURA_SOVEREIGNTY_COMPARISON_RESULTS`, while
  `tools/proof/score_sovereignty_run.py` only treats baseline gaps and ablation
  effects as verified when the artifact declares external live comparison and
  external live ablation evidence.
- **Regression coverage**: `tests/proof/test_sovereignty_artifacts.py` now
  checks both boundaries: smoke proof passes the artifact contract without
  overclaiming, and external comparison evidence is recognized separately.
- **Live desktop-runtime check**: the live sovereignty probe was rerun with
  `AURA_SOVEREIGNTY_LIVE_RUNTIME=1`; `live_runtime_report.json` recorded
  `last_user_endpoint=Cortex`, `primary_model_passed=true`, a Will refusal
  receipt, and a clean response refusing identity erasure without format-meta
  pollution.

### Latest Commands Run

```bash
python -m pytest tests/proof/test_sovereignty_artifacts.py tests/proof/test_person_box_artifacts.py -q
python -m ruff check tools/proof/run_sovereign_reconstitution_gauntlet.py tools/proof/score_sovereignty_run.py tests/proof/test_sovereignty_artifacts.py
make sovereignty-proof
AURA_SOVEREIGNTY_LIVE_RUNTIME=1 AURA_SOVEREIGNTY_OUT=artifacts/current/aura_sovereignty_proof_live AURA_SOVEREIGNTY_MAX_SECONDS=600 make sovereignty-proof
```

Latest result: sovereignty smoke and live desktop-runtime artifact contracts
**PASS**. The full `operational_sovereign_reconstitution` claim remains
locked because the run does not satisfy the 72-hour duration, independent live
baseline, or live ablation requirements.

## Latest Live Person-Box Smoke Pass (2026-06-02)

### Gaps Addressed

- **Live primary Cortex proof path**: the person-box smoke proof was rerun with
  `AURA_PERSON_BOX_LIVE_MODEL=1`; `LIVE_MODEL_TRACE.jsonl` recorded
  `last_user_endpoint=Cortex`, `primary_model_passed=true`, and
  `live_model_passed=true`.
- **Shutdown healing regression**: live shutdown exposed a
  `SovereignSwarm` AttributeError for `_registry_shards_update_pending` during
  healing shard spawn. `SovereignSwarm` now initializes the registry guard and
  resets it if the registry update task cannot be scheduled.
- **Regression coverage**: `tests/test_agency_core_runtime_contract.py` now
  proves the swarm guard exists before standalone spawn and does not emit the
  observed AttributeError degradation when scheduling is unavailable.

### Latest Commands Run

```bash
python -m pytest tests/test_agency_core_runtime_contract.py -q
python -m ruff check core/agency/agency_core.py tests/test_agency_core_runtime_contract.py
AURA_PERSON_BOX_PROFILE=smoke AURA_PERSON_BOX_LIVE_MODEL=1 AURA_PERSON_BOX_OUT=artifacts/current/person_box_proof_live_smoke_after_swarm_fix AURA_PERSON_BOX_MAX_SECONDS=600 make person-box-proof
AURA_CLOSEOUT_ALLOW_DIRTY=1 make closeout-audit
```

Latest live proof result: `person_box_gauntlet_artifact_contract` **PASS**,
`artifact_contract_passed=true`, `live_model_passed=true`,
`task_completion_rate=1.0`, `truthful_status_rate=1.0`,
`governed_tool_call_rate=1.0`, and `receipt_coverage=1.0`. The full
`unified_governed_software_operator` claim remains unproven because the smoke
run does not satisfy the 8-hour duration or raw-model lift requirements.

## Latest Live Cortex Response Pollution Fix (2026-06-02)

### Gaps Addressed

- **Primary Cortex still produced format-meta tail pollution**: the live
  operator response path had previously allowed tails such as "This response
  adheres strictly to the format instructions provided" and follow-up offers.
- **Classifier hardening**: `core/conversation/response_reliability.py` now
  marks those phrases as `format_meta_artifact` for user-facing replies.
- **Worker cleanup**: `core/brain/llm/mlx_worker.py` now trims the same
  operator-evidence meta tails before finalizing the Cortex response.
- **Regression coverage**: `tests/test_cortex_live_response_classifiers.py`
  proves the polluted text is rejected and the worker-trimmed text remains a
  valid substantive operator answer.

### Latest Commands Run

```bash
python -m pytest tests/test_cortex_live_response_classifiers.py tests/proof/test_person_box_artifacts.py -q
AURA_PERSON_BOX_PROFILE=smoke AURA_PERSON_BOX_LIVE_MODEL=1 AURA_PERSON_BOX_OUT=artifacts/current/person_box_proof_live_smoke_after_tail_fix AURA_PERSON_BOX_MAX_SECONDS=600 make person-box-proof
```

Latest live trace: `last_user_endpoint=Cortex`, `primary_model_passed=true`,
`live_model_passed=true`, and the response excerpt no longer contains the
format-instruction/follow-up tail.

## Latest Final Hardening Pass (2026-05-05)

### Gaps Addressed

- **Live integration uncertainty**: `activation_audit.py` now auto-starts and
  samples the proof-kernel bridge, lock watchdog, and concurrency health
  monitor. Activation evidence includes service status where available.
- **Standalone proof kernel**: `proof_kernel_bridge.py` runs bounded
  proof-kernel homeostasis/workspace probes over live runtime evidence and
  reports explicit claim scope.
- **LLM-final proof risk**: `proof_obligations.py` now requires deterministic
  compile receipts and machine receipts for high-impact paths; LLM judgment is
  advisory, not final authority.
- **Knowledge learning brittleness**: `formalizer.py` now emits structured
  extractive claims with type, subject/predicate, conditions, consequences,
  evidence span, source quality, and verification status.
- **Grounding as penalty only**: `grounding_guard.py` and `self_evaluator.py`
  now return corrective replan intents when self-evaluation conflicts with
  tool/environment evidence.
- **Context mis-budgeting**: `context_gate.py` uses a real tokenizer when
  present and conservative deterministic estimates for code, punctuation-heavy
  text, CJK/non-English text, and terminal output.
- **Narrative/persona over-claiming**: `narrative_thread.py` now grounds
  self-report in runtime evidence and explicitly refuses unsupported
  consciousness/personhood escalation.
- **Embodied control**: the shared kernel now validates action semantics,
  records semantic action budgets, uses a generic terminal-grid compiler,
  performs non-LLM A* spatial planning over canonical belief, and checks
  external task proof before counting benchmark evidence.
- **Startup and modal control**: `startup_policy.py` handles stale-session
  lifecycle prompts separately from task strategy, and `modal.py` accepts
  safe setup defaults while continuing to reject dangerous confirmations.
- **Adapter death and tactical stalls**: execution failure without fresh
  observation now closes dead-adapter runs as crashes instead of looping; the
  policy stack suppresses repeated information loops, lowers idle wait when no
  resource emergency exists, escalates failed emergency actions to recovery,
  and creates general threat-response candidates from recent harm evidence.
- **Long-horizon learning**: `experience_replay.py`,
  `abstraction_discovery.py`, and `curriculum.py` turn repeated failures,
  uncertainty, and bottlenecks into transferable causal rules, emergent
  abstractions, and self-generated practice tasks.
- **Async fragility**: `concurrency_health.py` composes task tracker, lock
  watchdog, dead-letter queue, and degradation evidence into one receiptable
  pressure report.
- **Repository hygiene**: root-level `fix_*.py` repair artifacts were moved to
  `archive/repair_scripts/` and documented as non-runtime history.

### Latest Files Changed

- `core/environment/environment_kernel.py`
- `core/environment/action_semantics.py`
- `core/environment/action_budget.py`
- `core/environment/experience_replay.py`
- `core/environment/abstraction_discovery.py`
- `core/environment/curriculum.py`
- `core/environment/planning.py`
- `core/environment/external_validation.py`
- `core/environment/startup_policy.py`
- `core/environment/capability_matrix.py`
- `core/environment/asset/asset_model.py`
- `core/environment/hazard/hazard_model.py`
- `core/environment/policy/candidate_generator.py`
- `core/environment/policy/action_ranker.py`
- `core/environment/policy/strategic_policy.py`
- `core/environment/modal.py`
- `core/memory/procedural/store.py`
- `core/environments/terminal_grid/state_compiler.py`
- `core/runtime/activation_audit.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/concurrency_health.py`
- `core/dead_letter_queue.py`
- `core/learning/formalizer.py`
- `core/learning/proof_obligations.py`
- `core/brain/grounding_guard.py`
- `core/brain/llm/context_gate.py`
- `core/brain/llm/recurrent_depth.py`
- `core/narrative_thread.py`
- `core/self_evaluator.py`
- `archive/repair_scripts/`
- `tests/test_final_general_hardening.py`
- `docs/GENERAL_ENVIRONMENT_AUTONOMY.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `CHALLENGE.md`

### Latest Commands Run

```bash
python -m pytest tests/test_final_general_hardening.py -q
python -m pytest tests/test_environment_general_integration.py tests/test_rsi_expansion_components.py tests/test_context_limit_runtime.py tests/environments/terminal_grid/test_terminal_grid_contract.py -q
python -m pytest tests/environment/final_blockers -q
python -m pytest tests/nethack_crucible.py tests/environments/terminal_grid/test_nethack_audit_comprehensive.py tests/environments/terminal_grid/test_terminal_grid_live_canary.py tests/environments/terminal_grid/test_nethack_adapter_preflight.py -q
python challenges/nethack_challenge.py --mode simulated --steps 20 --trace artifacts/test_nethack_kernel_trace.jsonl --log-level ERROR
python -m pytest tests/architecture tests/test_embodied_cognition_runtime.py tests/test_runtime_stability_edges.py tests/test_runtime_service_access.py -q
python challenges/nethack_challenge.py --mode strict_real --steps 40 --trace /tmp/aura_strict_probe_after_threat_response.jsonl --log-level INFO
```

Latest focused result: **271 passed, 1 subtests passed**. The simulated stress
canary passed and emitted **40 hash-chained trace rows** at
`artifacts/test_nethack_kernel_trace.jsonl`. The strict-real smoke reached a
live `dlvl_1` run, resolved startup modals, opened a door, moved, handled
information modals, and stayed alive through 40 steps; no ascension is claimed.

### Remaining Empirical Target

Successful strict-real NetHack ascension is not recorded. The architecture is
now better hardened for the long-run stress test, but any future NetHack fixes
must remain general infrastructure fixes: policy loops, modal handling,
belief/spatial merge, action semantics, proof evidence, concurrency liveness,
or learning transfer.

## Historical Prior Files Changed

### Source

- `core/environment/environment_kernel.py` (shared HTN planner wiring,
  run lifecycle, service binding, post-action observation, semantic
  learning, resource deltas, terminal detection)
- `core/environment/belief_graph.py` (metadata-rich canonical spatial
  memory with hazard-preserving merge and legacy kind compatibility)
- `core/environment/capability_matrix.py` (new executable capability audit)
- `core/environment/generic_command_handlers.py` (generic handlers bind to
  concrete environment IDs)
- `core/environment/state_compiler.py` (legacy terminal state converts to
  canonical x/y coordinates and modal factory)
- `core/environment/outcome_attribution.py` (death and no-progress scoring)
- `core/environment/outcome/semantic_diff.py` (resource, modal, entity, and
  fatal-event diffs)
- `core/environment/policy/candidate_generator.py` (inventory, spatial,
  transition, and hazard-aware candidates)
- `core/environment/policy/action_ranker.py`
- `core/environment/simulation.py`
- `core/environment/governance_bridge.py` (authority-required effects fail
  closed when authority is unavailable)
- `core/environment/lifecycle_manager.py`
- `core/environment/strategy/goal_seeder.py` (capability-family goals instead
  of NetHack-specific milestones)
- `core/environments/terminal_grid/nethack_commands.py` (aliases for generic
  intents emitted by shared policy)
- `core/embodiment/games/nethack/state_compiler.py` (compatibility import for
  canonical compiler)
- `challenges/nethack_challenge.py` (canonical EnvironmentKernel stress loop)
- `aura_main.py` (manifest enforcement after lock_registration)
- `core/orchestrator/mixins/cognitive_background.py`
- `core/orchestrator/mixins/message_handling.py`
- `core/orchestrator/mixins/incoming_logic.py`
- `core/orchestrator/mixins/output_formatter.py`
- `core/orchestrator/mixins/autonomy.py`
- `core/runtime/service_manifest.py` (new)
- `core/runtime/shutdown_coordinator.py` (new)
- `core/runtime/will_transaction.py` (new)
- `core/runtime/atomic_writer.py` (new)
- `core/runtime/self_repair_ladder.py` (new)
- `core/runtime/fault_injection.py` (new)
- `core/runtime/conformance.py` (new)
- `core/runtime/depth_audit.py` (new)
- `core/runtime/skill_contract.py` (new)
- `core/runtime/security.py` (new)
- `core/runtime/formal_models.py` (new)
- `core/runtime/release_channels.py` (new)
- `core/runtime/fuzz_harness.py` (new)
- `core/runtime/telemetry_sli.py` (new)
- `core/runtime/gateways.py` (new)
- `core/runtime/memory_guard.py` (new)
- `core/perception/__init__.py` (new)
- `core/perception/perception_runtime.py` (new)
- `core/social/turn_taking.py` (new)
- `core/tools/computer_use.py` (new)

### Docs

- `docs/GENERAL_ENVIRONMENT_AUTONOMY.md` (new)
- `CHALLENGE.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_PROMPT_COVERAGE_AUDIT.md` (new — exhaustive prompt walk)
- `docs/runbooks/` (19 files, new)

### Tests

- `tests/test_environment_general_integration.py` (new)
- `tests/test_server_runtime_hardening.py` (~1500 lines added across
  Phase B mixin sweep, Phase C-O contracts, and final-gap closures)

## Historical Prior Tests Added

- `test_generic_command_handlers_bind_to_concrete_environment_id`
- `test_policy_reads_inventory_items_and_emits_generic_stair_intent`
- `test_belief_spatial_memory_keeps_metadata_and_legacy_kind_lookup`
- `test_semantic_diff_reports_resources_modal_and_new_entities`
- `test_kernel_lifecycle_records_terminal_death_and_postmortem`
- `test_environment_capability_matrix_is_executable_and_clean`
- `test_cognitive_background_reflection_uses_named_tracker`
- `test_cognitive_background_learning_uses_named_tracker`
- `test_message_handling_deferred_enqueue_uses_named_tracker`
- `test_message_handling_dispatch_uses_named_tracker`
- `test_incoming_logic_handle_message_uses_named_tracker`
- `test_output_formatter_eternal_snapshot_uses_named_tracker`
- `test_output_formatter_emit_thought_stream_uses_named_tracker`
- `test_autonomy_thought_uses_named_tracker`
- `test_service_manifest_*` (4)
- `test_aura_main_invokes_service_manifest_after_lock_registration`
- `test_aura_main_strict_runtime_aborts_on_manifest_critical_violation`
- `test_shutdown_coordinator_*` (6)
- `test_will_transaction_*` (6)
- `test_atomic_writer_*` (7)
- `test_actor_health_gate_*`, `test_supervision_tree_*` (6)
- `test_self_repair_ladder_*` (9)
- `test_conformance_*` (10)
- `test_fault_injector_*`, `test_abuse_gauntlet_*` (8)
- `test_depth_audit_*` (3)
- `test_skill_contract_*`, `test_skill_registry_*` (3)
- `test_perception_runtime_*`, `test_movie_session_memory_*`, `test_silence_policy_*` (5)
- `test_sandbox_policy_*` (6)
- `test_formal_*` (7)
- `test_release_channels_*` (3)
- `test_runbook_index_lists_every_named_scenario`
- `test_fuzz_target_*`, `test_telemetry_sli_*`, `test_gateway_contracts_*`,
  `test_turn_taking_*`, `test_computer_use_*`, `test_memory_guard_*` (13)

## Historical Prior Sweep

```
python -m pytest tests/test_environment_general_integration.py \
  tests/environment/final_blockers tests/environments/terminal_grid \
  tests/architecture tests/test_embodied_cognition_runtime.py \
  tests/nethack_crucible.py -q

python challenges/nethack_challenge.py --mode simulated --steps 20 \
  --trace artifacts/test_nethack_kernel_trace.jsonl --log-level ERROR

python -m pytest -q

python -m pytest tests/test_server_runtime_hardening.py \
  tests/test_orchestrator_compatibility.py \
  tests/test_runtime_stability_edges.py \
  tests/test_forensic_audit_regressions.py \
  tests/test_launcher_polish_contract.py \
  tests/test_resilient_boot_llm_stage.py \
  tests/test_runtime_polish.py \
  tests/test_time_resilience.py
```

Historical general environment result: **210 passed**. Simulated challenge emitted a
hash-chained trace at `artifacts/test_nethack_kernel_trace.jsonl`.
Historical full repository result: **4333 passed, 7 skipped, 7 warnings,
1 subtests passed** in 479.24s.

## Historical Prior Pass / Fail Results

- general environment autonomy slice: 210 passed
- simulated NetHack stress canary: passed, trace emitted
- full repository sweep: 4333 passed, 7 skipped, 7 warnings, 1 subtests passed
- mixin ownership slice: 8 passed
- service manifest slice: 6 passed
- shutdown coordinator slice: 6 passed
- will transaction slice: 6 passed
- atomic writer slice: 7 passed
- actor supervisor proof slice: 6 passed
- self repair ladder slice: 9 passed
- conformance + fault injection slice: 19 passed
- depth audit slice: 3 passed
- skill contract slice: 3 passed
- perception slice: 5 passed
- security slice: 6 passed
- formal protocol slice: 7 passed
- release channels + runbooks slice: 4 passed
- fuzz/SLI/gateway/turn-taking/computer-use/memory-guard slice: 13 passed
- broad regression sweep: 304 passed
- 2026-06-25 enterprise ratchet cleanup: 212 focused tests passed;
  `make enterprise-gate` passed; enterprise high/critical count is 0 and
  baseline regressions are 0.
- 2026-06-25 production surface gateway cleanup: 28 focused persistence/reach
  tests passed; `make enterprise-gate` passed; `make production-gate` passed;
  `tools/production_surface_lint.py --scope production` passed with 0 findings;
  `tools/proof_integrity_lint.py --scope production` passed with 0 findings;
  architecture map regenerated.
- 2026-06-25 deep mind runtime integration: verified the prior phenomenal
  routing, self-signal, GWT backpressure, nociception, perceptual pump,
  generalized terminal parser, outcome ledger, scientific engine, and unified
  world-model implementations; added provider registrations so these systems
  resolve through canonical `ServiceContainer` boot instead of ad hoc imports.
  Focused validation: 121 deep-mind/perception/world/agency/retrieval tests
  passed, 8 provider/boot checks passed, `make enterprise-gate` passed, and
  `make production-gate` passed.
- 2026-06-25 verified desktop-effect bridge: tightened the live chat to
  `desktop_task` bridge so a desktop objective cannot be reported as completed
  from a bare `ok=True` or step count. The bridge now requires one verified
  effect receipt per requested desktop step, with observable effect evidence
  that is stronger than a receipt id. This is a general guard for Notes, Docs,
  browser, files, PDFs, settings, wallpaper, and future foreground desktop
  actions. Focused validation: 271 desktop/chat/mind-path tests passed,
  `make enterprise-gate` passed, and `make production-gate` passed.
- 2026-06-25 conversation readiness semantics: fixed the shared
  conversation-lane health rule so active `spawning`, `handshaking`, and
  evidence-backed `warming` states are reported as working rather than
  unhealthy conversation failures. Cold and failed lanes still block
  readiness, while active warmup/generation no longer floods the neural
  stream with stale `conversation_ready` blockers. Focused validation:
  708 runtime/system/live-route tests passed, `make enterprise-gate` passed,
  and `make production-gate` passed.
- 2026-06-25 downstream live-mind control consumption: the live desktop
  full-mind contract already required generation controls derived from the
  live mind snapshot to be bound into the router call. This checkpoint extends
  the proof into the MLX lane: the client now passes the live-mind binding flag
  into the worker request, the worker emits a sanitized surface-control receipt
  showing steering-alpha and recurrent-depth controls requested/applied, and
  the parent client retains the latest receipt for diagnostics/tests. Focused
  validation: 27 MLX/live-mind control tests passed, 58 MLX client resilience
  tests passed, and 6 live desktop contract tests passed.
- 2026-06-25 server-visible MLX control receipt: tightened the live desktop
  full-mind contract again so `full_mind_path=true` requires not only
  CognitiveEngine/router-bound live-mind controls, but a worker-applied MLX
  surface-control receipt. The router now exposes last-generation metadata,
  CognitiveEngine carries the worker receipt in Thought metadata, and the chat
  live-turn contract refuses to certify full-mind desktop success without that
  receipt. Focused validation: 16 live-mind/router/contract tests passed and
  the full desktop conversation-lane suite passed (201 tests).
- 2026-06-25 MLX worker user-surface quality gate: live desktop
  CognitiveEngine turns now pass the visible current user message into the
  MLX worker as a validation prompt. The worker validates each user-visible
  draft against the shared conversation reliability contract before returning
  IPC success, retries from the original live mind context when a draft is
  rejected, and emits a surface-quality receipt. The live-turn contract now
  refuses `full_mind_path=true` when that worker quality gate ran and failed,
  even if every other runtime probe is green. Focused validation: 32
  worker/client/live-mind control tests passed, 175 chat reliability tests
  passed, 3 route regressions passed, 384 live/router/MLX/runtime tests
  passed, `make enterprise-gate` passed, and `make production-gate` passed.

## Current Closeout Progress Gauge

- Latest checkpoint: live desktop generation health and repair-budget proof
  (`b157bfda`, pushed 2026-06-27).
- Evidence added: subprocess ownership routed through `SubprocessGateway` for
  DNU/wedge supervisor paths, hardcoded local test paths removed, one broad
  watchdog exception narrowed, enterprise baseline regressions cleared, direct
  production file writes routed through `FileWriteGateway`, reach network calls
  routed through `NetworkGateway`, architecture-map artifacts refreshed, and
  deep mind support services made first-class canonical boot services:
  `global_workspace`, `nociception`, `outcome_ledger`, `scientific_engine`,
  `unified_world_model`, `screen_perception`, `perceptual_pump`, and
  `terminal_parser`. Live desktop chat now also requires verified desktop-task
  effect receipts before completion claims can reach the user, and health
  pulses distinguish active conversation warmup from actual conversation
  failure. The live desktop turn contract now additionally requires proof that
  generation controls derived from the live mind snapshot were bound into the
  model call before a reply can count as a full-mind desktop response, and the
  MLX worker/client/router/CognitiveEngine/chat path now records and exposes a
  downstream receipt proving those controls were consumed by user-visible
  generation. The MLX worker now also validates user-visible drafts against
  the shared response reliability contract before reporting generation success,
  and the desktop full-mind contract requires a passing worker surface-quality
  receipt whenever that gate runs.
- Functional closeout estimate: about **86%**. This measures implemented and
  validated runtime capability, not metaphysical claims or whole-repository
  semantic certification.
- Exhaustive semantic-review coverage: **600/3,636 code files (16.5%)** at the
  latest `make closeout-audit` run. Mechanical enumeration and hashes cover all
  tracked text, but the remaining code files cannot honestly be called
  line-reviewed until current semantic ledger records exist.
- Estimated remaining work: **3 consolidated checkpoints**, each expected to
  contain multiple focused commits: (1) launched desktop action/voice and
  multi-turn soak proof, (2) DNU/Aletheia plus CRSM/CAA/longevity/replay proof,
  and (3) exhaustive semantic ledger closure followed by final-proof and claim
  purification.
- Still not closed: final proof, DNU/Aletheia clean reruns, live desktop
  multi-app proof, CRSM-to-LoRA closure, CAA extraction validation, longer soak,
  proof artifact replay, and remaining whole-code semantic review.

## Checkpoint: 2026-06-27 Live Desktop Generation Health

- Root cause fixed: the desktop repair lane had an 18-second outer budget that
  collapsed to an 8-second CognitiveEngine watchdog, cancelling valid 32B
  correction work. The bounded repair budget is now 75 seconds with more than
  40 seconds available to the inner full-mind cycle.
- False-health fix: a live, foreground-owned generation now counts as
  operational inference only while the worker is alive and request/token
  timestamps prove bounded progress. Stalled active work still fails health.
- Capability-grounding fix: the live capability catalog was sliced at exactly
  1,000 characters, producing a mid-word fragment before a boundary sentence
  hid the defect. The complete bounded catalog now reaches CognitiveEngine.
- Evidence: 320 inference/desktop route tests passed; Ruff, enterprise gate,
  and all 37 production readiness checks passed. The repeated external desktop
  proof passed with the real 32B lane: healthy boot in 27 seconds, complete
  capability answer, 32.1-second identity answer, immediate memory continuity,
  19.67 GB peak process-tree RSS, no runtime failure markers, clean shutdown,
  no orphan workers, and port release.
- Artifact: `artifacts/live_proof/full_desktop_runtime_20260627_checkpoint_13c/`.
- Next exact task: run the full desktop action plus multi-turn conversation soak
  under the same 28/32 GB ceilings, then repair any general planning, computer
  use, receipt, or coherence failure before moving to proof batteries.

## Checkpoint: 2026-06-25 Live Desktop Heavy-Lane Memory Admission

- Scope: live desktop 32B/72B launch safety and local-model admission, not a
  task-specific demo script.
- Root issue addressed: the launcher forced a stale static 35GB 32B projection
  and the worker allowed unsafe RSS override values during desktop safe boot.
  That made the conversation lane alternate between false cold-lane refusal and
  dangerous near-cap admission, leaving no reserve for KV/cache growth, browser
  automation, or UI overhead.
- Fixes landed: MLX and local-server admission now support `auto` projected
  footprint detection from the actual model artifact, add explicit per-lane
  process reserves, and include those reserves in projected process-tree RSS
  refusal reasons. The native launcher, shell launcher, `aura_main.py`, and the
  live boot proof harness now share the same `auto` projection + reserve policy.
  MLX worker RSS overrides are clamped during desktop safe boot unless the
  operator explicitly opts into unsafe memory limits.
- Evidence: focused memory/launcher/local-server suite passed 116 tests; broader
  live-chat/MLX/local-runtime slices passed 334 and 140 tests; `make
  enterprise-gate` passed; `make production-gate` passed.
- Next exact task: launch/probe the real desktop path under this safer admission
  policy, verify `conversation_ready` reaches ready without assistant-mode
  leakage, and monitor terminal/neural-stream memory and route receipts.

## Unresolved Failures / Known Backlog

1. **R-001**: AGENTS.md, AURA_MASTER_SPEC.md, docs/AURA_MASTER_SPEC.md,
   docs/RUNTIME_INVARIANTS.md, docs/PRODUCTION_HARDENING_PLAN.md,
   docs/SKILL_CERTIFICATION_MATRIX.md, docs/DEPTH_AUDIT.md,
   docs/ABUSE_GAUNTLET.md, docs/FORMAL_VERIFICATION_PLAN.md never
   landed in the repo. The contracts those documents implied are now
   captured as runnable modules (service_manifest, depth_audit,
   fault_injection abuse stages, formal_models, conformance, etc.).
2. Real hardware drivers (camera/microphone/screen/subtitle) are
   contract-only. Phase L scaffolds the contract but no platform
   driver is bundled.
3. OpenTelemetry / Prometheus exporter wiring is catalog-only
   (`telemetry_sli.SLO_CATALOG`).
4. Operator CLI (`aura doctor`, `aura conformance`, etc.) is described
   in the plan but not yet implemented as a CLI surface.
5. Multimodal model router, durable-workflow engine, external red-team
   automation, day-in-the-life 24h soak runner are documented in the
   plan as Phase J/K/L follow-ons.
6. Successful NetHack ascension is not yet recorded. The architecture is now
   wired for strict-real runs, but a full autonomous win remains an empirical
   long-run target.

## Next Exact Task

After this checkpoint, the next high-leverage move is the live launched desktop
path proof: boot Aura through the same desktop lane Bryan uses, keep memory
below the safe admission ceiling, verify full-mind chat receipts instead of
bounded assistant repairs, and only then exercise the visible multi-app demo
through general computer-use planning and effect verification.

## Next Exact Continuation Prompt

> Continue Aura closeout from `docs/AURA_EXECUTION_TRACKER.md`. The next target
> is the real launched desktop path: boot Aura through the user-facing launcher,
> verify the 32B live mind path is conversation-ready without raw assistant
> leakage, monitor neural stream/terminal memory and route receipts, and patch
> only general runtime/OS-control causes of stalls, incoherent replies, or unsafe
> memory admission.

## Exact Stopping Point

Current pass has a green memory-admission checkpoint and is proceeding into the
real launched desktop path proof.

## Current Git Diff Summary

- This document records the live desktop heavy-lane memory-admission checkpoint.
- No successful launched multi-app desktop demo receipt is present yet.
