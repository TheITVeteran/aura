# Aura Closeout Tracker

This tracker records checkpoint evidence for the final closeout effort. It is
not a claim that Aura is finished; it marks what has been verified and what
remains open.

## Current Estimate

- Overall closeout: 84%
- Remaining checkpoints: 4 total
- Current phase: live desktop runtime reliability and full-mind conversation
  path hardening

## Checkpoint 2026-06-27-01: Live Desktop Full-Mind Path

Status: verified locally, ready for commit.

Evidence:

- Live path: `/Users/bryan/Desktop/aura` resolves to
  `/Users/bryan/.aura/live-source`, the same source tree used by the desktop
  launcher.
- Live proof:
  `artifacts/live_proof/full_desktop_runtime_20260627T_desktop_checkpoint_13/live_proof_20260627_091256_verdict.json`
- Result: `LIVE PROOF PASSED`
- Boot: desktop system reached ready state while preserving the distinction
  between system readiness and unproven conversation readiness.
- Conversation: capability inventory, identity, and continuity probes passed
  through the required CognitiveEngine desktop path.
- Memory: RSS stayed near 19.5 GB during the proof and did not reproduce the
  previous 100 GB runaway.
- Shutdown: graceful stop, no orphan process, port freed.
- Runtime stream scan: no failure markers in runtime stdout.

Code areas hardened:

- `core/brain/cognitive_engine.py`
- `core/brain/inference_gate.py`
- `interface/routes/chat.py`
- `core/health/boot_status.py`
- `core/senses/voice_engine.py`
- `tools/live_boot_proof.py`

Tests run:

- Capability inventory contract propagation to the worker boundary.
- Protected capability inventory token floor under resource envelope pressure.
- Capability inventory accept/reject gates.
- Desktop capability inventory routing through CognitiveEngine.
- Boot health cold-standby and warming-lane semantics.
- Voice transcript dispatch gating and explicit direct-eventbus opt-in.

Still open:

- Full visible multi-app demo proof remains open: Notes, Chrome, Google Docs,
  PDF export, article synthesis, and wallpaper must be validated without
  hardcoded task logic.
- Long multi-turn live chat soak should be extended beyond the current focused
  probes.
- CRSM to LoRA train/fuse loop and CAA extraction remain certification
  blockers until real training/extraction validation lands.
- Full `make final-proof` remains open.
- Broad suite failures listed in `artifacts/closeout/OPEN_FINDINGS_2026-06-09.md`
  still need closure or refreshed triage against the current tree.

## Checkpoint 2026-06-27-02: Conversational Amplifier Safety

Status: verified locally, ready for commit.

Why:

- A concurrent change wired best-of-N conversational amplification into the
  primary `UnitaryResponsePhase`.
- Extra foreground model calls can improve voice, but can also recreate the
  lag and RAM spikes seen on the live desktop lane if enabled unconditionally.

What changed:

- Live conversational amplification now requires explicit
  `AURA_CONVERSATIONAL_AMPLIFIER_LIVE=1`.
- The hook checks the runtime memory-pressure snapshot before spawning extra
  foreground model calls.
- The hook is bounded to a smaller time budget and uses fewer candidates under
  tighter budget.
- Normal live chat remains single-generation by default; the taste-learning
  module remains callable and separately tested.

Evidence:

- `.venv/bin/python3 -m py_compile core/phases/response_generation_unitary.py tests/test_conversational_amplifier.py`
- `.venv/bin/python3 -m pytest -q tests/test_conversational_amplifier.py`
- Result: `17 passed`

Estimate update:

- Overall closeout: 77%
- Remaining checkpoints: 10 total

## Checkpoint 2026-06-27-03: Live Capability Inventory Stability

Status: verified locally, ready for commit.

Why:

- Live desktop conversations consistently destabilized when Bryan asked what
  external tools Aura could use.
- The capability/tool question was being treated like a broad operational
  status turn, forcing repeated Cortex retries. The retries outlived the
  client timeout, opened the Cortex circuit, and produced `No response
  returned` ASGI failures.

What changed:

- Capability inventory turns now stay inside CognitiveEngine but answer from
  the live governed capability catalog instead of allocating a foreground 32B
  generation for a static runtime-fact question.
- Worker-side surface quality validation no longer treats concise governed
  capability inventories as too-thin operational-status replies.
- Capability inventory prompt/context construction remains compact and avoids
  duplicate style, speech, preflight, recall, and challenge context.
- The live-turn contract recognizes the catalog-grounded CognitiveEngine path
  as a valid non-fallback path when live mind context and governance evidence
  are bound.

Evidence:

- `.venv/bin/python3 -m py_compile core/brain/cognitive_engine.py core/brain/llm/mlx_worker.py interface/routes/chat.py tests/test_cognitive_engine_background_hardening.py tests/test_strict_contract_steering_clamp.py`
- `.venv/bin/python3 -m pytest -q tests/test_cognitive_engine_background_hardening.py::test_cognitive_engine_capability_inventory_contract_uses_catalog_without_worker tests/test_strict_contract_steering_clamp.py::test_live_user_surface_quality_gate_accepts_concise_capability_inventory tests/test_server_conversation_lane.py::test_desktop_capability_inventory_uses_cognitive_engine_with_catalog_context tests/test_chat_reliability_proof.py::test_capability_inventory_gate_accepts_governed_effect_verified_answer tests/test_chat_reliability_proof.py::test_capability_inventory_gate_rejects_generic_tool_claim tests/test_inference_gate_tiering.py::test_protected_capability_inventory_keeps_min_budget_under_resource_envelope`
- Result: `6 passed`
- Live proof:
  `artifacts/live_proof/full_desktop_runtime_20260627T_desktop_soak_07/live_proof_20260627_101850_verdict.json`
- Result: `LIVE PROOF PASSED`
- Capability inventory latency: `0.5s`, status
  `cognitive_engine_capability_inventory`, RSS delta `4MB`.
- Conversation soak: `2/2 turns passed`.
- Shutdown: graceful stop, no orphan process, port freed.
- Runtime stream scan: no failure markers in runtime stdout.

Estimate update:

- Overall closeout: 78%
- Remaining checkpoints: 9 total

## Checkpoint 2026-06-27-04: Browser Editor Focus Proof

Status: verified locally, ready for commit.

Why:

- Live desktop-action attempts could still paste/type generated document
  content into Chrome's address/search field or another generic browser text
  field instead of the intended web document body.
- The previous guard only proved that focus was not obviously the URL bar. That
  is too weak for Google Docs and similar canvas-heavy editors.

What changed:

- Web editor focus now requires positive editor-like evidence before
  `desktop_task` allows paste/type into a browser document surface.
- Generic browser `AXTextField` and `AXComboBox` controls are rejected even when
  they are not explicitly labeled as the address bar.
- Accepted browser editor focus is limited to editor-like roles with document,
  editor, body, canvas, page, or Google web-document metadata.
- This is a general desktop-control invariant, not a demo-specific script.

Evidence:

- `python -m py_compile core/skills/computer_use.py core/skills/desktop_task.py`
- `python -m pytest tests/test_hardened_computer_use.py::test_browser_paste_refuses_generic_text_field_when_doc_focus_required tests/test_hardened_computer_use.py::test_browser_type_refuses_generic_text_field_when_doc_focus_required tests/test_hardened_computer_use.py::test_web_editor_focus_rejects_generic_browser_text_field tests/test_hardened_computer_use.py::test_web_editor_focus_accepts_editor_like_surface tests/test_hardened_computer_use.py::test_open_url_targets_named_browser -q`
- Result: `5 passed`
- `python -m pytest tests/test_hardened_computer_use.py tests/test_desktop_task_skill.py -q`
- Result: `115 passed`

Estimate update:

- Overall closeout: 79%
- Remaining checkpoints: 8 total

## Checkpoint 2026-06-27-05: Autonomous Runtime Health Truthfulness

Status: verified locally, ready for commit.

Why:

- Normal desktop launch should mean the full Aura runtime is alive, not just
  the foreground chat path.
- The autonomous initiative loop was started by boot, but `/system` did not
  list it as a required full-runtime organ. That allowed health/readiness to
  look better than reality if world watching, knowledge-gap monitoring,
  self-development, social initiative, or mission advancement were dead.

What changed:

- `AutonomousInitiativeLoop` now exposes `get_status()` with task-level liveness
  for world, knowledge, self-development, social, and mission watchers.
- Full desktop runtime status now includes `autonomous_initiative` as a required
  component.
- Full-runtime readiness now fails if the initiative loop is missing or any
  core initiative task is dead.
- This is a general reliability invariant: full Aura cannot claim full-runtime
  health while autonomous external/background initiative is dormant.

Evidence:

- `python -m py_compile core/autonomous_initiative_loop.py interface/routes/system.py`
- `python -m pytest tests/test_full_desktop_runtime_contract.py tests/test_autonomy_visibility.py tests/test_autonomous_initiative_loop_hardening.py -q`
- Result: `23 passed`

Estimate update:

- Overall closeout: 80%
- Remaining checkpoints: 7 total

## Checkpoint 2026-06-27-06: Substrate Health Requires Live Tasks

Status: verified locally, ready for commit.

Why:

- The desktop UI showed PNEUMA/MHAF as offline, and full-runtime health must not
  rely on weak flags that can drift from real task liveness.
- PNEUMA and MHAF expose stronger `get_state_dict().online` signals that prove
  the background task exists and has not already died. The status routes were
  still partially keyed off `_running`.

What changed:

- PNEUMA and MHAF subsystem status endpoints now reject dead background tasks
  even if an object still exists.
- `/api/health` system collection now reports PNEUMA/MHAF `online` from the
  modules' task-liveness state, not only from `_running`.
- Full-runtime readiness therefore tracks the actual substrate loops, not just
  initialized substrate objects.

Evidence:

- `python -m py_compile interface/routes/system.py interface/routes/subsystems.py tests/test_system_route_hardening.py`
- `python -m pytest tests/test_system_route_hardening.py tests/test_full_desktop_runtime_contract.py tests/test_pneuma_runtime_hardening.py -q`
- Result: `28 passed`

Estimate update:

- Overall closeout: 81%
- Remaining checkpoints: 6 total

## Checkpoint 2026-06-27-07: Autonomous Initiative Admission Visibility

Status: verified locally, ready for commit.

Why:

- Aura's autonomous initiative loop could be alive while visibly doing nothing,
  leaving no clear distinction between “dead,” “waiting for idle/boot grace,”
  “blocked by pressure,” and “allowed.”
- For daily runtime reliability, the live path needs to expose the actual
  admission state of autonomous world-watching, self-development, and social
  initiative without weakening their governance/resource gates.

What changed:

- `AutonomousInitiativeLoop.get_status()` now includes admission state for:
  world/knowledge initiative, self-development, and passive social initiative.
- The status reports `allowed` only when the exact production background policy
  permits work; otherwise it reports the real blocker such as recent user
  activity, boot grace, memory pressure, or failure lockdown.
- This keeps autonomous actions governed and bounded while making dormant
  autonomy diagnosable from the normal live runtime status path.

Evidence:

- `python -m py_compile core/autonomous_initiative_loop.py tests/test_autonomy_visibility.py`
- `python -m pytest tests/test_autonomy_visibility.py tests/test_autonomous_initiative_loop_hardening.py tests/test_full_desktop_runtime_contract.py -q`
- Result: `24 passed`

Estimate update:

- Overall closeout: 81.5%
- Remaining checkpoints: 6 total

## Checkpoint 2026-06-27-08: Full Runtime Status Visible In Desktop UI

Status: verified locally, ready for commit.

Why:

- Backend full-runtime truth is not enough if the live desktop UI still leaves
  Bryan guessing whether full Aura, safe boot, autonomy, and initiative
  admission are actually active.
- The telemetry panel showed substrate and learning values, but did not expose
  the canonical full-runtime readiness or autonomous initiative admission state.

What changed:

- Added a `FULL RUNTIME` telemetry section to the desktop UI.
- The section shows profile, full organ readiness, autonomous initiative
  liveness, self-development admission, and social/autonomous admission.
- The UI reads these values from `/api/health` `full_runtime.components`, not
  from decorative local state.

Evidence:

- `python -m py_compile tests/test_runtime_polish.py`
- `python -m pytest tests/test_runtime_polish.py::test_desktop_shell_surfaces_full_runtime_autonomy_status tests/test_runtime_polish.py::test_desktop_shell_does_not_treat_socket_liveness_as_runtime_health tests/test_full_desktop_runtime_contract.py tests/test_autonomy_visibility.py -q`
- Result: `20 passed`

Estimate update:

- Overall closeout: 82%
- Remaining checkpoints: 6 total

## Checkpoint 2026-06-27-09: Resource Guard Is Not Safe Boot

Status: verified locally, ready for commit.

Why:

- Normal desktop launch must run full Aura under RAM/process protection.
- The inference gate still treated the desktop resource guard as
  `_desktop_safe_boot_enabled()`, which could make normal app launch look like
  recovery safe boot and suppress/defer background local cognition.
- That directly matched the live symptom where full-mind/background systems
  appeared dormant even though the user had not requested safe boot.

What changed:

- Split inference-gate semantics:
  - `_desktop_safe_boot_enabled()` now means only explicit
    `AURA_SAFE_BOOT_DESKTOP`.
  - `_desktop_resource_guard_enabled()` now means normal RAM/process guard.
- Eager Cortex warmup remains RAM-protected by the resource guard.
- Background local cognition is no longer permanently disabled just because
  Aura launched from the desktop app with resource protection enabled.
- Successful visible MLX readiness probes now set the visible-readiness anchor,
  so `conversation_ready` cannot remain false after a real visible probe
  returns text.

Evidence:

- `python -m py_compile core/brain/inference_gate.py core/brain/llm/mlx_client.py tests/test_boot_runtime_safety.py`
- `python -m pytest tests/test_mlx_client_resilience.py::TestMLXClientResilience::test_warmup_precompile_requires_visible_readiness_after_empty_compile tests/test_boot_runtime_safety.py -q`
- Result: `42 passed`
- `python -m pytest tests/test_full_desktop_runtime_contract.py tests/test_autonomy_visibility.py tests/test_autonomous_initiative_loop_hardening.py tests/test_system_route_hardening.py -q`
- Result: `45 passed`
- `python -m pytest tests/test_mlx_client_resilience.py -q`
- Result: `59 passed`
- `python -m pytest tests/test_inference_gate_tiering.py::test_desktop_safe_boot_skips_deferred_cortex_prewarm tests/test_inference_gate_tiering.py::test_desktop_safe_boot_respects_deferred_cortex_prewarm_opt_out tests/test_inference_gate_tiering.py::test_desktop_safe_boot_allows_explicit_auto_deferred_prewarm_when_admitted tests/test_inference_gate_tiering.py::test_desktop_safe_boot_refuses_explicit_auto_deferred_prewarm_under_pressure tests/test_inference_gate_tiering.py::test_background_local_deferral_protects_cold_cortex_during_safe_boot tests/test_inference_gate_tiering.py::test_background_local_deferral_reserves_ready_cortex_during_safe_boot tests/test_inference_gate_tiering.py::test_background_local_deferral_honors_ready_cortex_foreground_quiet_window -q`
- Result: `7 passed`

Estimate update:

- Overall closeout: 83%
- Remaining checkpoints: 5 total

## Checkpoint 2026-06-27-10: Exhausted Full-Mind Failures Enter Immune Repair

Status: verified locally, ready for commit.

Why:

- Exhausted live CognitiveEngine failures produced a durable degradation
  receipt, but the incident explicitly reported `repair_requested: false` and
  never reached adaptive immunity or SelfHealing.
- A single malformed generation must not rewrite runtime code, while repeated
  failures must become causal repair evidence instead of recurring forever.
- The full conversation suite also exposed an over-broad capability classifier:
  merely asking whether governed tools were available was incorrectly treated
  as a request for a full multi-category capability inventory.

What changed:

- Every exhausted desktop full-mind reply now records its failure signature in
  the canonical adaptive immune system.
- Adaptive immune recurrence pressure controls escalation. A first transient
  incident remains evidence only; pressure at or above the existing `0.35`
  escalation floor can schedule a governed deep repair.
- Repair targets are narrowed by failure class: inference timeout/empty-output
  failures route to MLX, response-quality failures route to response
  generation, and unclassified engine failures route to CognitiveEngine.
- SelfHealing retains validation/promotion ownership, deduplicates active work,
  and a 15-minute per-target cooldown prevents repair storms.
- Operational tool-status questions no longer trigger the full capability
  inventory contract. Explicit requests to list or explain capabilities still
  require categories, governance, effect evidence, and a hypothetical boundary.
- Conversation-lane tests now reset response-history state between cases, so
  stale-response evidence cannot leak across test sessions.

Evidence:

- `python -m py_compile interface/routes/chat.py tests/test_server_conversation_lane.py`
- `python -m pytest -q tests/test_server_conversation_lane.py`
- Result: `206 passed`
- `python -m pytest -q tests/test_chat_reliability_proof.py tests/test_adaptive_immune_system.py tests/test_reconstruction_deep_repair.py`
- Result: `118 passed`
- `python -m pytest -q tests/test_full_desktop_runtime_contract.py tests/test_system_route_hardening.py tests/test_autonomy_visibility.py tests/test_boot_runtime_safety.py`
- Result: `80 passed`

Estimate update:

- Overall closeout: 84%
- Remaining checkpoints: 4 total
