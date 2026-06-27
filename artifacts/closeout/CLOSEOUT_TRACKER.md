# Aura Closeout Tracker

This tracker records checkpoint evidence for the final closeout effort. It is
not a claim that Aura is finished; it marks what has been verified and what
remains open.

## Current Estimate

- Overall closeout: 78%
- Remaining checkpoints: 9 total
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
