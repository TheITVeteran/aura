# Settings Panel Wiring Audit

**Finding (2026-06-15):** the settings panel (`interface/routes/settings.py`,
"binds 1:1 with the API") persists every toggle to `~/.aura/data/settings/
runtime.json`, but almost none are **read** by the runtime — they are controls
that look functional and do nothing. This is the same fake-control class as the
fabricated proofs fixed this cycle, and it's the concrete backlog behind
task #19 ("verify every button/gauge works") and #22 (daily reliability).

Method: scanned `core/` + `interface/` + `aura_main.py` for each setting key.
A setting is **wired** only if a subsystem actually consumes it (directly or via
the settings→runtime bridge). Frontend-only settings are legitimately client-side.

## Status

| Setting | Status | What wiring it needs (if dead) |
| :-- | :-- | :-- |
| `safety.safe_mode` | ✅ **wired** (this cycle) | restricts runtime via `core.safe_mode.set_safe_mode` |
| `autonomy.level` (`paused`) | ✅ **wired** (this cycle) | `paused` → restricted runtime via the same bridge |
| `theme.mode` | 🎨 frontend-only | client renders theme; no backend wiring needed |
| `theme.reduced_motion` | 🎨 frontend-only | client honors it; no backend wiring needed |
| `model.local_path` | ❌ dead | `core/brain/llm/model_registry.get_runtime_model_path` should prefer this over env/default |
| `model.deep_path` | ❌ dead | same, for the deep lane |
| `model.cloud_fallback_enabled` | ❌ dead | gate cloud routing in `llm_health_router` when local is down |
| `voice.input_enabled` | ❌ dead | wake/STT loop (`core/voice/*`, perceptual pump) must not capture when off |
| `voice.output_enabled` | ❌ dead | gate TTS in `voice_engine._play_locally` |
| `voice.output_rate` | ❌ dead | pass to the TTS synth rate |
| `permissions.camera` | ❌ dead | in-app gate before camera capture (distinct from macOS TCC) |
| `permissions.screen` | ❌ dead | in-app gate before screen perception |
| `permissions.files_workspace` | ❌ dead | gate file_io to the workspace sandbox |
| `autonomy.proactive_messaging` | ❌ dead | gate/throttle the autonomous output path |
| `autonomy.self_modification` (`blocked/staged/open`) | ❌ dead | map to the self-mod gate (finer than safe_mode) |
| `memory.retention_days` | ❌ dead | feed the episodic memory reaper |
| `memory.review_window` | ❌ dead | feed narrative-arc consolidation window |
| `privacy.mode` | ❌ dead | drive perception redaction (`perception_runtime.privacy_mode` is an internal bool, not bound to this setting) + telemetry/world-bridge tightening |
| `dev.developer_mode` | ❌ dead | gate `/api/trace` + raw subsystem panels |
| `dev.diagnostics_enabled` | ❌ dead | gate the boot self-diagnostic |
| `notify.enabled` | ❌ dead | gate the OS-notification sender |
| `notify.quiet_hours_start` / `_end` | ❌ dead | suppress proactive notifications in the window |

## How to wire honestly (the pattern proven this cycle)

The safe-mode/autonomy wiring is the template:

1. A **settings→runtime bridge** subscriber (`get_settings().subscribe(...)`)
   applies the change to the live subsystem immediately (no reboot).
2. **Boot** reads the persisted value and applies it (so it survives restart).
3. The subsystem reads the value (via the bridge-applied state or directly).
4. An **offline unit test** proves the toggle changes behavior — and, ideally, a
   "no dead settings" guard that fails CI if a non-frontend setting has neither a
   bridge nor a consumer (analogous to `tools/proof_fabrication_guard.py`).

## Why these weren't all wired in this pass

Each dead setting gates a **live subsystem** (voice capture, camera, perception,
memory reaper, notifications). Wiring without confirming on real hardware that the
toggle actually changes behavior would risk shipping a control that *looks* wired
but misbehaves — a different flavor of the fake-control problem. The two
**safety kill switches** (safe_mode, autonomy paused) were wired now because they
reuse a proven mechanism and are fully offline-testable. The rest are a scoped
daily-usability pass (#22) best done with the app running for live verification.
