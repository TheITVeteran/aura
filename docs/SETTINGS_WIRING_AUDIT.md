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
| `model.local_path` | ✅ **wired** (2026-06-21) | `model_registry.get_runtime_model_path` prefers it for the primary cortex lane via `_user_model_path_override` (only when set AND the file exists, else falls through). Tests: `tests/test_runtime_settings.py` |
| `model.deep_path` | ✅ **wired** (2026-06-21) | same override, mapped to the deep solver lane (`DEEP_MODEL`) |
| `model.cloud_fallback_enabled` | ✅ **wired** (2026-06-21) | authoritative in `autonomous_brain_integration`: `allow_cloud_fallback = requested AND model.cloud_fallback_enabled` (default off), so no caller can route off-box unless the user permits it. Tests: `tests/test_runtime_settings.py` |
| `voice.input_enabled` | ✅ **wired** (2026-06-21) | gated in `LocalVoiceCortex.listen_loop` via `_user_voice_input_enabled` (`get_runtime_setting`): off ⇒ the loop never opens the mic capture stream (polls so re-enabling resumes, no restart). Tests: `tests/test_runtime_settings.py` |
| `voice.output_enabled` | ✅ **wired** (2026-06-21) | gated in `SovereignVoiceEngine.synthesize_speech` + `speak_stream` via `core.runtime.runtime_settings.get_runtime_setting` (`_user_voice_output_enabled`); off ⇒ TTS short-circuits before synth/lock. Tests: `tests/test_runtime_settings.py` |
| `voice.output_rate` | ❌ dead | pass to the TTS synth rate |
| `permissions.camera` | ✅ **wired** (2026-06-21) | `core.runtime.permission_gates.camera_allowed` gates all camera capture entry points: `VisionSystem.capture` (returns `camera_permission_denied`), `SensoryMotorCortex` opencv stream + per-frame loop, `ProactivePerceptionV2._camera_loop`. Offline-tested; live-verify on hardware. |
| `permissions.screen` | ✅ **wired** (2026-06-21) | `permission_gates.screen_allowed` gates `ComputerUseSkill._default_screenshot` (raises before screencapture AND the pyautogui fallback) and `ScreenSensor.read` (returns unavailable). Offline-tested; live-verify on hardware. |
| `permissions.files_workspace` | ⚠️ deferred | `permission_gates.workspace_files_allowed` exists, but gating the central file gateway has high blast radius — wire deliberately with the gateway owner + live verification, not blind. |
| `autonomy.proactive_messaging` | ✅ **wired** (2026-06-21) | `ProactiveCommunicationManager._process_messages` skips all initiation when off (pending deque waits, resumes if re-enabled). Tests: `tests/test_runtime_settings.py` |
| `autonomy.self_modification` (`blocked/staged/open`) | ✅ **wired** (2026-06-21) | `GrowthLadder.propose_modification` refuses outright when `blocked` (default `staged`/`open` proceed to the existing canary-gated path). Tests: `tests/test_runtime_settings.py` |
| `memory.retention_days` | ✅ **wired** (2026-06-21) | `SovereignPruner._score_memory` uses it as the recency-decay horizon (default 365, replacing a hardcoded 90): longer retention keeps old memories competitive in the prune ranking. Ranking weight only — nothing hard-deleted purely by age. Tests: `tests/test_runtime_settings.py` |
| `memory.review_window` | ⚠️ deferred | no existing age-windowed consolidation consumer to bind to (`knowledge_curator` uses ad-hoc thresholds). Needs a real narrative-arc consolidation window built first, not a wire. |
| `privacy.mode` | ✅ **partial** (2026-06-21) | `WorldBridge.call` (the single gate for consequential world actions) now blocks ALL external actions when `isolated` and external posting when `private`. Telemetry-tightening + perception-redaction effects remain separate/diffuse. Tests: `tests/test_runtime_settings.py` |
| `dev.developer_mode` | ✅ **wired** (2026-06-21) | gates the `/api/trace/{receipt_id}` route in `dashboard.py` (403 `developer_mode_disabled` when off, default off). Tests: `tests/test_runtime_settings.py` |
| `dev.diagnostics_enabled` | ❌ dead | gate the boot self-diagnostic |
| `notify.enabled` | ✅ **wired** (2026-06-21) | `DesktopNotifier.send` short-circuits before the `osascript` toast when off (default on). Tests: `tests/test_runtime_settings.py` |
| `notify.quiet_hours_start` / `_end` | ✅ **wired** (2026-06-21) | `DesktopNotifier.send` suppresses toasts inside the window (`_within_quiet_hours`, wraps past midnight; default 22:00-08:00) |

## How to wire honestly (the patterns proven this cycle)

**Bridge pattern** (safe-mode/autonomy — for settings that reconfigure a
running subsystem):

1. A **settings→runtime bridge** subscriber (`get_settings().subscribe(...)`)
   applies the change to the live subsystem immediately (no reboot).
2. **Boot** reads the persisted value and applies it (so it survives restart).
3. The subsystem reads the value (via the bridge-applied state or directly).
4. An **offline unit test** proves the toggle changes behavior.

**Read-at-gate pattern** (voice.output_enabled, 2026-06-21 — for simple
per-action gates): the subsystem reads the persisted setting at the point of use
via `core.runtime.runtime_settings.get_runtime_setting(key, default)`. This is
**layering-clean** — core reads the JSON the UI writes (the file is the contract
boundary) and never imports the interface layer; an mtime cache keeps it cheap and
any read error falls back to the default. Reflects user changes on the next call,
no reboot. Best for boolean/value gates (voice, notifications, permissions). A
"no dead settings" guard that fails CI when a non-frontend setting has neither a
bridge nor a `get_runtime_setting`/consumer reference would lock this in
(analogous to `tools/proof_fabrication_guard.py`).

## Why these weren't all wired in this pass

Each dead setting gates a **live subsystem** (voice capture, camera, perception,
memory reaper, notifications). Wiring without confirming on real hardware that the
toggle actually changes behavior would risk shipping a control that *looks* wired
but misbehaves — a different flavor of the fake-control problem. The two
**safety kill switches** (safe_mode, autonomy paused) were wired now because they
reuse a proven mechanism and are fully offline-testable. The rest are a scoped
daily-usability pass (#22) best done with the app running for live verification.
