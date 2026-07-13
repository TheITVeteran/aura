from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_legacy_shell_keeps_constitutional_health_slots():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    required_ids = [
        "brand-status-dot",
        "hud-status",
        "c-policy-mode",
        "c-fragmentation",
        "c-contradictions",
        "c-contested",
        "c-commitments",
        "c-tools-available",
        "health-flags",
        "rolling-summary",
        "phenomenal-summary",
        "tool-available-count",
        "tool-degraded-count",
        "tool-voice-state",
        "tool-last-stage",
        "tool-last-detail",
    ]

    for item in required_ids:
        assert f'id="{item}"' in html, f"legacy shell missing {item}"


def test_legacy_shell_frontend_uses_bootstrap_and_tool_catalog():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "/api/ui/bootstrap" in js
    assert "/api/tools/catalog" in js
    assert "tool_event" in js
    assert "hydrateBootstrap" in js
    assert "renderToolCatalog" in js


def test_legacy_shell_presents_cold_standby_as_not_ready_shell_state():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneIsStandby" in js
    assert "cortex preparing" in js
    assert "CORTEX PREPARING" in js
    assert "Aura is ready. Cortex will warm on first turn." not in js
    assert "syncSplashState(payload);" in js
    assert "Live shell is still syncing. Aura is stabilizing background channels..." in js
    assert "setTimeout(() => dismissSplash(), 8000)" not in js


def test_legacy_shell_placeholder_is_system_status_not_aura_speech():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "Conversation lane initializing. Waiting for verified Aura reply path..." in html
    assert "Conversation lane initializing. Waiting for verified Aura reply path..." in js
    assert "Aura: Infinity online. Synchronizing cognitive drives" not in html
    assert "Aura: Infinity online. Synchronizing cognitive drives" not in js


def test_legacy_shell_presents_active_generation_as_working_not_unavailable():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneHasActiveGeneration" in js
    assert "active_generation_in_flight" in js
    assert "Number(lane.active_generations || 0) > 0" in js
    assert "if (laneHasActiveGeneration(lane)) return 'cortex thinking';" in js
    assert "laneText === 'cortex thinking' ? 'CORTEX THINKING'" in js
    assert "lane.conversation_ready === false && !laneHasActiveGeneration(lane)" in js
    assert "lane.conversation_ready === false && !laneHasActiveGeneration(lane)" in js


def test_legacy_shell_does_not_hide_lane_failures_as_generic_warming():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneFailureClass" in js
    assert "memory_pressure_refused_worker_spawn" in js
    assert "projected_process_tree_rss" in js
    assert "model_load_headroom" in js
    assert "visible_conversation_probe_missing" in js
    assert "endpoint_timeout" in js
    assert "if (failureClass === 'memory_guard') return 'cortex memory guard';" in js
    assert "if (failureClass === 'cognitive_engine') return 'cortex route blocked';" in js
    assert "if (failureClass === 'timeout') return 'cortex timeout';" in js
    assert "CORTEX MEMORY GUARD" in js
    assert "CORTEX ROUTE BLOCKED" in js
    assert "CORTEX TIMEOUT" in js


def test_legacy_shell_matches_conversation_lane_timeout_budget():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function conversationLaneRequestTimeoutMs" in js
    assert "CHAT_REQUEST_TIMEOUT_READY_MS = 335000" in js
    assert "CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000" in js
    assert "const requestTimeoutMs = conversationLaneRequestTimeoutMs(state.conversationLane);" in js
    assert "const requestTimeoutMs = 90000;" not in js


def test_legacy_shell_has_neural_feed_backpressure_controls():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "THOUGHT_QUEUE_MAX = 160" in js
    assert "function normalizeThoughtEvent" in js
    assert "function buildThoughtFingerprint" in js
    assert "function coalesceThoughtQueueItem" in js
    assert "state.thoughtQueue.splice(0, state.thoughtQueue.length - THOUGHT_QUEUE_MAX + 1);" in js


def test_legacy_shell_has_dark_boot_recovery_guard_before_external_assets():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    critical_style = html.index('id="aura-critical-boot-style"')
    first_external_script = html.index('<script src="/static/error_banner.js">')
    assert critical_style < first_external_script
    assert "window.__auraLegacyShellGuardInstalled" in html
    assert "window.__auraShowHardRecovery" in html
    assert "legacy_shell_load_timeout" in html
    assert "/api/ui/shell-error" in html
    assert "/api/dashboard/snapshot" in html
    assert 'href="/logs"' not in html


def test_legacy_shell_marks_ready_only_after_full_script_install():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function markLegacyShellReady()" in js
    assert "window.__auraLegacyShellReady = true;" in js
    assert "document.body.dataset.auraShell = 'ready';" in js
    assert js.rfind("markLegacyShellReady();") > js.rfind("$('regen-btn')?.addEventListener('click', regenerateResponse);")


def test_legacy_shell_neural_feed_receives_health_liveness_pulses():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "lastNeuralPulseAt" in js
    assert "lastSemanticThoughtAt" in js
    assert "NEURAL_LIVENESS_PULSE_MS = 30000" in js
    assert "function queueNeuralLivenessCard" in js
    assert "function publishHealthNeuralPulse" in js
    assert "function conversationPayloadReady" in js
    assert "conversationPayloadReady(payload, blockers)" in js
    assert "lane.conversation_ready === true || payload.conversation_ready === true" not in js
    assert "publishHealthNeuralPulse(payload, 'websocket_heartbeat');" in js
    assert "publishHealthNeuralPulse(d, 'health_poll');" in js
    assert "function recordHealthPollFailure" in js
    assert "health endpoint unavailable; retaining last known state" in js
    assert "endpoint recovered after" in js


def test_server_keeps_legacy_shell_as_default_route():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")

    assert 'ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert 'fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert '"shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell"' in server


def test_react_shell_is_opt_in_so_original_hud_stays_canonical():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    system = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(encoding="utf-8")

    assert "AURA_ENABLE_REACT_SHELL" in server
    assert "if LEGACY_UI_INDEX.exists() and not _react_shell_enabled():" in server
    assert '"canonical_shell": "legacy_shell"' in server
    assert "experimental_shell_enabled" in system
