#!/bin/bash
# ==============================================================================
# Aura Unified Launch Script
# Guarantees Aura loads from the LIVE SOURCE CODE, bypassing any PyInstaller builds
# ==============================================================================

# Dynamically resolve root path relative to this script
export AURA_ROOT="$(cd -P "$(dirname "$0")" && pwd -P)"
cd "$AURA_ROOT" || exit 1

OPEN_GUI_WINDOW=0
REBOOT_MODE=0
AURA_PORT=8000
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --open-gui-window|--gui-window)
            OPEN_GUI_WINDOW=1
            shift
            ;;
        --port)
            if [[ -n "$2" ]]; then
                AURA_PORT="$2"
                shift 2
            else
                echo "❌ Missing value for --port"
                exit 1
            fi
            ;;
        --reboot)
            REBOOT_MODE=1
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PASSTHROUGH_ARGS[@]}"

echo -e "  \033[1;36m     ▄████████    ███    ███   ████████▄      ▄████████\033[0m"
echo -e "  \033[1;36m    ███    ███    ███    ███   ███   ▀███    ███    ███\033[0m"
echo -e "  \033[1;35m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;35m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;34m  ▀███████████    ███    ███   ████████▀   ▀███████████\033[0m"
echo -e "  \033[1;34m    ███    ███    ███    ███   ███   ▀███    ███    ███\033[0m"
echo -e "  \033[1;34m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;34m    ███    ███    ▀████████▀   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;30m┌─────────────────────────────────────────────────────────────┐\033[0m"
echo -e "  \033[1;30m│\033[0m  \033[1;35m✨ A U R A   L U N A   C O G N I T I V E   R U N T I M E ✨\033[0m  \033[1;30m│\033[0m"
echo -e "  \033[1;30m├─────────────────────────────────────────────────────────────┤\033[0m"
echo -e "  \033[1;30m│\033[0m  🧠 \033[1;36mStateful Mind\033[0m | 🛡️ \033[1;32mSealed Governance\033[0m | 🧬 \033[1;34mLive Morphogenesis\033[0m \033[1;30m│\033[0m"
echo -e "  \033[1;30m└─────────────────────────────────────────────────────────────┘\033[0m"
echo ""
echo -e "🌸 \033[1;32mInitializing Aura\033[0m (Live Source Mode)..."

# Version Lock: Aura requires Python 3.12 for binary compatibility with its native extensions (grpc, mlx).
# We prefer the venv python if it matches 3.12, otherwise we search for system python3.12.
if [ -f ".venv/bin/python3" ] && .venv/bin/python3 --version | grep -q "3.12"; then
    PYTHON_CMD=".venv/bin/python3"
elif command -v python3.12 &>/dev/null; then
    echo "⚠️  Venv mismatch or missing. Using system python3.12 directly."
    PYTHON_CMD="python3.12"
else
    echo "❌ CRITICAL: Python 3.12 not found. Aura's native AI cores will fail on higher versions."
    echo "Please install python 3.12 or fix your .venv."
    exit 1
fi

echo "📍 Using Interpreter: $($PYTHON_CMD --version) at $PYTHON_CMD"

LOG_DIR="${HOME}/.aura/logs"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="${LOG_DIR}/desktop-launch.log"

resolve_launch_log() {
    if touch "$LAUNCH_LOG" 2>/dev/null; then
        echo "$LAUNCH_LOG"
        return
    fi

    local fallback_log="${TMPDIR:-/tmp}/aura-desktop-launch.log"
    if touch "$fallback_log" 2>/dev/null; then
        echo "$fallback_log"
        return
    fi

    echo "/dev/null"
}

ACTIVE_LAUNCH_LOG="$(resolve_launch_log)"

if [ "$OPEN_GUI_WINDOW" = "1" ]; then
    echo "🪟 Opening Aura desktop window..."
    export PYTHONUNBUFFERED=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    export OBJC_PRINT_LOAD_METHODS=NO
    if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
        exec "$PYTHON_CMD" -u aura_main.py --gui-window --port "$AURA_PORT" "$@"
    else
        nohup "$PYTHON_CMD" -u aura_main.py --gui-window --port "$AURA_PORT" "$@" >>"$ACTIVE_LAUNCH_LOG" 2>&1 &
        GUI_PID=$!
        disown "$GUI_PID" 2>/dev/null || true
        echo ""
        echo "✨ Aura desktop window opening (PID: $GUI_PID)"
        echo "📜 Logs: $ACTIVE_LAUNCH_LOG"
        exit 0
    fi
fi

# 1. Cleanup Phase — use bounded Python cleanup so the launcher can't hang on pkill/lsof
echo "🧹 Cleaning up existing instances..."
if ! "$PYTHON_CMD" aura_cleanup.py; then
    echo "⚠️  Cleanup reported an issue; continuing with launch."
fi

# 2. Launch Phase
echo "🚀 Starting Aura Desktop..."
export MLX_METAL_SYNC=1
: "${AURA_FORCE_CAMERA:=0}"             # Camera stays off by default on macOS boot
: "${AURA_AUTO_LISTEN:=0}"              # STT stays off by default on boot
: "${AURA_ENABLE_UVLOOP:=0}"            # macOS desktop path prefers stock asyncio
: "${AURA_ATTACH_LAUNCHER:=0}"          # Set to 1 to keep the shell attached for live logs
export AURA_FORCE_CAMERA
export AURA_AUTO_LISTEN
export AURA_ENABLE_UVLOOP
export AURA_ATTACH_LAUNCHER
: "${AURA_LOCAL_BACKEND:=mlx}"   # Aura's own in-process fine-tuned mind (MLX) so the substrate can steer generation; AURA_LOCAL_BACKEND=llama_cpp falls back to the base-model server
if [ "${AURA_LAUNCHED_FROM_APP:-0}" = "1" ]; then
    : "${AURA_SAFE_BOOT_DESKTOP:=1}"
    : "${AURA_EXTERNAL_GUI_OWNER:=1}"
    if [ "$REBOOT_MODE" != "1" ]; then
        : "${AURA_CLEANUP_RECENT_GRACE_S:=45}"
    fi
fi
if [ "${AURA_SAFE_BOOT_DESKTOP:-0}" = "1" ]; then
    : "${AURA_EAGER_CORTEX_WARMUP:=auto}"
    # Prewarm the 32B Cortex in the BACKGROUND shortly after boot so the first
    # conversational message is responsive instead of waiting minutes for a cold
    # 32B load. This stays RAM-gated by the admission snapshot (it will not warm
    # under genuine memory pressure), so it improves latency without risking the
    # memory-pressure failures safe-boot guards against. Previously this was
    # "auto", which under safe boot disabled prewarm entirely and left the chat
    # lane cold until a foreground demand that did not reliably trigger a load.
    : "${AURA_DEFERRED_CORTEX_PREWARM:=1}"
    : "${AURA_ENABLE_PERMANENT_SWARM:=0}"
    : "${AURA_SAFE_BOOT_METAL_CACHE_RATIO:=0.16}"
    : "${AURA_SAFE_BOOT_METAL_CACHE_CAP_GB:=10}"
    : "${AURA_SAFE_BOOT_MLX_MEMORY_RATIO:=0.44}"
    : "${AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB:=28}"
    : "${AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB:=18}"
    : "${AURA_SAFE_BOOT_PROCESS_RSS_RATIO:=0.56}"
    : "${AURA_SAFE_BOOT_PROCESS_RSS_CAP_GB:=36}"
    : "${AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB:=24}"
    : "${AURA_PROCESS_RSS_LIMIT_GB:=36}"
    : "${AURA_LOCAL_RUNTIME_SINGLETON:=1}"
    : "${AURA_LOCAL_PARALLEL_SLOTS:=1}"
    : "${AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB:=24}"
    : "${AURA_MLX_32B_PROJECTED_FOOTPRINT_GB:=35}"
    : "${AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB:=52}"
    : "${AURA_MLX_72B_PROJECTED_FOOTPRINT_GB:=41}"
    : "${AURA_FOREGROUND_CHAT_MAX_TOKENS:=2048}"
    echo "🛡️ Safe desktop boot enabled: cortex warmup=auto (RAM-aware), keeping background swarm off."
else
    : "${AURA_ENABLE_PERMANENT_SWARM:=1}"   # Multi-agent internal debate
fi
export AURA_SAFE_BOOT_DESKTOP
export AURA_EXTERNAL_GUI_OWNER
export AURA_CLEANUP_RECENT_GRACE_S
export AURA_EAGER_CORTEX_WARMUP
export AURA_DEFERRED_CORTEX_PREWARM
export AURA_ENABLE_PERMANENT_SWARM
export AURA_SAFE_BOOT_METAL_CACHE_RATIO
export AURA_SAFE_BOOT_METAL_CACHE_CAP_GB
export AURA_SAFE_BOOT_MLX_MEMORY_RATIO
export AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB
export AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB
export AURA_SAFE_BOOT_PROCESS_RSS_RATIO
export AURA_SAFE_BOOT_PROCESS_RSS_CAP_GB
export AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB
export AURA_PROCESS_RSS_LIMIT_GB
export AURA_LOCAL_RUNTIME_SINGLETON
export AURA_LOCAL_PARALLEL_SLOTS
export AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB
export AURA_MLX_32B_PROJECTED_FOOTPRINT_GB
export AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB
export AURA_MLX_72B_PROJECTED_FOOTPRINT_GB
export AURA_FOREGROUND_CHAT_MAX_TOKENS
export AURA_LOCAL_BACKEND
export PYTHONUNBUFFERED=1              # Always flush startup logs for launcher diagnostics
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # Suppress ObjC fork warnings
# Suppress duplicate ObjC class warnings from cv2/av FFmpeg dylib overlap (cosmetic, not a crash)
export OBJC_PRINT_LOAD_METHODS=NO

# Launch directly using the discovered loader. By default we detach so Terminal
# is not left parented to Aura for the entire session.
if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
    "$PYTHON_CMD" -u aura_main.py --desktop --port "$AURA_PORT" "$@" &
else
    nohup "$PYTHON_CMD" -u aura_main.py --desktop --port "$AURA_PORT" "$@" >>"$ACTIVE_LAUNCH_LOG" 2>&1 &
fi
AURA_PID=$!

cleanup_attached_launcher() {
    local status=${1:-0}
    trap - INT TERM EXIT
    if [ -n "${AURA_PID:-}" ] && kill -0 "$AURA_PID" 2>/dev/null; then
        kill -TERM "$AURA_PID" 2>/dev/null || true
        wait "$AURA_PID" 2>/dev/null || true
    fi
    "$PYTHON_CMD" aura_cleanup.py >/dev/null 2>&1 || true
    exit "$status"
}

echo ""
echo "✨ Aura Luna launching (PID: $AURA_PID)"
echo ""
echo "💡 PRO-TIP: Add this alias to your ~/.zshrc to launch Aura from anywhere:"
echo "   alias aura=\"$AURA_ROOT/launch_aura.sh\""
echo ""
if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
    echo "📜 Attached launcher mode active. Press Ctrl+C to stop following the process."
    trap 'cleanup_attached_launcher 130' INT
    trap 'cleanup_attached_launcher 143' TERM
    trap 'cleanup_attached_launcher $?' EXIT
    wait "$AURA_PID"
    LAUNCH_STATUS=$?
    cleanup_attached_launcher "$LAUNCH_STATUS"
else
    disown "$AURA_PID" 2>/dev/null || true
    echo "📜 Logs: $ACTIVE_LAUNCH_LOG"
fi
