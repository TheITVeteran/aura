#!/bin/bash
# scripts/bundle_app.sh
# Build a thin macOS app bundle that launches Aura from the live workspace.

set -euo pipefail

ROOT_DIR="$(cd -P "$(dirname "$0")/.." && pwd -P)"
DIST_DIR="${ROOT_DIR}/dist"
APP_BASENAME="${AURA_APP_NAME:-Aura}"
APP_NAME="${APP_BASENAME}.app"
APP_DIR="${DIST_DIR}/${APP_NAME}"
INSTALL_PATH="${AURA_INSTALL_PATH:-}"
DEFAULT_CODESIGN_IDENTITY="-"

run_with_timeout() {
    local timeout_s="$1"
    shift
    local status_file
    status_file="$(mktemp "${TMPDIR:-/tmp}/aura-run-timeout-status.XXXXXX")"
    (
        set +e
        "$@"
        printf '%s\n' "$?" > "${status_file}"
    ) &
    local child_pid="$!"
    local deadline=$((SECONDS + timeout_s))
    while [ ! -s "${status_file}" ]; do
        if ! kill -0 "${child_pid}" 2>/dev/null; then
            break
        fi
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            kill -TERM "${child_pid}" 2>/dev/null || true
            sleep 0.2
            kill -KILL "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" 2>/dev/null || true
            rm -f "${status_file}"
            return 124
        fi
        sleep 0.2
    done
    wait "${child_pid}" 2>/dev/null || true
    local status="1"
    if [ -s "${status_file}" ]; then
        status="$(cat "${status_file}")"
    fi
    rm -f "${status_file}"
    return "${status}"
}

# Prefer Aura's persistent local code-signing identity when it exists.  macOS TCC
# grants attach to the app identity, and ad-hoc signatures can drift on rebuilds.
AURA_AUTO_USE_LOCAL_CODESIGN="${AURA_AUTO_USE_LOCAL_CODESIGN:-1}"
if [ "${AURA_AUTO_USE_LOCAL_CODESIGN}" = "1" ] && command -v security >/dev/null 2>&1; then
    LOCAL_AURA_IDENTITY="$(
        security find-identity -v -p codesigning 2>/dev/null \
            | sed -n 's/.*"\(Aura Local Code Signing[^"]*\)".*/\1/p' \
            | head -n 1
    )"
    if [ -n "${LOCAL_AURA_IDENTITY}" ]; then
        SIGN_PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aura-codesign-probe.XXXXXX")"
        trap 'rm -rf "${SIGN_PROBE_DIR}"' EXIT
        printf '#!/bin/sh\nexit 0\n' > "${SIGN_PROBE_DIR}/probe"
        chmod +x "${SIGN_PROBE_DIR}/probe"
        if run_with_timeout "${AURA_CODESIGN_PROBE_TIMEOUT_S:-8}" codesign --force --sign "${LOCAL_AURA_IDENTITY}" "${SIGN_PROBE_DIR}/probe" >/dev/null 2>&1; then
            DEFAULT_CODESIGN_IDENTITY="${LOCAL_AURA_IDENTITY}"
        else
            echo "⚠️ Local Aura code-signing identity exists but cannot sign from this shell within ${AURA_CODESIGN_PROBE_TIMEOUT_S:-8}s; using ad-hoc signing." >&2
        fi
    fi
fi
CODESIGN_IDENTITY="${AURA_CODESIGN_IDENTITY:-${DEFAULT_CODESIGN_IDENTITY}}"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
EXECUTABLE_NAME="aura-launcher"
EXECUTABLE_PATH="${MACOS_DIR}/${EXECUTABLE_NAME}"
ICON_SOURCE="${ROOT_DIR}/aura_icon.icns"
ROOT_LINK="${RESOURCES_DIR}/aura-root"
ROOT_PATH_FALLBACK="${RESOURCES_DIR}/aura-root-path"
VERSION_FILE="${RESOURCES_DIR}/aura-version"
VERSION_FULL_FILE="${RESOURCES_DIR}/aura-version-full"
INFO_PLIST="${CONTENTS_DIR}/Info.plist"
ENTITLEMENTS_PLIST="${DIST_DIR}/aura.entitlements"
LAUNCHER_SOURCE="${ROOT_DIR}/scripts/AuraLauncher.swift"

cd "${ROOT_DIR}"

echo "📦 Building ${APP_NAME} (live source mode)..."

if [ -x "${ROOT_DIR}/.venv/bin/python3" ] && [ -f "${ROOT_DIR}/scripts/build_launcher_icon.py" ]; then
    "${ROOT_DIR}/.venv/bin/python3" "${ROOT_DIR}/scripts/build_launcher_icon.py" >/dev/null
fi

if [ ! -f "${LAUNCHER_SOURCE}" ]; then
    echo "❌ Missing launcher source: ${LAUNCHER_SOURCE}"
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "❌ xcrun is required to build the native Aura launcher."
    exit 1
fi

SWIFTC_PATH="$(xcrun --find swiftc 2>/dev/null || true)"
if [ -z "${SWIFTC_PATH}" ]; then
    echo "❌ swiftc is required to build the native Aura launcher."
    exit 1
fi

SDKROOT_PATH="$(xcrun --show-sdk-path --sdk macosx 2>/dev/null || true)"
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

ln -sfn "${ROOT_DIR}" "${ROOT_LINK}"
printf '%s\n' "${ROOT_DIR}" > "${ROOT_PATH_FALLBACK}"

PYTHON_FOR_VERSION="${ROOT_DIR}/.venv/bin/python3"
if [ ! -x "${PYTHON_FOR_VERSION}" ]; then
    PYTHON_FOR_VERSION="$(command -v python3 || true)"
fi
if [ -z "${PYTHON_FOR_VERSION}" ]; then
    echo "❌ python3 is required to write Aura.app metadata."
    exit 1
fi

APP_SEMVER="2026.3.31"
APP_FULL_VERSION="Aura Luna v${APP_SEMVER}"
APP_SEMVER="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.version import VERSION
semver = VERSION.split("-", 1)[0]
print(semver)
PY
)"
APP_FULL_VERSION="$("${PYTHON_FOR_VERSION}" - <<'PY'
from core.version import version_string
print(version_string("full"))
PY
)"

printf '%s\n' "${APP_SEMVER}" > "${VERSION_FILE}"
printf '%s\n' "${APP_FULL_VERSION}" > "${VERSION_FULL_FILE}"

CLANG_MODULE_CACHE_PATH="${TMPDIR:-/tmp}/aura-launcher-clang-cache" xcrun swiftc \
    -O \
    -framework AppKit \
    -framework CoreGraphics \
    -framework Foundation \
    -framework WebKit \
    "${LAUNCHER_SOURCE}" \
    -o "${EXECUTABLE_PATH}"

chmod +x "${EXECUTABLE_PATH}"

if [ -f "${ICON_SOURCE}" ]; then
    cp "${ICON_SOURCE}" "${RESOURCES_DIR}/Aura.icns"
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_FOR_VERSION}" - "${INFO_PLIST}" "${ENTITLEMENTS_PLIST}" "${APP_SEMVER}" <<'PY'
import plistlib
import sys
from pathlib import Path

from core.security.macos_bundle_manifest import (
    info_plist_overrides,
    write_entitlements_plist,
)

info_path = Path(sys.argv[1])
entitlements_path = Path(sys.argv[2])
app_semver = sys.argv[3]

payload = {
    "CFBundleDevelopmentRegion": "en",
    "CFBundleDisplayName": "Aura",
    "CFBundleExecutable": "aura-launcher",
    "CFBundleIconFile": "Aura.icns",
    "CFBundleIdentifier": "com.aura.desktop",
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": "Aura",
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": app_semver,
    "CFBundleVersion": app_semver,
    "NSHighResolutionCapable": True,
}
payload.update(info_plist_overrides())
with info_path.open("wb") as handle:
    plistlib.dump(payload, handle, sort_keys=True)
write_entitlements_plist(entitlements_path)
PY

echo "✅ Built ${APP_DIR}"
echo "🧠 Live source link: ${ROOT_DIR}"
echo "✍️ Edit the repo normally — this launcher always runs the current workspace code."

sign_bundle() {
    local target="$1"
    local timeout_s="${AURA_CODESIGN_TIMEOUT_S:-45}"
    if run_with_timeout "${timeout_s}" codesign "${CODESIGN_ARGS[@]}" "${target}" >/dev/null; then
        return 0
    fi
    if [ "${CODESIGN_IDENTITY}" != "-" ]; then
        echo "⚠️ Codesigning ${target} with ${CODESIGN_IDENTITY} failed or timed out; falling back to ad-hoc signing." >&2
        local fallback_args=(--force --sign "-" --entitlements "${ENTITLEMENTS_PLIST}")
        run_with_timeout "${timeout_s}" codesign "${fallback_args[@]}" "${target}" >/dev/null
        return $?
    fi
    return 1
}

if command -v codesign >/dev/null 2>&1; then
    CODESIGN_ARGS=(--force --sign "${CODESIGN_IDENTITY}" --entitlements "${ENTITLEMENTS_PLIST}")
    if [ "${CODESIGN_IDENTITY}" != "-" ]; then
        CODESIGN_ARGS+=(--options runtime)
        if [ "${AURA_CODESIGN_TIMESTAMP:-0}" = "1" ]; then
            CODESIGN_ARGS+=(--timestamp)
        fi
    fi
    sign_bundle "${APP_DIR}"
fi

if [ -n "${INSTALL_PATH}" ]; then
    echo "📥 Installing ${APP_NAME} to ${INSTALL_PATH}..."
    rm -rf "${INSTALL_PATH}"
    cp -R "${APP_DIR}" "${INSTALL_PATH}"
    if command -v codesign >/dev/null 2>&1; then
        sign_bundle "${INSTALL_PATH}"
    fi
    echo "✅ Installed ${INSTALL_PATH}"
fi
