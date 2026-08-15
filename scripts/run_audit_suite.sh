#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}"

# A git worktree has no .venv of its own — the repo keeps one, in the main
# worktree. Falling straight through to `python3` there picks up Homebrew's
# 3.14, which is not the runtime, so the audit measured an interpreter nobody
# ships. Resolve the main worktree before giving up.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  # --git-common-dir points at the main worktree's .git for every linked
  # worktree. Reading `git worktree list` through `awk '...; exit'` instead
  # kills git with SIGPIPE once the list is long, and `pipefail` then aborts
  # the whole script with 141 before it prints anything.
  GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --git-common-dir 2>/dev/null || true)"
  MAIN_WORKTREE="$(dirname "${GIT_COMMON_DIR:-/nonexistent}")"
  if [[ -n "$MAIN_WORKTREE" && -x "$MAIN_WORKTREE/.venv/bin/python" ]]; then
    PYTHON="$MAIN_WORKTREE/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

run() {
  echo "+ $*"
  "$@"
}

# Every pytest target named below. `quick` used to name crucible_test.py, which
# was deleted in 494cb0a4b; because `set -e` is on and pytest exits 4 on a
# missing path, `quick` failed before it ran a single test — while three docs
# called this script the live validation entrypoint. Paths are checked up front
# so the next deletion says which target vanished instead of failing as a
# pytest usage error. tools/check_script_targets.py is the standing gate.
QUICK_TARGETS=(
  tests/test_audit_contracts.py
  tests/test_response_contract.py
  tests/test_effect_closure.py
  tests/test_cognitive_pipeline_2026.py
)

require_paths() {
  local missing=()
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || missing+=("$path")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "audit suite: missing test target(s): ${missing[*]}" >&2
    exit 3
  fi
}

echo "Aura Luna audit suite (${MODE})"
echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON"

case "$MODE" in
  quick)
    require_paths "${QUICK_TARGETS[@]}"
    run env AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY=1 "$PYTHON" -m pytest "${QUICK_TARGETS[@]}" -q
    ;;
  full)
    run env AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY=1 "$PYTHON" -m pytest -q
    if command -v npm >/dev/null 2>&1; then
      run npm --prefix interface/static/shell run build
      run npm --prefix interface/static/memory run build
    else
      echo "npm not found; skipping frontend builds." >&2
    fi
    ;;
  *)
    echo "Usage: $0 [quick|full]" >&2
    exit 2
    ;;
esac

echo "Audit suite complete."
