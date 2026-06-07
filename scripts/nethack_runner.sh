#!/bin/bash
# scripts/nethack_runner.sh

set -euo pipefail

export AURA_TEST_MODE=1
export AURA_MODEL=Qwen2.5-32B-Instruct-8bit
export AURA_LOCAL_BACKEND=mlx
export AURA_NETHACK_LOG=~/.aura/logs/nethack/kernel_trace.jsonl
: "${AURA_NETHACK_STEPS:=5000}"

mkdir -p ~/.aura/logs/nethack/

echo "Launching Aura NetHack Gameplay..." > ~/.aura/logs/nethack/runner.log
date >> ~/.aura/logs/nethack/runner.log

if [[ "${AURA_SAFE_BOOT_DESKTOP:-0}" == "1" || "${AURA_LAUNCHED_FROM_APP:-0}" == "1" ]]; then
    if [[ "${AURA_ALLOW_DESKTOP_NETHACK:-0}" != "1" && "${AURA_ALLOW_DESKTOP_LONGRUNS:-0}" != "1" ]]; then
        echo "Refusing NetHack strict-real run during desktop-safe Aura session." >> ~/.aura/logs/nethack/runner.log
        echo "Set AURA_ALLOW_DESKTOP_NETHACK=1 for an intentional operator-started proof run." >> ~/.aura/logs/nethack/runner.log
        exit 64
    fi
fi

if ! [[ "${AURA_NETHACK_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "Refusing invalid AURA_NETHACK_STEPS=${AURA_NETHACK_STEPS}." >> ~/.aura/logs/nethack/runner.log
    exit 64
fi

if [[ "${AURA_ALLOW_LONG_NETHACK_RUN:-0}" != "1" && "${AURA_NETHACK_STEPS}" -gt 5000 ]]; then
    echo "Refusing ${AURA_NETHACK_STEPS} NetHack steps without AURA_ALLOW_LONG_NETHACK_RUN=1." >> ~/.aura/logs/nethack/runner.log
    exit 64
fi

# Run challenges/nethack_challenge.py only after explicit resource guards pass.
.venv/bin/python challenges/nethack_challenge.py \
    --mode strict_real \
    --steps "${AURA_NETHACK_STEPS}" \
    --trace ~/.aura/logs/nethack/kernel_trace.jsonl \
    --log-level INFO >> ~/.aura/logs/nethack/runner.log 2>&1
