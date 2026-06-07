#!/bin/bash
# scripts/nethack_runner.sh

export AURA_TEST_MODE=1
export AURA_MODEL=Qwen2.5-32B-Instruct-8bit
export AURA_LOCAL_BACKEND=mlx
export AURA_NETHACK_LOG=~/.aura/logs/nethack/kernel_trace.jsonl

mkdir -p ~/.aura/logs/nethack/

echo "Launching Aura NetHack Gameplay..." > ~/.aura/logs/nethack/runner.log
date >> ~/.aura/logs/nethack/runner.log

# Run challenges/nethack_challenge.py with a high number of steps
.venv/bin/python challenges/nethack_challenge.py \
    --mode strict_real \
    --steps 100000 \
    --trace ~/.aura/logs/nethack/kernel_trace.jsonl \
    --log-level INFO >> ~/.aura/logs/nethack/runner.log 2>&1
