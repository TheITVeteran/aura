#!/usr/bin/env python3
"""NetHack live proof: drive the REAL game through Aura's environment stack.

Not a simulation and not a screenshot: STRICT_REAL adapter → pexpect-spawned
NetHack → pyte terminal → Observation/CommandSpec pipeline → receipts. The
policy here is deliberately simple (modal hygiene + rotating exploration +
periodic search); the claim proven is the ARCHITECTURE one — Aura's
environment loop can start the real game, perceive it, act in it turn after
turn, and account for every action — not a skill claim.

    python tools/nethack_live_proof.py [--turns 60]

Artifact: artifacts/environment/nethack_live_proof.json
Exit codes: 0 proof passed, 1 loop degraded, 2 environment unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment.command import ActionIntent  # noqa: E402
from core.environments.terminal_grid.nethack_adapter import (  # noqa: E402
    EnvironmentMode,
    NetHackTerminalGridAdapter,
)
from core.environments.terminal_grid.nethack_commands import (  # noqa: E402
    NetHackCommandCompiler,
)

OUT_PATH = ROOT / "artifacts" / "environment" / "nethack_live_proof.json"
STATUS_RE = re.compile(r"HP:?\s*(\d+)\((\d+)\)", re.IGNORECASE)
def _policy(turn: int, screen: str) -> ActionIntent:
    """Reactive one-step policy: prompt hygiene first, then exploration.

    Prompt taxonomy matters: ESC at the character-selection prompt QUITS
    NetHack (observed: game dead in two turns), and 'y' at a Really-quit
    prompt ends the run. Affirm setup prompts, refuse destructive ones,
    acknowledge pagination, and only then explore.
    """
    if "Shall I pick" in screen or "pick a character" in screen:
        return ActionIntent(name="resolve_modal", parameters={"response": "y"})
    if "--More--" in screen:
        return ActionIntent(name="resolve_modal", parameters={"response": " "})
    if "Really" in screen or "(y/n)" in screen or "[yn" in screen:
        return ActionIntent(name="resolve_modal", parameters={"response": "n"})
    if turn % 7 == 6:
        return ActionIntent(name="search")
    direction = ("east", "south", "west", "north")[turn % 4]
    return ActionIntent(name="move", parameters={"direction": direction})


async def run_proof(turns: int) -> dict:
    run_id = f"nethack-live-proof-{int(time.time())}"
    adapter = NetHackTerminalGridAdapter(mode=EnvironmentMode.STRICT_REAL)
    compiler = NetHackCommandCompiler()

    await adapter.start(run_id=run_id)
    if adapter._simulated:
        raise RuntimeError("adapter fell back to simulation under STRICT_REAL")

    steps: list[dict] = []
    screens_seen: set[int] = set()
    hp_readings: list[tuple[int, int]] = []
    executed_ok = 0
    started = time.monotonic()

    try:
        for turn in range(turns):
            obs = await adapter.observe()
            screen = obs.text or ""
            screens_seen.add(hash(screen))
            hp = STATUS_RE.search(screen)
            if hp:
                hp_readings.append((int(hp.group(1)), int(hp.group(2))))

            intent = _policy(turn, screen)
            spec = compiler.compile(intent, trace_id=f"{run_id}:{turn}")
            result = await adapter.execute(spec)
            ok = bool(getattr(result, "ok", getattr(result, "success", True)))
            executed_ok += 1 if ok else 0
            steps.append(
                {
                    "turn": turn,
                    "intent": intent.name,
                    "params": intent.parameters,
                    "ok": ok,
                    "alive": adapter.is_alive(),
                }
            )
            if not adapter.is_alive():
                break
    finally:
        final_obs = None
        try:
            final_obs = await adapter.observe()
        except (RuntimeError, OSError, ValueError):
            final_obs = None
        await adapter.close()

    elapsed = time.monotonic() - started
    survived = steps and steps[-1]["alive"]
    report = {
        "schema": "aura.nethack_live_proof.v1",
        "run_id": run_id,
        "mode": "STRICT_REAL",
        "binary": adapter.nethack_path,
        "requested_turns": turns,
        "turns_executed": len(steps),
        "commands_ok": executed_ok,
        "distinct_screens": len(screens_seen),
        "hp_first": hp_readings[0] if hp_readings else None,
        "hp_last": hp_readings[-1] if hp_readings else None,
        "survived_full_run": bool(survived and len(steps) == turns),
        "elapsed_s": round(elapsed, 2),
        "final_screen_tail": (final_obs.text or "").splitlines()[-3:] if final_obs else [],
        "steps": steps,
        "at_unix": time.time(),
    }
    report["passed"] = bool(
        report["survived_full_run"]
        and report["commands_ok"] == report["turns_executed"]
        and report["distinct_screens"] >= 3
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=60)
    args = parser.parse_args()

    try:
        report = asyncio.run(run_proof(args.turns))
    except (RuntimeError, OSError) as exc:
        print(f"ENVIRONMENT UNAVAILABLE: {exc}")
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"turns={report['turns_executed']}/{report['requested_turns']} "
        f"ok={report['commands_ok']} screens={report['distinct_screens']} "
        f"hp={report['hp_first']}→{report['hp_last']} "
        f"survived={report['survived_full_run']} ({report['elapsed_s']}s)"
    )
    print(f"artifact: {OUT_PATH}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
