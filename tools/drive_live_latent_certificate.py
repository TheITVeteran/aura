#!/usr/bin/env python
"""Drive one compound latent turn through the LIVE desktop app → certificate.

Sends a real user message to the installed app's /api/chat (desktop surface),
extracts the latent-cortex trace + episode receipt from the live response,
grades the PASS conditions the CP-series certificates use, and writes
artifacts/current/cp<NN>_live_latent_turn.json.

Owner-operated, bounded, and honest: a turn that fell back, truncated, or
failed any receipt contract writes verdict FAIL with the reasons — never a
silent pass. Run only when the live app is up and idle.

  .venv/bin/python tools/drive_live_latent_certificate.py --checkpoint 119 \
      [--message "..."] [--timeout 240]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MESSAGE = (
    "Compare how your latent workspace and your ordinary decode path handle a "
    "multi-part question, choose which one you would trust for high-stakes "
    "arithmetic, explain why, and verify your choice with one concrete check."
)


def _git_head() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    session_id = f"cp{args.checkpoint}-live-latent"
    payload = json.dumps(
        {"message": args.message, "session_id": session_id}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{args.host}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Aura-Surface": "desktop",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - operator tool: report, don't mask
        print(f"live turn failed: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.monotonic() - started

    receipt = body.get("latent_cortex_receipt") or {}
    reasons: list[str] = []
    if status != 200:
        reasons.append(f"http_{status}")
    if body.get("latent_cortex_succeeded") is not True:
        reasons.append("latent_cortex_not_succeeded")
    if body.get("latent_cortex_identity_bound") is not True:
        reasons.append("identity_not_bound")
    if receipt.get("params_unchanged") is not True:
        reasons.append("params_unchanged_unproven")
    termination = str(receipt.get("decode_termination") or "")
    if termination not in {"eos", "token_limit", "token_limit_sentence_grace"}:
        reasons.append(f"decode_termination:{termination or 'missing'}")
    if receipt.get("fast_weights_applied") and receipt.get("fast_weights_erased") is not True:
        reasons.append("fast_weight_erase_unproven")
    text = str(body.get("response") or "")
    if len(text.split()) < 30:
        reasons.append("answer_too_thin")

    verdict = "PASS" if not reasons else "FAIL"
    certificate = {
        "schema": "aura.live_latent_certificate.v1",
        "checkpoint": args.checkpoint,
        "exact_commit": _git_head(),
        "request": {"message": args.message, "session_id": session_id},
        "http": {"status": status, "elapsed_s": round(elapsed, 3)},
        "response": {
            "response": text[:4000],
            "status": body.get("status"),
            "latent_cortex_succeeded": body.get("latent_cortex_succeeded"),
            "latent_cortex_identity_bound": body.get("latent_cortex_identity_bound"),
            "latent_cortex_output_quality_proven": body.get(
                "latent_cortex_output_quality_proven"
            ),
            "authentic_cognitive_reply": body.get("authentic_cognitive_reply"),
        },
        "latent_receipt": receipt,
        "fail_reasons": reasons,
        "verdict": verdict,
        "generated_at": time.time(),
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts" / "current" / f"cp{args.checkpoint}_live_latent_turn.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(certificate, indent=1, sort_keys=True))
    print(f"verdict={verdict} elapsed={elapsed:.1f}s reasons={reasons or 'none'}")
    print(f"📄 certificate → {out_path}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
