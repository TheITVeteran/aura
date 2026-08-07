#!/usr/bin/env python
"""Autonomous decision pipeline for the RLC reconciliation campaign.

Runs the phases in order, gates each on the previous one's evidence, and
writes a decision nobody has to be present to make. Every gate fails closed:
a phase that cannot prove its predecessor's precondition writes the decision
and exits zero rather than proceeding on hope.

  1 frozen sweep      can the recurrent path reach an ordinary decode at all?
  2 checkpoint sweep  does any retained adapter beat the frozen path?
  3 fusion candidate  does merging that adapter leave the model better?
  4 decision          what is true, and what was therefore done.

Phase 3 exists because "fuse the adapter" is not a merge on this program. The
recurrence adapter is a ScopedLoRALinear: its delta applies at latent slot
positions and nowhere else. Standard fusion folds that delta into the linear
weights unconditionally, so the fused model is a different function on every
ordinary token too. A fused candidate is therefore a new model that has to
re-earn ordinary decode, not a packaging step -- and this pipeline refuses to
activate one that regresses it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PIPELINE_SCHEMA = "aura.rlc_reconciliation_pipeline.v1"


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _phase_status(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "pipeline_status.json"
    body = _read_json(path) or {}
    body.update(fields)
    body["heartbeat_unix"] = _now()
    body["pid"] = os.getpid()
    _atomic_write(path, json.dumps(body, indent=1, sort_keys=True) + "\n")


def _wait_for_sweep(run_dir: Path, sweep_dir: Path, *, timeout_s: float) -> dict | None:
    """Wait on the already-detached sweep. Stall is a result, not a hang."""
    deadline = time.monotonic() + timeout_s
    last_committed = -1
    stalled_since = time.monotonic()
    while time.monotonic() < deadline:
        verdict = _read_json(sweep_dir / "verdict.json")
        if verdict is not None:
            return verdict
        status = _read_json(sweep_dir / "status.json") or {}
        committed = int(status.get("committed_cells", -1))
        if committed != last_committed:
            last_committed = committed
            stalled_since = time.monotonic()
            _phase_status(
                run_dir,
                phase="waiting_for_sweep",
                sweep_committed_cells=committed,
                sweep_phase=status.get("phase"),
            )
        elif time.monotonic() - stalled_since > 3600.0:
            _log(run_dir, f"sweep stalled at {committed} cells for over an hour")
            return None
        time.sleep(60.0)
    _log(run_dir, "sweep wait timed out")
    return None


def _run(run_dir: Path, argv: list[str], *, timeout_s: float) -> int:
    _log(run_dir, "run: " + " ".join(argv))
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=str(REPO_ROOT),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _log(run_dir, "phase exceeded its wall budget")
        return 124
    return completed.returncode


def _decide(run_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    body = {"schema": PIPELINE_SCHEMA, "decided_unix": _now(), **body}
    _atomic_write(run_dir / "DECISION.json", json.dumps(body, indent=1, sort_keys=True) + "\n")
    lines = [
        "# RLC reconciliation decision",
        "",
        f"**Decision:** `{body['decision']}`",
        "",
        body.get("summary", ""),
        "",
        "## Evidence",
        "",
    ]
    for key, value in sorted(body.get("evidence", {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Not claimed",
        "",
        "Nothing here awards a reasoning gain, a frontier result, or a",
        "production activation. Those require the preregistered powered",
        "campaign with independent trust roots, which this pipeline does not",
        "run and cannot authorize.",
        "",
    ]
    _atomic_write(run_dir / "DECISION.md", "\n".join(lines))
    _log(run_dir, f"DECISION: {body['decision']}")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--sweep-timeout-s", type=float, default=72_000.0)
    parser.add_argument("--phase-timeout-s", type=float, default=36_000.0)
    # A candidate re-validated on its selection seed would be grading its own
    # homework, so activation is decided on questions it has not seen.
    parser.add_argument("--revalidation-seed", type=int, default=20260814)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = run_dir / "sweep"

    _phase_status(run_dir, phase="started", started_unix=_now())
    _log(run_dir, "pipeline started")

    # ── Phase 1 ──────────────────────────────────────────────────────────
    sweep_verdict = _read_json(sweep_dir / "verdict.json")
    if sweep_verdict is None:
        sweep_verdict = _wait_for_sweep(
            run_dir, sweep_dir, timeout_s=args.sweep_timeout_s
        )
    if sweep_verdict is None:
        _decide(
            run_dir,
            {
                "decision": "incomplete_sweep_did_not_finish",
                "summary": (
                    "The frozen sweep did not produce a verdict inside its "
                    "budget. Nothing is concluded about the recurrent path, "
                    "and no weights were touched. Resume by re-running "
                    "launch_sweep.sh, which skips every committed cell."
                ),
                "evidence": {"sweep_verdict": "absent"},
            },
        )
        return 1

    arms = sweep_verdict.get("arms", {})
    scores = {name: bucket.get("correct") for name, bucket in arms.items()}
    _log(run_dir, f"sweep scores: {scores}")
    _phase_status(run_dir, phase="sweep_complete", sweep_scores=scores)

    if not sweep_verdict.get("reaches_parity_with_ordinary_decode"):
        _decide(
            run_dir,
            {
                "decision": "no_fusion_recurrent_path_below_ordinary_decode",
                "summary": (
                    "On frozen weights, no recurrent configuration answered as "
                    "many held-out questions as an ordinary decode of the same "
                    "checkpoint. Training against this path would start in a "
                    "hole no adapter has been shown to climb out of, so no "
                    "checkpoint was promoted and no weights were fused. The "
                    "arm scores below say which factor -- terminal-disposition "
                    "injection or recurrent depth -- carries the deficit."
                ),
                "evidence": {
                    "arm_scores": scores,
                    "vanilla_correct": sweep_verdict.get("vanilla_correct"),
                    "best_recurrent_arm": sweep_verdict.get("best_recurrent_arm"),
                    "best_recurrent_correct": sweep_verdict.get("best_recurrent_correct"),
                },
            },
        )
        return 0

    # ── Phase 2 ──────────────────────────────────────────────────────────
    if not args.checkpoint_root:
        _decide(
            run_dir,
            {
                "decision": "parity_reached_no_checkpoint_root_configured",
                "summary": (
                    "A recurrent configuration reached parity with ordinary "
                    "decode on frozen weights, which is the precondition this "
                    "program has never met before. No checkpoint root was "
                    "configured, so no adapter was evaluated and no weights "
                    "were fused."
                ),
                "evidence": {"arm_scores": scores},
            },
        )
        return 0

    _phase_status(run_dir, phase="checkpoint_sweep")
    code = _run(
        run_dir,
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_rlc_checkpoint_sweep.py"),
            "--model",
            args.model,
            "--checkpoint-root",
            args.checkpoint_root,
            "--out-dir",
            str(run_dir / "checkpoints"),
            "--reference-arm",
            str(sweep_verdict.get("best_recurrent_arm") or "rlc_nodisp"),
            "--vanilla-correct",
            str(int(sweep_verdict.get("vanilla_correct") or 0)),
        ],
        timeout_s=args.phase_timeout_s,
    )
    checkpoint_verdict = _read_json(run_dir / "checkpoints" / "verdict.json")
    if code != 0 or checkpoint_verdict is None:
        _decide(
            run_dir,
            {
                "decision": "incomplete_checkpoint_sweep_failed",
                "summary": (
                    "The frozen path reached parity, but the retained-checkpoint "
                    "sweep did not produce a verdict. No weights were fused."
                ),
                "evidence": {"arm_scores": scores, "checkpoint_sweep_exit": code},
            },
        )
        return 0

    if not checkpoint_verdict.get("beats_ordinary_decode"):
        _decide(
            run_dir,
            {
                "decision": "no_fusion_no_checkpoint_beats_ordinary_decode",
                "summary": (
                    "The recurrent path reached parity on frozen weights, but no "
                    "retained adapter out-answered an ordinary decode. There is "
                    "nothing to fuse: fusing an adapter that does not beat the "
                    "model it modifies would make the model worse."
                ),
                "evidence": {
                    "arm_scores": scores,
                    "best_checkpoint": checkpoint_verdict.get("best_checkpoint"),
                    "best_checkpoint_correct": checkpoint_verdict.get("best_correct"),
                    "vanilla_correct": sweep_verdict.get("vanilla_correct"),
                },
            },
        )
        return 0

    # ── Phase 3 ──────────────────────────────────────────────────────────
    # A positive checkpoint is a candidate, not a promotion. Fusing a
    # slot-scoped adapter changes ordinary decode too, so the fused model
    # re-earns both arms on a fresh seed before anything is activated.
    _phase_status(run_dir, phase="fusing")
    candidate_dir = run_dir / "fused_candidate"
    best_checkpoint = str(checkpoint_verdict.get("best_checkpoint") or "")
    checkpoint_path = Path(args.checkpoint_root) / best_checkpoint
    code = _run(
        run_dir,
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "fuse_rlc_candidate.py"),
            "--model",
            args.model,
            "--adapter",
            str(checkpoint_path),
            "--out",
            str(candidate_dir),
        ],
        timeout_s=args.phase_timeout_s,
    )
    if code != 0 or not (candidate_dir / "config.json").exists():
        _decide(
            run_dir,
            {
                "decision": "no_fusion_candidate_merge_failed",
                "summary": (
                    "A retained adapter beat ordinary decode, but merging it "
                    "into the base weights failed. The resident is untouched."
                ),
                "evidence": {"arm_scores": scores, "fuse_exit": code},
            },
        )
        return 0

    # Re-earn both arms on a seed the candidate has never seen.
    _phase_status(run_dir, phase="revalidating_candidate")
    revalidation_dir = run_dir / "candidate_sweep"
    code = _run(
        run_dir,
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_rlc_reconciliation_sweep.py"),
            "--model",
            str(candidate_dir),
            "--out-dir",
            str(revalidation_dir),
            "--seed",
            str(int(args.revalidation_seed)),
            "--arms",
            "vanilla," + str(sweep_verdict.get("best_recurrent_arm") or "rlc_nodisp"),
        ],
        timeout_s=args.phase_timeout_s,
    )
    candidate_verdict = _read_json(revalidation_dir / "verdict.json")
    if code != 0 or candidate_verdict is None:
        _decide(
            run_dir,
            {
                "decision": "no_activation_candidate_revalidation_incomplete",
                "summary": (
                    "The fused candidate exists but did not finish "
                    "re-validation. It was not activated; the resident is "
                    "untouched."
                ),
                "evidence": {"arm_scores": scores, "revalidation_exit": code},
            },
        )
        return 0

    candidate_arms = candidate_verdict.get("arms", {})
    candidate_vanilla = int(candidate_arms.get("vanilla", {}).get("correct", -1))
    candidate_best = int(candidate_verdict.get("best_recurrent_correct", -1))
    resident_vanilla = int(sweep_verdict.get("vanilla_correct") or 0)

    # Fusing folded a slot-scoped delta into every position, so the ordinary
    # lane is the one most likely to have silently regressed. It is the gate.
    ordinary_preserved = candidate_vanilla >= resident_vanilla
    recurrent_gain = candidate_best > resident_vanilla
    if not (ordinary_preserved and recurrent_gain):
        _decide(
            run_dir,
            {
                "decision": "no_activation_fused_candidate_regressed",
                "summary": (
                    "The fused candidate did not re-earn its place. Folding a "
                    "slot-scoped adapter into the base weights changes every "
                    "ordinary token, and on a fresh seed the candidate did not "
                    "hold ordinary decode and improve the recurrent path at the "
                    "same time. The resident is untouched and the candidate is "
                    "retained for inspection."
                ),
                "evidence": {
                    "resident_vanilla_correct": resident_vanilla,
                    "candidate_vanilla_correct": candidate_vanilla,
                    "candidate_recurrent_correct": candidate_best,
                    "ordinary_decode_preserved": ordinary_preserved,
                    "recurrent_gain_reproduced": recurrent_gain,
                    "candidate_path": str(candidate_dir),
                },
            },
        )
        return 0

    _decide(
        run_dir,
        {
            "decision": "fused_candidate_passed_staged_for_activation",
            "summary": (
                "The fused candidate held ordinary decode and improved the "
                "recurrent path on a fresh seed. It is staged at the path "
                "below with the resident preserved byte-for-byte alongside it. "
                "Activation is a next-boot pointer swap and is reversible by "
                "restoring the recorded resident path."
            ),
            "evidence": {
                "resident_vanilla_correct": resident_vanilla,
                "candidate_vanilla_correct": candidate_vanilla,
                "candidate_recurrent_correct": candidate_best,
                "candidate_path": str(candidate_dir),
                "rollback_path": args.model,
                "best_checkpoint": best_checkpoint,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
