"""Schedule-search campaign: verified outcomes into the LIVE schedule library.

The last seam RLC_WIRING_HANDOFF.md names for component 3: `ScheduleSearch`
and `ScheduleLibrary` are live (the engine consults `best_for_domain` on
every episode), but nothing ever fed `record_paired_outcome` with REAL
verified task scores — so promotion could never happen and the library was
a mechanism present without firing.

This driver closes the loop end-to-end, operator-launched and bounded:

    seeded curriculum tasks (search / holdout DISJOINT, commitment hashed
    before any evaluation)
        → evolutionary ScheduleSearch scored by verified outcomes on the
          search split (holdout evaluator separate — same callable refused)
        → PAIRED candidate-vs-default trials on the holdout split,
          alternating run order, schedule the ONLY variable (latent opt and
          fast weights disabled in both arms)
        → PairedScheduleOutcome.create(...) with full provenance
        → library.record_paired_outcome(...)
        → the live engine's next `_resolve_schedule` sees the evidence.

Scoring reuses the accuracy ladder's calibrated scorer — the one whose
harness faults RAISE instead of manufacturing 0% — via module loading, not
a parallel implementation.

Memory: run with a small checkpoint (1.5B/7B). Never launch beside a
resident 32B or an active 32B training run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_SCHEMA = "aura.schedule_search_campaign.v1"


def _load_ladder():
    """The calibrated scorer lives in the ladder tool; load it, don't fork it."""
    ladder_path = Path(__file__).resolve().parent / "rlc_accuracy_ladder.py"
    spec = importlib.util.spec_from_file_location("rlc_accuracy_ladder", ladder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_manifest_sha256(model_dir: Path) -> str:
    """Cheap, stable checkpoint identity: config + weight-file manifest.

    Not a full-weight hash (that is the frontier evidence path's job); this
    binds the outcome to a specific checkpoint DIRECTORY STATE and is
    labeled as a manifest hash in the receipt.
    """
    rows = []
    for item in sorted(model_dir.glob("*")):
        if item.is_file() and item.suffix in {".json", ".safetensors"}:
            stat = item.stat()
            rows.append({"name": item.name, "size": stat.st_size})
    if not rows:
        raise ValueError(f"no checkpoint files found under {model_dir}")
    config = model_dir / "config.json"
    return _canonical_sha256(
        {
            "files": rows,
            "config_sha256": _file_sha256(config) if config.exists() else "",
        }
    )


def _task_manifest(tasks) -> list[dict[str, Any]]:
    return [
        {
            "family": task.family,
            "depth": task.depth,
            "seed": task.seed,
            "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
        }
        for task in tasks
    ]


def _build_engine(model, tokenizer, schedule_dict, *, n_slots: int, max_steps: int):
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    # Schedule attribution requires the schedule to be the ONLY variable:
    # latent optimization and fast weights stay off in BOTH arms.
    config = CortexConfig(
        workspace=WorkspaceConfig(n_slots=n_slots, seed=11),
        recurrence=RecurrenceConfig(max_steps=max_steps, min_steps=1),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        schedule=schedule_dict,
        decode_max_tokens=96,
        decode_temperature=0.0,
        telemetry_enabled=False,
    )
    return LatentCortexEngine(model, tokenizer, config)


def _run_task(engine, ladder, task, *, budget_layer_apps: int) -> tuple[bool, str, int]:
    """One verified episode: (success, outcome, layer_apps_spent)."""
    from core.brain.llm.latent_cortex.types import ComputeBudget

    budget = ComputeBudget(max_layer_apps=budget_layer_apps, wall_clock_s=180.0)
    result = engine.reason(
        messages=[{"role": "user", "content": task.prompt}],
        budget=budget,
    )
    outcome = ladder._score(task, result.text or "")
    success = outcome in {"correct", "correct_lenient"}
    return success, outcome, int(budget.spent_layer_apps)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_lm import load

    from core.brain.llm.latent_cortex.schedules import (
        LayerSchedule,
        PairedScheduleOutcome,
        ScheduleComputeReceipt,
        ScheduleLibrary,
        ScheduleSearch,
    )
    from core.learning import recurrence_curriculum as curriculum

    ladder = _load_ladder()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    search_tasks = curriculum.task_battery(
        families, [args.task_depth], args.search_per_cell, seed=args.seed
    )
    holdout_tasks = curriculum.task_battery(
        families, [args.task_depth], args.holdout_per_cell, seed=args.seed + 1000
    )
    search_prompts = {task.prompt for task in search_tasks}
    holdout_prompts = {task.prompt for task in holdout_tasks}
    overlap = search_prompts & holdout_prompts
    if overlap:
        raise ValueError(
            f"{len(overlap)} holdout prompts overlap the search split; "
            "a schedule selected on its own scoring tasks is an answer key"
        )
    # Commit the holdout manifest BEFORE any model evaluation. Each trial
    # then carries a PER-TASK commitment derived from this manifest hash —
    # the record layer refuses replayed commitments, so a shared one would
    # reject every trial after the first.
    holdout_manifest = _task_manifest(holdout_tasks)
    manifest_commitment_sha256 = _canonical_sha256(
        {
            "holdout": holdout_manifest,
            "search_count": len(search_tasks),
            "seed": args.seed,
        }
    )

    def task_commitment(index: int) -> str:
        return _canonical_sha256(
            {
                "manifest": manifest_commitment_sha256,
                "task": holdout_manifest[index],
                "index": index,
            }
        )

    from core.runtime.model_lane_control import standalone_model_lane

    model_dir = Path(args.model).expanduser().resolve()
    with standalone_model_lane(
        owner_id=f"schedule-search-campaign:{Path(args.out).name}",
        model_path=str(model_dir),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "schedule_search_campaign", "operator_launched": True},
    ):
        model, tokenizer = load(str(model_dir))
    tool_sha = _file_sha256(Path(__file__).resolve())
    model_sha = _model_manifest_sha256(model_dir)
    run_id = f"schedule-campaign-{args.seed}-{uuid.uuid4().hex[:12]}"
    protocol_sha = _canonical_sha256(
        {
            "schema": CAMPAIGN_SCHEMA,
            "families": families,
            "task_depth": args.task_depth,
            "seed": args.seed,
            "population": args.population,
            "generations": args.generations,
            "prelude_end": args.prelude_end,
            "coda_start": args.coda_start,
            "domain": args.domain,
        }
    )

    default_schedule = LayerSchedule.single_window(
        args.prelude_end, args.coda_start, args.default_repeats
    )

    def evaluate_on(tasks, schedule: LayerSchedule) -> float:
        engine = _build_engine(
            model,
            tokenizer,
            schedule.to_dict(),
            n_slots=args.n_slots,
            max_steps=max(2, schedule.total_layer_repeats),
        )
        wins = 0
        for task in tasks:
            success, _outcome, _spent = _run_task(
                engine, ladder, task, budget_layer_apps=args.budget_layer_apps
            )
            wins += int(success)
        return wins / max(1, len(tasks))

    def search_evaluator(schedule: LayerSchedule) -> float:
        return evaluate_on(search_tasks, schedule)

    def holdout_evaluator(schedule: LayerSchedule) -> float:
        return evaluate_on(holdout_tasks, schedule)

    search = ScheduleSearch(
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        max_repeats=args.max_repeats,
        seed=args.seed,
    )
    result = search.run(
        search_evaluator,
        population=args.population,
        generations=args.generations,
        seed_schedule=default_schedule,
        holdout_evaluator=holdout_evaluator,
        max_layer_apps=args.max_schedule_layer_repeats,
    )
    winner = result.best

    # ── Paired holdout trials: candidate vs default, alternating order ──
    outcomes: list[dict[str, Any]] = []
    library = ScheduleLibrary(Path(args.library) if args.library else None)
    candidate_engine = _build_engine(
        model, tokenizer, winner.to_dict(),
        n_slots=args.n_slots, max_steps=max(2, winner.total_layer_repeats),
    )
    default_engine = _build_engine(
        model, tokenizer, default_schedule.to_dict(),
        n_slots=args.n_slots, max_steps=max(2, default_schedule.total_layer_repeats),
    )
    recorded = 0
    # CP126 78c85746: the search is SEEDED with the default, so it can quite
    # legitimately conclude the default is best. Running paired trials then
    # compares a schedule against itself and credits it with "wins" over its
    # own results — which is exactly the baseline-aliasing this binding
    # exists to prevent. There is nothing to promote, and the receipt says so
    # rather than manufacturing three tie trials.
    winner_is_default = winner.schedule_hash == default_schedule.schedule_hash
    # NB: holdout_tasks is left intact — how many tasks were AVAILABLE is a
    # fact about the run, independent of whether trials were worth running.
    trial_tasks = [] if winner_is_default else holdout_tasks
    for index, task in enumerate(trial_tasks):
        candidate_first = index % 2 == 0
        first_engine, second_engine = (
            (candidate_engine, default_engine)
            if candidate_first
            else (default_engine, candidate_engine)
        )
        first = _run_task(first_engine, ladder, task, budget_layer_apps=args.budget_layer_apps)
        second = _run_task(second_engine, ladder, task, budget_layer_apps=args.budget_layer_apps)
        candidate_run, default_run = (
            (first, second) if candidate_first else (second, first)
        )
        task_id = f"{task.family}-d{task.depth}-s{task.seed}"
        scorer_receipt = {
            "task_id": task_id,
            "candidate_outcome": candidate_run[1],
            "default_outcome": default_run[1],
            "scorer": "rlc_accuracy_ladder._score",
        }
        outcome = PairedScheduleOutcome.create(
            schedule_hash=winner.schedule_hash,
            domain=args.domain,
            task_id=task_id,
            task_commitment_sha256=task_commitment(index),
            candidate_success=candidate_run[0],
            default_success=default_run[0],
            candidate_compute=ScheduleComputeReceipt(
                layer_apps=max(1, candidate_run[2]), estimator_sha256=tool_sha
            ),
            default_compute=ScheduleComputeReceipt(
                layer_apps=max(1, default_run[2]), estimator_sha256=tool_sha
            ),
            run_order=(
                "candidate_first" if candidate_first else "default_first"
            ),
            held_out=True,
            contamination_scan_passed=True,  # exact prompt disjointness, refused above
            scorer_receipt_sha256=_canonical_sha256(scorer_receipt),
            verifier_receipt_sha256=_canonical_sha256(
                {"task_id": task_id, "graded_at": "deterministic", "seed": task.seed}
            ),
            evaluation_run_id=run_id,
            evaluator_build_sha256=tool_sha,
            model_checkpoint_sha256=model_sha,
            evidence_protocol_sha256=protocol_sha,
            # CP126 78c85746: name the baseline this candidate was actually
            # compared against, so trials against different defaults cannot
            # aggregate as one comparator.
            default_schedule_hash=default_schedule.schedule_hash,
        )
        library.record_paired_outcome(winner, args.domain, outcome)
        recorded += 1
        outcomes.append(scorer_receipt)

    saved = library.save()
    if args.library and not saved:
        raise RuntimeError(
            "schedule library did not persist; recorded evidence would "
            "evaporate with this process"
        )
    promoted = library.best_for_domain(
        args.domain,
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        default_repeats=args.default_repeats,
    )
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": run_id,
        "seed": args.seed,
        "families": families,
        "search_tasks": len(search_tasks),
        "holdout_tasks": len(holdout_tasks),
        "task_commitment_sha256": manifest_commitment_sha256,
        "winner_schedule_hash": winner.schedule_hash,
        "winner_ops": winner.to_dict(),
        "search_score": result.best_score,
        "holdout_score": result.holdout_score,
        "generalization_gap": result.generalization_gap(),
        "overfit_warning": result.overfit_warning(),
        "paired_outcomes_recorded": recorded,
        "winner_is_default": winner_is_default,
        "paired_trials_skipped_reason": (
            "search returned the seed default; a schedule cannot be paired "
            "against itself" if winner_is_default else ""
        ),
        "library_status": library.status(),
        "promoted_schedule_hash": promoted.schedule_hash,
        "promotion_happened": promoted.schedule_hash != default_schedule.schedule_hash,
        "finished_at": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--library", default="data/latent_cortex/schedule_library.json")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=int, default=8)
    parser.add_argument("--search-per-cell", type=int, default=8)
    parser.add_argument("--holdout-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--prelude-end", type=int, default=16)
    parser.add_argument("--coda-start", type=int, default=48)
    parser.add_argument("--default-repeats", type=int, default=2)
    parser.add_argument("--max-repeats", type=int, default=8)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--budget-layer-apps", type=int, default=2_000_000)
    parser.add_argument("--max-schedule-layer-repeats", type=int, default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    try:
        receipt = run_campaign(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: every failure becomes exit code + stderr
        print(
            f"schedule_search_campaign: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    rendered = json.dumps(receipt, indent=1, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).expanduser().write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
