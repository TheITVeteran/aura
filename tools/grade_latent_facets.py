#!/usr/bin/env python
"""Operator grading for latent-cortex facet judgments — the held-out loop.

Every verifier-guided episode records its winning candidate's per-facet
judgments (facet, satisfied, excerpt) as ungraded Foundry verdicts. This
tool shows them to a human WITH the excerpt and takes the verdict-level
ground truth. Ten grades on a facet activate its reliability weight inside
future episodes' verifiers: a facet whose cue-detector humans keep
overruling gets muted, so "add the word 'because'" stops paying.

Usage:
  .venv/bin/python tools/grade_latent_facets.py list [--domain general]
  .venv/bin/python tools/grade_latent_facets.py grade <verdict_id> pass|fail
  .venv/bin/python tools/grade_latent_facets.py reliability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _foundry():
    from core.brain.verifiers.foundry import get_verifier_foundry

    return get_verifier_foundry()


def _facet_events(foundry) -> dict[str, dict]:
    """verdict_id → event body for latent-facet verdicts, from the ledger."""
    events: dict[str, dict] = {}
    try:
        with open(foundry.events_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                body = event.get("body", event)
                if body.get("event") != "verdict":
                    continue
                verifier = str(body.get("verifier", ""))
                if verifier.startswith("latent_facet_"):
                    events[str(body.get("verdict_id", ""))] = body
    except OSError as exc:
        print(f"cannot read foundry ledger: {exc}", file=sys.stderr)
    return events


def cmd_list(args: argparse.Namespace) -> int:
    foundry = _foundry()
    pending = set(foundry.pending_verdicts(domain=args.domain))
    events = _facet_events(foundry)
    rows = [
        (vid, body)
        for vid, body in events.items()
        if vid in pending
        and (args.domain is None or body.get("domain") == args.domain)
    ]
    if not rows:
        print("no pending latent-facet judgments")
        return 0
    for vid, body in rows[-int(args.limit) :]:
        excerpt = str((body.get("meta") or {}).get("excerpt") or "")
        print(
            f"{vid}  {body.get('verifier','?'):28s} "
            f"domain={body.get('domain','?'):10s} "
            f"said={'PASS' if body.get('hard_pass') else 'FAIL'}"
        )
        print(f"    excerpt: {excerpt or '(none — cue never earned a sentence)'}")
    print(
        f"\n{len(rows)} pending. Grade with:\n"
        "  tools/grade_latent_facets.py grade <verdict_id> pass|fail\n"
        "PASS means the excerpt GENUINELY addresses the facet."
    )
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    foundry = _foundry()
    ok = foundry.grade_verdict(
        args.verdict_id,
        truth_pass=(args.truth == "pass"),
        source="human",
    )
    if not ok:
        print(f"verdict {args.verdict_id} is not pending (already graded or unknown)")
        return 1
    foundry.flush_ledger()
    print(f"graded {args.verdict_id}: truth={args.truth}")
    return 0


def cmd_reliability(_args: argparse.Namespace) -> int:
    from core.brain.llm.latent_cortex.task_verifiers import _ANSWER_FACET_HINTS

    foundry = _foundry()
    status = foundry.status()
    printed = False
    for cell in status.get("cells", []):
        verifier = str(cell.get("verifier", ""))
        if not verifier.startswith("latent_facet_"):
            continue
        printed = True
        print(
            f"{verifier:28s} domain={cell.get('domain','?'):10s} "
            f"recorded={cell.get('recorded',0):4d} graded={cell.get('graded',0):4d} "
            f"accuracy_lb={cell.get('accuracy_lb', 0.0):.3f} "
            f"weight={foundry.weight_for(verifier, str(cell.get('domain','?'))):.3f}"
        )
    if not printed:
        print(
            "no latent-facet reliability cells yet "
            f"(facets: {', '.join(sorted(_ANSWER_FACET_HINTS))})"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list", help="show pending facet judgments")
    list_parser.add_argument("--domain", default=None)
    list_parser.add_argument("--limit", default=25)
    list_parser.set_defaults(func=cmd_list)
    grade_parser = sub.add_parser("grade", help="grade one judgment")
    grade_parser.add_argument("verdict_id")
    grade_parser.add_argument("truth", choices=["pass", "fail"])
    grade_parser.set_defaults(func=cmd_grade)
    reliability_parser = sub.add_parser(
        "reliability", help="show facet reliability cells"
    )
    reliability_parser.set_defaults(func=cmd_reliability)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
