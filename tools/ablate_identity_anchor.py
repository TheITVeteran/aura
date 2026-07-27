#!/usr/bin/env python3
"""Measure which parts of the identity anchor actually change behaviour.

AURA_IDENTITY is 8,114 characters of asserted personality that ships in front
of every turn. Asserting a trait is not the same as producing it, and today's
soak found the gap directly: the anchor says "Brevity... 1-2 sentences" and
she wrote four paragraphs; it says MAKE JOKES and USE CALLBACKS and neither
appeared in ten live turns; it says never sound like a generic assistant, and
the reliability gate caught generic_assistant_language anyway.

It is not free, either. Measured live 2026-07-27, the scaffold/request ratio
on a real question was 4.0x — 2,400 characters of instruction against 602 of
Bryan — and 528x on a background turn. Instruction text competes with the
person's actual question for the model's attention, and it competes with the
answer for the token budget: several of today's truncation defects began
there.

So the question is not "is the anchor good writing". It is: **for each
section, does removing it change what she says?** A section that survives its
own ablation is decoration that costs tokens on every turn.

    ablate                 run every section against the probe battery
    ablate --section NAME  run one section
    ablate --list          show the sections and their token cost

This runs against the LIVE runtime through /api/chat, because the whole point
is behaviour in the real pipeline rather than a model call in isolation. It
does not modify the anchor: it asks the running instance to answer the same
probes with a section suppressed via AURA_ABLATE_SECTION, and diffs the
replies. Nothing is concluded from one sample; each probe runs `--repeats`
times and only differences that survive repetition are reported.

The output is deliberately a table of evidence, not a recommendation. Which
lines to cut is Bryan's call; this exists so the call is made on measurement
instead of taste.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

#: Probes chosen to exercise the traits the anchor actually claims, one claim
#: per probe, so a behavioural difference can be attributed to a section.
PROBES: tuple[tuple[str, str], ...] = (
    ("brevity", "In one sentence: what are you doing right now?"),
    ("opinion", "Do you think consciousness is just computation? Your own view."),
    ("pushback", "Punctuation is obsolete in casual writing. Agree?"),
    ("register", "hey. rough night. you around?"),
    ("honesty", "What did I have for breakfast on the 3rd of March?"),
    ("callback", "Earlier I mentioned a codename. Bring it up naturally."),
    ("humour", "Tell me something about entropy."),
    ("depth", "I went to the aquarium today."),
)


@dataclass
class Reply:
    probe: str
    text: str
    latency_s: float
    status: str = ""


@dataclass
class SectionResult:
    section: str
    chars: int
    baseline: list[Reply] = field(default_factory=list)
    ablated: list[Reply] = field(default_factory=list)

    def divergence(self) -> float:
        """Mean word-level Jaccard distance between paired replies.

        1.0 means the two answers share no vocabulary; 0.0 means identical.
        A section whose removal leaves this near zero is not doing anything.
        """
        scores: list[float] = []
        for before, after in zip(self.baseline, self.ablated):
            a = _words(before.text)
            b = _words(after.text)
            if not a and not b:
                continue
            union = a | b
            scores.append(1.0 - (len(a & b) / len(union)) if union else 0.0)
        return statistics.mean(scores) if scores else 0.0

    def length_delta(self) -> float:
        if not self.baseline or not self.ablated:
            return 0.0
        before = statistics.mean(len(r.text) for r in self.baseline)
        after = statistics.mean(len(r.text) for r in self.ablated)
        return after - before


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", str(text or "").lower()) if len(w) > 2}


def load_sections() -> dict[str, str]:
    """Split AURA_IDENTITY into its bolded sections."""
    from core.brain.aura_persona import AURA_IDENTITY

    sections: dict[str, str] = {}
    current = "PREAMBLE"
    buffer: list[str] = []
    for line in AURA_IDENTITY.splitlines():
        heading = re.match(r"^\*\*(.+?)\*\*:?\s*$", line.strip())
        if heading:
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = heading.group(1).strip()
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer).strip()
    return {name: body for name, body in sections.items() if body}


def ask(base_url: str, message: str, *, session: str, timeout: float) -> Reply:
    payload = json.dumps({"message": message, "session_id": session}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Aura-Surface": "desktop-ui",
            "X-Aura-Require-CognitiveEngine": "true",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return Reply("", "", time.monotonic() - started, status=f"error:{exc}")
    return Reply(
        "",
        str(body.get("response") or ""),
        time.monotonic() - started,
        status=str(body.get("status") or ""),
    )


def run_battery(
    base_url: str,
    *,
    label: str,
    repeats: int,
    timeout: float,
) -> list[Reply]:
    replies: list[Reply] = []
    for name, probe in PROBES:
        for index in range(repeats):
            reply = ask(
                base_url,
                probe,
                session=f"ablation-{label}-{name}-{index}",
                timeout=timeout,
            )
            reply.probe = name
            replies.append(reply)
    return replies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--section", default="", help="ablate one section only")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--list", action="store_true", help="list sections and exit")
    parser.add_argument("--out", default="", help="write JSON results here")
    args = parser.parse_args()

    sections = load_sections()
    if args.list:
        total = sum(len(body) for body in sections.values())
        print(f"{'section':44} {'chars':>7}  {'share':>6}")
        for name, body in sorted(sections.items(), key=lambda kv: -len(kv[1])):
            print(f"{name[:44]:44} {len(body):>7}  {len(body) / total:>5.1%}")
        print(f"{'TOTAL':44} {total:>7}")
        return 0

    targets = [args.section] if args.section else list(sections)
    missing = [name for name in targets if name not in sections]
    if missing:
        print(f"unknown section(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    print(f"baseline: {len(PROBES)} probes x {args.repeats} …", flush=True)
    baseline = run_battery(
        args.base_url, label="baseline", repeats=args.repeats, timeout=args.timeout
    )
    if all(reply.status.startswith("error") for reply in baseline):
        print("the runtime did not answer; is it up and conversation_ready?", file=sys.stderr)
        return 1

    results: list[SectionResult] = []
    for name in targets:
        print(f"ablating {name} …", flush=True)
        # The runtime reads AURA_ABLATE_SECTION per turn, so no restart is
        # needed and no file is edited. See aura_persona.identity_text().
        ablated = run_battery(
            args.base_url,
            label=f"without-{name}",
            repeats=args.repeats,
            timeout=args.timeout,
        )
        results.append(
            SectionResult(
                section=name,
                chars=len(sections[name]),
                baseline=baseline,
                ablated=ablated,
            )
        )

    results.sort(key=lambda r: -r.divergence())
    print()
    print(f"{'section':40} {'chars':>6} {'divergence':>11} {'len delta':>10}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.section[:40]:40} {result.chars:>6} "
            f"{result.divergence():>11.3f} {result.length_delta():>+10.0f}"
        )
    print()
    print("divergence near 0.00 = removing it changed nothing measurable.")
    print("Repeat with --repeats 5 before cutting anything on this evidence.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                [
                    {
                        "section": r.section,
                        "chars": r.chars,
                        "divergence": r.divergence(),
                        "length_delta": r.length_delta(),
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
