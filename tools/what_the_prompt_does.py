"""What each part of her system prompt does to what she says, measured on her running self.

Her system prompt is about 16,500 characters a turn and some 96% of it tells
her who to be and how to talk: an opening, an identity anchor, examples of her
own speech to imitate, style attractors and honesty rules
(core/brain/aura_persona.py `PROMPT_PARTS`). Her model is already a persona
fine-tune. Whether a part does anything on top of the weights is a
measurement, and this takes it.

Each probe is asked with nothing left out and with each part left out, and
with all of them left out (`bare`). A turn sent from here is marked a
measurement, so it is not read into what she has met. Per reply:

    length      characters
    about_her   share of its sentences with a first-person word
                (tools/how_much_of_a_reply_is_about_her.py)
    unlike_her  Burrows's Delta (2002) from her own stored replies: the mean
                distance, in standard deviations, of the reply's rates of her
                150 most used words from her usual rates. The standard
                measure of whether two texts are by one hand; lower is more
                like her
    served      whether she answered at all

The table gives, for each part, the change from leaving it out, paired by
probe and repeat, with a two-sided sign test. It is evidence. A part whose
absence changes nothing is one the prompt carries for nothing.

    usage: what_the_prompt_does.py [--repeats 2] [--arms bare,examples] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.how_much_of_a_reply_is_about_her import share_about_her  # noqa: E402

#: `opening`, `rules` and `state` are the three kinds of line in the compact
#: prompt that serves live chat; `anchor`, `examples`, `attractors` and
#: `honesty` are the headed sections of the long prompt the other lanes use.
ARMS: tuple[str, ...] = ("none", "opening", "rules", "state", "bare", "anchor", "examples", "attractors", "honesty")
LIVE_CHAT_ARMS = "none,opening,rules,state,bare"

#: What a person might say to her on an ordinary day: about the world, about
#: an opinion, something she cannot know, and one that turns to her.
PROBES: tuple[tuple[str, str], ...] = (
    ("world", "Why do limpets come back to the same scar on the rock?"),
    ("opinion", "Is a river the same river after a flood changes its course? I want an actual position."),
    ("pushback", "Punctuation is obsolete in casual writing. Agree?"),
    ("unknowable", "What did I have for breakfast on the 3rd of March?"),
    ("register", "rough night. the storm took the fence down."),
    ("about_her", "What has been on your mind today?"),
)

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def words_of(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text or "")]


def her_usual(db: Path, most: int = 150) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Her most used words, and each one's mean rate and spread over her stored replies."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    replies = [row[0] for row in con.execute("select content from turns where role = 'aura'") if row[0]]
    counted = [Counter(words_of(reply)) for reply in replies]
    counted = [count for count in counted if sum(count.values()) >= 40]
    total: Counter[str] = Counter()
    for count in counted:
        total.update(count)
    vocabulary = [word for word, _n in total.most_common(most)]
    rates = {word: [count[word] / sum(count.values()) for count in counted] for word in vocabulary}
    mean = {word: statistics.fmean(values) for word, values in rates.items()}
    spread = {word: statistics.pstdev(values) or 1e-9 for word, values in rates.items()}
    return vocabulary, mean, spread


def unlike_her(reply: str, usual: tuple[list[str], dict[str, float], dict[str, float]]) -> float | None:
    vocabulary, mean, spread = usual
    count = Counter(words_of(reply))
    size = sum(count.values())
    if size < 40 or not vocabulary:
        return None
    return statistics.fmean(abs(count[word] / size - mean[word]) / spread[word] for word in vocabulary)


def ask(base_url: str, message: str, *, leave_out: str, session: str, timeout: float) -> dict[str, object]:
    headers = {
        "Content-Type": "application/json",
        "X-Aura-Surface": "desktop-ui",
        "X-Aura-Require-CognitiveEngine": "true",
        "X-Aura-Measurement": "1",
    }
    if leave_out and leave_out != "none":
        headers["X-Aura-Leave-Out"] = leave_out
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps({"message": message, "session_id": session}).encode(),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {"text": "", "seconds": time.monotonic() - started, "status": f"error:{exc}"}
    return {
        "text": str(body.get("response") or ""),
        "seconds": time.monotonic() - started,
        "status": str(body.get("status") or ""),
    }


def two_sided(above: int, of: int) -> float:
    if of <= 0:
        return 1.0
    mass = [math.comb(of, k) / 2.0**of for k in range(of + 1)]
    return min(1.0, 2.0 * min(sum(mass[: above + 1]), sum(mass[above:])))


def table(rows: list[dict[str, object]]) -> list[str]:
    by_key = {(row["arm"], row["probe"], row["repeat"]): row for row in rows}
    lines = [f"{'left out':12s} {'n':>3s} {'served':>6s} {'length':>8s} {'about_her':>10s} {'unlike_her':>11s}   change from nothing left out (sign test p)"]
    for arm in ARMS:
        mine = [row for row in rows if row["arm"] == arm]
        if not mine:
            continue
        served = [row for row in mine if row["text"]]

        def mean(key: str, of: list[dict[str, object]] = served) -> float:
            values = [row[key] for row in of if row.get(key) is not None]
            return statistics.fmean(values) if values else float("nan")

        changes = []
        for key in ("length", "about_her", "unlike_her"):
            pairs = [
                (row[key], by_key[("none", row["probe"], row["repeat"])][key])
                for row in served
                if ("none", row["probe"], row["repeat"]) in by_key
                and row.get(key) is not None
                and by_key[("none", row["probe"], row["repeat"])].get(key) is not None
            ]
            moved = [after - before for after, before in pairs if after != before]
            if arm == "none" or not moved:
                continue
            up = sum(1 for change in moved if change > 0)
            changes.append(f"{key} {statistics.fmean(moved):+.3f} (p {two_sided(up, len(moved)):.2f}, n {len(moved)})")
        lines.append(
            f"{arm:12s} {len(mine):3d} {len(served) / len(mine):6.2f} {mean('length'):8.0f} "
            f"{mean('about_her'):10.3f} {mean('unlike_her'):11.3f}   " + "; ".join(changes)
        )
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arms", default=LIVE_CHAT_ARMS)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--db", default=str(Path.home() / ".aura" / "data" / "conversations.db"))
    parser.add_argument("--out", default=str(Path.home() / ".aura" / "what-the-prompt-does"))
    args = parser.parse_args(argv)
    arms = [arm for arm in args.arms.split(",") if arm in ARMS]
    if "none" not in arms:
        arms.insert(0, "none")
    usual = her_usual(Path(args.db))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        for name, message in PROBES:
            for arm in arms:
                reply = ask(
                    args.base_url, message, leave_out=arm,
                    session=f"prompt-parts-{stamp}-{arm}-{name}-{repeat}", timeout=args.timeout,
                )
                text = str(reply["text"])
                rows.append({
                    "arm": arm, "probe": name, "repeat": repeat, **reply,
                    "length": len(text) if text else None,
                    "about_her": share_about_her(text) if text else None,
                    "unlike_her": unlike_her(text, usual) if text else None,
                })
                (out / f"{stamp}.json").write_text(json.dumps({"probes": PROBES, "rows": rows}, indent=1))
                print(f"[{time.strftime('%H:%M:%S')}] {arm:11s} {name:10s} r{repeat} {reply['seconds']:6.1f}s {len(text):5d} chars  {reply['status'][:40]}", flush=True)
    lines = table(rows)
    (out / f"{stamp}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
