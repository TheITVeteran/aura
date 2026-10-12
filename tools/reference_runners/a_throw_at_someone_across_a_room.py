"""Reference runner: a contest of throws at someone across a room, the player not in the picture (Food Bash, case 2).

What it knows, as a person who has read the rules knows it: the pointer aims, a click throws, Space dodges, each side's
bar of cells along the top is what it has left. What it does:

- the room is what stays put: a picture taken as the median of the last few seconds' pictures, so the one running to
  and fro is never part of it;
- what differs from the room, in its biggest piece, is the opponent; its place over the last pictures gives its going;
- a throw goes where the opponent will be when the throw arrives (its going times the lead), and the lead is changed
  by which leads splatted it;
- something that differs from the room and grows from picture to picture is food coming; Space as it grows.

    python -m tools.reference_runners.a_throw_at_someone_across_a_room --seconds 120 --out /tmp/rr/food
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage

from tools.reference_runners.a_game_at_hand import GameAtHand, report

#: Pictures a room is made of, and how far apart in time.
ROOM_OF = 15
ROOM_EVERY_S = 0.25
#: How far a pixel must be from the room to be something in it.
DIFFERS = 40
#: Seconds ahead a throw is aimed, tried in turn and kept by what splats.
LEADS = (0.0, 0.2, 0.4, 0.6)
#: How long after the opponent stops to wind up a dodge is made, in seconds: tried in turn and kept by what spared.
DODGE_AT = (0.0, 0.15, 0.3)
#: Held this long, the dodge key.
DODGE_HELD_S = 0.3
#: How little the opponent may move over a moment and still be standing to wind up, in shares of the picture.
STANDS = 0.006
STANDS_FOR_S = 0.1
#: The least share of the picture something must cover to count, and to be judged as coming.
A_THING = 0.0015
#: The share of the picture something growing must have reached to be coming at her and not someone moving about.
GROWN_TO = 0.13


def _cells_left(picture: np.ndarray, left: float, right: float) -> float | None:
    """How full a bar of orange cells along the top is, from its share of orange among its cells; None where what is
    there is not cells (the screen zoomed, flashed or gone to another)."""
    tall, wide = picture.shape[:2]
    strip = picture[int(0.035 * tall):int(0.06 * tall), int(left * wide):int(right * wide)].astype(int)
    orange = (strip[:, :, 0] > 200) & (strip[:, :, 1] > 120) & (strip[:, :, 1] < 185) & (strip[:, :, 2] < 60)
    white = strip.min(axis=2) > 225
    if (orange | white).mean() < 0.75:
        return None
    return float(orange.sum() / max(1, (orange | white).sum()))


class Tracks:
    """Each thing that differs from the room, followed from picture to picture by where it is."""

    def __init__(self) -> None:
        self.tracks: dict[int, deque] = {}
        self.next = 0

    def saw(self, now: float, blobs: list[tuple[float, float, float]]) -> None:
        taken: set[int] = set()
        for x, y, area in sorted(blobs, key=lambda b: -b[2]):
            best, best_d = None, None
            for number, track in self.tracks.items():
                if number in taken:
                    continue
                _t, tx, ty, ta = track[-1]
                d = math.hypot(x - tx, y - ty)
                if d < 0.6 * math.sqrt(max(area, ta)) + 0.05 and (best_d is None or d < best_d):
                    best, best_d = number, d
            if best is None:
                best = self.next
                self.next += 1
                self.tracks[best] = deque(maxlen=12)
            self.tracks[best].append((now, x, y, area))
            taken.add(best)
        for number in [n for n in self.tracks if n not in taken]:
            if now - self.tracks[number][-1][0] > 0.3:
                del self.tracks[number]

    def coming(self, number: int) -> bool:
        """Something grown on each of the last two pictures, by more than a share each time, and big by now: what flies
        at the view. What shows up big and shrinks is going away (her own throw); what wavers in size is someone
        moving about."""
        track = self.tracks[number]
        if len(track) < 3:
            return False
        a0, a1, a2 = track[-3][3], track[-2][3], track[-1][3]
        return a1 > 1.15 * a0 and a2 > 1.3 * a1 and a2 > GROWN_TO

    def contact_in(self, number: int) -> float | None:
        """Seconds until a thing that grows fills the view: its size over how fast its size grows (Lee 1976)."""
        track = self.tracks[number]
        if len(track) < 3:
            return None
        (t0, _x0, _y0, a0), (t1, _x1, _y1, a1) = track[-3], track[-1]
        s0, s1 = math.sqrt(a0), math.sqrt(a1)
        if t1 - t0 <= 0 or s1 <= s0 * 1.05:
            return None
        return s1 / ((s1 - s0) / (t1 - t0))

    def stopped_for(self, number: int, now: float) -> float:
        """How long a thing that was moving has stood where it is: 0 while it moves, or if it never moved."""
        track = self.tracks[number]
        x, y = track[-1][1], track[-1][2]
        since = None
        for t, tx, ty, _a in reversed(track):
            if math.hypot(tx - x, ty - y) > STANDS:
                return now - since if since is not None else 0.0
            since = t
        return 0.0

    def going(self, number: int) -> tuple[float, float]:
        track = self.tracks[number]
        (t0, x0, y0, _a0), (t1, x1, y1, _a1) = track[0], track[-1]
        return (x1 - x0) / max(1e-3, t1 - t0), (y1 - y0) / max(1e-3, t1 - t0)


async def _into_play(game: GameAtHand, out: Path | None = None, log: list[str] | None = None) -> None:
    for step in range(20):
        picture, _ = await game.look()
        said = game.said(picture).upper()
        if log is not None:
            log.append(f"menu step {step}: {said[:80]}")
        if out is not None:
            game.keep(picture, out / f"menu_{int(time.monotonic())}_{step}.png")
        if "YOU" in said and "MENU" in said and _cells_left(picture, 0.11, 0.38) is not None:
            return
        for word in ("START", "PLAY", "NEXT", "CONTINUE", "AGAIN"):
            if word in said and await game.click_word(picture, word):
                break
        else:
            await game.click(0.9, 0.92, arrive_s=0.3, held_s=0.1)
        await asyncio.sleep(1.5)


def _settled(readings: deque, value: float) -> float | None:
    """A bar's reading once the last three agree: a picture mid-flash or zoomed reads anything."""
    readings.append(value)
    if len(readings) == readings.maxlen and max(readings) - min(readings) < 0.03:
        return float(np.mean(readings))
    return None


def _recording_into(out: Path | None):
    if out is None:
        return None
    out.mkdir(parents=True, exist_ok=True)
    return (out / "frames.jsonl").open("w")


async def play(game: GameAtHand, seconds: float, out: Path | None, dodge: bool = True) -> dict:
    recording = await asyncio.to_thread(_recording_into, out)
    await asyncio.sleep(12)
    await _into_play(game)
    room: deque[np.ndarray] = deque(maxlen=ROOM_OF)
    room_at = 0.0
    background = None
    tracks = Tracks()
    tries, splats = dict.fromkeys(LEADS, 0), dict.fromkeys(LEADS, 0)
    dodges, spared = dict.fromkeys(DODGE_AT, 0), dict.fromkeys(DODGE_AT, 0)
    thrown_at = dodged_at = 0.0
    last_lead, last_dodge = LEADS[0], None
    mine_seen, theirs_seen = deque(maxlen=3), deque(maxlen=3)
    mine_before = theirs_before = None
    lost = won = rounds_won = rounds_lost = 0
    log: list[str] = []
    ends = time.monotonic() + seconds
    pictured = 0.0
    filming_until = 0.0
    while time.monotonic() < ends:
        picture, now = await game.look()
        small = picture[::4, ::4].astype(np.int16)
        if now - room_at > ROOM_EVERY_S:
            room.append(small)
            room_at = now
            if len(room) >= 5:
                background = np.median(np.stack(room), axis=0)
        if now < filming_until and out is not None:
            game.keep(picture[::2, ::2], out / "film" / f"{now:.2f}.jpg")
        mine_now, theirs_now = _cells_left(picture, 0.11, 0.38), _cells_left(picture, 0.575, 0.845)
        mine = _settled(mine_seen, mine_now) if mine_now is not None else None
        theirs = _settled(theirs_seen, theirs_now) if theirs_now is not None else None
        if out is not None and now - pictured > 5:
            game.keep(picture, out / f"play_{int(now)}.png")
            pictured = now
        if theirs is not None:
            if theirs_before is not None and theirs < theirs_before - 0.05:
                splats[last_lead] += 1
                won += 1
                log.append(f"{now:.1f} splatted them (lead {last_lead}) theirs={theirs:.2f}")
            theirs_before = theirs
        if mine is not None:
            if mine_before is not None and mine < mine_before - 0.05:
                lost += 1
                log.append(f"{now:.1f} splatted me mine={mine:.2f} (last dodge {last_dodge} at {now - dodged_at:.2f}s ago)")
                if last_dodge is not None and now - dodged_at < 1.5:
                    spared[last_dodge] -= 1
            mine_before = mine
        if mine_before is not None and theirs_before is not None and (mine_before < 0.02 or theirs_before < 0.02):
            if theirs_before < 0.02:
                rounds_won += 1
            else:
                rounds_lost += 1
            await asyncio.sleep(2.5)
            picture, _ = await game.look()
            log.append(f"{now:.1f} round over ({'won' if theirs_before < 0.02 else 'lost'}): {game.said(picture)[:120]}")
            game.keep(picture, (out or Path("/tmp")) / f"over_{int(now)}.png")
            await asyncio.sleep(2)
            await _into_play(game, out, log)
            room.clear()
            background = None
            mine_seen.clear()
            theirs_seen.clear()
            mine_before = theirs_before = None
            continue
        if background is None:
            await asyncio.sleep(0.03)
            continue
        differs = np.abs(small - background).max(axis=2) > DIFFERS
        differs[: int(0.08 * differs.shape[0])] = False          # the bars along the top
        labels, count = ndimage.label(differs)
        tall, wide = differs.shape
        blobs = []
        for number, found in enumerate(ndimage.find_objects(labels), 1):
            if found is None:
                continue
            part = labels[found] == number
            area = float(part.sum()) / differs.size
            if area < A_THING:
                continue
            ys, xs = np.nonzero(part)
            blobs.append(((xs.mean() + found[1].start) / wide, (ys.mean() + found[0].start) / tall, area))
        tracks.saw(now, blobs)
        if recording is not None:
            recording.write(json.dumps({"t": round(now, 3), "mine": mine_now, "theirs": theirs_now, "dodged": round(dodged_at, 3),
                                        "tracks": {n: [round(v, 4) for v in tr[-1][1:]] for n, tr in tracks.tracks.items()}}) + "\n")
        untried = [d for d in DODGE_AT if dodges[d] < 3]
        dodge_at = untried[0] if untried else max(DODGE_AT, key=lambda d: (spared[d] + dodges[d] + 0.5) / (dodges[d] + 1))
        steady = [n for n in tracks.tracks if len(tracks.tracks[n]) >= 4 and not tracks.coming(n)]
        them = max(steady, key=lambda n: tracks.tracks[n][-1][3]) if steady else None
        winding_up = them is not None and tracks.stopped_for(them, now) >= STANDS_FOR_S + dodge_at
        if dodge and winding_up and now - dodged_at > 1.0:
            asyncio.ensure_future(game.tap(" ", held_s=DODGE_HELD_S))
            if out is not None:
                filming_until = now + 1.2
            dodges[dodge_at] += 1
            last_dodge, dodged_at = dodge_at, now
            log.append(f"{now:.1f} dodged {dodge_at}s after they stopped")
            continue
        if now - thrown_at > 0.35 and them is not None:
            _t, x, y, area = tracks.tracks[them][-1]
            going = tracks.going(them)
            untried = [lead for lead in LEADS if tries[lead] < 4]
            lead = untried[0] if untried else max(LEADS, key=lambda ahead: (splats[ahead] + 0.5) / (tries[ahead] + 1))
            tries[lead] += 1
            last_lead = lead
            await game.click(x + going[0] * lead, y - 0.3 * math.sqrt(area) + going[1] * lead)
            thrown_at = now
        await asyncio.sleep(0.005)
    return {"game": game.title, "rounds_won": rounds_won, "rounds_lost": rounds_lost, "splatted_them": won,
            "splatted_me": lost, "tries": tries, "splats": splats, "dodges": dodges, "spared": spared, "log": log[-80:]}


async def _main(seconds: float, out: Path | None, dodge: bool) -> None:
    async with GameAtHand(2) as game:
        report(out, await play(game, seconds, out, dodge))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-dodge", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main(args.seconds, args.out, not args.no_dodge))


if __name__ == "__main__":
    main()
