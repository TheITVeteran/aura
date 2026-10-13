"""Reference runner: signs coming along a strip to a mark, each one a key (Zap to It!, case 15).

What it knows, as a person who has read the rules and watched a round knows it: arrows come out of a book and travel
along the strip at the foot of the picture to a pad at its far end; the arrow at the front is judged on the pad by
which arrow keys are down at that moment, and by nothing else; a miss costs one of three lives. What it does:

- the strip's ink is found by its colour, each piece of it cut out and read as an arrow by its taper; ink that stays
  where it is is the strip's own, however like an arrow it reads;
- arrows drawn close together are one sign, and it asks for all their keys at once;
- each sign is read while it is still out on the clear strip and that reading kept, since on the pad it swells;
- the keys of the sign furthest along are held, and no others, from the moment the one before it has gone, which is
  known by the front one being suddenly further back. No mark is followed from picture to picture: only their order
  is used, so a slow picture or two loses nothing.

    python -m tools.reference_runners.signs_coming_along_a_strip --seconds 240 --out /tmp/rr/zap
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from tools.reference_runners.a_game_at_hand import GameAtHand, report

#: Where the strip is in the picture, top and bottom as shares of its height.
STRIP = (0.86, 1.0)
#: The arrows' ink: a dark green, on a strip of lighter greens.
INK_RED_UNDER, INK_GREEN, INK_BLUE_UNDER = 75, (70, 132), 75
#: A mark's sides, in pixels of a picture about a thousand wide, and the least of it that is ink. On the pad a mark
#: swells to twice its size and is still the mark.
SIDES = (10, 150)
LEAST_INK = 60
#: Where along the strip a mark is read: clear of the book it comes out of, and short of the pad it swells on.
READ_BETWEEN = (0.33, 0.78)
#: Where the lives are drawn, and the yellow of the heads that count them.
LIVES_IN = (0.0, 0.33, 0.03, 0.19)
#: Where the pad begins, as a share of the picture's width, and where a sign on it has been judged: the program
#: judges the front sign over the last sixty of the stage's 432 units before 420.
PAD_FROM = 0.833
JUDGED_AT = 0.86
#: How far behind the front arrow, as a share of the picture, another is that is part of the same sign.
ONE_SIGN_WITHIN = 0.075
#: Keys as the page takes them.
KEYS = {"up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight"}


def ink_on_the_strip(picture: np.ndarray) -> np.ndarray:
    """Where the strip, at the foot of the picture, is the arrows' ink."""
    tall = picture.shape[0]
    band = picture[int(STRIP[0] * tall):int(STRIP[1] * tall), :, :3].astype(int)
    return ((band[:, :, 0] < INK_RED_UNDER) & (band[:, :, 1] > INK_GREEN[0]) & (band[:, :, 1] < INK_GREEN[1])
            & (band[:, :, 2] < INK_BLUE_UNDER))


def marks_on_the_strip(picture: np.ndarray, furniture: np.ndarray | None = None) -> list[tuple]:
    """Each piece of the arrows' ink on the strip: how far across the picture its middle is, its shape, and where its
    top left corner is on the strip. Not what
    was ink there before any arrow came (``furniture``): the pad's own shading is the ink's colour, stands at the far
    end, and read by its taper it is a down arrow, held for a whole game by a runner that took it for the front one."""
    wide = picture.shape[1]
    ink = ink_on_the_strip(picture)
    if furniture is not None and furniture.shape == ink.shape:
        ink = ink & ~furniture
    labels, _count = ndimage.label(ink)
    found = []
    for number, place in enumerate(ndimage.find_objects(labels), 1):
        high, broad = place[0].stop - place[0].start, place[1].stop - place[1].start
        shape = labels[place] == number
        if SIDES[0] <= high <= SIDES[1] and SIDES[0] <= broad <= SIDES[1] and int(shape.sum()) >= LEAST_INK:
            found.append(((place[1].start + broad / 2) / wide, shape, (place[0].start, place[1].start)))
    return sorted(found, key=lambda one: one[0])


def lives_left(picture: np.ndarray) -> int:
    """How many heads are drawn at the top left: each a patch of the hair's yellow."""
    tall, wide = picture.shape[:2]
    left, right, top, bottom = LIVES_IN
    part = picture[int(top * tall):int(bottom * tall), int(left * wide):int(right * wide), :3].astype(int)
    yellow = (part[:, :, 0] > 225) & (part[:, :, 1] > 200) & (part[:, :, 2] < 120)
    labels, _count = ndimage.label(yellow)
    sizes = ndimage.sum(yellow, labels, range(1, _count + 1)) if _count else []
    return int(sum(size > 0.0006 * tall * wide for size in sizes))


async def _until_it_says(game: GameAtHand, word: str, within_s: float = 30.0) -> np.ndarray | None:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        picture, _at = await game.look()
        if game.where_it_says(picture, word) is not None:
            return picture
        await asyncio.sleep(0.4)
    return None


async def _into_play(game: GameAtHand, log: list[str]) -> bool:
    """Through the title, the choice of who to be, and the two pages of story, to the room."""
    picture = await _until_it_says(game, "START")
    if picture is None or not await game.click_word(picture, "START"):
        return False
    picture = await _until_it_says(game, "Mandy", 15)
    if picture is None:
        return False
    where = game.where_it_says(picture, "Mandy")
    await game.click(where[0], where[1] - 0.12, arrive_s=0.3, held_s=0.1)     # the picture over the name
    for _page in range(2):
        picture = await _until_it_says(game, "NEXT", 15)
        if picture is None:
            return False
        await game.click_word(picture, "NEXT")
        await asyncio.sleep(0.8)
    log.append("in the room")
    return True


def _as_one_mark(pieces: list[tuple]) -> np.ndarray:
    """The pieces of one sign put together as they lie: one shape."""
    top = min(corner[0] for _across, _shape, corner in pieces)
    left = min(corner[1] for _across, _shape, corner in pieces)
    bottom = max(corner[0] + shape.shape[0] for _across, shape, corner in pieces)
    right = max(corner[1] + shape.shape[1] for _across, shape, corner in pieces)
    whole = np.zeros((bottom - top, right - left), dtype=bool)
    for _across, shape, corner in pieces:
        whole[corner[0] - top:corner[0] - top + shape.shape[0], corner[1] - left:corner[1] - left + shape.shape[1]] |= shape
    return whole


def signs_on_the_strip(picture: np.ndarray, furniture: np.ndarray | None,
                       known: dict[str, np.ndarray] | None = None) -> list[tuple[float, frozenset[str], np.ndarray]]:
    """The signs on the strip now, the one furthest along first: how far across each is, the keys it shows (none where
    it could not be read), and its shape. Arrows drawn within a third of the gap between signs are one sign. A sign of
    one arrow is read by its taper; once the four single arrows are known by their shapes (``known``), every sign is
    read as the ones it is made of, which reads two arrows drawn over one another as both."""
    from core.perception.keys_drawn_on_screen import which_way_it_points
    from core.perception.signs_made_of_signs import made_of

    pieces = [piece for piece in marks_on_the_strip(picture, furniture) if piece[0] >= READ_BETWEEN[0] - 0.03]
    signs: list[list[tuple]] = []
    for piece in sorted(pieces, key=lambda one: -one[0]):
        if signs and signs[-1][-1][0] - piece[0] <= ONE_SIGN_WITHIN:
            signs[-1].append(piece)
        else:
            signs.append([piece])
    read = []
    for sign in signs:
        whole = _as_one_mark(sign)
        if known is not None and len(known) >= 4:
            keys = made_of(whole, known)
        else:
            keys = frozenset(way for way in (which_way_it_points(shape) for _across, shape, _corner in sign) if way)
        read.append((max(across for across, _shape, _corner in sign), keys, whole))
    return read


async def play(game: GameAtHand, seconds: float, out: Path | None, keep_after: float = 0.0) -> dict:
    log: list[str] = []
    if not await _into_play(game, log):
        return {"game": game.title, "error": "never got into play", "log": log}
    began = time.monotonic()
    # What is ink on the strip before anything comes along it is the strip's own, and so is whatever stays ink where
    # it is: kept as how often each place has been ink lately.
    picture, _at = await game.look()
    stays = ink_on_the_strip(picture).astype(np.float32)
    held: frozenset[str] = frozenset()
    known: dict[str, np.ndarray] = {}
    seen_as: dict[str, list[np.ndarray]] = {"up": [], "down": [], "left": [], "right": []}
    read_of_front: Counter = Counter()      # what the sign at the front has been read as, while out on the clear strip
    read_of_next: Counter = Counter()       # and the one behind it
    front_was_at, front_seen_at = 0.0, 0.0
    pace = 0.35                             # shares of the picture a second, until seen
    keys_of_the_last: frozenset[str] = frozenset()
    held_until = 0.0
    lives_were, judged, rooms = 3, 0, 0
    answered: Counter = Counter()
    traced: list[str] = []
    looked = kept = 0
    empty_since: float | None = None
    last_said_at = 0.0
    ended = "out of time"
    while time.monotonic() - began < seconds:
        picture, at = await game.look()
        looked += 1
        ink = ink_on_the_strip(picture)
        if ink.shape == stays.shape:
            stays = 0.96 * stays + 0.04 * ink
        furniture = ndimage.binary_dilation(stays > 0.75, iterations=2)
        # Only the clear strip is gone by: on the pad a sign swells, changes colour and is not always ink.
        signs = [sign for sign in signs_on_the_strip(picture, furniture, known) if sign[0] <= PAD_FROM + 0.01]
        for across, keys, whole in signs:
            # The single arrows, each kept by its shape the first few times it is read plainly out on the clear strip:
            # what the two-arrow signs of later showings are read by.
            if len(keys) == 1 and READ_BETWEEN[0] + 0.1 <= across <= READ_BETWEEN[1] and max(whole.shape) <= 45:
                (key,) = keys
                seen_as[key].append(whole)
                if key not in known and len(seen_as[key]) >= 3:
                    known[key] = seen_as[key][1]
        lives = lives_left(picture)
        if lives and lives != lives_were:
            if lives < lives_were:
                log.append(f"{at - began:.1f} a life lost ({lives} left); held {sorted(held)}")
            lives_were = lives
        if signs:
            front_at, front_keys, _shape = signs[0]
            if front_at < front_was_at - 0.1:
                # The one that was at the front has gone onto the pad, and the one behind it is the front now. The one
                # gone on is judged a moment after, where the pad begins: its keys stay down until then, and no
                # longer, for the next one's keys must be the only ones down when it gets there in its turn.
                judged += 1
                keys_of_the_last, held_until = held, front_seen_at + max(0.0, (JUDGED_AT - front_was_at) / pace) + 0.05
                read_of_front, read_of_next = read_of_next, Counter()
            elif front_at > front_was_at and at > front_seen_at and front_was_at > 0:
                pace = 0.8 * pace + 0.2 * (front_at - front_was_at) / (at - front_seen_at)
            front_was_at, front_seen_at = front_at, at
            if front_keys and front_at <= READ_BETWEEN[1]:
                read_of_front[front_keys] += 1
            if len(signs) > 1 and signs[1][1] and READ_BETWEEN[0] <= signs[1][0] <= READ_BETWEEN[1]:
                read_of_next[signs[1][1]] += 1
            wanted = read_of_front.most_common(1)[0][0] if read_of_front else frozenset()
            if empty_since is not None and at - empty_since > 1.5:
                log.append(f"{at - began:.1f} a showing begins")
            empty_since = None
        else:
            wanted = frozenset()
            if empty_since is None:
                empty_since = at
                if front_was_at > 0:
                    # The last of a showing has gone onto the pad: its keys stay down until it is judged.
                    judged += 1
                    keys_of_the_last, held_until = held, front_seen_at + max(0.0, (JUDGED_AT - front_was_at) / pace) + 0.05
            if at - empty_since > 0.6:
                read_of_front, read_of_next, front_was_at = Counter(), Counter(), 0.0
        if at < held_until:
            wanted = keys_of_the_last
        if wanted != held:
            for key in held - wanted:
                await game.key_up(KEYS[key])
            for key in wanted - held:
                await game.key_down(KEYS[key])
            if wanted:
                answered["+".join(sorted(wanted))] += 1
            held = wanted
        if keep_after and at - began >= keep_after and len(traced) < 500:
            traced.append(f"{at - began:.2f} " + " ".join(f"{across:.2f}:{'+'.join(sorted(keys)) or '?'}" for across, keys, _shape in signs)
                          + f" | held {'+'.join(sorted(held)) or '-'} lives {lives}")
        if out is not None:
            # Each sign as drawn, cut out as it passes the middle of the strip, named by how it was read: the game's own
            # artwork for whoever writes or tests a reader of signs.
            for across, keys, _shape in signs:
                if 0.50 <= across <= 0.53:
                    tall, wide = picture.shape[:2]
                    cut = picture[int(STRIP[0] * tall):, int((across - 0.16) * wide):int((across + 0.045) * wide)]
                    game.keep(cut, out / "signs" / f"{'+'.join(sorted(keys)) or 'unread'}_{at - began:06.1f}.png")
        if out is not None and kept < 120 and signs and at - began >= keep_after and looked % 3 == 0:
            kept += 1
            game.keep(picture, out / f"play{kept:03d}_{at - began:05.1f}.png")
        if empty_since is not None and at - empty_since > 2.5 and at - last_said_at > 2.5:
            # Nothing on the strip for a while: between two showings, or over. What the screen says tells which.
            last_said_at = at
            said = game.said(picture)
            if "MESS AGAIN" in said.upper():
                # Its last line: every room is clean, and the first is a mess again. There is no more of it than that.
                rooms += 1
                ended = f"won: every room cleaned ({rooms}); it says {said[:90]!r}"
                break
            if "PLAY AGAIN" in said.upper() or "DADS HOME" in said.upper().replace("'", ""):
                ended = f"lost: {said[:80]}"
                break
            if "NEXT" in said.upper():
                rooms += "EXCELLENT" in said.upper()
                log.append(f"{at - began:.1f} it says {said[:70]!r}: going on")
                await game.click_word(picture, "NEXT")
            elif said.strip():
                log.append(f"{at - began:.1f} it says {said[:70]!r}")
        await asyncio.sleep(0.003)
    for key in held:
        await game.key_up(KEYS[key])
    picture, _at = await game.look()
    if out is not None:
        game.keep(picture, out / "end.png")
    return {"game": game.title, "ended": ended, "seconds": round(time.monotonic() - began, 1), "signs_judged": judged,
            "rooms_cleaned": rooms,
            "lives_at_the_end": lives_were, "answered": dict(answered),
            "pictures_a_second": round(looked / max(1.0, time.monotonic() - began), 1), "pace": round(pace, 3),
            "says_at_the_end": game.said(picture)[:120], "log": log[-60:], "traced": traced}


async def _main(seconds: float, out: Path | None, keep_after: float) -> None:
    async with GameAtHand(15) as game:
        report(out, await play(game, seconds, out, keep_after))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--keep-after", type=float, default=0.0, help="keep pictures of play from this many seconds in")
    args = parser.parse_args()
    asyncio.run(_main(args.seconds, args.out, args.keep_after))


if __name__ == "__main__":
    main()
