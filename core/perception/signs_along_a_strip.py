"""The strip signs come along, read directly once it is known: each sign on it, in order, and what it shows.

Following each mark from picture to picture is the wrong way to read a line of
signs. Two arrows drawn a finger apart are told from one another only by
which is which, a slow picture or two loses the thread, and a sign that
swells where it is judged becomes a new thing at the worst moment. What a
person reads is the strip: the row of signs on it now, front to back.

So once the line is known (where it runs, which way, what ink the signs are
drawn in and how big one is), the band it runs in is read whole at each look:

- the signs' ink, where it is not what is usually there. What is usually
  at each place of the band is kept as a picture, learned from every look at
  it that no sign was passing. The strip's own edge is often near the signs'
  colour, and the place they are judged on is often shaded like one of them:
  neither is a sign, and a sign passing over the edge is still seen whole
  wherever the two are not one colour. (Kept as places that stay ink, with a
  little left out round them, the edge cut the head off the lower arrow of
  every pair that lay on it, and the pair read as its upper arrow alone.);
- of that, only what has come along the line since the look before: the
  ink of the last look, moved on by as far as the signs have gone, lies under
  a sign on its way and not under anything else. What the signs come out of
  may show them whirling in it, in line with the strip, and those are not on
  their way;
- its pieces gathered into signs, pieces within a couple of sign-widths along
  the line being one sign, since a sign may be two drawn together;
- each sign read by its taper where it is one arrow, and as the single signs
  it is made of once those are known by their shapes
  (core/perception/signs_made_of_signs.py). The single shapes are learned
  here, from the signs that come one at a time. A sign of several pieces
  that the known ones do not explain is left unread: two arrows joined at a
  corner taper as neither, and a wrong key is worse than none.

A reference runner written to beat one such game outright won by exactly
this and by nothing that named the game (docs/design-docs/WHAT_THE_RUNNERS_WON_BY.md).

numpy and scipy only: her own process refuses OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SeenSign", "StripOfSigns", "ink_of"]

#: How far a pixel may be from the signs' ink, of 255, and still be it.
INK_WITHIN = 40
#: How far a mark stands from its ground to be a mark, of 255.
MARKED = 36.0

#: How unlike what is usually at a place a pixel must be, of 255, to be a sign passing over it (and never less
#: unlike it than it is unlike the signs' ink); how fast what is usually there is learned from a look that shows
#: something in the signs' ink staying, and how fast from one that shows no ink there at all.
UNLIKE_THE_USUAL = 14.0
STAYS_FORGETS = 0.04
CLEAR_FORGETS = 0.3

#: The band read, either side of the line, in sizes of one sign; a piece's sides and area, in sizes of one sign; and
#: how far apart along the line, in sizes of one sign, two pieces of one sign may be.
BAND = 1.7
SIDES = (0.3, 4.5)
LEAST_AREA = 0.12
ONE_SIGN_WITHIN = 2.2

#: How much of a piece must lie under the last look's ink moved on along the line for it to be on its way; the
#: fastest a sign goes, in lengths of the line a second (one crossing the whole line in under a second is past
#: answering); and the longest wait for a picture that the one after it may be making up for, in seconds. (A
#: picture come again shows nothing as having come along: whoever takes the signs in turn forgives a look.)
COMES_ALONG = 0.7
MOST_A_SECOND = 1.2
A_PICTURE_WAITS_S = 0.15

#: How much bigger or smaller than a usual single sign a mark may be and still be taught as one, and how many times a
#: single sign is seen before its shape is kept: the one of them most like the others, since any one may be cut short.
A_SINGLE_UP_TO = 1.35
A_SINGLE_FROM = 0.6
SEEN_BEFORE_KEPT = 3


@dataclass
class SeenSign:
    """One sign on the strip at a look: how far along the line it is (a share of the picture, more being further
    along), the pieces it is drawn in, and all of them as one shape."""

    along: float
    whole: np.ndarray
    pieces: list[np.ndarray] = field(default_factory=list)
    #: How much of the line it takes up, as a share of the picture.
    span: float = 0.0


def ink_of(patch: Any) -> tuple[tuple[int, int, int], int] | None:
    """The ink a mark in a patch is drawn in, and how big the mark is (its longer side, in pixels): what is unlike the
    patch's ground, in its largest piece. None where nothing in the patch stands off its ground."""
    from scipy import ndimage

    pixels = np.asarray(patch).astype(np.int16)
    if pixels.ndim != 3 or min(pixels.shape[:2]) < 6:
        return None
    ground = np.median(pixels.reshape(-1, pixels.shape[2]), axis=0).astype(np.int16)
    unlike = np.abs(pixels[:, :, :3] - ground[:3]).max(axis=2) > MARKED
    labels, count = ndimage.label(unlike)
    if not count:
        return None
    sizes = ndimage.sum(unlike, labels, range(1, count + 1))
    number = int(np.argmax(sizes)) + 1
    place = ndimage.find_objects(labels)[number - 1]
    mark = labels == number
    colour = tuple(int(v) for v in np.median(pixels[:, :, :3][mark], axis=0))
    return colour, max(place[0].stop - place[0].start, place[1].stop - place[1].start)


class StripOfSigns:
    """A line signs come along, read whole at each look."""

    def __init__(self, going: tuple[float, float], on_line: tuple[float, float], ink: tuple[int, int, int],
                 size: float) -> None:
        #: Which way they go (shares of the picture a second), a place on the line (shares), the ink they are drawn in,
        #: and how big a single sign is (pixels of the picture, its longer side).
        self.going, self.on_line, self.ink, self.size = going, on_line, tuple(int(v) for v in ink), float(size)
        self.level = abs(going[0]) >= 3 * abs(going[1])
        self.readable = self.level or abs(going[1]) >= 3 * abs(going[0])
        self._usual: np.ndarray | None = None                   # what is usually at each place of the band
        self._was: tuple[np.ndarray, float] | None = None       # the ink that does not stay, at the look before, and when
        #: The single signs known by their shapes, by the key each shows; and the ones seen so far, before they are kept.
        self.known: dict[str, np.ndarray] = {}
        self._seen_as: dict[str, list[np.ndarray]] = {}

    # -- the signs on it now ------------------------------------------------------------------------------------------

    def signs(self, picture: Any, at: float | None = None) -> list[SeenSign]:
        """The signs on the strip now, the one furthest along first. ``at`` is when the picture was taken: with it,
        only what has come along the line since the look before is a sign."""
        from scipy import ndimage

        pixels = np.asarray(picture)
        tall, wide = pixels.shape[:2]
        reach = max(4, round(BAND * self.size))
        if self.level:
            middle = round(self.on_line[1] * tall)
            first, last = max(0, middle - reach), min(tall, middle + reach)
            band = pixels[first:last, :, :3]
        else:
            middle = round(self.on_line[0] * wide)
            first, last = max(0, middle - reach), min(wide, middle + reach)
            band = pixels[:, first:last, :3]
        now = band.astype(np.float32)
        from_ink = np.abs(now - np.asarray(self.ink, dtype=np.float32)).max(axis=2)
        ink = from_ink <= INK_WITHIN
        if self._usual is None or self._usual.shape != now.shape:
            self._usual = now.copy()
        from_usual = np.abs(now - self._usual).max(axis=2)
        moving = ink & (from_usual > np.maximum(UNLIKE_THE_USUAL, from_ink))
        # What is usually there is learned at every look: fast where there is no ink, and slowly where there is, so
        # that a sign passing does not become what is usual in the time it takes to pass, and a thing in the signs'
        # ink that comes and stays does.
        learns = np.where(ink, STAYS_FORGETS, CLEAR_FORGETS).astype(np.float32)
        self._usual += learns[:, :, None] * (now - self._usual)
        gone = self._gone_along(moving, at)
        labels, _count = ndimage.label(moving)
        pieces = []
        for number, place in enumerate(ndimage.find_objects(labels), 1):
            high, broad = place[0].stop - place[0].start, place[1].stop - place[1].start
            shape = labels[place] == number
            if not (SIDES[0] * self.size <= max(high, broad) <= SIDES[1] * self.size
                    and int(shape.sum()) >= LEAST_AREA * self.size * self.size):
                continue
            if gone is not None and not self._came_along(shape, place, *gone):
                continue
            pieces.append((self._along(place, tall, wide, first), shape, (place[0].start, place[1].start)))
        pieces.sort(key=lambda piece: -piece[0])
        long = wide if self.level else tall
        together = ONE_SIGN_WITHIN * self.size / long
        gathered: list[list[tuple]] = []
        for piece in pieces:
            if gathered and gathered[-1][-1][0] - piece[0] <= together:
                gathered[-1].append(piece)
            else:
                gathered.append([piece])
        seen = []
        for sign in gathered:
            whole = _as_one(sign)
            seen.append(SeenSign(along=sign[0][0], whole=whole, pieces=[shape for _along, shape, _corner in sign],
                                 span=whole.shape[1 if self.level else 0] / long))
        return seen

    def _gone_along(self, moving: np.ndarray, at: float | None) -> tuple[np.ndarray, range] | None:
        """The ink of the look before, and how far along the line a sign may have come since, in pixels. None where no
        time was given (then everything is taken); nothing to have come from at the first look."""
        from scipy import ndimage

        if at is None:
            return None
        before, self._was = self._was, (moving, at)
        if before is None or before[0].shape != moving.shape or at <= before[1]:
            return np.zeros_like(moving), range(0)
        long = moving.shape[1] if self.level else moving.shape[0]
        # Pictures do not come as evenly as their times say: two may show a tenth of a second's going a thirtieth of
        # a second apart. So by anything from a pixel to as far as the signs go in the longest wait for a picture.
        # How fast the signs were first seen to go says which way they go and no more: by anything from a pixel to
        # as far as a sign may have come in the longest wait for a picture, at any pace a sign can be answered at.
        # Offline 2026-10-12 the line was found by a thing going a quarter as fast as the signs; they were looked
        # for no further on than that thing could have got, seen in the first room, and not in the faster second.
        furthest = MOST_A_SECOND * long * max(at - before[1], A_PICTURE_WAITS_S)
        # A pixel's give either way: a picture scaled to the screen puts an edge on one side of a pixel or the other.
        return ndimage.binary_dilation(before[0], iterations=1), range(1, min(long - 1, int(furthest) + 3))

    def _came_along(self, shape: np.ndarray, place: tuple, was: np.ndarray, tries: range) -> bool:
        """Whether a piece has come along the line since the look before: the ink of that look, moved on by some one
        of the distances a sign may have come, lies under most of it."""
        forward = (self.going[0] if self.level else self.going[1]) > 0
        rows, columns = place
        # A thing that has stayed where it was lies under itself as it is, and under itself moved on a little if it
        # is long the way the line goes: it has come along only where moving the last look's ink on covers more of
        # it than leaving that ink where it was. (Offline 2026-10-12: a strip of the line's own ink, shown again when
        # a sign of another colour had passed over it, was taken for a sign, and the sign coming up to it for part of it.)
        most = max(COMES_ALONG * int(shape.sum()), int((was[rows, columns] & shape).sum()) + 1)
        for by in tries:
            back = -by if forward else by
            if self.level:
                from_, to = columns.start + back, columns.stop + back
                under = was[rows, from_:to] if 0 <= from_ and to <= was.shape[1] else None
            else:
                from_, to = rows.start + back, rows.stop + back
                under = was[from_:to, columns] if 0 <= from_ and to <= was.shape[0] else None
            if under is not None and int((under & shape).sum()) >= most:
                return True
        return False

    def what_stays(self) -> Any:
        """What is usually on the band, for whoever reads the same line next: without it the first signs of the next
        showing are read against a picture that had signs in it."""
        return self._usual

    def stays_as_before(self, kept: Any) -> None:
        if isinstance(kept, np.ndarray) and kept.ndim == 3:
            self._usual = kept.astype(np.float32, copy=True)

    def _along(self, place: tuple, tall: int, wide: int, first: int) -> float:
        """How far along the line a piece is: its middle along the way they go, as a share of the picture."""
        if self.level:
            share = (place[1].start + place[1].stop) / 2.0 / wide
            return share if self.going[0] > 0 else 1.0 - share
        share = (place[0].start + place[0].stop) / 2.0 / tall
        return share if self.going[1] > 0 else 1.0 - share

    # -- what a sign shows --------------------------------------------------------------------------------------------

    def read(self, sign: SeenSign) -> frozenset[str]:
        """The keys a sign shows: the single signs it is made of where those are known, else each arrow of it by its
        taper. A sign of one arrow, read by its taper, teaches its shape."""
        from core.perception.keys_drawn_on_screen import which_way_it_points
        from core.perception.signs_made_of_signs import made_of

        single = (len(sign.pieces) == 1
                  and A_SINGLE_FROM * self.size <= max(sign.whole.shape) <= A_SINGLE_UP_TO * self.size)
        by_taper = which_way_it_points(sign.whole) if single else ""
        if by_taper and by_taper not in self.known:
            seen = self._seen_as.setdefault(by_taper, [])
            seen.append(sign.whole)
            if len(seen) >= SEEN_BEFORE_KEPT:
                self.known[by_taper] = _most_like_the_others(seen)
        if len(self.known) >= 2:
            made = made_of(sign.whole, self.known)
            if made:
                return made
        return frozenset({by_taper}) if by_taper else frozenset()


def _most_like_the_others(shapes: list[np.ndarray]) -> np.ndarray:
    """Of several shapes seen of one sign, the one most like the rest."""
    from core.perception.signs_made_of_signs import how_alike

    return max(shapes, key=lambda one: sum(how_alike(one, other) for other in shapes if other is not one))


def _as_one(pieces: list[tuple]) -> np.ndarray:
    """The pieces of one sign put together as they lie: one shape."""
    top = min(corner[0] for _along, _shape, corner in pieces)
    left = min(corner[1] for _along, _shape, corner in pieces)
    bottom = max(corner[0] + shape.shape[0] for _along, shape, corner in pieces)
    right = max(corner[1] + shape.shape[1] for _along, shape, corner in pieces)
    whole = np.zeros((bottom - top, right - left), dtype=bool)
    for _along, shape, corner in pieces:
        whole[corner[0] - top:corner[0] - top + shape.shape[0], corner[1] - left:corner[1] - left + shape.shape[1]] |= shape
    return whole
