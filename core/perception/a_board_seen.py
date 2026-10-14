"""A board seen: the squares a picture is ruled into, and what stands on each.

A board game is played on a lattice, and a person sees the lattice before the
pieces: rows and columns of squares of one size, coloured in turn or ruled with
lines, and on some of them pieces of two sides. Nothing on it is written, so it
is read from the picture itself. The edges between squares come at one
spacing across the board, so the board is where the picture's edges keep a
period; the period is the size of a square and the run of edges the extent of
the board. A square is empty where it looks like the empty squares of its own
colour, and holds a piece where it does not. Pieces come in sides, told apart by
colour; and a piece that looks unlike the rest of its side (a crown, a stack, a
figure standing on it) is one of a different standing.

Nothing here knows a game: not the number of squares, nor the colours, nor which
side is whose. Checkers, a connect-four grid, a tic-tac-toe square, an Othello
board are all seen the same way. Her own side is told by where its pieces start
and by what she is able to move (core/agency/playing_a_board.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.verify import invariant

__all__ = ["Board", "a_board_in", "the_lattice"]

#: The fewest squares a side of a board has, and the most.
FEWEST, MOST = 3, 19
#: How much stronger than the picture's edges at large the edges at a period must be for it to be a lattice.
STANDS_OUT = 1.6
#: How unlike the empty squares of its colour a square must look to hold something, in colour distance.
HOLDS = 38.0
#: How unlike its side a piece must look to be of a different standing, in colour distance.
STANDS_APART = 30.0


@dataclass
class Board:
    """A lattice and what stands on it: ``cells[row][column]`` is "" for empty, else the side ("a", "b", ...) and,
    for a piece of a different standing, that side with "+" ("a+")."""

    rows: int
    columns: int
    #: Where the lattice is in the picture: its left and top edges, and a square's width and height, in pixels.
    left: float
    top: float
    width: float
    height: float
    cells: list[list[str]] = field(default_factory=list)
    #: Each side's colour, for telling the sides apart in the next picture.
    sides: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: The colour of an empty square of each of the two colours squares come in by turns (those whose row and column
    #: add up to an even number, then the rest): the board known again by its ground.
    ground: tuple[tuple[float, float, float], ...] = ()

    def centre(self, row: int, column: int) -> tuple[float, float]:
        """The middle of a square, in the picture's pixels."""
        return self.left + (column + 0.5) * self.width, self.top + (row + 0.5) * self.height

    def pieces(self, side: str) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.columns) if self.cells[r][c].rstrip("+") == side]

    def says(self) -> str:
        return "\n".join("".join(cell[:1].upper() if cell.endswith("+") else (cell or ".") for cell in row)
                         for row in self.cells)


#: How nearly a smaller spacing must score what the best does to be the square's size: every other edge of a lattice
#: is a lattice too, at twice the spacing, and a square is the smallest spacing the edges keep.
NEARLY = 0.85


def _period(profile: np.ndarray, least: int, most: int) -> tuple[int, int, float]:
    """The smallest spacing whose edge score is nearly the best, with its phase and score."""
    overall = float(profile.mean()) or 1.0
    # Keep the last pixel's zero neighbor and the original float64 promotion.
    near = np.maximum(profile, np.concatenate([profile[1:], np.zeros(1)])[:profile.size])
    found: list[tuple[int, int, float]] = []
    for spacing in range(max(4, least), max(5, most) + 1):
        scores = _phase_scores(near, spacing, overall)
        # The first strictly positive maximum wins; NaNs and negative scores never did.
        accepted = np.where(scores > 0.0, scores, 0.0)
        phase = int(accepted.argmax())
        found.append((spacing, phase, float(accepted[phase])))
    if not found:
        return (0, 0, 0.0)
    top = max(score for _s, _p, score in found)
    return next(one for one in found if one[2] >= NEARLY * top)


def _phase_scores(near: np.ndarray, spacing: int, overall: float) -> np.ndarray:
    """Score phases together where they have the same number of observed teeth."""
    rows, extra = divmod(near.size, spacing)
    scores = np.zeros(spacing, dtype=np.float64)
    full = near[:rows * spacing].reshape(rows, spacing)
    if extra and rows + 1 >= FEWEST:
        taller = np.concatenate([full[:, :extra], near[rows * spacing:][None, :]], axis=0)
        scores[:extra] = np.asarray(np.median(taller, axis=0), dtype=np.float64) / overall
    if rows >= FEWEST:
        scores[extra:] = np.asarray(np.median(full[:, extra:], axis=0), dtype=np.float64) / overall
    return scores


@invariant("perception.lattice_phases_keep_their_observed_teeth", scope="perception",
           owner="core/perception/a_board_seen.py", observational=False)
def _lattice_phase_invariant() -> tuple:
    """Grouping phases keeps tooth counts, neighbor pixels, ties, and the smallest spacing."""
    assert _period(np.ones(31, np.float32), 4, 12) == (4, 0, 1.0)
    assert _period(np.ones(7, np.float32), 4, 12) == (4, 0, 0.0)
    assert _period(np.array([5., 0., 0., 0.] * 12), 4, 16) == (4, 0, 4.0)
    tail = np.array([-1., -1., -5., -5., -2., -2., -5., -5., -3.], np.float32)
    assert _period(tail, 4, 4) == (4, 0, -1. / float(tail.mean()))
    return ()


#: How near two edges must be to be one line's two sides, and how nearly evenly spaced a chain of lines must be.
ONE_LINE_PX = 4
EVEN_PX = 3


def _the_chain(profile: np.ndarray) -> tuple[float, float, int] | None:
    """The lattice along one axis from where its edges stand out: the strongest edges, a ruled line's two sides taken as
    one, and the longest run of them evenly spaced; where it starts, its spacing, and how many squares it spans."""
    if profile.size < 8:
        return None
    strong = float(profile.mean() + 2.0 * profile.std())
    peaks = [i for i in range(profile.size) if profile[i] >= strong
             and profile[i] == profile[max(0, i - 2):i + 3].max()]
    merged: list[list[int]] = []
    for at in peaks:
        if merged and at - merged[-1][-1] <= ONE_LINE_PX:
            merged[-1].append(at)
        else:
            merged.append([at])
    lines = [sum(group) / len(group) for group in merged]
    best: list[float] = []
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            spacing = lines[second] - lines[first]
            if spacing < 6:
                continue
            chain = [lines[first], lines[second]]
            for nxt in lines[second + 1:]:
                if abs(nxt - chain[-1] - spacing) <= EVEN_PX:
                    chain.append(nxt)
            if len(chain) > len(best):
                best = chain
    if len(best) < FEWEST + 1:
        return None
    return best[0], (best[-1] - best[0]) / (len(best) - 1), len(best) - 1


def _run(profile: np.ndarray, spacing: int, phase: int) -> tuple[int, int]:
    """The longest run of edges at the spacing: where the board starts and how many squares it spans."""
    strong = float(np.percentile(profile, 75))
    teeth = [max(float(profile[i]), float(profile[min(i + 1, profile.size - 1)])) > strong
             for i in range(phase, profile.size, spacing)]
    best_start, best_length, start = 0, 0, None
    for index, edge in enumerate([*teeth, False]):
        if edge and start is None:
            start = index
        elif not edge and start is not None:
            if index - start > best_length:
                best_start, best_length = start, index - start
            start = None
    return phase + best_start * spacing, max(0, best_length - 1)


def the_lattice(picture: np.ndarray) -> tuple[float, float, float, float, int, int] | None:
    """Where a picture is ruled into squares: left, top, square width and height, columns and rows; None where it is not.
    """
    grey = np.asarray(picture, dtype=np.float32).mean(axis=2) if picture.ndim == 3 else np.asarray(picture, np.float32)
    tall, wide = grey.shape
    across = np.abs(np.diff(grey, axis=1)).mean(axis=0)
    down = np.abs(np.diff(grey, axis=0)).mean(axis=1)
    found = _by_the_comb(across, down, wide, tall)
    if found is not None:
        return found
    # Squares told apart by ruled lines, not by colour, have an edge each side of each line: the lattice is the longest
    # chain of lines evenly spaced, each line's two sides taken as one.
    by_peaks = _the_chain(across), _the_chain(down)
    if all(by_peaks):
        (left, width, columns), (top, height, rows) = by_peaks
        if FEWEST <= columns <= MOST and FEWEST <= rows <= MOST:
            return float(left + 1), float(top + 1), float(width), float(height), columns, rows
    return None


def _by_the_comb(across: np.ndarray, down: np.ndarray, wide: int, tall: int
                 ) -> tuple[float, float, float, float, int, int] | None:
    """The lattice from the spacing at which the edges peak most regularly: squares told apart by their colours."""
    width, x0, x_score = _period(across, wide // MOST, wide // FEWEST)
    height, y0, y_score = _period(down, tall // MOST, tall // FEWEST)
    if min(x_score, y_score) < STANDS_OUT or not width or not height:
        return None
    left, columns = _run(across, width, x0)
    top, rows = _run(down, height, y0)
    # The picture's own border is the last edge of a board drawn up to it.
    columns += int(left + (columns + 1) * width <= wide + 2 and columns < MOST)
    rows += int(top + (rows + 1) * height <= tall + 2 and rows < MOST)
    if not (FEWEST <= columns <= MOST and FEWEST <= rows <= MOST):
        return None
    return float(left + 1), float(top + 1), float(width), float(height), columns, rows


def a_board_in(picture: np.ndarray, sides: dict[str, tuple[float, float, float]] | None = None,
               known: Board | None = None) -> Board | None:
    """The board a picture shows, with what stands on each square; None where it shows none. ``sides`` are the sides'
    colours as seen before, so a side keeps its name from one picture to the next. ``known`` is the board as seen
    before in a game going on: a board does not move or change its squares in play, so its lattice is kept, and the
    board is there where most of its squares show their own ground (LIVE 2026-10-11 a board of eight by eight read
    in one picture as eight by ten, a piece of hers on a square that is not there, and a win claimed from it)."""
    lattice = (known.left, known.top, known.width, known.height, known.columns, known.rows) if known else \
        the_lattice(picture)
    if lattice is None:
        return None
    left, top, width, height, columns, rows = lattice
    image = np.asarray(picture, dtype=np.float32)
    looks = np.zeros((rows, columns, 3), np.float32)
    bare = np.zeros((rows, columns, 3), np.float32)
    for r in range(rows):
        for c in range(columns):
            looks[r, c] = _the_look(image, left, top, width, height, r, c, (0.3, 0.7), (0.3, 0.7))
            # A piece stands in the middle of its square: the square's corners are the square itself.
            corners = [_the_look(image, left, top, width, height, r, c, ys, xs)
                       for ys in ((0.06, 0.18), (0.82, 0.94)) for xs in ((0.06, 0.18), (0.82, 0.94))]
            bare[r, c] = np.median(np.array(corners), axis=0)
    ground = _the_ground(bare)
    if known is not None and known.ground and not _the_same_ground(bare, known.ground):
        return None
    held = [(r, c) for r in range(rows) for c in range(columns) if np.linalg.norm(looks[r, c] - bare[r, c]) > HOLDS]
    # A tall piece (a figure standing on its piece) reaches up into the square above it: what crosses a square's lower
    # border into a held square below is that piece, not one of its own.
    tall_ones = {(r + 1, c) for r, c in held if (r + 1, c) in held and _crosses_down(image, bare, left, top, width, height, r, c)}
    held = [(r, c) for r, c in held if (r + 1, c) not in tall_ones]
    board = Board(rows, columns, left, top, width, height, [["" for _ in range(columns)] for _ in range(rows)],
                  ground=ground)
    if not held:
        return board
    # A piece's side is told by its own pixels (those unlike its bare square), its lower half's where the piece fills it:
    # a figure standing on a piece still shows the piece there.
    bases = {(r, c): _the_pieces_colour(image, bare[r, c], left, top, width, height, r, c) for r, c in held}
    colours = np.array([bases[(r, c)] for r, c in held])
    named = _sides_of(colours, sides)
    board.sides = {name: tuple(float(v) for v in colour) for name, colour in named["colours"].items()}
    for (r, c), side in zip(held, named["of"], strict=True):
        board.cells[r][c] = side
    # A piece that looks unlike the rest of its side is of a different standing.
    for side in named["colours"]:
        mine = [(r, c) for (r, c), s in zip(held, named["of"], strict=True) if s == side]
        usual = np.median(np.array([looks[r, c] for r, c in mine]), axis=0) if mine else None
        for r, c in mine:
            if (r, c) in tall_ones or (len(mine) > 2 and np.linalg.norm(looks[r, c] - usual) > STANDS_APART):
                board.cells[r][c] = f"{side}+"
    return board


#: The share of a board's squares that must show their own ground for the board to be there.
SAME_GROUND = 0.7


def _the_ground(bare: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """The colour of the squares of each of the two turns, as the middle of their bare corners."""
    rows, columns = bare.shape[:2]
    turns = [[bare[r, c] for r in range(rows) for c in range(columns) if (r + c) % 2 == k] for k in (0, 1)]
    return tuple(tuple(float(v) for v in np.median(np.array(one), axis=0)) for one in turns if one)


def _the_same_ground(bare: np.ndarray, ground: tuple[tuple[float, float, float], ...]) -> bool:
    """Whether most squares show the ground of their own turn: the board is still there, where it was."""
    rows, columns = bare.shape[:2]
    alike = sum(1 for r in range(rows) for c in range(columns)
                if np.linalg.norm(bare[r, c] - np.array(ground[(r + c) % len(ground)])) <= HOLDS)
    return alike >= SAME_GROUND * rows * columns


def _the_pieces_colour(image: np.ndarray, bare: np.ndarray, left: float, top: float, width: float, height: float,
                       row: int, column: int) -> np.ndarray:
    """The colour of what stands in a square: the middle colour of its pixels unlike the bare square, from its lower
    half where those are many there, else from all of it."""
    tall, wide = image.shape[:2]
    x0, x1 = int(left + (column + 0.12) * width), int(left + (column + 0.88) * width)
    y0, ym, y1 = int(top + (row + 0.12) * height), int(top + (row + 0.5) * height), int(top + (row + 0.88) * height)
    whole = image[max(0, y0):min(tall, y1), max(0, x0):min(wide, x1)].reshape(-1, 3)
    lower = image[max(0, ym):min(tall, y1), max(0, x0):min(wide, x1)].reshape(-1, 3)
    for part in (lower, whole):
        unlike = part[np.linalg.norm(part - bare, axis=1) > HOLDS] if part.size else part
        if len(unlike) >= max(4, 0.2 * len(part)):
            return np.median(unlike, axis=0)
    unlike = whole[np.linalg.norm(whole - bare, axis=1) > HOLDS] if whole.size else whole
    return np.median(unlike, axis=0) if len(unlike) else bare


def _crosses_down(image: np.ndarray, bare: np.ndarray, left: float, top: float, width: float, height: float,
                  row: int, column: int) -> bool:
    """Whether what stands in a square runs on across its lower border into the square below: unlike the bare square
    along the last of the one and the first of the other, across the middle of them."""
    tall, wide = image.shape[:2]
    x0, x1 = int(left + (column + 0.25) * width), int(left + (column + 0.75) * width)
    border = int(top + (row + 1) * height)
    above = image[max(0, border - 3):max(1, border - 1), max(0, x0):min(wide, x1)].reshape(-1, 3)
    below = image[min(tall - 1, border + 1):min(tall, border + 3), max(0, x0):min(wide, x1)].reshape(-1, 3)
    if not above.size or not below.size:
        return False
    unlike_above = (np.linalg.norm(above - bare[row, column], axis=1) > HOLDS).mean()
    unlike_below = (np.linalg.norm(below - bare[row + 1, column], axis=1) > HOLDS).mean()
    return bool(unlike_above > 0.3 and unlike_below > 0.3)


def _the_look(image: np.ndarray, left: float, top: float, width: float, height: float, row: int, column: int,
              ys: tuple[float, float], xs: tuple[float, float]) -> np.ndarray:
    """The middle colour of a part of a square, given as shares of it across and down."""
    tall, wide = image.shape[:2]
    y0 = min(tall - 1, max(0, int(top + (row + ys[0]) * height)))
    y1 = min(tall, max(y0 + 1, int(top + (row + ys[1]) * height)))
    x0 = min(wide - 1, max(0, int(left + (column + xs[0]) * width)))
    x1 = min(wide, max(x0 + 1, int(left + (column + xs[1]) * width)))
    patch = image[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0) if patch.size else np.zeros(3, np.float32)


def _sides_of(colours: np.ndarray, sides: dict[str, tuple[float, float, float]] | None) -> dict:
    """The pieces in two sides by colour: the two colours furthest apart, each piece to the nearer; named as before
    where the sides were seen before."""
    if sides and len(sides) >= 2:
        names = list(sides)
        centres = np.array([sides[n] for n in names], np.float32)
    else:
        # Each side seeded where most pieces look alike, the second as far from the first as a crowd of them stands:
        # one odd piece (a figure standing on a piece) seeds no side.
        distances = np.linalg.norm(colours[:, None, :] - colours[None, :, :], axis=2)
        crowd = (distances < HOLDS).sum(axis=1)
        first = int(np.argmax(crowd))
        apart = distances[first] > HOLDS
        second = int(np.argmax(np.where(apart, crowd, -1))) if apart.any() else int(np.argmax(distances[first]))
        centres = np.array([colours[first], colours[second]], np.float32)
        for _ in range(8):
            nearest = np.argmin(np.linalg.norm(colours[:, None, :] - centres[None, :, :], axis=2), axis=1)
            centres = np.array([np.median(colours[nearest == k], axis=0) if (nearest == k).any() else centres[k]
                                for k in range(2)], np.float32)
        names = ["a", "b"]
    nearest = np.argmin(np.linalg.norm(colours[:, None, :] - centres[None, :, :], axis=2), axis=1)
    return {"of": [names[int(k)] for k in nearest], "colours": {names[k]: centres[k] for k in range(len(names))}}
