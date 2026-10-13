# What a world's code says, put to the world running it, 13 October 2026

The owner asked for the games' code to be read for their physics, written
into Aura as general understanding, and each piece of her that acts on it
tested against the real code, falsified where it could be. The Flash files
were not reachable from this cloud session: the network policy refused
`archive.org`. The 31 test worlds under `tests/fixtures/worlds_that_move/`
are complete game code, and were used. Paintcan Panic's grid, field and
quota come from the earlier reading of its real program and its card.

## The oracle

`tools/oracles/a_world_run_from_its_code.js` runs a world's two scripts as
its page does, outside a browser: a canvas that draws nowhere, the page's
listeners, and a clock this side drives. Under the same keys, clicks and
time, all 31 worlds agreed with Chromium on every value of their tally and
their game state. A clock 0.05% off failed the comparison. This is the
ground truth every claim below is checked against.

## Reading the code

`core/perception/reading_a_world_s_code.py` reads a frame's function
without running it. Each quantity is written as what it was at the start of
the frame, the time, and the keys held, from the code's own arithmetic.
Of the 31 worlds, 18 move a body of hers by keys, and all 18 were read for
how: paces, a lander's pushes, a jump's lift and pull, a car's turning and
driving. The paint world was read in full: 150 px/s, kept in (0, 40) to
(464, 304), 20 px cells from (0, 40), 24 by 14, marked at her middle, 85%
wins, a 150 s clock, 3 lives, bugs that wander at 80 px/s and cost a life.

80 checks put each finding to the running world. All passed. Fourteen
readings were then made wrong on purpose: a pace 1 px/s off, a cell 1 px
wider, the grid 2 px lower, 84% for 85%, a clock 1 s short, a bound 1 px off
either way, a touch's effect, a speed 10% low, a lift and a push 5% off,
four lives for three, a reverse limit of 80 for 90. Every one failed. The
first version of the pace check had a 1 px/s tolerance and let the first
mutation through; it was tightened to 0.01 px/s.

The checks found nine real reader defects before they passed, among them a
clock read from an invulnerability timer, a jump counted as a walking pace,
a clamp on her place taken for a clamp on her speed, and speeds of things
given a random sign over-read threefold.

Card, pairs, typing and board worlds have no physics of her body and are
not read by this reader.

## The decision piece: ground to be gone over

Put to the paint world's running code with what she would have measured
handed over exactly, her choosing painted 29%, 11% and 5% of the floor in
150 s on seeds 1 to 3. The world needs 85%. Two causes showed in the trace.
She measured ground in cells of her own size over the whole picture, so the
strip above the floor was ground: she held up against the floor's top edge
from 20 s to 38 s. And the nearest cell not yet gone over was a few pixels
off, worth about 0.02 to a plan, so any risk outweighed it: she stood still
for the first 4 s.

With the code's ground laid on her picture
(`core/agency/ground_her_code_names.py`), and each imagined plan worth
the cells it would go over (`core/agency/imagining_ahead.py`), she won
seeds 1 to 3 in 46, 59 and 41 s at 30 decisions a second, losing no life.
At 15 a second she won seeds 1 to 5 in 43, 55, 37, 44 and 40 s, no life
lost. Seeds 4 to 7 at 30 a second won in 97, 50, 42 and 87 s.

Each half alone was measured on seeds 1 to 3 at 30 a second:

| her choosing | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| before | 29% | 11% | 5% |
| her own cells, scored as ground | 33%, all lives lost | 62% | 41% |
| the code's ground, not scored | 32% | 31% | 27% |
| the code's ground, scored | won, 46 s | won, 59 s | won, 41 s |

Neither half wins. At every one of 180 choices over 12 s, the cells she
counted as gone over were exactly the cells the world counted as painted.

Paintcan Panic's real code says its grid is 30 by 25 cells, its hero kept
in (10, 110, 590, 390), and its coverage the painted count over 2.4. Laid
from the first two, the cells the hero can reach are 20 by 12, which is 240:
the third. The two facts were read independently from the same program.

## What is not shown

Her whole live play, from pixels, was run on the paint world in this
container with and without the code read in, seeds 1 to 3 at once. Her
eyes ran at 6.1, 5.0 and 4.5 pictures a second with it and 5.2, 5.3 and 6.1
without; alone, 8.1. She lost all her lives in every game either way. Her vision model needs MLX, so
the full desktop path could not leave the title screen at all.

With one picture of input delay at 15 decisions a second, she still won
seeds 1 to 5, in 133, 65, 52, 98 and 70 s. At 8 decisions a second with
125 ms of input delay, the decision piece
alone lost seeds 1 and 3. Handing her planner the delay, as live play does,
made it worse, 1 win in 3. The traces show two causes: a wandering bug that
turned toward her 64 px away when she had one decision left, and her plan
flipping between up and down each picture while a bug came from 75 px to
29 px. That is the next piece to isolate. It is not fixed here.

## Corrections, later on 13 October

Two findings above were wrong.

The planner was not flipping between up and down because of how it uses
the delay. The test harness handed her choosing the key the world was
applying, which with one picture of delay was the key she chose two
pictures before. Live play hands her the key she last sent. With the
harness corrected, at 15 decisions a second she won seeds 1 to 5 with 67 ms
of delay and seeds 1 to 5 with 133 ms. At 8 a second with 125 ms she won
seeds 2 to 6 and lost seed 1, losing 7 lives in all.

The loss was a bug her straight-line reckoning had passing 1.5 px clear,
which then turned into her: its code turns a bug by up to 1.2 radians at a
time. She now keeps clear of what costs her by as far as it could swerve
before she can answer it: the most its kind's going has been seen to change
at once, times a picture and her delay
(`core/agency/how_far_a_thing_could_swerve.py`). At 8 a second with 125 ms
she won seeds 1 to 6, losing 1 life in all with the swerve measured in
play, and 3 with it taken from the code. On the dodge world nothing
changed. On the collect world no life was lost either way, and she took
265 coins for 280 and 280 for 310 on two seeds of three: foes that rebound
off walls count as swerving.

The finding that neither half of the ground wins alone does not hold with
the swerve berth in. On seeds 1 to 6 at 15 a second: before, none won, 6%
to 52% painted. Her own cells scored as ground: none won, 5% to 45%. The
code's ground not scored: seeds 2 and 4 won, at 132 and 133 s, the rest 34%
to 62%. Both together: all six won, in 44, 49, 40, 44, 44 and 51 s, no
life lost.

## The whole of her play, later on 13 October

The paint world was handed to the entry a live game is handed to
(`tools/measure_whole_play_on_worlds.py`), from its title screen, in this
container with no models. She played at 3.1 to 5.1 pictures a second over the
stretches measured. Her eyes took 190 ms of a picture on average, 336 ms at
the 90th percentile, and the canvas read 31 ms.

What the code says reached her play: she said "Its code says the bugs cost
me." and "Its code says which one is me" within half a second of play
starting, and her eyes matched her body, the bugs and the painted floor to
the code's colours on every picture checked.

Faults found there, each now fixed with a test:

- Rows of painted floor were read as bars. One was named for the
  "PAINTED 2%" written above it; each shortening counted as a loss.
- A stretch she was winning ended as "nothing here answers to me, and it is
  going against me": her going is measured only between pictures at most
  0.35 s apart, and it went unmeasured for 8 s.
- In the world's own code at three decisions a second, her plans changed keys
  a sixth of a second ahead, and she went left and right on the spot. At four
  a second she crossed a row between two pictures each time, never saw
  herself in it, and went up and down across it for six seconds.

Measured afterwards in the world's own code, seeds 1 to 6, keys a decision
late, waits uneven by up to 30%, marks seen: at three decisions a second six
won, the slowest in 68.6 s, 6 lives lost; at two, two won, 15 lives lost.
Plans changing before her next decision: five won and 8 lives lost at three,
none won and every life lost at two.

The ground ablation at 15 decisions a second, with the cells she passes
between choices counted: both halves won all six, the slowest in 49.5 s, no
life lost. The code's ground not scored won five, in 73.0, 92.4, 108.1, 116.2
and 127.5 s, and seed 3 ran out of time at 81%. Her own cells: none won. No
reading of the code: none won.

The whole of her play after the fixes, nothing remembered: the first game
lost at 67% in 59 s, the second won in 77 s with no life lost. Lives were
lost where she decided on a picture 0.25 to 0.59 s old and her keys took
about as long again to land; one bug she ran into had bounced off a wall in
that time.

Nineteen tests fail in this container on the code before these changes and
the same nineteen after them, among them the brawl punch, the beam charge,
the needle press, the paddle's returns, and the eyes in a process of their own.
