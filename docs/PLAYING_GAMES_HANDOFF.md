# Playing games: what is proven, what is not, and what to run

Status: Handoff · Written 2026-10-13 · Cloud container, no models, no Flash games

Aura is to play three games drawn at random from the 56 Flash games of the
Cartoon Network list, on the owner's Mac. This session could not reach those
games: archive.org is blocked here and the offline bundle lives only on the
Mac. Everything below was built and measured against the test worlds in
`tests/fixtures/worlds_that_move/`, played from their own code, and through
the whole of her play as a live game is handed to it.

The owner's direction for this work: code and architecture, no fine-tuning.
Nothing here trains a model.

## What she has now

**She reads a game's own code before she plays it.**
`core/perception/reading_a_world_s_code.py` reads a page's scripts for her
body (its size, start, paces per key, the rectangle it is kept in, jumps,
pushes, the colour it is drawn in), the things in the world (what touching
each does, how each moves, how fast, what it looks like), the ground to be
gone over (its grid, the point of her that marks it, the share that wins,
the colour of its marks), the clock and the lives. For Flash programs,
`core/perception/reading_a_program.py` turns the decompiled ActionScript into
the same shape (`from_how_it_plays`). The page's reading is cached per page
as "a world read" in `~/.aura/state/worlds/`.

**What the code says reaches her play.** Through the guide to the place:

- which thing on the screen is her body, by the colour the code draws it in;
- which kinds cost her and which are worth getting, before she meets them;
- how fast her keys move her, where her trials have not measured it, and
  that a key the code paces her by is no jump;
- that nothing inside the play area stops her, where the code says so;
- how far each kind could swerve, for the berth she keeps;
- the ground's own grid and which of its cells are marked, read off the
  screen in the colour the code marks them in;
- that a strip on that ground is ground and not a bar of what she has left.

**Her decisions are tested against the world running its own code.**
`tools/oracles/a_world_run_from_its_code.py` runs a world's JavaScript in
Node with a driven clock, checked equal to Chromium.
`tools/oracles/her_choices_on_a_world.py` feeds her real choosing the exact
state at a chosen pace, with her keys arriving late by whole decisions and,
with `jitter`, uneven waits between decisions, as live pictures come.

## Proven

Each line has a test that fails when it stops holding.

**Her reading of a world's code is what the code does.** 83 checks against
the running code, 14 deliberate misreadings caught
(`tests/test_what_a_world_s_code_says_is_what_it_does.py`). The colour read
for each thing is the colour on the canvas, in 15 worlds
(`tests/test_what_a_world_s_code_draws_is_what_is_on_its_canvas.py`).

**She wins the paint world at the pace she decides.** In its own code, on
seeds 1 to 6, each key arriving a decision late, the waits between decisions
uneven by up to 30%, the floor's marks seen
(`tests/test_she_plays_at_the_pace_she_decides.py`):

| Decisions a second | Won | Lives lost of 18 | Slowest win |
|---|---|---|---|
| 2 | 2 | 15 | 90 s |
| 3 | 6 | 6 | 69 s |
| 15, keys on time | 6 | 0 | 50 s |

With plans that change keys before her next decision, she won 5 at three a
second and lost 8 lives, and none at two a second, losing every life.

**At 15 decisions a second the code's ground decides it.** On seeds 1 to 6:
with the code's ground and plans worth the cells they cover, six wins, the
slowest in 50 s, no life lost. Plans that do not count cells won five, the
quickest in 73 s. Her own cells as ground, or no reading of the code: no win
(`tests/test_the_ground_her_code_names_is_the_ground_she_goes_over.py`).

**A painted floor is not a bar.** Rows of floor painted in a browser are
read as bars that fall without the code, and none is taken for what she has
left with it
(`tests/test_a_strip_of_floor_she_paints_is_not_what_she_has_left.py`).

## Found in the whole of her play, and fixed

`tools/measure_whole_play_on_worlds.py paint` hands the paint world to the
same entry a live game is handed to (`core/skills/sovereign_browser_drawing.py`),
from its title screen. Here it played at 3.1 to 5.1 pictures a second over the
stretches measured. Each fault below was found there and is fixed in code,
with a test.

1. **Plans changed keys faster than she decides.** A plan turned a sixth of a
   second ahead when her next decision was a third of a second ahead, so she
   went right and left on the spot while a bug reached her. Plans now change
   no sooner than her next decision (`core/agency/imagining_ahead.py`).
2. **Cells crossed between pictures were not counted.** She went back over
   them for seconds at a time. Cells along a straight row or column between
   two pictures now count, but not along a cell's edge, where the world's
   choice of side is a guess (`core/agency/ground_her_code_names.py`).
3. **A cell her steps kept carrying her past held her for 15 s.** A cell she
   goes for near at hand for 2.5 s without marking is left for 12 s.
4. **The floor was read as bars.** One strip was named for the "PAINTED 2%"
   written above it. Each shortening counted as a loss, and she came to think
   holding down cost her (`core/agency/what_its_code_says_she_sees.py`,
   `a_strip_of_ground`).
5. **"Nothing here answers to me" ended a stretch she was winning.** Her going
   is measured only between pictures 0.35 s apart or less; at three a second
   it went unmeasured for 8 s. Her body seen to go somewhere while a key is
   held now counts (`core/agency/which_one_answers_to_her.py`, `went_at`).
6. **Code paces were applied before she knew she follows the pointer.**
   Fixed by order (`core/agency/choosing_where_to_be.py`).

After the fixes, from the title screen with nothing remembered: the first
game was lost at 67% painted in 59 s, the second won in 77 s with no life
lost. Before them, three runs won their later game in 112, 114 and 124 s, with lives
lost. In both, she said "Its code says the bugs cost me." and "Its code says
which one is me" within half a second of play starting.

**Why she still loses lives here.** Recorded beside the world's own record of
each hit: she decides on a picture taken 0.25 to 0.59 s before, and her keys
take about as long again to land. A bug that bounces off a wall or turns in
that time is where she did not plan for it; one hit came as she ran into a
corner a bug was bouncing towards. Her eyes take 190 ms of each picture
here. On the Mac they are measured at 20 pictures a second alone. Cutting
the age of the picture she decides on is the work that most changes this.

## Not proven

- **The 56 Flash games.** None was run here. The reader's Flash path was
  checked against one hand-written card (game 25, Camp Lazlo: Paintcan Panic,
  `tests/fixtures/flash_games_as_their_code_says/25.json`), with the program
  itself absent.
- **Her pace on the Mac.** `core/perception/seeing_play_elsewhere.py`
  records 7 to 9 pictures a second live with her eyes in her own process,
  and 20 with them alone. This session measured only this container.
- **Two decisions a second.** With a decision's delay of half a second she
  won the paint world on 2 seeds of 6 in its own code.
- **Nineteen tests fail in this container on the code as it was before this
  session's changes, and the same nineteen after.** Among them the brawl
  punch, the beam charge, the needle press, the paddle's returns, the order
  of key trials, OCR word mending and the eyes in a process of their own.
  Whether each is this container or a real fault is not known.

## Run it on the Mac

This plays three games drawn at random from the offline bundle, the way
the demo draws them, with no model loaded and nothing on port 8000.

1. Make sure that the bundle exists:
   `ls ~/.aura/flash-campaign/offline-bundle/programs | wc -l` must show 56.
2. Run:

   ```bash
   cd ~/.aura/live-source && .venv/bin/python tools/measure_on_archived_games.py \
     --bundle ~/.aura/flash-campaign/offline-bundle --random 3 --minutes 6 \
     --trace /tmp/aura-three --out /tmp/aura-three/report.json
   ```

3. Read the first line: it gives the games drawn and the seed. To draw the
   same three again, add `--seed` with that number.
4. Read one line per game: `runs`, `won`, `error`, and `last_seen`. The
   folder `/tmp/aura-three` holds each game's log and its last screen.

To let the next cloud session play the same games, push the bundle to a
private repository and add it to that session:

```bash
cd ~/.aura/flash-campaign/offline-bundle && git init -b main && git add -A \
  && git commit -qm bundle && gh repo create youngbryan97/aura-flash-museum --private --source . --push
```

## Next, by what it is worth to the demo

1. Get the 56 games where a session can play them, and put each program's
   reading to its running, as was done here for the test worlds.
2. Find out which of the nineteen failing tests are this container and which
   are faults, starting with the ones that play: the paddle, the punch, the
   charge and the needle.
3. Her eyes cost 190 ms of a 270 ms picture in the whole of her play here
   (`WhatMoves.see`). Exact speed-ups not yet tried: gradients computed once
   a picture in `_refine`, `_with_the_view` (8 ms), `_blobs` (16 ms). A
   rewrite of the sampling in numpy gave the same numbers and was slower, and
   was taken out.
4. Survival at two decisions a second.
