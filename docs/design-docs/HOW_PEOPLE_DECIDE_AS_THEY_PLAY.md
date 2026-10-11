# How people decide as they play, and how her choices are bound to theirs

Companion to [HOW_PEOPLE_AND_SYSTEMS_ACT_TOWARD_AN_END.md](HOW_PEOPLE_AND_SYSTEMS_ACT_TOWARD_AN_END.md)
and [THE_56_GAMES_AND_WHAT_THEY_ASK.md](THE_56_GAMES_AND_WHAT_THEY_ASK.md).
The question here is narrower than either: when a person playing one of these
games picks one act over another, what are they going on, why that act, and
what do they expect it to do? And how is each of her choices tied to an act,
a reason, a strategy and something she knows, so that the next choice can
use what the last one came to?

## What people do

**They recognise, then check.** Experienced people rarely compare options.
A situation seen as a typical one brings with it what to watch, what to
expect, what to aim at and a typical act; where there is time they run the
act in their head, and an expectation that breaks sends them back to look
again (Klein's recognition-primed decision model, 1993; studied in fire
commanders, pilots and athletes). A player asked why they moved names the
cue, the aim and what they counted on.

**They bring a theory of the world, and learn its rules in minutes.** People
reach competent play of a new Atari-style game in a few minutes because they
come with intuitive theories of objects, agents and their interactions, and
plan with them (Tsividis et al. 2017; EMPA, Tsividis et al. 2021, which
matched human learning speed across 90 games). Take away their priors and
they slow down: in Dubey et al. 2018 a game whose objects' looks, meanings and
affordances were hidden took people more than 20 minutes instead of about 2.
Platforms mean standing, ladders climbing, a key a door.

**They act to think.** Tetris players rotate a falling piece more than its
placement needs, to see where it fits rather than to place it (Kirsh and
Maglio 1994, "epistemic action"). Trying a control where nothing can hurt is
the same thing.

**They sample, simulate, update.** In the Virtual Tools game people place one
tool to get a ball into a goal. They start from a guess drawn from similar
situations, imagine its effect, try the best one, and move their guesses
toward what the real try showed (Allen, Smith and Tenenbaum 2020, PNAS).
A miss is information about the setting that caused it.

**Physics they lean on, by kind of play.**

- Catching and dodging: time to contact, read off how fast a thing grows or
  closes (Lee 1976). People intercept by where a thing will be, and their
  hand paths change with the angle it comes from (Ledouit et al. 2024).
- Aimed shots and launches: change one setting at a time and correct by the
  last miss: short, long, left, right. Bank shots reflect at the angle they
  came in.
- Falling and jumping: a jump traces a parabola; game physics is often
  asymmetric (heavier on the way down) and forgiving at ledges, so players
  jump from the edge and look for the landing first.
- A lander: kill sideways speed first, then come down slowly; full thrust
  late is efficient and unforgiving.
- Timing presses: people tap slightly before the beat (negative mean
  asynchrony, Repp 2005); the press is made on the predicted arrival, not on
  seeing it.

**Logistics: what first, what to keep, what to spend.**

- Soonest first, then most worth among what can still be reached. Chasing
  what cannot be reached in time and missing two that could is the catcher's
  classic mistake.
- Routes: people's tours of many points follow the outline of the whole set
  and link near neighbours inside it (MacGregor and Ormerod 1996; Vickers et
  al. 2003); they rarely cross their own path.
- Held value is at risk in proportion to how much is held: bank early.
- Locks and keys: adventure players keep a list of what they hold and what
  each place wants, and work back from the goal. Designers draw this as a
  puzzle dependency chart (Ron Gilbert, Maniac Mansion 1987), built from the
  end back.
- Boards in turns: count material, keep the back row, hold the centre, trade
  when ahead, and look one move ahead for every move, which removes most
  blunders. A capture that must be taken can be used as a trap.

**Shooters.** Know which shots are aimed at you and which are a fixed
pattern; aimed shots are drawn to one side and then passed through the gap
they leave ("herding"); keep your dodges small, since a wide dodge spends
the room you need for the next.

## What systems do with people's play

- **Inverse dynamics.** VPT (Baker et al. 2022) labelled 70,000 hours of
  Minecraft video with the actions that must have caused each change, using a
  model trained on a small labelled set; Pixels2Play (2025) and a 2025 3D
  game agent did the same across many games. An inverse model looks at what
  happened next, which a policy cannot.
- **Learning from what changed, without the actions.** Behavioural cloning
  from observation (Torabi, Warnell and Stone 2018) learns its own inverse
  model from its own play, then reads an expert's states with it.
- **Objects as the interface.** Robots learn from human video by the motion
  of the object, not the hand: Im2Flow2Act (object flow, CoRL 2024), ATM
  (any-point tracks, RSS 2024), Track2Act (tracks from web video, ECCV 2024).
  What a person made a thing do is what the robot is asked to do with its own
  body.
- **A video as a path of checkpoints.** Aytar et al. 2018 embedded one
  YouTube video of each game and rewarded reaching its frames in order;
  their agent passed human scores on Montezuma's Revenge, Pitfall! and
  Private Eye with no game reward at all.
- **Reasons from actions.** Bayesian inverse planning (Baker, Saxe and
  Tenenbaum 2009) reads goals off actions by asking which goal makes them
  the sensible thing to do. Players speaking their reasons after each act in
  Frogger gave a model the words to explain an agent's acts (Ehsan et al.
  2018).

## The binding

Every choice her fast play makes is kept as one record
(core/agency/a_choice_and_its_reasons.py):

| Part | What it holds | Where it comes from |
|---|---|---|
| cue | what she noticed | the thing, where it will be, when |
| act | the key held, the click, nothing | her play's own choice |
| because | the reason, for this choice | the aim and what the act does for it |
| craft | the strategy it is a case of | core/agency/the_craft_of_play.py, drawn from people's play |
| knows | what it counts on, each with how it is known | her own measurements (her speed, a kind's pull, an edge) |
| expects | what should follow, and by when | the kind of choice |
| came to | what did follow | what met her, what she gained and lost |

A choice is recorded when it changes (a new aim or a new reason), not every
picture. When its time is up it is settled against what happened. How each
craft's choices have come out here is kept with the game, and by craft name
across games, because a craft is said for anywhere. Her narration says the
reason for each new kind of choice once, from the record; the guide to the
place gets the same lines to reason with.

The catchers' triage is in her choosing: an arrival she cannot reach in time
gives way to one she can (core/agency/a_choice_and_its_reasons.py
`the_arrival_to_meet`).

## Boards, as the people who play them well judge them

A board is seen as a lattice of squares and what stands on each, by side or by
kind (core/perception/a_board_seen.py). Which family of rules it is played by
comes from its look, its words and every move seen made on it, a family that
could not have made a move all but dropped and draughts' variations tried where
none explains it (core/agency/which_board_game.py). The families are draughts, a
line of your own (put down or dropped down columns), flanking, and swapping to
line up (core/agency/rules_of_a_board.py), and one search plays them all
(core/agency/playing_a_board.py).

How a position stands is judged as the person whose recorded game against a
hard draughts engine the general-reasoning work studied judged it
(core/agency/judging_a_board_as_people_do.py): they won five pieces to eight by
taking the engine's options, not its pieces. The reasoning modules drawn from
that game and their own account of it are bound to the search: room to move for
each side counts beside the family's own count (what_is_still_open); the way of
winning is chosen from how her earlier games of the family ended and what told
those endings apart (which_way_to_win); a single legal move is made without
weighing (when_the_situation_decides); well ahead she plays the move that leaves
the other side least, and behind the one that leaves her most
(enough_rather_than_most); and a move that leaves open to capture a piece she
cannot do without is passed over for one nearly as good
(what_she_cannot_afford_to_lose).

## Reading someone else's play

core/perception/watching_someone_play.py reads a recording: the player's thing
is the one that starts, stops and turns of its own accord and lasts; each change
of its going is an act, read as the key of hers that makes that going where she
has played the game; going toward a thing that it then took was going for it,
going away from one that was closing was keeping clear of it; and the places
along the way are kept as checkpoints. What it showed is kept with the game
(tools/watch_someone_play.py hands her a recording) and used for any kind she
has not met yet, matched by look, ahead of what her situation would suppose.

## What she keeps

Per game, between sittings: which thing is hers and what each key does to it,
the mouse, the looks of every kind and what meeting each did, the physics she
measured, the aim settings that last got there, how each craft's choices came
out, and what a recording showed (core/agency/what_she_keeps_of_a_game.py).
Per place, with its guide: controls and where each was learned, goals, named
parts, mechanics, strategy, sources, what the place is, the instructions she
heard (re-read from the store of every sentence she has read), her plan with its
means, steps and what each was to show, the locks she met, seen or said, and the
board family she settled it is with its variation and her side
(core/cognition/a_guide_to_a_place.py). Everywhere: every sentence of rules she
has read, and how her board games of each family ended.

## The games, by what people do in them

The 56 games of the design map fall into kinds that people play the same way.
Each row names the crafts (the_craft_of_play) that carry the people's way of
playing it. Search for walkthroughs and footage of these particular titles
found little: for most of them people's play survives as unlabelled video.
The evidence here is per kind of play; footage of each game is for her to
read herself, the way the systems above read video.

| Kind | Games (# from the map) | People's way, as crafts |
|---|---|---|
| catching what falls | 20, 22, 23, 38, 40 | getting under what to catch; sidestepping what falls |
| dodging and lasting | 9, 14, 16, 18, 25, 35, 41, 48, 53 | sidestepping what falls; knowing which one is me |
| shooting what comes | 2, 12, 16, 26, 30, 32, 33, 34, 37, 39, 47, 49, 52 | lining up to throw or shoot; fighting one on one |
| jumping a way along | 12, 14, 27, 30, 41, 43, 47, 49, 50, 53 | climbing and jumping a way up; going screen to screen |
| aimed sends | 4, 6, 10, 19, 24, 36, 55 | launching for distance; lining up to throw or shoot |
| timed presses and sequences | 15, 36, 44, 51, 54 | learning lesson by lesson; reading before acting |
| finding and clicking | 0, 13, 21, 46 | finding what is shown; opening every door |
| using things, errands | 3, 7, 13, 45 | fetching and trading; opening every door |
| building a way through | 11, 28, 46 | building a way through; trying again, differently |
| a board in turns, grid rules | 8, 31, 54 | planning moves before they play out; setting off a chain |
| making, typing | 1, 5, 17, 29, 42 | reading before acting |
