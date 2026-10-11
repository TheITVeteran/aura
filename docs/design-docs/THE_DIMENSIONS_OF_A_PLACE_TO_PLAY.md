# The dimensions of a place to play

Status: Design map · Every way places to play differ, the values each takes, what does each, and the count over the 56 games

A game is a point in a space of independent dimensions. Which screen of it she
is on is one; whose turn it is is another; what each of her controls does to
what is hers is a third. A way of playing answers one value of one dimension
and works whatever the other dimensions are. So the ways needed are the sum of
the values, not their product, and a place she has never seen is played by the
ways for each of its values together.

The registry is `core/agency/dimensions_of_play.py`. The counts below are
made by `tools/the_dimensions_of_play.py` from what is already kept of each
game: the mechanics, ways and situations its own file showed
(`docs/design-docs/what_it_takes_to_beat.json`) and the ways the design map
gives it (`THE_56_GAMES_AND_WHAT_THEY_ASK.md`).

## The count, 11 October 2026

| | |
|---|---|
| Dimensions | 16 |
| Values across them | 85 |
| Kinds of place they make, multiplied out | 222,953,472,000 |
| Values with an answer | 79 |
| Games | 56 |
| Distinct combinations among the 56 | 52 |
| Values a game needs | 0 to 41, 17.8 on average |
| Games with an answer for every value they need | 21 |

Two of the 56 need the same combination as another game; the rest are each a
place of their own. Answering the 6 values with none answers every value the
56 need, where the dimensions are independent.

Eight values were answered on 11 October, each by a module shown working in a
world of its own and then joined to her play: the pull, the clock, more and
faster and urgency (`core/agency/what_the_place_does_to_her.py`), a thing gone
for (`core/agency/a_thing_gone_for.py`), meeting by what she is now
(`core/agency/meeting_by_what_she_is.py`), what comes for her
(`core/agency/what_comes_for_her.py`) and keys that turn her
(`core/agency/which_way_she_faces.py`). A way through
(`core/agency/a_way_through.py`) finds its way in its own world and is joined
to her play, but in whole play on a world of rooms she has not yet got through
a gap, so it stays on the list below.

## Where the dimensions meet

Independence fails where two answers read or write the same thing. Four such
seams are kept to one owner each, so that the answers compose:

- **Where she will be.** What a control does, what the place does to her and
  how others behave all feed one prediction of the next moment
  (`core/agency/imagining_ahead.py`). A pull, a heading and a thing coming for
  her are terms in it, not separate predictions.
- **How much it matters now.** A clock running down, what she has left and
  things coming faster all feed one urgency, which the choice of what to go for
  reads.
- **What a thing is to her.** A legend, the words, what meeting did, and her
  own state feed one stance per kind (`core/agency/what_meeting_things_does.py`).
- **What she is keeping to.** The way taken up, the plan, and the thing gone
  for are held in one place per game, so one is not dropped by another.

## The values with no answer yet, by the games each stops

| Games | Dimension: value | What answers it |
|---|---|---|
| 18 | seeing: sound | what a page plays, heard as events, and what each came with (a point, a life lost) |
| 13 | searching: a way through | a way across the cells she can stand in, round what she has found in the way |
| 9 | meeting: pushed to a place | a thing she runs into moves; got to its place by pushing it from its far side |
| 9 | sides: allies to keep | someone on her side, kept clear of what would cost her through them |
| 6 | logistics: spend a supply | a meter spent where it does most, and saved where it does least |
| 5 | searching: where a piece goes | each place a falling piece could land, judged by what it would clear and leave |

Each is built as a module of its own, shown working in a world of its own
before it is joined to her play, and listed in the registry with its test.
