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
| Dimensions | 18 |
| Values across them | 107 |
| Kinds of place they make, multiplied out | 26,754,416,640,000 |
| Values with an answer | 105 |
| Games | 56 |
| Distinct combinations among the 56 | 52 |
| Values a game needs | 0 to 53, 23.4 on average |
| Games with an answer for every value they need | 40 |

Two of the 56 need the same combination as another game; the rest are each a
place of their own. Answering the 2 values with none answers every value the
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

## Sound

What a page plays is heard the way its drawing is read: a script loaded before
the page's own wraps the calls that start a sound (an audio buffer, an
oscillator, a media element) and notes each as it starts, with a signature by
which the same sound is known again (`core/perception/what_the_page_plays.py`).
The sounds come to her with each picture. What each means is learned from what
happens around it, against what chance would give
(`core/agency/what_a_sound_comes_with.py`): it comes with a point, with a blow,
with a press of hers, or it is a warning, which the blows she takes follow at a
steady while. A warning is read from the blows and not from its hearings, so a
warning she heeds is still one. While a warning's blow is due, the spot she was
on as it sounded is weighed as danger, as a thing coming at her is; where the
blow found her wherever she went, the spot is let go. In a world heard and not
seen, a siren a second before a strike on her spot, heeding it cut the strikes
after the first minute from 54 to 11 over ten runs, and every sound's meaning
was learned in all ten. In whole play, in a browser world whose siren is heard
and not seen, she learned what each sound meant in every run, and heeding the
siren cost her 23 lives over three runs of two and a half minutes against 39
when deaf to it.

## Allies to keep

What is on her side is found by what losing it cost her (who is who, below);
what comes for it is found as what comes for her is, by how its going lines up
with the way to it, picture after picture (`core/agency/keeping_allies.py`). A
threat coming for an ally is aimed at before one that is not, and is worth
taking out by what losing the ally would cost her, by how soon it gets there.
In a world where threats came for her and for an ally walking to and fro,
played for herself she lost the ally about twenty times in two minutes; keeping
it, not once, and she was hit no more.

## Spending a supply

Which counter a key spends is found by the counter falling just after the key
is pressed, press after press; then each press must clear a bar
(`core/agency/spending_a_supply.py`). The bar is the old answer to spending a
budget on chances that come over time: of the chances still to come in the time
left, as often as they have come, only as many as there is supply for, counting
what refills, are taken, and the bar is the worth that share of them reach. A
shot's worth is what taking its target out is worth. Ten units over a minute of
chances took 1,396 spent above the bar against 727 spent on whatever came first;
where the supply refilled faster than chances came, the two did the same.

## Pushing a thing to its place

What she pushes is a thing she was touching, holding a way, that went that way
with her, press after press; where it goes is the still thing it was on when a
count went up (`core/agency/pushing_a_thing.py`). To get it there she lines up
behind it, on the side away from its place, along the way it has furthest to
go, and pushes; to get behind it she steps back clear of it, or goes out to a
lane beside it and along the lane, never touching it on the way. In a world of
a crate and a pad, set afresh each time it was got home, she got it home 80
times in six minutes; making straight for the crate, not once.

## What she plays for, and who is who

Two dimensions are about her as much as the place: what she plays for, and
what each thing in it is to her. Both were added on 11 October.

**Motivation** (`core/agency/what_she_plays_for.py`). One end leads: the
person's ask, else what the game's words say, else what the screen counts. A
target to reach, the most to raise, to last, or to find out. Every other pull is
priced in that end's terms, from her record of the run and the runs before:

- Self-preservation. What she can lose is counted in blows: a hit off a bar, or
  a life where there is no bar. Blows are taken to come at random at the rate
  her record shows (a Poisson process), so a blow is worth the play it would
  cut short. Toward a target it is the chance of getting there it takes away,
  against what one gain adds. Being hit often does not make a blow cheap, since
  how often she is hit depends on her care. Her care is that worth against the
  thirty seconds a life is believed to last, which is what her play's berth was
  set for. In a race to twenty against a clock, with each gain taken a quick
  risky way or a slow safe way, care that follows the stakes did as well as care
  held fixed at every berth tried, and where the fixed berth played too safe to
  get there in time, it reached the target 89 per cent of the time against none.
- Threat elimination. A threat left alone deals blows at the rate its kind has
  dealt them, for as long as threats stay; taking it out is worth that by what a
  blow is worth, and is done where that comes to more than the time it takes.
  In six kinds of place, never shooting and always shooting each did less than
  half as well as the best somewhere; priced from the two runs before, her third
  run did at least 85 per cent as well as the better of them everywhere, and
  better than both under a long clock (45 against 34).
- Finding out. A thing not yet known is worth the value of knowing it, from what
  the kinds met so far turned out to be.
- When it stops paying. A count that has stopped moving for much longer than it
  took to move before is spent: she says so and tries what she does not know.

**Who is who** (`core/agency/who_is_who.py`). Each thing is hers, a threat, a
prize, friendly, friendly competition, neutral, in her way, negligible, part of
the place, or unknown, with nothing in the game to say which. The progression is
the one military symbology standardised (MIL-STD-2525, APP-6): pending, then
unknown, then assumed friend or suspect on signs, then friend, neutral or
hostile on its own acts. Everything rests on what play measures: what meeting it
did to her counts, whether it keeps coming for her, whether it takes the prizes
she goes for, whether it stopped her, whether a blow came as it went, whether it
has ever mattered. In a world of every kind of thing, sixty runs of a minute
each designated every thing rightly but one companion, never lost and just come
back, which was taken for neutral, as what it had done said; a companion lost
only once is left as assumed friendly. A thing that goes off the edge of the
picture as a blow comes got past her, as an invader that lands does, and is a
threat; one that goes in the middle of the picture as a blow comes was hers to
keep, and is friendly. Threats and suspects are what threat
elimination prices; what is friendly, friendly competition, neutral or part of
the place is never shot at unless shooting its kind is what pays.

## Where the dimensions meet

Independence fails where two answers read or write the same thing. Five such
seams are kept to one owner each, so that the answers compose:

- **Where she will be.** What a control does, what the place does to her and
  how others behave all feed one prediction of the next moment
  (`core/agency/imagining_ahead.py`). A pull, a heading and a thing coming for
  her are terms in it, not separate predictions.
- **How much it matters now.** A clock running down, what she has left and
  things coming faster all feed one urgency, which the choice of what to go for
  reads.
- **What a thing is to her.** A legend, the words, what meeting did, and her
  own state feed one stance per kind (`core/agency/what_meeting_things_does.py`),
  and what each thing is to her is read from the same record
  (`core/agency/who_is_who.py`), not kept beside it.
- **What anything is worth.** Care, a detour to find out, and a threat taken
  out are all priced in the one end that leads (`core/agency/what_she_plays_for.py`).
- **What she is keeping to.** The way taken up, the plan, and the thing gone
  for are held in one place per game, so one is not dropped by another.

## The values with no answer yet, by the games each stops

| Games | Dimension: value | What answers it |
|---|---|---|
| 13 | searching: a way through | a way across the cells she can stand in, round what she has found in the way |
| 5 | searching: where a piece goes | each place a falling piece could land, judged by what it would clear and leave |

Each is built as a module of its own, shown working in a world of its own
before it is joined to her play, and listed in the registry with its test.
