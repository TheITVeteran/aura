# What the runners won by

Status: Working ledger · Programs written to beat one kind of game each, outside Aura, and what made each one win, said
as a general way of playing and where it went in her

A reference runner (`tools/reference_runners/`) is a program that plays one bundled game the way a person who has
read its rules and watched it played would: it may know the game, where its bars are, what its keys do. Aura never
runs one. Each runner is the proof that something wins, so the reason it wins can be taken into her as a way of
playing keyed to what any place shows and says. Nothing a runner knows of its game goes into her: not a name, a
place on the screen, a colour or a number of seconds.

All runners share `tools/reference_runners/a_game_at_hand.py`: the offline bundle served on loopback, the pinned
emulator in a headless browser, the movie's frames streamed (about 27 a second; Ruffle draws with WebGL, so its
canvas reads back blank and the devtools screencast is used), and a hand and keys on it.

    python -m tools.reference_runners.look_around 2 --out /tmp/rr/g2 word:START wait:2 key:Space
    python -m tools.reference_runners.a_throw_at_someone_across_a_room --seconds 120 --out /tmp/rr/food
    python -m tools.reference_runners.signs_coming_along_a_strip --seconds 300 --out /tmp/rr/zap

## Read first, then run

A game's program and artwork say what it will show and ask before anything plays it: every sign it draws, what is
shaded like what, where and by what each thing is judged, and every rate. Each of the faults the second runner met
was readable there (the pad's shading is artwork; the colour change on the pad and the two-key signs are in the
program). So the order of work for a game is: its sheet from the program and artwork; the abilities that sheet asks
of her, each with a test on the real artwork; the runner, to confirm the sheet; her own run, to confirm the build.
A runner that finds something the sheet did not say has found a gap in the sheet.

## Found by every runner

- **A button answers a pointer that came over it.** A click made by putting the pointer down on a Flash button and
  pressing at once did nothing, twenty times running, on a game's "TRY AGAIN"; the same click arriving from beside
  the button and resting there before the press worked at once. Her hand already clicks this way off the screen's
  play (`core/skills/screen_pursuit_on_a_page.py`, `as_a_hand_clicks`); in play it presses at once, which is right
  for a throw and wrong for a button shown mid-play.
- **A bar is read only where it is drawn.** A game zooms, shakes or flashes when a point is lost; a bar read from
  where it stands in the calm picture then reads empty and a round is taken to be over. A reading counts where what
  is there is the bar's own colours, and once a few pictures agree.

## A contest of throws, the player not in the picture (Food Bash, case 2)

Words: "Take aim with your mouse and click to throw. Dodge food by pressing the space bar. The first to splat the
other five times wins!"

What wins:

- **The room is what stays put.** The median of the last few seconds' pictures has no one in it who runs about; what
  differs from it is the opponent, what is thrown, and the pointer's sight. Thrown at, he is the biggest steady thing.
- **Throws are led, and many.** He runs to and fro; throws aimed 0.2 to 0.4 s ahead of him took nearly every splat,
  and a throw every third of a second won rounds against two opponents before dodging worked at all. Aura's
  `core/agency/leading_a_click.py` tries leads and keeps the one that takes.
- **A dodge is made at the tell, not at the throw.** Filmed picture by picture, the food is in the air for 0.2 to
  0.3 s and doubles in size every twentieth of a second; by the time three pictures show it growing it is on the
  screen. Dodging on growth came too late every time (13 of 13 hit). The opponent stops running and stands for 0.2 to
  0.5 s before each throw; dodging when he stops spared most throws, and the runner won rounds against two opponents.
  Aura learns tells in `core/agency/warning_signs.py` and ducks as one begins, but only after losses have taught it.
- **How big a thing grows is not how soon it arrives, in a drawn world.** The time to contact read from growth (Lee's
  tau) holds for a thing coming nearer in a real view, whose size goes as one over its distance. A drawn throw is
  scaled up by a steady amount each picture, and tau read from it said 0.1 s while it landed 0.6 s later.

What it still lacks: the second opponent throws without stopping first, and was dodged by nothing.

## Signs coming along a strip to a mark, each one a key (Zap to It!, case 15)

Words: "Follow the arrow keys in Grim's Magic Book to cast the correct clean-up spell. Be careful! A mistake could
create an even bigger mess!"

What its program says, read before the runner was written: the stage is 432 units wide; arrows are made 100 units
apart and go 5, 6 or 7 units a frame; the front arrow is judged over the last 60 units before 420 by exactly which
arrow keys are down then; an arrow that passes 420 unanswered costs one of three lives, which come back at each new
spell; from the second spell a sign may be two arrows, asking for two keys at once; twenty spells in three rooms.

The runner cleans all three rooms in 3 minutes 37 seconds (371 signs judged, pairs among them) and reaches the game's
last line, "thanks to Billy the kitchen is a mess again". What it took, each thing a lost game before it was done:

- **Ink that stays is the strip's, however like a sign it reads.** The pad at the end is shaded with a shape in the
  arrows' colour that reads by its outline as a down arrow, and the strip's edges are that colour too. The runner
  held "down" for a whole game. A sign is ink that comes along.
- **A sign is not to be seen while it is judged.** On the pad it swells and changes colour. Its keys stay down for a
  moment after it goes from sight, timed from the signs' pace, and what shows beyond where the front ones go from
  sight is one already judged.
- **Order, not following.** Matching each mark to the nearest mark of the last picture took the two halves of a
  double sign for one mark read two ways, and lost the thread at twenty pictures a second. Only their order along the
  strip is used: the front one has changed when the furthest sign is suddenly further back.
- **A sign made of signs she knows.** The pairs are two arrows drawn over one another, the lower one lying on the
  strip's edge and losing its foot with it. Read by outline a pair is one of its halves. Laid over by the four single
  arrows learned in the first spell, and scored by intersection over union, every pair reads as both
  (`core/perception/signs_made_of_signs.py`).
- **The emulator's splash is pressed until the movie plays.** One press as it came up took nothing.

Where it went in her: `core/perception/signs_along_a_strip.py` reads the line whole once its ink and size are seen,
`core/agency/each_sign_in_its_turn.py` holds each sign's keys until it is judged, and
`core/agency/signs_that_come_along.py` finds the line in the first place and hands over to them. None of them has a
colour, a place or a pace of this game in it: the ink, the size, the pace and where signs go from sight are all seen
in play. Proven on an invented strip with pairs, an edge in the signs' ink, a pad shaded like an arrow and signs that
change colour where they are judged (`tests/test_a_sign_made_of_signs_is_answered_with_all_its_keys_in_its_turn.py`).

What her own run needed that the runner did not, each a round lost offline on 12 October before it was built:

- **A loop that does only this.** Read inside the loop that plays everything else as it happens, the line was seen
  three to ten times a second, and a sign crosses the place it is judged in a third of a second. A loop that looks,
  reads the line and holds (`answer_signs_as_they_come`) sees twenty-nine.
- **The menu she just left is not the screen she is on.** After the NEXT of the instructions, that NEXT was still
  what her reflexes went by, and the round was over before the screen was read again. The old menu's own pixels are
  held against the frame before anything is read (`core/skills/current_words_before_play.py`), and a place whose
  words ask for signs to be followed is taken to go on without her until a line is seen or ruled out.
- **A reading slower than a frame still stands where its words still are.** Reading a frame's words took longer
  than the frame stayed current and was thrown away, so a screen that asked her to choose a character was not known
  for one, and an arrow sliding beside the choices was held to for 22 seconds.
- **Ink by agreement, kept while it answers.** The line's ink is what four readings agree on, and what is known of
  a line is forgotten when two stretches running judge nothing on it. One run took its ink from another screen and
  read an empty line for the rest of the game.

With these she cleaned the first room on her own from a frozen copy: 61 seconds of play at 29 pictures a second,
72 signs held for, and the game's "Excellent! This place is clean as a whistle". She lost the second.

What the frames before each life lost then showed, her general loop driven straight against the game with the
pictures kept (the runner left some signs unread too, and got through on the two spare lives each spell gives):

- **A sign is ink that has come along since the look before.** The book the arrows come out of shows arrows in
  their ink whirling on its page, in line with the strip, and they were read as signs for the whole game. The ink
  of the last look, moved on by some distance a sign may have come, lies under a sign on its way and under nothing
  else (`StripOfSigns._came_along`).
- **Only what has come along the line is answered, and what shows ahead of the front one came from nowhere.** A
  sign that was not answered is drawn on, bigger, past where the others went from sight, and one crossing the rim
  of the pad goes from sight and comes back. Each put the order out by one, and every sign after it was answered
  with its neighbour's key until the line was empty.
- **Keys stay down until midway to the next sign's going from sight.** A sign goes from sight at the pad's rim
  and is judged a twentieth of the picture further on. Keys let go a moment after it went were let go too soon
  about one time in six.
- **A look that misses every sign is a look that missed them.** Pictures do not come as evenly as their times say,
  and one comes again now and then. A front sign goes from sight alone; where the one behind it is gone too, the
  look is forgiven.
- **Each sign is read again once it is the front one.** What it was read as while behind counts for two readings
  of the four, so a reading handed on by a mistake of order is outvoted.
- **Whatever reads the line is loaded before the first look.** The first sign of a game was read 0.7 seconds late,
  the time its reader took to load.

Driven so, the same code cleans all three rooms in 224 seconds without losing a life. Before these it lost
fifteen, and the game in the third room.
Each of these is in the invented strip now: a page of whirling arrows, a sign drawn again bigger on the pad, and a
place of judging well onto it.

Then her own run, from a frozen copy, the prompt's terms and her pursuit around the loop: all three rooms
cleaned, 61, 72 and 81 seconds of play at 30 pictures a second, 46, 63 and 75 signs held for. Two more things it
took: a thing has come some way in a straight line before it is one of a stream (one run took a whirling arrow for
the first sign and looked for the line on the book's page), and signs being answered are not left to go by while
the round is looked at (a round's four minutes were up twelve signs from the end of the last room, and the game
was lost in the ten seconds the looking took).

Still open: after its third room the game says the kitchen is a mess again and begins again from the kitchen,
faster. Nothing on any screen says she has won, and she does not yet take coming round to the first room for
having done all of it.

Still open: a sign made of one single she knows and one she has not yet seen alone. It is fitted well enough by the
known one and another laid over the same part of it, and both their keys are held. The game shows every single
many times before its first pair, so it never meets this; the invented strip does when pairs start early.

## Timing, from the programs and people that do it well

What the programs that time well do, read for this ledger (October 2026):

- Fighting-game AI (the DareFightingICE competition) splits every attack into startup, active and recovery frames;
  the players that do well predict the opponent's next attack from its current state (nearest neighbours over past
  states) and answer within the startup, since the platform shows every state a quarter second late. The same split
  holds here: the windup is the startup, and the answer belongs inside it.
- Rhythm games calibrate. A player taps to a beat; the mean of how early or late each tap lands is their offset, and
  every later press is moved by it (the osu! offset wizard, Rhythm Doctor's two-stage calibration of picture against
  sound, then of the hand). A press timed by her has the same three delays: seeing, deciding, the press arriving.
- Endless-runner bots that last jump on the time an obstacle will take to arrive, its distance over the game's speed,
  not at a fixed distance; those keyed to distance fail as the game speeds up.
- Human catching and batting start a movement on a predicted time to contact and steer it to the end by the closing
  gap (tau coupling, Lee and colleagues). The predictive start absorbs the delay of the hand.

Sources: [DareFightingICE](https://ice.ci.ritsumei.ac.jp/~ftgaic/index-2a.html),
[FightingICE, CIG 2014](https://www.ice.ci.ritsumei.ac.jp/~ruck/PAP/ieeeCIG2014.pdf),
[reaction times in fighting games](https://ki.infil.net/reaction.html),
[Rhythm Quest: latency calibration](https://ddrkirby.com/rhythm-quest/devlog/10.html),
[osu! offset wizard](https://osu.ppy.sh/wiki/fi/Client/Options/Offset_Wizard),
[run-dino-run](https://github.com/dperyel/run-dino-run),
[guiding contact by coupling the taus of gaps](https://experts.umn.edu/en/publications/guiding-contact-by-coupling-the-taus-of-gaps/).
