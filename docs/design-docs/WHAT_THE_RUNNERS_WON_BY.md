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
