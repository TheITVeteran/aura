# What it takes to beat the 56, and the general capability that does it

Status: Design map and build ledger · For each Cartoon Network game of the Web Design Museum's list, what beating it
asks for (from the game's own words, read offline from its program by `tools/what_it_takes_to_beat.py`; the reading is
kept in `what_it_takes_to_beat.json`), and which general capability of Aura's does that. The game files are a design
map read once and offline, never by her while she plays.

Nothing below is built for one game. Each capability is general to every place of its kind, and is held by the test
that names it. A game is listed under the capability its winning rests on; most also need others (menus, reading
rules, counters), which every game shares.

The first measurement, on 10 October 2026 (`tools/measure_on_archived_games.py`, 6 minutes a game, no language model,
the code of that morning): of 46 games with an Internet Archive copy, none won; 13 reached a readable end; 33 never
reached one in the time. That is the bar what follows is measured against.

## The capabilities

| Capability | What it is, for any place | Where | State |
|---|---|---|---|
| Imagining ahead | Her learned world (how her keys move her, how each kind falls, bounces and leaves, what meeting each does) run forward under every plan she could follow over the next second, scored by what it gets and costs, robust to her model's error; she does the best plan's first part and plans again | `core/agency/imagining_ahead.py` | built 11 Oct; to measure on the games |
| The situation | Catching what falls, keeping clear of what comes, keeping a thing away until help comes, holding on, getting them all against a clock, tossing to whoever takes it, ...: told by words, setting what she supposes of things not yet met, how wide she keeps, which ways she plays, what the clock's end means | `core/agency/situations_she_knows.py` | built |
| Aimed clicks | A click aimed at a moving thing, led by the lead that has taken things; a shot sent toward the pointer, pointed first at what it is for | `core/agency/leading_a_click.py` | built; to measure |
| Sending | Pull or hold and let go, how hard and which way found from where each went | `core/agency/playing_by_shots.py` | built; read as told since 10 Oct |
| A press in time | Press when a moving thing is where presses have paid | `core/agency/when_a_press_pays.py` | built |
| A board seen and searched | A wordless board's squares and pieces seen, its family known, its rules as its words change them, every move searched | `core/agency/playing_a_board.py` | built 11 Oct for draughts; other families to add |
| Pieces placed | Every column and orientation of a falling piece tried and scored | `core/agency/where_a_piece_goes.py` | built by another session; not yet joined to play |
| A build from parts | Parts carried to places, a chain put on from where the last that worked went, each step held to what it shows | `core/agency/putting_things_in_place.py`, `core/cognition/a_plan_to_an_end.py` | partial |
| An errand | A thing taken and used on another; doors and rooms gone through | `core/agency/taking_and_using.py`, `core/agency/where_things_lead.py` | partial |
| Done again | What was shown, done again in order; keys shown, pressed while lit | `core/agency/doing_again_what_was_shown.py`, `core/agency/pressing_what_is_shown.py` | built |
| Pairs | What each place showed when turned, and two alike turned together | `core/agency/things_that_go_together.py` | built |
| Typing | What a screen asks to be typed, where it points | `core/agency/typing_what_is_asked.py` | built |
| Making | Made, not won: tried out, and what was made said in the place's own words | `core/skills/sovereign_browser_drawing.py` | built |

## The games, by what beating them rests on

### Steering a body among things, by keys or the pointer (imagining ahead, the situation)

| # | Game | To beat it, from its own words |
|---|---|---|
| 2 | Food Bash | Aim with the mouse and click to throw; space to dodge food. "The first to splat the other five times wins." |
| 8 | Courage: Nightmare Vacation | Arrow keys; get through each place. |
| 9 | Cow and Chicken: Ballet Parking | Drive (forward, reverse, turn) and park each car. |
| 12 | Samurai Jack: Code of the Samurai | Arrows to move, keys to strike and jump; get through each level. |
| 14 | Scooby-Doo: Scooby Trap | Left and right; set off the trap at the right time. |
| 16 | Toonami: Tunnel Rush | "Avoid or shoot the obstacles"; score against the clock. |
| 18 | KND: Operation Tommy | A lander: up thrusts against the pull; land on the pad slowly. |
| 20 | Foster's: Coco's Egg Scramble | Mouse moves Mac, click throws to crack the eggs, space jumps; "catch the prizes, watch out for the eggs carrying a different surprise". |
| 22 | Foster's: Mid-Flight Snack | Click to jump and start the catapult; mouse to move and grab food. |
| 23 | Foster's: Simply Smashing | Mouse moves Frankie to catch items before they crash; click to bank them; stack for bonus. |
| 25 | Camp Lazlo: Paintcan Panic | Arrows move, z jumps; cover the whole campground in paint. |
| 26 | KND: Tummy Trouble | Arrows move, mouse aims, click fires; destroy containers to find the keycard; reach the control room. |
| 27 | The Batman: Cobblepot Caper | Arrows walk, space jumps (held to glide), S punches, D kicks, A throws; find the lab and save the doctor. |
| 32 | Ben 10: Blockade Blitz | A paddle by the mouse, click to shoot; clear the aliens. |
| 33 | Ben 10: Krakken Attack | "Fire at the Kraken to wear down its energy"; aim for the chest; watch out for its fire. |
| 34 | Ready, Im, Fire! | Arrow keys to dodge snowballs, and to throw. |
| 35 | Operation Z.E.R.O. Out-Mandy'd | Arrows, jump and fire keys; get through. |
| 37 | KND: Rainbow Monkey Rundown | "Move the mouse to control your boat. Click to hit enemies with your yo-yo." |
| 38 | Foster's: Team Work | Help Bloo escape: jump, grab the eggs Coco lays and use them on the bombs. |
| 39 | Powerpuff Girls: Puppybots | Each girl's power; rescue the puppies; health shared. |
| 41 | Ben 10: Cavern Run | Run, jump; get through the cavern. |
| 43 | Adventure Time: Jumping Finn | Jump and steer. (No Flash copy found.) |
| 47 | SnowBrawl Fight! | "Turn all the opposing players into snowmen" to win. |
| 48 | Teen Titans: Grab that Grub | Steer and grab. (No copy found.) |
| 49 | Regular Show: Killer Z's | Steer, jump, shoot. (No copy found.) |
| 50 | Regular Show: Ride 'Em Rigby | Jump with a click, steer. (Only phone versions found.) |
| 53 | Regular Show: High Flying Halloween | Steer and jump. (No copy found.) |

### A shot aimed and sent (sending, aimed clicks)

| # | Game | To beat it |
|---|---|---|
| 4 | Ed, Edd n Eddy: Spin Stadium | "Move your mouse to aim, then click to release your top"; knock the other tops out. |
| 6 | Cartoon Cove Mini Golf | "Click on your ball and hold down the mouse button", pull back and release; sink each hole. |
| 10 | Dexter: Clone-A Doodle Doo | Click to place a fence post, drag to set its way; fence the clones in before the day is out. |
| 19 | Foster's: A Friend in Need | Aim and toss Bloo to a friend (short, medium, far) until Mac arrives; keep him from the girl. |
| 24 | Foster's: Wilt's Wash N' Swoosh | Mouse moves Wilt; click and hold to aim, release to shoot. |
| 36 | KND: Flight of the Hamsters | Click to start the hamster swinging, click again when it lines up with the pillow; hold to glide. |
| 52 | Gumball: Water Sons | "Use the mouse to aim and click to throw"; space switches characters. |
| 55 | Gumball: Battle Bowlers | Bowl at the enemies; number keys switch characters. |

### Clicking what comes, or pressing in time

| # | Game | To beat it |
|---|---|---|
| 0 | Scooby-Doo: Scooby Snapshot | Set up a scene and take photos: made, not won. |
| 21 | Foster's: Door to Door | "Click on the doors" before the time is out. |
| 44 | Regular Show: All-Nighter | Click in time; keep awake. |
| 46 | Annoying Orange: Escape | "Click on a balloon", "click to jump, then cut the rope". |
| 54 | Sonic Boom: Link 'N Smash | Move the tile block left and right; link to smash. |

### A board in turns (a board seen and searched)

| # | Game | To beat it |
|---|---|---|
| 31 | Foster's: Big Shot Checkers | Draughts with a Big Shot: capture all, block every move, or bring the Big Shot to the king row; jumps must be made; the Big Shot swaps with its own. |

### A build from parts, and errands

| # | Game | To beat it |
|---|---|---|
| 3 | Dexter: Runaway Robot | Move room to room with the mouse; collect the glasses and items and use them to find the robot. |
| 7 | Operation S.T.A.T. | Click on items as you find them, drag them here, and assemble them before time runs out. |
| 11 | Candy Machine Deluxe | Add parts (pipes) and place them so the dropped jawbreaker rolls into the cup. |
| 13 | Scooby-Doo: Creepy Castle | Click objects into the inventory, click doors open and shut, use the objects to keep the ghosts away; rescue Fred and Velma. |
| 28 | Tom's Trap-O-Matic | Choose devices from the library and build a chain that springs the trap. |
| 40 | Pokemon: Towering Legends | Stack crystals of one colour, click to fuse; keep the tower from crystals of another colour. |
| 45 | Gumball: Blind Fooled | WASD to move; pick up and stack boxes to make steps; get the others through. |

### Done again, typed, made

| # | Game | To beat it |
|---|---|---|
| 1 | Batman: Riddler's Secret Identity Inventor | Answer the riddle: typed. |
| 5 | Scooby-Doo: Ask Swami Shaggy | Type a yes-or-no question, click Ask: made, not won. |
| 15 | Billy & Mandy: Zap to It! | The arrows in the book, done again in order. (No copy found.) |
| 17 | KND: Numbuh Generator | Create your own operative and generate it: made. |
| 29 | Ben 10: Hero Matrix | Made. (No copy found.) |
| 30 | Code Lyoko: Monster Swarm | Steer, jump, strike, charge, shoot. (No copy found.) |
| 42 | Total Drama: H-Bomb's Killah Beatz | Make a beat: made. |
| 51 | Regular Show: Dance of Doom | The moves shown, done again. (No copy found.) |

Games 15, 29, 30, 43, 48, 49, 50, 51 and 53 have no Flash copy in the Internet Archive that a browser plays; picked
live, she looks for each elsewhere and, finding none, goes on by the rule to the next (the request's count is kept).
