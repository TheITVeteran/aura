# Current visual controls and reopened selectors, 10 October 2026

Base revision: `e7dabe356`. Presentation qualification remains **unproved**.

An ordinary app request at 17:59 reached the construction board. Aura cycled
through clicks without placing anything. Its observer offered the earlier PLAY
control at remembered coordinates after the screen had replaced that control.
`_where_the_way_on_was` inserted a shape from memory without examining current
pixels. This is a general observation failure: a remembered object became a
current affordance without evidence.

Remembered labels now retain their measured extent, immutable appearance,
surface, viewport, capture and episode. Reacquisition checks fresh pixels before
the observer can offer that control again. Local movement and a bounded uniform
brightness or contrast change are accepted when the appearance remains distinct.
Missing, ambiguous, stale, occluded or mismatched evidence returns unknown.
Measured duplicates also prevent reacquisition. The search is bounded to a small
neighborhood and current measured extents; it does not claim arbitrary motion,
exhaustive global identity or enabledness from presence.

[OpenCV's template-matching documentation](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html)
describes locating a retained image patch in a current image and inspecting
multiple matching locations. It informed the measured search and ambiguity
check. Aura uses its existing immutable pixel custody and NumPy; this change
does not install another vision model or encode a game's buttons or solution.

Procedures now recognize categorical target phrases such as "click a type of
component" without requiring the separate verb "choose". Quoted names remain
literal. A category selection can bind newly exposed choices of the same entity.
A completed selector can reopen when its dependent choice has no current bound
options; failed reopening preserves the earlier completion. The tests cover
component libraries, font panels, chart menus and asset catalogs.

The preceding inference checkpoint also has live Stop evidence. At 17:59:04,
ordinary UI Stop cancelled the owned model generation, job sequence 4. The worker
observed cancellation at token 7 and acknowledged it at 17:59:04.208. The browser
closed, and a subsequent health capture reported zero active generations with
the existing worker ready. Repeating the ordinary request then entered the game.
Stopping that unsuccessful repeat at 18:03:50 cancelled its owned job sequence 7;
the worker acknowledged it at 18:03:50.994. This proves these observed Stop cases,
not completed gameplay or instantaneous cancellation of every background task.

Local evidence is retained under
`/Users/bryan/.aura/flash-proof-2026-10-10/`: `live-e7dabe356-complete.log`,
`live-e7dabe356-stopped-health.json` and `visual-anchor-checks.json`.

Validation used the shared virtual environment and isolated test logs:

- Seven stale-control regressions failed before the repair.
- Related procedure, vision, transfer and validation checks: **273 passed**,
  30.07 seconds. The final visual-anchor registry check: **24 passed**,
  3.52 seconds. These runs overlap and are not additive counts.
- Smoke: **165 passed, 1 skipped**, 60.10 seconds.
- Lint, compile and layering passed; layering retains 36 grandfathered edges.
- Thirty synthetic reacquisitions took a median **2.65 ms**, maximum **7.05 ms**.
- The changed production modules have **zero AST effect bucket changes** against
  the base. The repository governance baseline remains open.
- Registered claims describe synthetic measurements. Full-suite validation,
  current live construction, completed gameplay and the random three-game
  request remain outstanding.
