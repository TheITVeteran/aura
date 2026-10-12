# Point paths, 12 October 2026

The body in the landing run remained visible, and its texture points kept
moving. Its foreground box also contained exhaust and pixels left at an
earlier place. The box grew from one edge, and its middle did not follow
the body's marks. Requiring that middle to agree with the marks repeatedly
ended the observation of a delivered control.

`position_from_the_same_points.py` keeps a separate displacement path. Its
origin is the box's middle when the marks are first kept; that origin is
not claimed to be the physical centre. At least three original named marks
must remain on current owned support, span two directions, and agree in
their displacement. New marks cannot gradually take over the path. A gap,
lost support, changed birth, deformation or replacement set ends it.

The observed box retains its geometry. Key trials, press curves and pointer
trials use the separate measured position. Losing that path leaves an
unmeasured interval before the box's coordinate origin can be used again.
Box translation checks remain in place when points cannot supply a path.

## Evidence

Baseline run: `/private/tmp/codex-steer/run-18p`, revision `51e41fe8f`.
It finished in 193.48 seconds, recorded three losses and no body controls.
The next baseline, `/private/tmp/codex-steer/run-18q`, used `a5f61fe41`
with the intervening appearance changes. It also learned no body controls.

A passive replay compared the same 106 lossless tracker inputs with and
without the point path, on the current implementation. Of 91 sightings of
the original tracked body, the box path supplied 21 measured sightings;
the point path supplied 63 point measurements and two box measurements.
The longest continuous path increased from six samples to 36. The replay
took 1.363 seconds with boxes and 1.317 seconds with points. This single
replay does not establish a performance advantage or successful gameplay.

Replay output: `/private/tmp/codex-steer/point-path-comparison.json`.
The rejected foreground-only experiment is preserved separately in
`/private/tmp/codex-steer/foreground-cleanup-comparison.json`.

The invented tests cover growing boxes, stationary marks, lost or replaced
marks, gradual replacement, collinearity, deformation, birth changes,
capture gaps, invalid support, press curves and keyboard samples. The
expanded related suite passed 202 tests in 19.25 seconds, including the
browser runner and thrust/braking tests. Gameplay proof remains open.

## Research and reference controllers

Bryan proposed comparing stronger projects and successful reference runners
with Aura. The following sources were checked on 12 October. A source in
this list is research, not a claim that its whole implementation is in Aura.

| Source | Mechanism worth comparing | Current status |
| --- | --- | --- |
| [Playwright actionability](https://playwright.dev/docs/actionability) | Wait for a unique, visible, stable target that can receive the action; check the result | Research; no new actionability implementation in this checkpoint |
| [WebDriver actions](https://www.w3.org/TR/webdriver2/#actions) | Explicit input state, ordered ticks, durations and release actions | Research; the earlier input-receipt changes cover delivered intervals, not the whole protocol |
| [CoTracker](https://github.com/facebookresearch/co-tracker) and [TAPIR](https://arxiv.org/abs/2306.08637) | Named points, trajectories and visibility distinct from detection boxes | The new deterministic point path uses Aura's existing corner/Lucas–Kanade tracker; neither learned model was installed |
| [Gymnasium landing controller](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | A reference policy supplied for testing and demonstration, with explicit state | Research; it has not been run against the archived game and is not a proven solver for it |
| [MIT feedback motion planning](https://underactuated.mit.edu/feedback_motion_planning.html) | Replan using current feedback rather than replaying an open-loop action sequence | Research; existing Aura planning needs a measured comparison with a reference controller |

A reference controller can establish what succeeds under its own
observations and action interface. Comparisons should then remove its
privileged state, introduce capture and delivery delays, and use Aura's
ordinary observation and input paths. Reproducing one successful game
policy does not prove generalization. Invented plants and untouched games
remain separate tests. No held-out game data or unofficial assets were read
for this work.
