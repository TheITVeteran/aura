# Grouped points and retained visible marks, 13 October 2026

The original-program run at 572c481ae finished in 136.687808 s without a
harness error and without a win. It retained 567 measured inputs. Of 251
heading records, 164 carried a body identifier. Image inspection placed
the early and middle identifiers on the moving player. The motion record
contained 332 owned rows, 251 with current qualified motion. This improves
on the direct learned run's zero owned records, but is not presentation
qualification. Four births remained; one provisional steering input was
delivered, with no verified prediction outcome. Menu reading, continuous
identity, calibration and selecting the correct task target remain open.

Grouped tracking now reaches the actual observation factory. One worker
receives bounded sequences of consecutive retained captures. CoTracker
advances its real causal windows; no frames are repeated to fill a window.
If retained history or batch capacity is exceeded, new queries start.
Window warm-up supplies no visible points. A window answer retains the
capture it measured even when later batch inputs are still warming.
New changing texture can replace a window containing only resting marks.
Continuing moving queries keep their slots. This is query scheduling,
not motion authority across an unobserved interval.

The first grouped benchmark with eight-frame batches produced no moving
patch support. Increasing the batch to the 32 retained captures exposed
another delay: transporting a late answer through many pictures took
0.691053 s on the observation worker. That work now runs separately.
Only the last two intervening steps can run on the current observation
worker; a longer history retains one owned background operation.
A new scene invalidates its answer. There is no second operation queued
behind unfinished inference or transport.

Shrinking the source also changed fine texture between frames. The
independent pixel check rejected valid grouped positions. Source pixel
budget is now explicit and bounded, with its existing 256,000 default
and a maximum of 1,000,000. Preserving native pixels recovered support.
An incoming refinement then discarded still-visible moving marks when
its new queries described scenery. Refinement now retains marks measured
in the same current capture and geometry, admits distinct new marks,
and bounds its retained support to 192. Duplicate locations cannot
multiply evidence; another reference cannot inherit the marks.

The final 240-frame generated-pixel run used the actual CoTracker MPS
backend and observation factory, with the explicit native pixel budget.
It supplied at least three continuing moving marks on the patch in
97 frames. Median observation call was 0.068805 s, 95th percentile
0.094924 s and maximum 0.156962 s. Earlier native transport without
retaining visible marks supplied support in 27 of 240 frames. Both
paired classical replays used the final run's identical pixels and
capture times: neither the 256,000 nor 1,000,000 pixel budget supplied
three moving patch marks on any of those frames. Their median costs
were 0.020078 and 0.037884 s. Startup and missing support remain visible
in these counts. This is a correspondence experiment, not a game win.

SAM 2 was also replayed on 38 actual recorded player images spanning
13.833874 s, with a supplied initial box. Its first seeded mask supplied
no observation. The next 37 masks reported visible presence; the least
presence was 0.985325. Viewed overlays followed the player through
motion and rotation across a color-track replacement. Median later call
was 0.701246 s; the range was 0.489975 to 2.026453 s. This is a bounded
component replay. Automatic object prompting and current mask transport
into the actor remain open; the replay does not establish either.

The final related suites passed 114 tests in 16.84 s. All nine repository
gates passed. Capture-sequence and visible-mark retention claims are
registered with their tests. Classical observation remains the default;
CoTracker and TAPIR require explicit local weights and compute selection.
No live desktop process was restarted for this work.

Records:

- Original run: `/private/tmp/codex-steer/run-9-background-point-corrections-1013/`.
- Failed batch and transport experiments: `grouped-pixel-benchmark-1013/`,
  `grouped-pixel-benchmark-retained-sequence-1013/` and
  `grouped-pixel-benchmark-background-transport-1013/` under `/private/tmp/codex-steer/`.
- Native and final experiments: `grouped-pixel-benchmark-native-texture-1013/`
  and `grouped-pixel-benchmark-retained-visible-marks-1013/` under that same folder.
- SAM replay: `/private/tmp/codex-steer/sam-on-observed-body-1013/`.
- Gates: `/private/tmp/codex-steer/grouped-capture-final-gates-1013.log`.

Fresh original-program and ordinary desktop qualification remain open.
