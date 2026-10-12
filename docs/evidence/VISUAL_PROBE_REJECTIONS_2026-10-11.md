# Visual probe rejections, 11 October 2026

A failed board detection previously suppressed later detections while the
screen's words stayed alike. A failed falling-piece detection used the same
word-only test. Changed geometry could remain unseen behind an unchanged HUD.

Both callers now capture the current picture before reusing a miss. A general
visual witness includes all pixels, their dimensions, the surface context and
the interpretation inputs. A miss expires after twelve seconds. Missing or
unsupported pixel measurements cannot create a reusable rejection. Temporal
probes retain their final watched picture rather than their initial picture.

The regression renders a blank picture, then a ruled board with unchanged
words. The real board detector admits the later board. Additional tests cover
one-pixel changes, changed surfaces and inputs, absent captures, expiry, and
the final picture of a temporal probe. The focused slice passed 57 tests in
69.43 seconds. The synthetic invariant is registered with model validation.

Smoke passed 165 tests with one skip in 42.09 seconds. Compile and lint passed.
The first layering run identified the new validation import; its exact edge
was added to organism/DEPS. Layering, module-size and writing then passed.
No baseline was raised.

Claude's preserved pass4 workers finished their 54 available original-game
rows. Their result files report no wins and no row-level errors, with 58
unread runs, 40 lost runs and three finished runs. The diagnostic score script
reports only levels zero through two. Those counters are controller reports,
not independent normal-app qualification. Two original programs remain missing.
The worker and desktop processes were absent after completion; stale ledger
records are left for normal liveness reconciliation.

The final picture of case zero shows a selected background and treasure chest,
with no placed character. Its finished run is not sufficient proof that the
requested photo was made. Creation completion requires further repair.

Logs: `/Users/bryan/.aura/flash-proof-2026-10-11/visual-probe-rejections.log`,
`visual-probe-checkpoint-gates.log`, `visual-probe-followup-gates.log`;
retained campaign: `/tmp/claude-501/campaign/pass4/`.
