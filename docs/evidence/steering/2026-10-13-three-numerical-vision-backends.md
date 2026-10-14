# Three numerical vision backends, 13 October 2026

The user asked for CoTracker3, causal TAPIR and SAM 2 to be implemented.
All three now have numerical adapters with explicit local weights and
captured-frame contracts. This checkpoint does not yet connect their
results to the production motion eyes or establish a controlled body in
the original programs. That connection remains work to do.

TrackingFrame owns immutable RGB pixels, capture time, sequence, generation
and geometry. Point and mask results retain those fields. Hidden point
coordinates remain predictions. Supplied masks and newly seeded objects
cannot grant visible observation or control authority. Unique query ids
survive inference; invalid geometry or clocks cannot advance a session.
Failed inference retires partially advanced state. Local checkpoint
selection verifies SHA-256 before allocating model memory.

CoTracker3 uses real sixteen-frame online windows advancing eight frames
at a time. Only the newest captured frame is returned; between windows
the result is warming and empty. Its actual visibility/confidence logits
are retained. History is bounded and requires fresh queries on exhaustion.
Causal BootsTAPIR advances exactly one captured frame, preserving its query
features and causal context. SAM 2 uses an adapter-owned live frame store
around the pinned video predictor. Each call propagates exactly its current
frame, with bounded feature, conditioning and object memory. Object ids
survive internal slot remapping. This is an implemented streaming adapter,
not a claim that the standard video loader accepts a live camera.

Actual checkpoints were downloaded from their official publishers and
tested one at a time on this Mac's MPS device. No second language model or
desktop runtime was started, stopped or restarted. Added dependencies are
installed in `/Users/bryan/.aura/vision-runtime/`; the shared repository
virtual environment remains unchanged. PyTorch is 2.11.0.

On an invented 192 by 256 RGB moving textured patch with four point queries:

| Backend | Measured behavior |
| --- | --- |
| CoTracker3 | First sixteen-frame inference 1.833 s; warmed window 0.425 s; median newest-point error 0.088 and 0.106 pixels. |
| Causal BootsTAPIR, CPU | About 1.8 s per frame; finite visible tracks. |
| Causal BootsTAPIR, MPS after repair | Initialization 0.531 s; four subsequent frames 0.081, 0.075, 0.079 and 0.074 s; median point error below 0.786 pixels. |
| SAM 2.1 tiny, MPS after repair | Seed 0.722 s; subsequent frames 0.252, 0.339, 0.319 and 0.342 s; mask intersection over union at least 0.991. |

These are component measurements, not complete interactive deadline or
gameplay proof. First SAM 2 masks are marked seeded even when they match
the supplied region exactly. No seeded mask supplies observation credit.

Two real integration failures were repaired rather than hidden. TAPIR's
finite single-frame MPS 3D grid sample returned nonfinite query features;
the same inputs were finite on CPU. Singleton-time border interpolation
is now computed by its equivalent 2D sampler. Tests compare the original
CPU operation at fractional, edge and outside coordinates and preserve
the input tensors. Real MPS inference then matches the CPU tracks.
SAM 2's asynchronous CPU state offload returned invalid masks; bounded
memory now remains on its compute device. The next actual checkpoint
run produced finite masks and correct moving-patch correspondence.

The known-body fallback also changed. Missing visual identity no longer
turns established steering into timing-only play. Stronger known timing or
no-body roles still outweigh incidental source keys; demonstrated timing
payoffs remain eligible. An invented test exercises the actual actor's
timing wait that had delayed rediscovery for 14.88 s in original game 9.

The final capture, custody, streaming, singleton sampling, control-role,
travel-speed and calibration suites passed 116 tests in 29.15 s.
Real benchmark records: `/private/tmp/codex-steer/learned-tracking-benchmark-1013/`.
First failing operator record:
`/private/tmp/codex-steer/learned-tracking-mps-diagnostics-tapir.log`.

Sources are pinned in `core/perception/learned_tracking/models.py`:
CoTracker3 at 82e02e8029753ad4ef13cf06be7f4fc5facdda4d;
TAPIR at 730cda1c730877cfedbe01bf87fb1cadb78a565d;
SAM 2 at 2b90b9f5ceec907a1c18123530e92e794ad901a4.
CoTracker3 code and weights use CC-BY-NC 4.0. TAPIR and SAM 2 code and
weights use Apache 2.0. Adapters do not change those upstream terms.
The official sources are [CoTracker](https://github.com/facebookresearch/co-tracker),
[TAPIR](https://github.com/google-deepmind/tapnet) and
[SAM 2](https://github.com/facebookresearch/sam2).

After extracting timing dispatch into its control-role module, the same
116 tests passed in 15.77 s. The continuous actor remains under its
2,000-line ceiling. All nine gates passed; smoke passed 165 tests with
one skip in 91.80 s. Gate log:
`/private/tmp/codex-steer/learned-backends-final-gates-1013.log`.
