# Desktop campaign protocol, 11 October 2026

This checkpoint follows `256d6a7f1`. It preserves Claude's campaign work and
adds two general mechanisms.

A making task that reaches its abandonment timeout without enough observed
changes returns an incomplete attempt. Its end no longer shares the reason
used for a thing made, and the browser completion caller does not request an
artifact description or certify success. This minimum guard does not prove
that every changed surface is a completed artifact.

`tools/run_desktop_campaign_case.py` submits one task to the existing desktop
through `/api/chat`, with the same desktop surface headers, a unique delivery
key and an explicit session. It reads the durable status for that exact identity.
The client neither starts a runtime nor invokes a reduced perception/controller
entry point. It requires a verified current runtime at the requested commit.

At its deadline the client explicitly requests cancellation, then waits for the
durable terminal status. A transport disconnect does not cause resubmission.
Client interruption also requests cancellation. An acknowledged cancel alone is
insufficient to establish a terminal state. Receipts preserve that uncertainty.
Every terminal case still requires independent review; the client never turns
an Aura reply or a successful HTTP response into a gameplay pass.

Verification:

- Making timeout and terminal-evidence slice: 147 passed in 25.77 seconds.
- Desktop delivery bounds, making canary, collection and cancellation slice:
  33 passed in 5.97 seconds.
- Smoke: 165 passed, one skipped in 80.34 seconds.
- Compile, lint, layering and module-size passed. Changed-file Ruff passed.
- Governance lint reports the same 107 findings on the Claude base and
  `256d6a7f1`; no finding bucket changed. Its baseline was not updated.

The desktop-client tests use a simulated HTTP transport to verify admission and
cancellation behavior. They do not measure Aura's model, browser or gameplay.
Original-game qualification and recorded-demo readiness remain unmeasured.
