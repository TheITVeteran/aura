# Full desktop Flash attempt, 11 October 2026

Runtime: `ece1733a0edf87fb42e24b0f60d4a92ba2796f29`, signed app launch.
Boot provenance reported verified, source current, matching source/workspace
hashes, and no identity issues. One launcher (34500) owned one desktop backend
(34502). The offline origin was scoped to `http://127.0.0.1:54582`.

At 19:21:54 PDT, the ordinary app input received the first original-game case,
with the request to win it, complete three scored plays, or make one thing,
according to what the environment offered. Delivery:
`aura-chat-6e045c8a-d53f-497e-9a85-eec94e8da313`.
Turn: `c02d91bb820447febd077beee88e703b`.

The browser loaded the local page, pinned Ruffle assets and original program
with HTTP 200 responses. The visible window showed the original instructions.
This settles the earlier local navigation failure; it does not settle gameplay.

The runtime reported model admission refusal because a vision lease owned by
Claude's still-active offline campaign could not be evicted. Semantic dependency
warmup also delayed ordinary startup, and durable delivery reads/cancellation
reported IO timeouts. Performance under this contention is unqualified.

Stop was requested through the app and its cancellation endpoint. Neither
produced a verified terminal delivery before shutdown. The app was then closed
through its normal Quit lifecycle. The backend log reported clean container
teardown and exit code zero; both launcher and backend disappeared. The four
Claude campaign processes remained running. The journal still held this
delivery as running, without terminal_at. Treat it as unresolved until durable
recovery adjudicates it. Do not advance or credit it as a completed game.

Repairs following this observation:

- The authenticated active execution can receive Stop from its existing
  durable admission binding without first waiting for journal reads. Its HTTP
  acknowledgement remains pending and cannot certify a terminal outcome.
- Research sources share a scoped preparation deadline. Retired sources lose
  publication authority and cannot extend the caller indefinitely by delaying
  cancellation. Results available within the deadline remain usable.

Regressions: 57 cancellation/journal/client tests passed in 53.54 seconds;
41 stock-taking and guidance-lifetime tests passed in 13.23 seconds.
These are synthetic mechanism tests. The fixes have not yet received a clean
ordinary-app retest. No presentation qualification or individual game pass
is claimed.

Logs: `/Users/bryan/.aura/flash-proof-2026-10-11/` and the retained desktop log.
