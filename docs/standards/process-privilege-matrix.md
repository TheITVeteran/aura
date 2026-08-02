# Process privilege matrix

**Source of the idea:** Chromium's process-model security architecture.
**License:** BSD-3-Clause (Chromium). **Code used: none.** This is a clean-room
adoption of a published design principle. No Chromium source was read, copied,
or translated; the module below was written against Aura's own process roles.

## The principle borrowed

Chromium's central security claim is not that renderers are bug-free — it is
that a compromised renderer *cannot do much*, because the privilege boundary is
declared per process type and enforced by the browser process. The component
that parses hostile input is structurally denied the authority to act on it.

The transferable part for Aura is narrower than the sandboxing machinery and
more useful: **write the privilege boundary down as a matrix the code consults,
instead of re-deciding it at every spawn site.** Boundaries that live in
individual call sites drift apart, and nothing notices.

## What Aura had already

Real process separation: the MLX worker is its own process, subprocesses go
through `core/runtime/subprocess_gateway.py`, several sandboxes exist. Three of
them (`sandbox_operator`, `symbolic_sandbox`, `local_sandbox`) already built a
scrubbed environment for their children.

What was missing was any statement of *what a role may hold*. So the fourth
sandbox — `core/sandbox/bash_daemon.py`, which runs arbitrary shell commands —
passed `os.environ.copy()` to its child and nobody noticed. That handed a
persistent bash session every credential in Aura's environment, including
`SSH_AUTH_SOCK`: code in that sandbox could authenticate as the user through
their SSH agent. Three sandboxes scrubbed, one did not, and there was no gate
that could tell the difference. That is the exact failure mode this adoption
addresses, and it was found by building the matrix.

## What was built

`core/runtime/process_privilege.py` — pure policy, no I/O:

* `ProcessRole`, ordered by trust: `DOCUMENT_DECODER` < `WEB_CONTENT` <
  `UNTRUSTED_CODE` < `TOOL_RUNNER` < `MODEL_WORKER` < `CRASH_HANDLER` <
  `COORDINATOR`.
* `Privilege`: network, filesystem read/write, request-authority, spawn-children,
  model-weights, user-surface, secrets.
* `ROLE_PRIVILEGES`: the matrix itself.
* `role_for_source()`: infers a role from the `source` label every gateway
  caller *already* passes — so no call site had to be rewritten to be covered.
* `check_spawn()`: the decision, naming the specific denied privilege.

Enforcement is in `subprocess_gateway._enforce_process_privilege`, called from
both `spawn` and `spawn_async`.

## Deliberate limits

**This is not a sandbox.** It applies no seccomp, entitlements, or namespaces.
It is the *declaration and admission* half. Saying otherwise would be the kind
of overclaim the module exists to prevent.

**Unknown roles are allowed, not refused.** The source vocabulary is large and
this matrix is young; refusing every unrecognised spawn would break the runtime
and get the check disabled, which is strictly worse than an incomplete matrix.
Unrecognised sources are reported so the vocabulary can be completed from real
traffic rather than guessed at.

**Only low-trust roles are constrained.** Roles above `UNTRUSTED_CODE` return
early. Constraining `COORDINATOR` would refuse the process that legitimately
holds everything.

**`env=None` is recorded, not refused.** Inheriting the parent environment is
the larger exposure, but refusing it blind would break every spawn that
legitimately relies on inheritance. It raises a degradation naming the call
site, so inheriting spawns become visible and can be given explicit
environments. This is the observe-before-enforce rollout used elsewhere in the
codebase for gates on ungraded surfaces.

**Privilege is not monotone in trust, on purpose.** `WEB_CONTENT` holds
`NETWORK` while the more-trusted `MODEL_WORKER` does not: network access is a
need, not a reward. A model that can reach the network can be made to exfiltrate
its context. `tests/test_process_privilege.py` asserts the weaker, correct
property — no role out-privileges the coordinator — rather than monotonicity,
which would force exactly the wrong fix.

## Tests

`tests/test_process_privilege.py` (21 tests) pins the structural invariants:
every role has an entry; no input-parsing role holds authority, secrets, or
spawn; only the coordinator holds secrets or may spawn; decoders and model
workers cannot reach the network; refusals name the specific privilege and do
not blame legitimately-held ones; and the gate never raises on hostile input.
