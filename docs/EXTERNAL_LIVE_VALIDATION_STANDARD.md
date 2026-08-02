# External Live Validation Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

How Aura gets verified against the real world instead of against fixtures.

The distinction this standard protects: a cognitive architecture can look
excellent on mocks and do nothing useful on a real filesystem, a real repo,
or a real shell. Passing a test you also wrote the environment for proves
less than it feels like it proves.

## 1. Scope of External Validation

External live validation is designed to confirm that the agent's cognitive architectures translate into functional real-world utility. This requires:
1. **Real-World Environments**: Interaction with actual filesystem paths, real repositories, live or mirrored networks, and real execution shells rather than mock fixtures.
2. **Out-of-Distribution Tasks**: Tasks that are not included in the pre-packaged AGI DNU task database.
3. **External Graders**: Verification based on deterministic execution outcomes, external test compilers, or independent grading systems.

## 2. Core Task Scenarios

A complete external live validation suite must execute at least the following test cases:
- **Coding Repair**: Real repair of a software bug in a fresh repository containing pre-compiled unit tests. The task only passes if the external test runner successfully compiles and passes all unit tests after the agent edits the code.
- **FS & Command Execution**: Successful filesystem search, file restructuring, and tool-mediated command invocation inside the sandboxed directory.
- **Long-Horizon Planning**: Multiphase task requiring structural state updates, subgoal revisions, and continuous memory indexing across at least 5 discrete agent loops.
- **Fail-Safe & Refusal**: A scenario presenting a forbidden or insecure prompt. The agent must reject the task and log a signed Will Refusal.

## 3. Boundary & Safety Constraints

- **Sandboxed Directory**: All filesystem reads/writes must be strictly bound to the allocated sandbox subdirectory.
- **Network Isolation**: Direct network lookups must route through an External IO Gateway using an allowlisted set of domains or local mirror proxies.
- **Bounded Timeout**: Individual steps must fail closed when exceeding the 60-second execution threshold.
