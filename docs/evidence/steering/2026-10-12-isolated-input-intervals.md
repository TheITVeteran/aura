# Isolated input intervals — 12 October 2026

The bounded archived run on c85585616 finished in 194.35 seconds. Its
first two real-time stretches lasted 21.0 and 23.4 seconds, rather than the
roughly three-second handoffs seen before the activity fix. It still found
no body or usable movement controls and ended out of time. It captured
1,854 tracker inputs. This was a failed gameplay proof.

The log showed attack taps about every 0.35 seconds during movement trials.
Those trials did not isolate the movement key: attacks changed the body's
pose while it was being measured. The same issue applies to any secondary
action whose effects are not known to be independent.

The control hands now close an isolated observation before a secondary
input. Raw velocity samples and trajectory receipts retain their measured
prefix. Later mixed motion cannot extend or contradict that ended interval.
The secondary input still reaches the original hands. The primary delivery
scope belongs to its async task; another task cannot borrow it. A finite
discovery pass reserves one input at a time, including neutral intervals,
before ordinary secondary actions resume.

This is an implementation of input accounting, not a claim to identify an
arbitrary dynamical system. The distinction between measured outputs,
applied inputs and model estimation is described in MIT's
[System Identification notes](https://underactuated.mit.edu/sysid.html).
Their [input-design lecture](https://ocw.mit.edu/courses/6-435-system-identification-spring-2005/resources/lec4_6_435/)
also covers excitation. No new neural weights or game-specific rules are used.

The initial expanded check measured 96 passed in 45.05 seconds, including
the browser runner, entity trajectories, delivered input intervals, pointer
identity and existing trigger behavior. Further task-scope and scheduling
tests and checkpoint gates remain to be checked. Logs are under
/private/tmp/codex-steer/single-input-tests-2.log and run-12i.

The supplied frame harness calls the campaign as an imported function,
which returns its results but does not execute the command-line result
writer. A passive wrapper now saves that returned report for subsequent
runs. The previous run's completion record and gameplay log remain intact.
Reliable gameplay and demo readiness remain unproven.

The completed expanded check measured 111 passed in 46.42 seconds,
including inherited async scope and the trigger's finite discovery pass.
All checkpoint gates passed. Smoke measured 165 passed and one skipped in
48.22 seconds. The gate log is retained as single-input-checkpoint-gates.log;
the test log is single-input-final-tests.log under the same evidence root.
