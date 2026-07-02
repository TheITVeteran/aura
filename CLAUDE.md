# CLAUDE.md — Agent guide for the Aura codebase

Operational facts an agent needs before touching anything. Architecture
rules live in [CONTRIBUTING.md](CONTRIBUTING.md); the deep spec is
[ARCHITECTURE.md](ARCHITECTURE.md).

## The live instance is sacred

A real Aura instance is usually running on this machine (port 8000,
`aura_main` process, logs streaming to `~/.aura/logs/`). **Never kill,
restart, or port-collide with it.** Do not boot a second full desktop
runtime or load another 32B model beside it — the host has 64GB and the
live model already holds ~20GB wired. Code fixes reach the live instance
only when the user restarts it themselves.

## Environment

- Python: use the repo venv — `/Users/bryan/.aura/live-source/.venv/bin/python`
  (Python 3.12). The Homebrew `python3` is 3.14 and is NOT the runtime.
- Makefile gates accept it: `PYTHON=/Users/bryan/.aura/live-source/.venv/bin/python make <target>`.
- Worktrees share this venv; there is no per-worktree venv.

## Build / test / gates

```bash
make compile      # syntax sweep (core + tests)
make lint         # ruff, three passes (surface E9, critical F-codes, curated files)
make smoke        # ~100 contract tests, <10s — run after every change
make test         # FULL offline suite (~7,400 tests) in 6 bounded process chunks
make governance-lint  security  enterprise-gate  # scrutiny gates
```

- Full suite: `tools/run_test_chunks.py --chunks 6 --marker "not live and not network and not external"`.
  Use `--continue-on-failure` to collect everything, `--only-chunks 5,6` to
  resume a partial run. One pytest process on the whole suite gets
  OOM-killed (~83%); always use the chunk runner.
- A test failing in-chunk but passing alone is an ORDER-DEPENDENCE defect —
  the runner's isolated-retry pass reports these separately.
- Never launch test chunks while editing Python files: chunks spawn fresh
  processes mid-run and will import half-written modules.
- Long runs: bound them (`caffeinate -dims`, explicit timeouts), check
  interim output at expected milestones, never poll unbounded.

## Conventions that will bite you

- **All consequential file writes go through `core/runtime/file_write_gateway.py`.**
  From async code use the `*_async` methods (or `async_atomic_*` in
  `core/runtime/atomic_writer.py`) — an on-loop fsync once froze the live
  event loop for 20 minutes. `tests/test_async_write_lane_ratchet.py`
  fails on new sync writes inside `async def`; its allowlist only shrinks.
- Internal maintenance writes need `local_internal_governed_scope(...)`
  (core/governance_context.py) or the live runtime refuses them as
  governance violations.
- Log through `logging`/structlog; the file sink JSON-wraps and redacts
  everything. Set `AURA_LOG_DIR` for anything test-like so you never write
  into the live `~/.aura/logs/`.
- Degradations: `record_degradation(subsystem, exc, action=...)` — never a
  silent `except: pass`. Modules on the fail-closed list (see
  `core/config.py`) escalate warning+ records to CRITICAL; for expected
  backpressure (timeouts under load), log at info and only record a
  degradation when the condition is persistent/total.
- ServiceContainer keys are the spine (`core/service_names.py`); health
  contract lives in `core/runtime/health_contract.py`.

## Session mechanics for this repo

- Work in a worktree under `.claude/worktrees/`; push checkpoints with
  `git push origin HEAD:main` (no remote side branches).
- A parallel agent (Zencoder, commits as "Zenflow") shares this checkout
  and may modify files under you. `git log`/`git status` before resuming
  anything — your half-remembered work may already be committed.
- Crash forensics when the runtime dies: `data/error_logs/crash/`
  (faulthandler + loop-wedge + memory-spike stacks), `data/error_logs/stalls/`,
  `data/error_logs/memory/` (sentinel ring, tombstones, death syslogs),
  plus `~/.aura/logs/desktop-launch.log` for the live stdout stream.
