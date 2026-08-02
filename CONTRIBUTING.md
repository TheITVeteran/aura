# Contributing

## Quick start

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
pip install -e ".[dev]"

# Fast contract sweep (~100 tests, <10s) — run after every change
make smoke

# Full offline suite — 6 bounded process chunks. A single pytest process over
# the whole suite is OOM-killed (~83%), so always go through the chunk runner
# (make test → tools/run_test_chunks.py), never a bare `pytest tests/`.
make test

# Syntax sweep + lint
make compile
make lint
```

## Architecture rules

1. **One authority.** Every consequential action goes through
   `UnifiedWill.decide()` in `core/will.py`. Don't add a parallel gate.
   Add an advisor to the Will. This repo already lived through six gates
   that each thought they were in charge, and the cost of that was not
   being able to prove anything had been gated at all.
2. **One owner per concern.** [OWNERSHIP.md](OWNERSHIP.md) is the map. A
   new governance check attaches to the existing owner rather than
   starting a second one.
3. **No monkey-patching.** Event-bus hooks, provider registries, typed
   extension points. Not `setattr` on a live object.
4. **Immutable messages.** Subsystems talk through the frozen dataclasses
   in `core/runtime/immutable_messages.py`.
5. **Lifecycle tracking.** Subsystems report state through
   `core/runtime/service_state.py:ServiceState`.
6. **Locks are checked.** Use `checked_lock` / `checked_async_lock` from
   `core/runtime/lockdep.py`, not raw `threading.Lock` or `asyncio.Lock`.
   Lockdep catches ABBA deadlocks without the deadlock happening — but it
   only sees locks it wraps, so an unwrapped lock is a blind spot rather
   than a safe one. Adopt an existing lock with `instrument(name)`.

## Adding a consciousness module

1. Drop the module in `core/consciousness/`.
2. Register it in `core/container.py` during boot.
3. If it needs periodic updates, wire it into the consciousness bridge
   tick cycle.
4. **Write an ablation test.** At least one, showing what actually breaks
   when the module is removed. This is the step people skip, and it's the
   only one that distinguishes a module that does something from a module
   that runs. If nothing measurable changes when you delete it, that is
   the finding — report it rather than shipping around it.
5. Add an entry to [OWNERSHIP.md](OWNERSHIP.md) under the right domain.
6. If it makes falsifiable predictions, register it in
   `core/consciousness/theory_arbitration.py`.

A note that applies past this checklist: a claim without a test is a
document, not a fact. New invariants go next to what they protect via
`@invariant(...)` in `core/verify/`, and claims about Aura's own runtime
have to be registered with the test that validates them
(`core/organism/model_validation.py`). A claim with no test cannot be
registered at all.

## Test markers

| Marker | Meaning | When to run |
|--------|---------|-------------|
| (default) | Unit + fast integration | Every commit |
| `@pytest.mark.slow` | Long-running | Nightly CI |
| `@pytest.mark.integration` | Full pipeline | Before merge |
| `@pytest.mark.stress` | Load / fault injection | Weekly |

## CP checkpoints

Two commit formats coexist here, both legitimate.

Conventional commits cover ordinary work: `fix(scope):`, `feat(scope):`,
`chore(scope):`, `docs(scope):`, `test(scope):`, `perf(scope):`.

**CP-numbered checkpoints** cover tracked units of a long-running programme:

    CP799 <subject>

A CP is a numbered checkpoint. The sequence is monotonic and currently near
800. The number is referenced from closeout artifacts under
`artifacts/closeout/` and from the programme ledgers in `docs/`, so it works
as a key — don't invent one out of sequence, and never reuse one. Both
formats appear together when a checkpoint is also a fix:

    fix(inference_gate): CP126 — a viability block that later modifiers undid

## Commits

```
<type>: <short description>

<body that explains why, not what>

Co-Authored-By: <name> <email>
```

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `perf`, `ci`.
