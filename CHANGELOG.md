# Changelog

Aura is calendar-versioned. The authoritative version string is `version` in
`pyproject.toml`.

This changelog starts at 2026-08-01 and is written forward from there.
Everything before it is summarised below from the commit history rather than
reconstructed release by release — 4,080 commits in five months, most of them
landing without a release boundary. The git log is the real record; this is
the shape of it.

## Format

Entries group by month, newest first. Each names what changed and why it
mattered. A change with no user-visible or operator-visible consequence
belongs in the commit log, not here.

Statuses used: **Added**, **Changed**, **Fixed**, **Removed**, and
**Not claimed** — the last for capability that shipped as infrastructure
without evidence to back a claim yet.

---

## 2026-08

### Added
- **Reality Reach** (`core/reality_reach/`) — a physical request compiles to a
  typed contract with declared channels, and reachability is proven before
  anything executes. Unmeetable requests return a typed limitation
  certificate rather than an optimistic simulation. Dispatch, execution and
  `EFFECT_VERIFIED` are separate states, so transport success can never be
  recorded as a verified effect. Registered hardware routes through
  `HardwareManager` and `BaseHardwareDevice.safe_execute`.
- **Kernel-boundary sandboxing for model-written Python** (`core/sandbox/`) —
  `sandbox-exec` on macOS, `bwrap` on Linux, network denied. When no boundary
  is available it **refuses to run the code** rather than running it
  unconfined and reporting a normal result.
- One shared bounded numeric guard for values accepted from outside the
  process, and one structural redaction primitive.

### Fixed
- Untrusted code inherited the parent's entire environment; the sandbox
  boundary binary is now resolved absolutely.
- Two benchmark harnesses executed Aura-written modules inside the privileged
  runner process.
- Cloud deployment accepted any host key at the target address.
- Importing the cloud launcher provisioned twenty regions as a side effect.
- A bench gate a model could pass while wrong on every case.

### Not claimed
No Aura physical actuation, physical effect, weakpoint, or ambient-law result
is claimed. The RR-10 acceptance battery is open and the P0–P6 evidence
promotion state machine is not implemented. See
[docs/REALITY_REACH.md](docs/REALITY_REACH.md).

### Documentation
Full reconciliation of all 179 living docs against the tree. Corrected a
documented test count that was 3× low, an architecture map ~400 files stale,
four documented environment variables and files that did not exist, and a
supply-chain instruction that pointed at the wrong requirements file. Added
[docs/DOC_STATUS.md](docs/DOC_STATUS.md),
[docs/README.md](docs/README.md), [AGENTS.md](AGENTS.md), and runbooks for
all 19 known failure modes.

---

## 2026-07 — 2,041 commits (346 features, 681 fixes)

The heaviest month, and the one where fixes outnumbered features two to one.

- **The endurance ceiling turned out not to be cognition.** The "15-turn
  ceiling" was a prompt cache that was never constructed and then cleared
  every turn, so each turn re-prefilled the whole conversation from token 0.
  Root cause in `artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`.
- **A standing self-model** — `core/metacognition/faculty_model.py`. Faculties
  declare metrics with units, floors, targets and ceilings; unmeasured reads
  as a blind spot rather than as healthy; priority is headroom weighted by
  how much of the stack a faculty gates.
- **Associative entity memory** — one place where a person, place, thing,
  organization or concept accumulates traits, facts, events and relations,
  plus what it has come to mean to her.
- **Structural screen perception and native OS control** — window ownership,
  geometry and z-order instead of aiming actions at OCR'd pixels.
- **The engineering spine** — taint register, lockdep, PSI, OOM shed ladder,
  telemetry dictionary, invariants in `core/verify/`, the `make layering`
  gate. Seven clean-room adoption waves.
- **Recursive latent cortex and SPARK** — resident 32B recurrent SFT and GRPO
  campaigns, preregistered canaries, holdout discipline.
- Voice, UI legibility, and conversational-organ coverage work.

## 2026-06 — 1,016 commits (106 features, 85 fixes)

Reasoning and evidence discipline. Verifier-gated reasoning with a measured
verifier foundry, the frontier discovery engine's
PROVEN/SUPPORTED/CONJECTURE/REFUTED taxonomy, program-DNA reconstruction,
whole-system φ, the flight recorder, source-body proprioception, and the
Ulysses Covenant.

## 2026-05 — 598 commits

Evidence standards and security posture. The twelve `*_STANDARD.md` bars,
compliance mappings (OWASP, NIST SSDF, MITRE ATLAS), the threat model, the
permission matrix, and the incident runbooks.

## 2026-04 — 327 commits

Production hardening. The governance fence and `make governance-lint`,
capability-token lifecycle, stem-cell reversion, the SLO contract, the
platform posture decisions, and the first runbook set.

## 2026-02 / 2026-03 — 3 commits

Repository initialised 2026-02-23.

---

## Known version drift

`pyproject.toml` reads `2026.4.20` on a calendar-versioned repo that is now
well past April. The version string has not tracked the work. Flagged here
rather than silently bumped, because choosing the next version is a release
decision, not a documentation one.
