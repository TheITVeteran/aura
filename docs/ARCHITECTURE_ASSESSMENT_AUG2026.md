# Architecture assessment — August 2026

Measured, not estimated. Every number below came from the current tree during
the coherence session on 2026-08-04; the commands are given so they can be
re-run rather than trusted.

## Scale

| Measure | Value |
| --- | ---: |
| Python files (`core` + `interface`) | 2,664 |
| Lines | 1,095,416 |
| Functions | 35,673 |
| Declared runtime dependencies | 48 |

## 1. Semantic-modification risk — the real fragility

The external review's conclusion survives measurement, and its headline number
was almost exact.

```bash
# reproduce
python - <<'EOF'
import ast, pathlib
rows=[]
for p in list(pathlib.Path('.').glob('core/**/*.py'))+list(pathlib.Path('.').glob('interface/**/*.py')):
    try: tree=ast.parse(p.read_text(errors='ignore'))
    except SyntaxError: continue
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            end=getattr(n,'end_lineno',n.lineno)
            branches=sum(1 for c in ast.walk(n) if isinstance(c,(ast.If,ast.For,ast.While,ast.Try,ast.BoolOp,ast.IfExp)))
            rows.append((end-n.lineno,branches,str(p),n.name,n.lineno))
rows.sort(reverse=True)
for r in rows[:12]: print(r)
EOF
```

| Function | Lines | Branches |
| --- | ---: | ---: |
| `interface/routes/chat.py:19194 api_chat` | **4,465** | **632** |
| `core/brain/llm/latent_cortex/engine.py:2660 _latent_episode` | 3,795 | 369 |
| `core/brain/llm/mlx_worker.py:4501 _mlx_worker_loop` | 3,364 | 468 |
| `core/brain/inference_gate.py:7046 generate` | 3,125 | 536 |
| `core/phases/response_generation_unitary.py:4178 execute` | 2,941 | 477 |
| `interface/routes/chat.py:7437 _run_cognitive_engine_chat_turn` | 2,329 | 360 |

14 functions exceed 1,000 lines; 46 exceed 500. **The risk is concentrated,
not diffuse** — 46 of 35,673 functions carry it, which makes it tractable.

### Duplicated responsibility, measured

The review's diagnosis ("a rule existed but was implemented at only one site")
was confirmed three separate times this session, twice by walking into it:

| Rule | Sites | How it surfaced |
| --- | ---: | --- |
| `_recent_completed_conversation_exchanges` | **10** | A single-site fix did nothing; the real cause took four steps to find |
| Prompt block assembly | **3** | Ledger and self-preference blocks each needed three separate edits |
| `assess_user_facing_reply` | **75 call sites** | — |
| `.reasons` consumers | **115 references** | Adding one reason converted a good answer into a refusal |

The last one is the sharpest illustration and is worth stating plainly.
Emitting a new reason from the single assessment chokepoint, *and* adding it
to `_DELIVERABLE_RESIDUAL_SURFACE_REASONS`, still broke three tests — because
some consumer among the 115 treats any reason as a failure. One chokepoint
with 115 disagreeing readers is not a chokepoint.

**Direction (correct, and already underway):** the repo is moving rules toward
structural chokepoints. This session added one (prompt block groups defined
once, rendered three times, with a structural test forbidding per-path
concatenation). The next highest-value target is the reason-list contract:
until every consumer agrees what a residual reason means, no new reason can be
added safely.

## 2. Modularity

**Layering exists but covers almost nothing.** `make layering` enforces DEPS
include-rules, and there are exactly **2** DEPS files in `core/`. The gate is
real; its coverage is not. Extending DEPS to the subsystem boundaries that
already exist by convention (`core/brain`, `core/phases`, `core/conversation`,
`core/memory`) would turn an aspiration into a check.

**ServiceContainer is a genuine seam.** Keys are the spine
(`core/service_names.py`), registration is centralised
(`core/service_registration.py`), and the `phenomenal_engine` bridge the
review flagged as unwired is in fact registered `required=True` and called at
boot — **that finding does not reproduce against current source.**

## 3. Portability

**Platform-locked, and honestly so.** 89 modules import MLX; 65 reference
Darwin/AppKit/Quartz/osascript. The substrate is Apple-silicon MLX by design —
[[project-aura-real-mind-mlx-substrate]] records that an external llama-server
could not be substrate-steered, which is why the coupling exists.

The cost is that `core/brain/llm` is not portable and cannot be tested off
this host. The mitigation already present is the fallback client boundary; the
gap is that no CI can exercise the primary lane. Worth knowing before anyone
plans a Linux deployment: this is not a packaging problem, it is an
architecture commitment.

## 4. Maintainability

**Strong where it is instrumented, weak where it is not.**

* Degradation discipline is real and enforced (`record_degradation`, fail-closed
  lists, no silent `except: pass`).
* The async-write ratchet, layering baseline, and enterprise gate all use
  "baseline only shrinks", which is the right instrument at this size.
* **Coverage was not measured at all** until this session — no line, branch, or
  mutation threshold existed, and `coverage` was not installed. Infrastructure
  now exists (`make coverage`, `make coverage-check`, `make mutation`); a
  baseline still requires a full 6-chunk run.
* **12 test files collected zero tests** and counted inside the ~24,900 figure.
  All twelve now have real tests (79 of them), and a gate refuses new ones.

## 5. External red teaming

**Absent as a practice.** The security surface has gates
(`make security`, `make enterprise-gate`, bandit config, a network sentinel),
and there is a defensive immune system — but no adversarial review by anyone
outside the two agents and the owner.

The nearest thing to red teaming this repo has is the transcript that started
this session: driving the real UI as a user and reading what came back. That
found four defects no gate had. **That is the practice to formalise** — a
scripted adversarial conversation battery, run against the live surface, with
the failure modes from this session as its first cases (topic abandonment,
confabulated provenance, contract misclassification, depth-driven decay).

## 6. Contributor diversity

```
3,623  Zenflow      (parallel agent)
  512  Codex        (this agent)
  241  Claude
   11  Bryan Young
    2  youngbryan97
    2  Bryan
```

**~99.6% of commits are agent-authored.** This is the single largest structural
risk in the list, and it is not a code problem. Every convention in
CLAUDE.md — the async write lane, lockdep, the layering baseline, telemetry
ids — exists because an agent burned itself on the absence of it and wrote it
down. That works, and it is also why the giant functions grew: no human review
step ever rejected a 4,000-line addition.

The concentration also means the review that produced this document's agenda
was the most valuable input the project has had in weeks, precisely because it
came from outside.

## 7. Ecosystem

48 declared dependencies is lean for the surface area. The notable couplings:
MLX (substrate, non-negotiable), pydantic (schemas), sqlite (ubiquitous, used
directly rather than through an ORM). No dependency injection framework, no
ORM, no message broker in the desktop build — all deliberate, all documented
in `pyproject.toml` comments.

## Ranked next actions

1. **Make the reason-list contract explicit.** 115 consumers, no agreed
   semantics. This blocks every future reliability signal, including the
   thread-continuity one this session had to leave as observability.
2. **Extend DEPS beyond 2 files.** The gate works; give it something to check.
3. **Establish the coverage baseline.** One overnight run turns the new
   ratchet from infrastructure into a floor.
4. **Formalise adversarial conversation testing.** Script what the Aug 4
   transcript did by hand.
5. **Decompose `api_chat`.** 4,465 lines / 632 branches, and it is the live
   desktop entry point. Not first, because 1–4 are cheaper and this one needs
   the reason-list contract settled to be safe.
