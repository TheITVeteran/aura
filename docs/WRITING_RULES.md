# Writing rules

Status: Guide · Standing conventions for prose in this repository

Everyone learned the em-dash trick, so the tells moved. These are the patterns
that now read as machine-generated, and the repository is checked against them:

```bash
make writing            # the front-facing docs, ratcheted
python tools/lint_ai_writing.py --all      # every guide
python tools/lint_ai_writing.py FILE       # one file
```

`tools/lint_ai_writing.py` is the executable half of this page. **Adding a
pattern here without adding it there leaves the rule unenforced**, which is how
the last set of writing conventions quietly stopped applying.

The ratchet lives in `config/ai_writing_baseline.json` and only goes down.

---

## The forbidden patterns

### 1. "That's not X. That's Y."

> That's not compliance. That's stalling.

Say the second half only: *They're stalling.* The negated first half exists to
make the second sound earned. It didn't earn it.

### 2. Two short bits stapled together

> Fast. Simple.
> No fluff. Just answers.

Pick one and write it as a sentence.

### 3. Two pictures and no advice

> Less a hammer, more a scalpel.

Say what to do instead.

### 4. Clapping for itself

> And that matters. That's the part everyone misses. Which is exactly the
> point. It's worth stating that…

Delete them all. Nothing is lost. If a sentence needs to be announced as
important, it isn't.

### 5. The analogy that assumes both referents

> It's the Excel of AI agents.

Only works if the reader knows both things. Usually they don't.

### 6. Warming up before talking

> Here's the thing. Let me be clear. The truth is.

Start one sentence later.

### 7. Always three things

> Faster, cheaper, smarter.

Real reasons come in twos and fives. A triad that keeps appearing is a rhythm
the writer fell into, not a count of anything.

### 8. Ranges standing in for measurements

> 5 to 10 minutes.

If you ran it, say what it took.

**Exception, and it matters here.** A range that *is* the measurement stays.
`10–50 ms per φ evaluation` and `30–60 seconds while Metal compiles shaders`
describe real distributions across inputs and machines. Collapsing them to one
invented number would be worse writing and factually wrong. The rule targets
estimates nobody made, not variance somebody measured. The linter flags both;
reviewed-and-kept ranges sit in the baseline.

### 9. The ending that repeats the whole thing

> In short… At the end of the day…

Stop typing.

---

## The four principles underneath

From William Zinsser, *On Writing Well*:

1. **Simplicity**
2. **Brevity**
3. **Clarity**
4. **Humanity**

ASD-STE100 — the controlled-English standard used for aircraft maintenance
manuals, where an ambiguous sentence can kill somebody — delivers the first
three and drops the fourth. Ask for it by name when writing procedures, gates,
runbooks, and API documentation. Do not ask for it when the writing has a
voice: it flattens anything with a heart into a parts catalogue.

## What this does not apply to

**Append-only records are never edited for style.** The execution tracker, the
RLC ledger, `docs/evidence/`, dated verdicts, and benchmark results say what
was true when they were written. Editing a July entry in August is falsifying
a record, not improving prose. The linter excludes them by path; see
[DOC_STATUS.md](DOC_STATUS.md) for which documents are which.

Quoted material keeps its original wording, including when the original
commits one of the nine.

## Extending this

Add a pattern the moment you catch one in a draft. Two edits:

1. A section here, with the example that caught it.
2. A rule in `RULES` in `tools/lint_ai_writing.py`, with a regex and the
   one-line fix shown to whoever trips it.

Then run `python tools/lint_ai_writing.py --all --baseline` to record where the
tree sits. The file gets better every time somebody notices something.
