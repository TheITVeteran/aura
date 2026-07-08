# Compounding weight-learning proof bundle — 2026-07-07, Qwen2.5-1.5B-4bit, 2 cycles

Two consecutive **unsupervised** learning cycles, run by `tools/learning_demo.py`
(reproduce yourself: `make demo-learning`, Apple Silicon, ~20–40 min). No human
intervened between the start of cycle 1 and the final verdict.

## What happened, mechanically

1. **Self-play harvest** (`selfplay_gen0.json`): the base model sampled 4
   attempts per task at temperature 0.8 against 32 seeded reasoning tasks
   (arithmetic, logic, string manipulation — each task computes its own exact
   ground truth). Every attempt was graded by the task's exact checker.
   Verified-correct vs verified-wrong attempts on the SAME prompt became DPO
   pairs (`verifiable_preferences.jsonl`). The verifier is the reward —
   there is no reward model to hack.
2. **Cycle 1** (`runs/g0000-*/`): DPO-trained a LoRA on those pairs
   (`data/` holds the exact training rows), evaluated base-vs-candidate on a
   **sealed held-out battery** (fresh task seeds ≥1000, disjoint by
   construction from all training seeds <1000, fingerprint-checked against
   contamination), passed the gate, fused, published, and wrote the manifest.
3. **Cycle 2** (`runs/g0001-*/`): resolved its base **from the manifest** —
   i.e. cycle 1's freshly published artifact — harvested new self-play data
   *with those weights* (`selfplay_gen1.json`), trained on top of them, and
   passed the same sealed gate. That chain is the compounding claim.
4. **Ledger** (`lineage.jsonl`): every generation appended to a hash-chained,
   tamper-evident ledger. The verdict below is computed from ledger records,
   not asserted by the demo.

## The honest result

| generation | candidate (sealed held-out) | incumbent on same battery | hidden battery | status |
|---|---|---|---|---|
| g0000 | **0.667** | 0.583 (raw base) | 0.667 | promoted |
| g0001 | 0.625 | 0.625 (g0000) | 0.750 | promoted |

Cycle 1 beat its raw base by +8.3pp on the sealed battery — a real single-step
gain from pure self-play. Cycle 2 held capability but did not exceed it.

**Ledger verdict: `BOUNDED_SELF_OPTIMIZATION`** — the loop runs end-to-end
unsupervised and compounds mechanically, but the cross-generation capability
curve is NOT strictly increasing (0.667 → 0.625 is within ±1-task noise on a
24-task battery, and we do not round noise up to "gain"). Capability growth
remains an open claim (CLAIMS_MATRIX.md claim 16); the mechanism is claim 23.

## Audit trail

- `runs/*/cycle_receipt.json` — full receipt per cycle (base source, data
  counts, eval numbers, promote/refuse reasons, ledger entry hash).
- `runs/*/candidate_eval.responses.jsonl` / `incumbent_eval.responses.jsonl` /
  `hidden_eval.responses.jsonl` — every raw model response on every eval task,
  regradeable offline against the task answers embedded alongside.
- `runs/*/data/` — the exact train/valid/test rows each cycle trained on.
- `verifiable_preferences.jsonl` — the full DPO store with provenance.
- `lineage.jsonl` — the hash-chained generation ledger; verify with
  `WeightCompoundingLoop.verify_ledger()` or by re-hashing records yourself.

Fused model weights (~800MB per generation) and LoRA adapters (~40MB each)
are not committed; every decision they fed is reproducible from the receipts
above, and the whole run can be regenerated with one command.
