# Reasoning-delta: 32B + amplifier vs 72B single-pass (hard suite)

**Date:** 2026-06-27 · **Host:** M5 Pro / 64 GB · **Backend:** MLX (mlx_lm 0.31.3)
**Models:** Qwen2.5-32B-Instruct-4bit (cortex), Qwen2.5-72B-Instruct-4bit (solver)
**Suite:** `aura_bench/hard_suite.py` (11 base-failing, objectively-graded tasks)
**Grading:** external + amplifier-independent — numeric final-answer extraction (math/logic),
executed hidden asserts in an AST-screened `-I` subprocess (code). No verifier rubber-stamp.
**Runs:** two sequential bench invocations (never both models resident at once); the
amplified condition runs `skip_cache` (read-only — a measurement can't poison production).

## Headline

| condition | mean score | avg time/task |
|---|:--:|:--:|
| 32B single-pass | 0.818 (9/11) | 13 s |
| **32B + amplifier** | **0.909 (10/11)** | 85 s |
| 72B single-pass | 0.818 (9/11) | 36 s |

**32B + amplifier (0.909) ≥ 72B single-pass (0.818)** and **+0.091 over 32B single-pass**,
with **zero regressions** (amplified ≥ single on every task).

## Per-task (1 = verified correct)

| task | 32B single | 32B + amp | 72B single |
|---|:--:|:--:|:--:|
| pow_17_4 | 1 | 1 | 1 |
| trailing_zeros_100 | 1 | 1 | 1 |
| gcd | 0 | **1** | 1 |
| mod | 0 | 0 | 0 |
| clock_angle | 1 | 1 | 1 |
| primes_sum | 1 | 1 | 1 |
| knights | 1 | 1 | 1 |
| trains | 1 | 1 | **0** |
| rle (code) | 1 | 1 | 1 |
| balanced (code) | 1 | 1 | 1 |
| roman (code) | 1 | 1 | 1 |

## Honest reading (do not overclaim)

- **Real win:** amplification rescued `gcd` (32B 0 → 1) — exactly what raw scale buys the
  72B (which gets `gcd` single-pass). Amplification substituted for parameters there.
- **Noise:** the 72B's `trains` miss is almost certainly sampling variance at temp 0.3 on an
  easy word problem, not a capability gap. With n=11, ±1 task = ±0.09 — so the margin between
  0.909 and 0.818 is within one-task noise. The defensible claim is **parity-or-better**, not
  "strictly beats the 72B."
- **Hard for everyone:** `mod` (123456 mod 7) fails on all three — a verifier/model blind spot,
  not a size issue. A target for hardening.
- **Cost is real:** 32B+amp averages 85 s/task vs 72B-single 36 s — amplification trades compute
  for capability. Where the 72B fits comfortably, single-pass 72B is the faster route to the same
  quality.
- **Why this matters on THIS box:** the 64 GB host auto-disables the local 72B deep solver
  (`AURA_LOCAL_DEEP_AUTO_MIN_TOTAL_GB=96`). So in practice the 72B isn't a comfortable runtime
  option here — and **32B + amplification reaches 72B-class accuracy on hardware that can't run
  the 72B.** That is the practical payoff.

## Caveats / next

- n=11 is small; widen the suite (and add tasks where the 32B fails *more*) for tighter CIs.
- `mod`-class arithmetic is unsolved by all conditions — harden the math verifier so amplification
  can catch it.
- A prior run regressed (−0.18) due to solved-cache poisoning by a vacuous math-verifier pass;
  fixed in `a2e4ab24` (cache/learn now gate on `verdict.checked`). These numbers are post-fix.
