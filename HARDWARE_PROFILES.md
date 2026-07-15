# Aura Hardware & Model Profiles

This document defines the supported hardware and model execution profiles for the Aura runtime, detailing what claims can be validated under each hardware constraint.

## 1. No-Model / Dev Profile
* **Target Hardware**: Standard laptop (e.g. Intel/M1 MacBook Air), 8GB RAM.
* **Required Models**: None (mocks/stubs only).
* **Memory/Compute**: Minimal resource requirement.
* **Allowed Claims**:
  - `governed runtime` (static verification only)
  - `production-sealed` (static gate validation)
* **Disallowed Claims**: All empirical claims, including `operational volition`, `autonomous agency`, `emergent intelligence`, `DNU AGI`, `synthetic cognitive entity`.
* **Tests That Can Run**:
  - `python -m compileall`
  - `pytest --collect-only`
  - Strict Flagship Readiness check
  - Production Surface Lint check
  - Static Enterprise/Readiness gates
* **Tests That Are Blocked**: All live capability runs, agent loop tests, longevity soak, and model-dependent tests.

---

## 2. CI / Proof-Short Profile
* **Target Hardware**: Virtualized CI Runner (e.g. GitHub Actions standard runner), 2-4 vCPUs, 7-14GB RAM.
* **Required Models**: Light local MLX-compatible models for bounded proof runs.
* **Memory/Compute**: Bounded.
* **Allowed Claims**:
  - `governed runtime` (receipt verification on light runs)
  - `persistent memory` (local persistent memory writes)
  - `operational volition` (bounded Will Decision receipt logging)
  - `production-sealed`
* **Disallowed Claims**: `emergent intelligence`, `external real-world validation`, `DNU AGI`, `AGI-candidate`, `mature RSI`, `synthetic cognitive entity`.
* **Tests That Can Run**:
  - All unit/integration tests (`pytest`)
  - Bounded Agency Emergence proof runs (with local/mocked LLMs)
  - Bounded Longevity soak (`proof_short` profile)
* **Tests That Are Blocked**: Full 100-task DNU AGI suite, multi-hour longevity soak, high-capacity model-reasoning evaluations.

---

## 3. Local Apple Silicon Profile
* **Target Hardware**: Mac Studio / MacBook Pro (M2/M3/M4 Max), 64GB - 128GB Unified Memory.
* **Required Models**: the three in-process MLX tiers — Cortex (32B, foreground), Brainstem (7B, background), and Reflex (1.5B, fast lane) — plus Qwen-2.5-Coder-7B-Instruct (local) for code work.
* **Memory/Compute**: High-throughput CPU/GPU memory bandwidth.
* **Allowed Claims**:
  - `governed runtime`, `persistent memory`, `causal internal state`, `affect steering`, `System 2 planning/search`, `self-repair`
  - `operational volition`, `autonomous agency`, `entity-in-a-box behavior`
  - `experience-adjacent functional indicators`
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, `external real-world validation` (requires cloud APIs and high-horizon scale), `indefinite autonomy`.
* **Tests That Can Run**:
  - Local model-aware agency emergence batteries
  - Local sandbox/boxed entity suites
  - Medium-duration longevity soak (e.g., `local_4h`)
* **Tests That Are Blocked**: Multi-day longevity soak (e.g., `local_72h`), full cloud-scale external validation.

---

## 4. Local High-Memory Profile
* **Target Hardware**: Dedicated Workstation / Server, 128GB+ System RAM, 2x NVIDIA RTX 4090 or A6000 GPUs.
* **Required Models**: Aura MLX 32B/72B lane artifacts, DeepSeek-Coder-33B (local quantized).
* **Memory/Compute**: Massive local GPU memory allocation.
* **Allowed Claims**: Same as Local Apple Silicon, plus:
  - `emergent intelligence` (locally evaluated on larger distributions)
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, `indefinite autonomy`.
* **Tests That Can Run**:
  - Heavy local model reasoning runs
  - Local System 2 search rollouts
  - Longer longevity soak (e.g., `local_24h`)
* **Tests That Are Blocked**: Full cloud-scale third-party benchmark gates.

---

## 5. Cloud / External Model Profile
* **Target Hardware**: Any host with network access to the Google Gemini API — the only cloud adapter Aura ships (`core/brain/llm/gemini_adapter.py`). Cloud is an opt-in fallback for the reasoning lanes, never the default substrate; the local MLX tiers remain primary.
* **Required Models**: Gemini 3.5 Flash (chat / deep lanes) and Gemini 3.5 Pro (thinking lane) by default, with per-model daily/minute rate-limit tiers overridable via `AURA_GEMINI_*` environment variables.
* **Memory/Compute**: Network-bound, infinite API compute resources.
* **Allowed Claims**:
  - All local properties, plus:
  - `DNU AGI` (requires complete API budget and execution unblocking)
  - `AGI-candidate`
  - `external real-world validation`
* **Disallowed Claims**: `subjective consciousness`, `personhood`, `metaphysical free will` (strictly banned).
* **Tests That Can Run**:
  - Full 100-task DNU AGI battery
  - External live validation scenarios using web/browser tools
* **Tests That Are Blocked**: Bounded only by rate limits and network connection status.

---

## 6. Live Hardware / Browser Profile
* **Target Hardware**: Dedicated robotic/embodied system or developer workstation with full system access and live web interface hooks.
* **Required Models**: Mixed local and cloud LLM runtime.
* **Memory/Compute**: Unconstrained host access.
* **Allowed Claims**: Bounded by authorization/compliance profiles.
* **Disallowed Claims**: `mature RSI` (unless sandboxed with rollback), subjective consciousness.
* **Tests That Can Run**:
  - Live browser-use and OS-control validation
  - Physical or simulation co-presence integration
* **Tests That Are Blocked**: Bounded by environment safety profiles and authority filters.
