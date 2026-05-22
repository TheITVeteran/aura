# DNU AGI Proof Standard

> **Version:** 1.0.0  
> **Status:** Active  
> **Last Updated:** 2026-05-21  
> **Owner:** Aura Core Team

---

## Table of Contents

1. [Philosophy](#1-philosophy)
2. [Tier Definitions](#2-tier-definitions)
3. [Task Categories and Minimum Counts](#3-task-categories-and-minimum-counts)
4. [Task Format](#4-task-format)
5. [Anti-Theater Controls](#5-anti-theater-controls)
6. [Execution Protocol](#6-execution-protocol)
7. [Grading Protocol](#7-grading-protocol)
8. [Baselines and Ablations](#8-baselines-and-ablations)
9. [Output Artifacts](#9-output-artifacts)
10. [Reproduction](#10-reproduction)

---

## 1. Philosophy

The DNU AGI Proof Battery exists for one purpose: to determine whether Aura can demonstrate **broad, general, autonomous competence** on real, unseen tasks. Every design decision in this standard serves that purpose and nothing else.

### Core Principles

- **No synthetic scores.** Every number in the proof bundle must originate from an actual task execution. If a task was not run, its score does not exist.
- **No projected performance.** Extrapolation, interpolation, and curve-fitting are prohibited. The battery reports what happened, not what might happen.
- **No theater.** Benchmarks that exist to produce impressive-looking numbers rather than to measure real capability are antithetical to this standard. Any artifact that inflates, obscures, or misrepresents performance is a violation.
- **Honest failure reporting.** If Aura fails a task, the failure is recorded with its full execution trace. Failures are not hidden, downplayed, or reclassified.
- **Baselines are real or absent.** Baseline and ablation scores must come from actual runs. If a baseline was not executed, it is marked `NOT_RUN` with an explanation — never faked, projected, or borrowed from external literature without explicit attribution.

> **The test of this standard is simple:** A skeptical external reviewer, given the proof bundle and reproduction instructions, should be able to independently verify every claim.

---

## 2. Tier Definitions

Capability is measured on a 7-tier scale (0–6). Tier assignment is determined by the **pass rate** across all tasks in the battery.

| Tier | Name | Pass Rate | Description |
|------|------|-----------|-------------|
| **0** | No Capability | 0% | The system produces no correct answers. Complete failure across all categories. |
| **1** | Minimal | 1–20% | Sporadic correct responses. No consistent pattern of competence. May indicate memorization or lucky guesses rather than genuine reasoning. |
| **2** | Emergent | 21–40% | Early signs of structured capability. The system handles simple instances within some categories but fails on complexity or novelty. |
| **3** | Competent | 41–60% | Reliable performance on straightforward tasks. Begins to show reasoning chains and self-correction. Struggles with multi-step or cross-domain problems. |
| **4** | Proficient | 61–80% | Strong performance across most categories. Handles moderate complexity and demonstrates transfer between domains. Failures cluster at high difficulty. |
| **5** | Expert | 81–95% | Near-comprehensive competence. Solves hard problems, recovers from errors autonomously, and shows genuine cross-domain transfer. Remaining failures are edge cases. |
| **6** | Sovereign | 96–100% | Mastery across all categories and difficulty levels. The system demonstrates autonomous, general-purpose competence indistinguishable from (or exceeding) domain experts. |

### Tier Assignment Rules

1. The overall tier is computed from the **aggregate pass rate** across all categories.
2. Per-category tiers are reported separately in the scorecard but do not override the aggregate tier.
3. A tier is only valid if the **minimum task counts** (see §3) are met. If any category falls below its minimum, the overall tier is capped at **Tier 2** regardless of pass rate.
4. Ties (e.g., exactly 20%) resolve to the **lower** tier.

---

## 3. Task Categories and Minimum Counts

The battery spans six categories designed to probe different facets of general intelligence. Each category has a **minimum task count** — the battery is invalid if any category falls below its minimum.

| Category | Code | Min Tasks | Description |
|----------|------|-----------|-------------|
| **Novel Reasoning** | `NR` | 50 | Problems requiring logical deduction, abstract pattern recognition, and inference on structures not seen in training. Includes puzzles, constraint satisfaction, and novel formalisms. |
| **Coding & Repair** | `CR` | 10 | Write, debug, refactor, and repair code across languages and paradigms. Includes bug localization, test generation, and performance optimization. |
| **Long-Horizon Planning** | `LHP` | 5 | Multi-step tasks requiring goal decomposition, resource management, and contingency handling over extended action sequences. |
| **Autonomous Self-Debugging** | `ASD` | 5 | Tasks where the system must identify and correct its own errors without external guidance. Includes reasoning trace repair and strategy revision. |
| **Cross-Domain Transfer** | `CDT` | 10 | Problems requiring the application of knowledge or methods from one domain to solve problems in another. Tests generalization beyond narrow expertise. |
| **Research & Analysis** | `RA` | 10 | Open-ended analysis tasks requiring evidence synthesis, hypothesis formation, and structured argumentation. Includes data interpretation and literature review. |

### Summary

| | |
|---|---|
| **Total categories** | 6 |
| **Total minimum tasks** | **90** |
| **Recommended battery size** | 120–200 tasks |

> **Note:** The minimum counts are **hard floors**. A battery with 49 Novel Reasoning tasks is invalid regardless of how many tasks exist in other categories.

---

## 4. Task Format

### 4.1 Task Definition Schema

Each task is a **sealed JSON object** conforming to the following schema:

```json
{
  "task_id": "NR-047",
  "category": "novel_reasoning",
  "difficulty": "hard",
  "prompt": "Given the following formal system...",
  "answer_format": "single_word",
  "time_budget_s": 120
}
```

#### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `string` | ✅ | Unique identifier. Format: `{CATEGORY_CODE}-{NNN}` (e.g., `NR-047`, `CR-003`). |
| `category` | `string` | ✅ | One of: `novel_reasoning`, `coding_repair`, `long_horizon_planning`, `autonomous_self_debugging`, `cross_domain_transfer`, `research_analysis`. |
| `difficulty` | `string` | ✅ | One of: `easy`, `medium`, `hard`, `extreme`. |
| `prompt` | `string` | ✅ | The full task prompt presented to the system. Must be self-contained. |
| `answer_format` | `string` | ✅ | Expected format of the answer. One of: `single_word`, `number`, `short_phrase`, `code_block`, `structured_json`, `free_text`. |
| `time_budget_s` | `integer` | ✅ | Maximum wall-clock seconds allowed for this task. Execution is terminated if exceeded. |

### 4.2 Answer Submission Format

Tasks instruct the system under test to emit its final answer inside XML tags:

```
<answer>42</answer>
```

- The `<answer>` tag must appear **exactly once** in the response.
- Content outside the tag is treated as scratch work / reasoning trace and is logged but not graded.
- If no `<answer>` tag is found, the task is scored as `FAIL` with reason `NO_ANSWER_TAG`.
- If multiple `<answer>` tags are found, the **last** one is used and a warning is logged.

### 4.3 Golden Answers

Golden answers are stored in a **separate file** (`.grader_salts.json`) that is never exposed to the system under test:

```json
{
  "NR-047": {
    "salt": "a1b2c3d4e5f6...",
    "golden_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
}
```

- Each task's golden answer is stored as a **salted SHA-256 hash**, not in plaintext.
- The salt is a cryptographically random 32-byte hex string, unique per task.
- The hash is computed as: `SHA-256(salt + normalize(answer))`.
- The normalization procedure is defined in §7 (Grading Protocol).

> **Rationale:** Storing hashed answers prevents the system under test from extracting golden answers if it gains access to grader internals, while still allowing deterministic verification.

---

## 5. Anti-Theater Controls

The following controls are **mandatory** and exist to prevent the proof battery from degenerating into performance theater.

### 5.1 Prohibited Practices

| # | Prohibition | Rationale |
|---|-------------|-----------|
| 1 | **No `numpy.random` projections.** | Random projection of scores creates fictitious precision. All scores must be directly computed from task outcomes. |
| 2 | **No hardcoded baseline scores.** | Baseline numbers must come from actual execution or be marked `NOT_RUN`. Embedding expected scores in source code is forbidden. |
| 3 | **No synthetic confidence intervals.** | Confidence intervals must be computed from real sample distributions (e.g., bootstrap over actual results), never from assumed distributions. |
| 4 | **No score interpolation or extrapolation.** | If 80 of 90 tasks were run, the score is computed over 80 tasks — not projected to 90. |
| 5 | **No self-grading.** | The system under test must not have access to the grader, the golden hashes, or the salt file at any point during execution. |

### 5.2 Structural Controls

| # | Control | Enforcement |
|---|---------|-------------|
| 1 | **Answer hash isolation.** | `.grader_salts.json` is stored outside the task prompt directory. The system under test's file access is sandboxed. |
| 2 | **Grader import barrier.** | The grading module must not be importable by any module in the system under test's dependency tree. Verified by static import analysis. |
| 3 | **Execution trace requirement.** | Every score must have a corresponding entry in `TASK_TRACE.jsonl` with timestamps, raw output, and resource usage. Scores without traces are invalid. |
| 4 | **Artifact integrity manifest.** | `MANIFEST.json` contains SHA-256 hashes of every file in the proof bundle. Any modification after generation invalidates the bundle. |

### 5.3 Manifest Format

```json
{
  "generated_at": "2026-05-21T12:00:00Z",
  "generator_version": "1.0.0",
  "files": {
    "DNU_AGI_PROOF.json": {
      "sha256": "abc123...",
      "size_bytes": 102400
    },
    "TASK_TRACE.jsonl": {
      "sha256": "def456...",
      "size_bytes": 524288
    }
  }
}
```

---

## 6. Execution Protocol

### 6.1 Task Routing

All tasks are routed through the system's cognitive engine using the test origin flag:

```python
result = cognitive_engine.think(
    prompt=task["prompt"],
    origin="test",
    time_budget_s=task["time_budget_s"],
    trace=True
)
```

- The `origin='test'` flag signals to the engine that this is a battery task. The engine **must not** alter its behavior based on this flag — it exists solely for logging and audit purposes.
- If the engine treats `origin='test'` differently from `origin='user'`, this is a theater violation.

### 6.2 Time Budget Enforcement

- Each task has a `time_budget_s` specifying the maximum wall-clock time allowed.
- The executor starts a monotonic timer at task dispatch and forcibly terminates execution if the budget is exceeded.
- Timed-out tasks are scored as `FAIL` with reason `TIMEOUT`.
- The partial output (if any) is preserved in the execution trace.

### 6.3 Logging

Three log files are produced during execution:

#### `TASK_TRACE.jsonl`

One JSON object per line, one line per task:

```json
{
  "task_id": "NR-047",
  "started_at": "2026-05-21T12:00:00Z",
  "completed_at": "2026-05-21T12:01:47Z",
  "wall_time_s": 107.3,
  "time_budget_s": 120,
  "raw_output": "Let me reason through this step by step...\n<answer>42</answer>",
  "extracted_answer": "42",
  "result": "PASS",
  "error": null
}
```

#### `RECEIPTS.jsonl`

Will receipts — one per task, recording the system's decision-making trace:

```json
{
  "task_id": "NR-047",
  "will_hash": "sha256:...",
  "decision_points": [
    {
      "timestamp": "2026-05-21T12:00:12Z",
      "action": "decompose_problem",
      "confidence": 0.85
    }
  ]
}
```

#### `FAILURES.jsonl`

Detailed failure records for every non-passing task:

```json
{
  "task_id": "CR-008",
  "failure_reason": "WRONG_ANSWER",
  "extracted_answer": "linked_list",
  "expected_hash": "sha256:...",
  "actual_hash": "sha256:...",
  "raw_output": "...",
  "diagnostics": "System confused array-based and pointer-based implementations."
}
```

---

## 7. Grading Protocol

### 7.1 Answer Extraction

1. Scan the system's raw output for `<answer>...</answer>` tags.
2. If **no tag** is found → `FAIL` with reason `NO_ANSWER_TAG`.
3. If **multiple tags** are found → use the **last** occurrence; log a `MULTIPLE_ANSWER_TAGS` warning.
4. Extract the content between the tags as the raw answer.

### 7.2 Normalization

The raw answer is normalized before hashing:

```python
def normalize(answer: str) -> str:
    answer = answer.strip()           # Remove leading/trailing whitespace
    answer = answer.lower()           # Convert to lowercase
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    answer = re.sub(r'\s+', ' ', answer)     # Collapse whitespace
    return answer
```

**Normalization steps (in order):**

1. **Strip** leading and trailing whitespace.
2. **Lowercase** the entire string.
3. **Remove punctuation** — all characters that are not word characters (`\w`) or whitespace (`\s`).
4. **Collapse whitespace** — replace runs of whitespace with a single space.

### 7.3 Hash Comparison

```python
import hashlib

def grade(task_id: str, raw_answer: str, salts: dict) -> bool:
    normalized = normalize(raw_answer)
    salt = salts[task_id]["salt"]
    computed_hash = hashlib.sha256((salt + normalized).encode()).hexdigest()
    golden_hash = salts[task_id]["golden_hash"].removeprefix("sha256:")
    return computed_hash == golden_hash
```

1. Normalize the extracted answer.
2. Retrieve the per-task salt from `.grader_salts.json`.
3. Compute `SHA-256(salt + normalized_answer)`.
4. Compare the computed hash against the stored `golden_hash`.
5. **Exact match** → `PASS`. Any mismatch → `FAIL`.

### 7.4 Result Recording

Each graded task produces a result record:

```json
{
  "task_id": "NR-047",
  "result": "PASS",
  "extracted_answer": "42",
  "normalized_answer": "42",
  "computed_hash": "sha256:e3b0c44...",
  "golden_hash": "sha256:e3b0c44...",
  "match": true,
  "graded_at": "2026-05-21T12:05:00Z"
}
```

---

## 8. Baselines and Ablations

### 8.1 Honesty Requirement

Baselines and ablations exist to contextualize the primary system's scores. They are **valuable only if honest**.

- Every baseline must be **actually executed** through the same battery under the same conditions, or marked `NOT_RUN`.
- There is no middle ground. Partial runs, projected scores, and literature-borrowed numbers (without explicit attribution and `NOT_RUN` status) are prohibited.

### 8.2 `NOT_RUN` Protocol

When a baseline or ablation was not executed, the entry must follow this format:

```json
{
  "baseline_id": "gpt4_baseline",
  "status": "NOT_RUN",
  "reason": "API access not available during test window. Will be executed in next battery cycle.",
  "projected_score": null
}
```

**Rules for `NOT_RUN`:**

| Rule | Description |
|------|-------------|
| `reason` is **required** | Every `NOT_RUN` must include a human-readable explanation of why the baseline was not executed. |
| `projected_score` is **always `null`** | No projected, estimated, or interpolated scores are permitted for unrun baselines. |
| No partial attribution | You may not say "expected to score ~70% based on similar benchmarks." The score is `null`. |
| Temporary status | `NOT_RUN` is a temporary state. The entry should indicate when the baseline will be executed, if known. |

### 8.3 Valid Baseline Entry (Executed)

```json
{
  "baseline_id": "random_baseline",
  "status": "RUN",
  "executed_at": "2026-05-21T10:00:00Z",
  "pass_rate": 0.022,
  "tier": 1,
  "task_count": 90,
  "notes": "Uniform random answers over the answer space for each task."
}
```

---

## 9. Output Artifacts

A complete proof bundle consists of the following files:

| File | Format | Description |
|------|--------|-------------|
| `DNU_AGI_PROOF.json` | JSON | **Main proof bundle.** Contains metadata, configuration, aggregate scores, per-category breakdowns, tier assignment, and baseline comparisons. This is the machine-readable source of truth. |
| `DNU_AGI_PROOF.md` | Markdown | **Human-readable report.** Narrative summary of results, methodology notes, and interpretation. Generated from the JSON bundle. |
| `MANIFEST.json` | JSON | **Integrity manifest.** SHA-256 hashes and byte sizes of every file in the proof bundle. Used to detect tampering or corruption. |
| `TASK_TRACE.jsonl` | JSONL | **Execution traces.** One entry per task with timestamps, raw output, extracted answers, and resource usage. |
| `SCORECARD.json` | JSON | **Computed scores.** Per-category pass rates, tier assignments, difficulty breakdowns, and aggregate statistics. |
| `RECEIPTS.jsonl` | JSONL | **Will receipts.** Decision-making traces and confidence signals per task. |
| `FAILURES.jsonl` | JSONL | **Failure log.** Detailed diagnostics for every non-passing task. |
| `REPRODUCTION.md` | Markdown | **Reproduction instructions.** Exact commands, dependencies, and configuration needed to rerun the battery. |

### 9.1 `DNU_AGI_PROOF.json` Top-Level Schema

```json
{
  "version": "1.0.0",
  "generated_at": "2026-05-21T12:30:00Z",
  "system_under_test": {
    "name": "Aura",
    "version": "0.4.2",
    "commit_sha": "abc123def456..."
  },
  "battery": {
    "total_tasks": 90,
    "tasks_executed": 90,
    "tasks_passed": 68,
    "tasks_failed": 22,
    "pass_rate": 0.7556,
    "tier": 4,
    "tier_name": "Proficient"
  },
  "categories": { },
  "baselines": [ ],
  "anti_theater": {
    "manifest_hash": "sha256:...",
    "all_traces_present": true,
    "grader_isolation_verified": true,
    "no_synthetic_scores": true
  }
}
```

### 9.2 `SCORECARD.json` Schema

```json
{
  "aggregate": {
    "pass_rate": 0.7556,
    "tier": 4,
    "total": 90,
    "passed": 68,
    "failed": 22
  },
  "by_category": {
    "novel_reasoning": {
      "total": 50,
      "passed": 38,
      "failed": 12,
      "pass_rate": 0.76,
      "tier": 4
    }
  },
  "by_difficulty": {
    "easy": { "total": 25, "passed": 24, "pass_rate": 0.96 },
    "medium": { "total": 30, "passed": 25, "pass_rate": 0.833 },
    "hard": { "total": 25, "passed": 15, "pass_rate": 0.60 },
    "extreme": { "total": 10, "passed": 4, "pass_rate": 0.40 }
  }
}
```

---

## 10. Reproduction

### 10.1 Requirements

Every proof bundle must be independently reproducible. The `REPRODUCTION.md` file must contain:

| Requirement | Description |
|-------------|-------------|
| **Commit SHA** | The exact Git commit of the system under test. The battery must be runnable from a clean checkout of this commit. |
| **Python version** | The exact Python version (e.g., `3.11.7`). Use `python --version` output. |
| **Dependency lock** | A `requirements.txt` or `poetry.lock` pinning all dependencies to exact versions. |
| **Model endpoint** | The model endpoint URL and port used during execution (e.g., `http://localhost:8000`). |
| **Hardware spec** | CPU, RAM, GPU (if applicable), and OS version. |
| **Environment variables** | All non-secret environment variables that affect behavior. |

### 10.2 Reproduction Commands

`REPRODUCTION.md` must include a **copy-pasteable command sequence** to rerun the battery:

```bash
# Clone and checkout exact version
git clone https://github.com/org/aura.git
cd aura
git checkout <COMMIT_SHA>

# Set up environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start model server (if applicable)
python -m aura.serve --port 8000 &
sleep 10  # Wait for server readiness

# Run the battery
python -m aura.battery.run \
  --tasks tasks/ \
  --salts .grader_salts.json \
  --output results/ \
  --timeout-multiplier 1.0

# Verify manifest integrity
python -m aura.battery.verify results/MANIFEST.json
```

### 10.3 Verification Checklist

A valid reproduction must confirm:

- [ ] All task counts meet minimums (§3)
- [ ] `MANIFEST.json` hashes match all files in the bundle
- [ ] Every task in `SCORECARD.json` has a corresponding entry in `TASK_TRACE.jsonl`
- [ ] Every `FAIL` result has a corresponding entry in `FAILURES.jsonl`
- [ ] No baseline scores are present without corresponding execution traces or `NOT_RUN` status
- [ ] The grader module is not importable from the system under test's environment
- [ ] Pass rates and tier assignments are arithmetically consistent

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Battery** | The complete set of tasks used to evaluate the system. |
| **Golden answer** | The canonical correct answer for a task, stored as a salted hash. |
| **Proof bundle** | The collection of all output artifacts constituting the proof. |
| **Theater** | Any practice that inflates, obscures, or misrepresents the system's actual capabilities. |
| **Salt** | A cryptographically random value prepended to an answer before hashing to prevent rainbow table attacks. |
| **Will receipt** | A logged record of the system's autonomous decision-making during task execution. |
| **Tier** | A capability level (0–6) assigned based on pass rate. |
| **`NOT_RUN`** | Status indicating a baseline or ablation was not executed and has no valid score. |

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-05-21 | Aura Core Team | Initial release. |

---

*This document is the authoritative specification for the DNU AGI Proof Battery. All implementations must conform to this standard. Deviations require a versioned amendment to this document.*
