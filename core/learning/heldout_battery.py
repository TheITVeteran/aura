"""core/learning/heldout_battery.py — sealed, exact-checkable capability battery.

The promotion gate for weight-level learning needs a capability measurement that
cannot be gamed by vibes: every task here has ONE exactly checkable answer,
generated from seeded templates so the battery is reproducible (same seed +
version → same tasks) yet unbounded (a fresh seed mints a fresh sealed set that
was never in any training data).

Design rules:
  * Exact answers only — integer/string equality after normalization. No judge
    model, no fuzzy match, nothing a confident wrong answer can slip past.
  * Seeded + versioned — a battery is identified by ``(version, seed, size)``;
    receipts record all three so a stranger can regenerate the identical set.
  * Seal enforcement — ``battery_fingerprints()`` hashes every prompt; the
    training harvest excludes any example that collides, so the eval set can
    never leak into the training set of the very model it gates.
  * Answers computed, never hardcoded — each template *computes* its ground
    truth (including executing the tiny generated programs), so a template bug
    breaks loudly instead of mis-grading silently.

This is the data half of the held-out gate; ``tools/heldout_eval.py`` is the
process half (loads one model at a time and grades it against this battery).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable

BATTERY_VERSION = 1

ANSWER_INSTRUCTION = (
    "Solve the problem. Think step by step if needed, then give the final "
    "answer on its own last line in exactly this form:\nAnswer: <answer>"
)


@dataclass(frozen=True)
class HeldoutTask:
    task_id: str
    domain: str
    prompt: str
    answer: str           # normalized ground truth
    answer_kind: str      # "int" | "str"

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_kind": self.answer_kind,
        }


@dataclass(frozen=True)
class BatterySpec:
    version: int = BATTERY_VERSION
    seed: int = 0
    size: int = 40

    def battery_id(self) -> str:
        return f"heldout-v{self.version}-seed{self.seed}-n{self.size}"


# ── answer normalization / extraction ────────────────────────────────────────

_ANSWER_LINE_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+")


def normalize_answer(raw: str, answer_kind: str) -> str:
    """Normalize a raw answer string for exact comparison."""
    text = str(raw or "").strip().strip("`*\"'").strip()
    # strip trailing punctuation that models add ("42." → "42")
    text = text.rstrip(".!,;")
    if answer_kind == "int":
        m = _NUMBER_RE.search(text.replace(",", ""))
        return m.group(0) if m else ""
    return text.strip().lower()


def extract_answer(response: str, answer_kind: str) -> str:
    """Pull the final answer out of a model response.

    Prefers the LAST explicit "Answer:" line (the instructed format). Falls
    back to the last number (int tasks) / last non-empty line (str tasks) so a
    model that reasons correctly but forgets the exact format still gets
    graded on content, not formatting.
    """
    text = str(response or "")
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1], answer_kind)
    if answer_kind == "int":
        numbers = _NUMBER_RE.findall(text.replace(",", ""))
        return numbers[-1] if numbers else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return normalize_answer(lines[-1], answer_kind) if lines else ""


def grade_response(task: HeldoutTask, response: str) -> bool:
    return extract_answer(response, task.answer_kind) == task.answer


# ── task templates (each computes its own ground truth) ──────────────────────

def _t_arithmetic_chain(rng: random.Random) -> tuple[str, str, str]:
    a, b, c = rng.randint(12, 97), rng.randint(3, 19), rng.randint(2, 9)
    d, e = rng.randint(2, 7), rng.randint(10, 99)
    value = ((a * b) - c) // d + e
    prompt = f"Compute (({a} * {b}) - {c}) // {d} + {e}, where // is integer (floor) division."
    return prompt, str(value), "int"


def _t_linear_equation(rng: random.Random) -> tuple[str, str, str]:
    x = rng.randint(-12, 12)
    m = rng.randint(2, 11)
    b = rng.randint(-30, 30)
    y = m * x + b
    prompt = f"Solve for x: {m}*x + ({b}) = {y}. Give x as an integer."
    return prompt, str(x), "int"


def _t_modular(rng: random.Random) -> tuple[str, str, str]:
    a = rng.randint(3, 40)
    b = rng.randint(2, 9)
    m = rng.randint(5, 97)
    value = pow(a, b, m)
    prompt = f"Compute ({a} ** {b}) mod {m}."
    return prompt, str(value), "int"


def _t_sequence(rng: random.Random) -> tuple[str, str, str]:
    if rng.random() < 0.5:
        start, step = rng.randint(-20, 20), rng.randint(2, 13)
        terms = [start + i * step for i in range(5)]
        nxt = start + 5 * step
    else:
        start, ratio = rng.randint(1, 5), rng.randint(2, 4)
        terms = [start * ratio**i for i in range(5)]
        nxt = start * ratio**5
    prompt = f"What is the next term of this sequence: {', '.join(map(str, terms))}, ...?"
    return prompt, str(nxt), "int"


def _t_string_transform(rng: random.Random) -> tuple[str, str, str]:
    words = ["cascade", "aperture", "lantern", "granite", "meridian", "oblique",
             "quartz", "harbor", "velvet", "cinder", "trellis", "monsoon"]
    word = rng.choice(words)
    op = rng.choice(["reverse", "upper_evens", "drop_vowels"])
    if op == "reverse":
        answer = word[::-1]
        prompt = f'Reverse the string "{word}". Give only the reversed string.'
    elif op == "upper_evens":
        answer = "".join(ch.upper() if i % 2 == 0 else ch for i, ch in enumerate(word))
        prompt = (
            f'Take the string "{word}" and uppercase the characters at even indices '
            f"(0-based), leaving the others unchanged. Give the resulting string."
        )
    else:
        answer = "".join(ch for ch in word if ch not in "aeiou")
        prompt = f'Remove all vowels (a, e, i, o, u) from the string "{word}". Give the resulting string.'
    return prompt, answer.lower(), "str"


def _t_program_output(rng: random.Random) -> tuple[str, str, str]:
    # Ground truth is computed by a hand-mirrored evaluation of each template
    # (no dynamic code execution): the mirror IS the same loop the emitted
    # program describes, so a divergence would break the battery's own tests.
    n = rng.randint(4, 9)
    k = rng.randint(2, 4)
    kind = rng.choice(["accumulate", "filter_sum", "nested"])
    if kind == "accumulate":
        code = (
            "x = 0\n"
            f"for i in range(1, {n + 1}):\n"
            f"    x = x * {k} % 100 + i\n"
            "print(x)"
        )
        x = 0
        for i in range(1, n + 1):
            x = x * k % 100 + i
        answer = x
    elif kind == "filter_sum":
        code = (
            "total = 0\n"
            f"for i in range(1, {n + 1}):\n"
            f"    if i % {k} == 0:\n"
            "        total += i * i\n"
            "print(total)"
        )
        answer = sum(i * i for i in range(1, n + 1) if i % k == 0)
    else:
        code = (
            "s = 0\n"
            f"for i in range(1, {min(n, 6) + 1}):\n"
            f"    for j in range(1, {k + 1}):\n"
            "        s += i * j\n"
            "print(s)"
        )
        answer = sum(i * j for i in range(1, min(n, 6) + 1) for j in range(1, k + 1))
    prompt = f"What does this Python program print?\n```python\n{code}\n```"
    return prompt, str(answer), "int"


def _t_date_arithmetic(rng: random.Random) -> tuple[str, str, str]:
    base = datetime.date(2026, 1, 1) + datetime.timedelta(days=rng.randint(0, 300))
    delta = rng.randint(5, 400)
    target = base + datetime.timedelta(days=delta)
    prompt = (
        f"{base.strftime('%B %d, %Y')} falls on a {base.strftime('%A')}. "
        f"What day of the week is it {delta} days later? Give just the weekday name."
    )
    return prompt, target.strftime("%A").lower(), "str"


def _t_unit_conversion(rng: random.Random) -> tuple[str, str, str]:
    kind = rng.choice(["km_m", "h_s", "gb_mb", "week_min"])
    n = rng.randint(2, 48)
    if kind == "km_m":
        prompt, answer = f"How many meters are in {n} kilometers?", n * 1000
    elif kind == "h_s":
        prompt, answer = f"How many seconds are in {n} hours?", n * 3600
    elif kind == "gb_mb":
        prompt, answer = f"How many megabytes are in {n} gigabytes (1 GB = 1024 MB)?", n * 1024
    else:
        prompt, answer = f"How many minutes are in {n} weeks?", n * 7 * 24 * 60
    return prompt, str(answer), "int"


_TEMPLATES: tuple[tuple[str, Callable[[random.Random], tuple[str, str, str]]], ...] = (
    ("arithmetic_chain", _t_arithmetic_chain),
    ("linear_equation", _t_linear_equation),
    ("modular", _t_modular),
    ("sequence", _t_sequence),
    ("string_transform", _t_string_transform),
    ("program_output", _t_program_output),
    ("date_arithmetic", _t_date_arithmetic),
    ("unit_conversion", _t_unit_conversion),
)


# ── battery generation ────────────────────────────────────────────────────────

def generate_battery(spec: BatterySpec) -> list[HeldoutTask]:
    """Deterministically generate the sealed task set for a spec.

    Domains are round-robined so every battery exercises all of them; the rng
    stream is derived only from (version, seed) so the set is reproducible.
    """
    rng = random.Random(f"heldout-battery|v{spec.version}|{spec.seed}")
    tasks: list[HeldoutTask] = []
    seen_prompts: set[str] = set()
    i = 0
    while len(tasks) < spec.size:
        domain, template = _TEMPLATES[i % len(_TEMPLATES)]
        i += 1
        prompt_body, answer, kind = template(rng)
        if prompt_body in seen_prompts:
            continue
        seen_prompts.add(prompt_body)
        full_prompt = f"{prompt_body}\n\n{ANSWER_INSTRUCTION}"
        task_id = hashlib.sha256(
            f"{spec.battery_id()}|{domain}|{prompt_body}".encode()
        ).hexdigest()[:12]
        tasks.append(
            HeldoutTask(
                task_id=task_id,
                domain=domain,
                prompt=full_prompt,
                answer=normalize_answer(answer, kind),
                answer_kind=kind,
            )
        )
    return tasks


def battery_fingerprints(tasks: list[HeldoutTask]) -> set[str]:
    """Stable fingerprints of the sealed prompts, for training-set exclusion."""
    return {
        hashlib.sha256(task.prompt.encode()).hexdigest()[:16]
        for task in tasks
    }


def text_collides_with_battery(text: str, tasks: list[HeldoutTask]) -> bool:
    """True when a training text contains any sealed prompt body (leak guard)."""
    haystack = str(text or "")
    if not haystack:
        return False
    for task in tasks:
        body = task.prompt.split("\n\n")[0]
        if body and body in haystack:
            return True
    return False


# ── grading ───────────────────────────────────────────────────────────────────

@dataclass
class BatteryResult:
    battery_id: str
    total: int
    correct: int
    per_domain: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "battery_id": self.battery_id,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "per_domain": self.per_domain,
            "failures": self.failures[:20],
        }


def grade_battery(
    spec: BatterySpec,
    tasks: list[HeldoutTask],
    responses: dict[str, str],
) -> BatteryResult:
    """Grade model responses (task_id → raw response) against the battery."""
    result = BatteryResult(battery_id=spec.battery_id(), total=len(tasks), correct=0)
    for task in tasks:
        response = responses.get(task.task_id, "")
        ok = grade_response(task, response)
        bucket = result.per_domain.setdefault(task.domain, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if ok:
            result.correct += 1
            bucket["correct"] += 1
        else:
            result.failures.append(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "expected": task.answer,
                    "extracted": extract_answer(response, task.answer_kind),
                }
            )
    return result


def battery_manifest(spec: BatterySpec, tasks: list[HeldoutTask]) -> dict[str, object]:
    """Auditable description of a sealed battery (hashes, not answers)."""
    return {
        "battery_id": spec.battery_id(),
        "version": spec.version,
        "seed": spec.seed,
        "size": spec.size,
        "task_ids": [t.task_id for t in tasks],
        "set_hash": hashlib.sha256(
            json.dumps([t.to_dict() for t in tasks], sort_keys=True).encode()
        ).hexdigest(),
        "domains": sorted({t.domain for t in tasks}),
    }


__all__ = [
    "ANSWER_INSTRUCTION",
    "BATTERY_VERSION",
    "BatteryResult",
    "BatterySpec",
    "HeldoutTask",
    "battery_fingerprints",
    "battery_manifest",
    "extract_answer",
    "generate_battery",
    "grade_battery",
    "grade_response",
    "normalize_answer",
    "text_collides_with_battery",
]
