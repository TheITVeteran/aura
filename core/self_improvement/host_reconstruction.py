"""core/self_improvement/host_reconstruction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reverse-engineer a REAL host binary from behavior only — the shared engine
behind both the user-facing skill ("reverse engineer base64 and prove it") and
the proof harness. Lawful, behavior-only: observe the real program's I/O, read
its man page, reconstruct a runnable equivalent via the model, then verify it
against HELD-OUT real outputs the model never saw.

No source is read; no decompilation. The verification uses the general
reconstruction sandbox, so realistic programs — not just toys — can be checked.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.HostReconstruction")


@dataclass
class HostBinaryTarget:
    name: str
    binary: str
    man_topic: str
    argv: Callable[[str], list[str]]
    fn_name: str = "reconstructed"
    case_key: str = "text"
    train_inputs: list[str] = field(default_factory=list)
    held_out_inputs: list[str] = field(default_factory=list)
    # Focused behavior description used INSTEAD of the man page when the man
    # page is huge/misleading (e.g. jq documents a whole language, but we only
    # exercise one invocation). The observed examples remain the ground truth.
    behavior_hint: str = ""


# Curated real targets whose observable behavior is reconstructable and whose
# use is lawful (user-owned host utilities, behavior-only observation).
KNOWN_TARGETS: dict[str, HostBinaryTarget] = {
    "base64": HostBinaryTarget(
        name="base64", binary="base64", man_topic="base64", argv=lambda _p: [],
        train_inputs=["hello", "Aura", "a", "The quick brown fox."],
        held_out_inputs=["reverse-engineered", "1234567890", "unit test", "Zenith"],
    ),
    "rev": HostBinaryTarget(
        name="rev", binary="rev", man_topic="rev", argv=lambda _p: [],
        train_inputs=["hello", "abc", "racecar", "Aura"],
        held_out_inputs=["reverse", "level", "Zenith", "9876"],
    ),
    "md5": HostBinaryTarget(
        name="md5", binary="md5", man_topic="md5", argv=lambda _p: ["-q"],
        train_inputs=["hello", "Aura", "abc", "12345"],
        held_out_inputs=["reverse-engineered", "Zenith", "held-out", "67890"],
    ),
}

KNOWN_TARGETS["jq"] = HostBinaryTarget(
    name="jq",
    binary="jq",
    man_topic="jq",
    argv=lambda _p: ["."],  # the identity filter: pretty-print JSON
    train_inputs=['{"b":2,"a":1}', '[1,2,3]', '{"name":"Aura","n":42}', '{"nested":{"k":[true,false,null]}}'],
    held_out_inputs=['{"list":[{"id":1},{"id":2}]}', '{"a":{"b":{"c":1}}}', '["x","y","z"]', '{"flag":true,"count":10}'],
    behavior_hint=(
        "This is `jq '.'` — the identity filter. It reads one JSON value from stdin and "
        "pretty-prints it to stdout: 2-space indentation, object keys kept in their INPUT "
        "order (not sorted), non-ASCII kept as UTF-8 (not \\u-escaped), and a trailing "
        "newline. Reconstruct exactly that formatting."
    ),
)

_ALIASES = {
    "base64": "base64", "b64": "base64", "the base64 tool": "base64",
    "rev": "rev", "reverse": "rev",
    "md5": "md5", "md5sum": "md5", "the md5 tool": "md5",
    "jq": "jq", "jq .": "jq", "the jq tool": "jq", "jq app": "jq",
}


def resolve_target(name_or_label: str) -> HostBinaryTarget | None:
    key = str(name_or_label or "").strip().lower()
    if key in KNOWN_TARGETS:
        return KNOWN_TARGETS[key]
    for alias, canonical in _ALIASES.items():
        if alias in key:
            return KNOWN_TARGETS[canonical]
    return None


def observe_binary(target: HostBinaryTarget, payload: str) -> str:
    binary = shutil.which(target.binary)
    if not binary:
        raise FileNotFoundError(f"real binary not found on host: {target.binary}")
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    completed = get_subprocess_gateway().run(
        [binary, *target.argv(payload)],
        input=payload,
        capture_output=True,
        timeout=10,
        check=False,
        source="tool_execution:host_reconstruction.observe_binary",
        accelerator_capability="none",
    )
    return completed.stdout


def read_man(topic: str, limit: int = 60) -> str:
    man = shutil.which("man")
    if not man:
        return f"reconstruct the observable behavior of `{topic}`."
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        out = get_subprocess_gateway().run(
            [man, topic],
            capture_output=True,
            timeout=10,
            check=False,
            env={"MANPAGER": "cat", "PAGER": "cat", "MANWIDTH": "80", "PATH": "/usr/bin:/bin"},
            source="tool_execution:host_reconstruction.read_man",
            accelerator_capability="none",
        )
        text = out.stdout.replace("\x08", "")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[:limit])[:4000]
    except (OSError, subprocess.SubprocessError):
        return f"reconstruct the observable behavior of `{topic}`."


def held_out_observations(target: HostBinaryTarget) -> list[dict[str, Any]]:
    return [
        {"input": {target.case_key: p}, "expected": observe_binary(target, p)}
        for p in target.held_out_inputs
    ]


def train_observations(target: HostBinaryTarget) -> list[dict[str, Any]]:
    return [
        {"input": {target.case_key: p}, "output": observe_binary(target, p)}
        for p in target.train_inputs
    ]


def spec_docs(target: HostBinaryTarget) -> list[str]:
    docs = [
        f"Reconstruct the observable stdout behavior of the `{target.name}` command.",
        f"The function receives one dict argument with key '{target.case_key}' (the stdin payload) "
        f"and must return the EXACT stdout the real program produces, including trailing newlines.",
    ]
    if target.behavior_hint:
        docs.append("Behavior: " + target.behavior_hint)
    else:
        docs.append("Specification (the program's own man page — observable, not source):")
        docs.append(read_man(target.man_topic))
    return docs


async def reverse_engineer_host_binary(
    engine: Any,
    target: HostBinaryTarget,
    *,
    authorization: str = "host_observation",
) -> dict[str, Any]:
    """Observe → spec → reconstruct (via the model) → differential-verify against
    held-out real outputs. Returns an honest, epistemically-labeled report."""
    held = held_out_observations(target)
    outcome = await engine.reconstruct_executable_via_cognition(
        target=f"real:{target.name}",
        spec_docs=spec_docs(target),
        train_examples=train_observations(target),
        held_out=held,
        fn_name=target.fn_name,
        authorization=authorization,
        objective=f"clean-room reconstruction of host command `{target.name}` from behavior",
        sandbox_profile="general",
    )
    return {
        "target": f"real:{target.name}",
        "binary": shutil.which(target.binary) or target.binary,
        "policy": "behavior-only: observed I/O + man page; NO source, NO decompilation",
        "status": outcome.get("status"),
        "held_out_passed": outcome.get("held_out_passed", 0),
        "held_out_total": outcome.get("held_out_total", len(held)),
        "equivalence": outcome.get("equivalence", 0.0),
        "failures": outcome.get("failures", []),
        "reconstructed_code": outcome.get("code", ""),
        "reason": outcome.get("reason", ""),
    }


__all__ = [
    "HostBinaryTarget",
    "KNOWN_TARGETS",
    "resolve_target",
    "observe_binary",
    "read_man",
    "held_out_observations",
    "train_observations",
    "spec_docs",
    "reverse_engineer_host_binary",
]
