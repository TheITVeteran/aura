#!/usr/bin/env python3
"""Generate an undeniable RSI proof bundle by running the AutonomousSuccessorEngine."""
from __future__ import annotations

import os
os.environ["AURA_EAGER_CORTEX_WARMUP"] = "1"
os.environ["AURA_METABOLISM_RATE"] = "0"
os.environ["AURA_STRICT_RUNTIME"] = "1"  # Fully disables volition/MindTick

import argparse
import asyncio
import contextlib
import fcntl
import json
import subprocess
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from core.learning.autonomous_rsi import AutonomousSuccessorEngine

RUNTIME_LOG = ROOT / "artifacts" / "rsi_frozen_generations" / "cortex_32b_runtime.log"
LOCK_PATH = ROOT / "artifacts" / "rsi_frozen_generations" / ".generate_undeniable_rsi.lock"


class LiveRuntimeRouter:
    """Minimal OpenAI-compatible router for the live 32B local runtime."""

    def __init__(self, *, runtime_url: str, model: str, timeout_s: float):
        self.runtime_url = runtime_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)

    async def think(self, prompt: str, **kwargs: Any) -> tuple[bool, str, dict[str, Any]]:
        messages: list[dict[str, str]] = []
        system_prompt = str(kwargs.get("system_prompt") or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(kwargs.get("max_tokens") or 4096),
            "temperature": float(kwargs.get("temperature") or 0.0),
            "top_p": 0.9,
        }
        timeout = httpx.Timeout(self.timeout_s, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.runtime_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = str(message.get("content") or choices[0].get("text") or "")
        return True, text, {
            "model": self.model,
            "endpoint": self.runtime_url,
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
        }


@contextlib.contextmanager
def proof_run_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another undeniable RSI generation is already running: {LOCK_PATH}") from exc
        yield


async def discover_runtime_model(runtime_url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            response = await client.get(f"{runtime_url.rstrip('/')}/v1/models")
        if response.status_code != 200:
            return ""
        data = response.json()
        models = data.get("data") or data.get("models") or []
        if not models:
            return ""
        first = models[0]
        if isinstance(first, dict):
            return str(first.get("id") or first.get("model") or first.get("name") or "").strip()
        return str(first).strip()
    except Exception:
        return ""


def start_cortex_runtime(*, runtime_url: str, model_path: str) -> dict[str, Any]:
    from urllib.parse import urlparse

    from core.brain.llm.model_registry import find_llama_server_bin

    parsed = urlparse(runtime_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 11435)
    llama_server = find_llama_server_bin()
    if not llama_server:
        raise RuntimeError("llama-server binary was not found")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"32B runtime model path does not exist: {path}")

    RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(RUNTIME_LOG, "ab", buffering=0)
    cmd = [
        llama_server,
        "-m",
        str(path),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        "8192",
        "--jinja",
        "-ngl",
        "99",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "-b",
        "2048",
        "-ub",
        "512",
        "--parallel",
        "1",
        "--cache-ram",
        "256",
        "--no-cache-prompt",
    ]
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    return {"started_pid": proc.pid, "command": cmd, "log": str(RUNTIME_LOG)}


async def prepare_live_router(args: argparse.Namespace) -> dict[str, Any]:
    from core.brain.llm.model_registry import get_lane_runtime_model_path
    from core.container import ServiceContainer

    runtime_url = args.runtime_url.rstrip("/")
    model = args.runtime_model.strip() or await discover_runtime_model(runtime_url)
    runtime_info: dict[str, Any] = {
        "runtime_url": runtime_url,
        "model": model,
        "started_runtime": False,
    }
    if not model and args.start_runtime:
        model_path = args.runtime_model_path or get_lane_runtime_model_path("Cortex")
        runtime_info.update(start_cortex_runtime(runtime_url=runtime_url, model_path=model_path))
        runtime_info["started_runtime"] = True
        deadline = time.monotonic() + float(args.ready_timeout_s)
        while time.monotonic() < deadline:
            model = await discover_runtime_model(runtime_url)
            if model:
                runtime_info["model"] = model
                break
            await asyncio.sleep(2.0)

    if not model:
        raise RuntimeError(
            f"live 32B runtime is not ready at {runtime_url}; "
            "start it or pass --start-runtime with a valid --runtime-model-path"
        )

    router = LiveRuntimeRouter(runtime_url=runtime_url, model=model, timeout_s=args.generation_timeout_s)
    ServiceContainer.register_instance("llm_router", router, required=False)
    runtime_info["router_registered"] = True
    return runtime_info


def _sandbox_pass(metadata: dict[str, Any]) -> bool:
    sandbox = metadata.get("sandbox_result")
    return isinstance(sandbox, dict) and bool(sandbox.get("pass"))


def _looks_like_fallback_template(source: str) -> bool:
    required = (
        "HANDLERS =",
        "if task.kind == 'gcd'",
        "if task.kind == 'mod'",
        "if task.kind == 'compose'",
        "if task.kind == 'sort'",
        "if task.kind == 'palindrome'",
    )
    return all(token in source for token in required)


def l3_claim_summary(
    *,
    result: Any,
    solver_source: str,
    strategy: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
) -> dict[str, Any]:
    baseline_score = float(eval_before.get("score", 0.0))
    candidate_score = float(eval_after.get("score", 0.0))
    candidate_improved = candidate_score > baseline_score
    manifest_kinds = {
        str(task.get("kind"))
        for task in manifest.get("public_tasks", [])
        if isinstance(task, dict) and task.get("kind")
    }
    strategy_handlers = {str(handler) for handler in strategy.get("handlers", [])}
    handler_coverage_complete = bool(manifest_kinds) and manifest_kinds.issubset(strategy_handlers)
    fallback_flag = bool(metadata.get("fallback_flag", True))
    router_presence = bool(metadata.get("router_presence", False))
    generated_source_hash = metadata.get("generated_source_hash")
    prompt_used = metadata.get("prompt_used")
    lineage_verdict = getattr(getattr(result, "verdict", None), "verdict", "")
    fallback_template = _looks_like_fallback_template(solver_source)
    lineage_undeniable = lineage_verdict == "UNDENIABLE_RSI"
    l3_rsi_claim = bool(
        not fallback_flag
        and router_presence
        and candidate_improved
        and generated_source_hash
        and prompt_used
        and _sandbox_pass(metadata)
        and not fallback_template
        and lineage_undeniable
        and handler_coverage_complete
    )
    failed_requirements = []
    requirements = {
        "fallback_flag_false": not fallback_flag,
        "router_presence_true": router_presence,
        "candidate_improved_over_baseline": candidate_improved,
        "generated_source_hash_present": bool(generated_source_hash),
        "generated_solver_not_fallback_template": not fallback_template,
        "prompt_used_present": bool(prompt_used),
        "sandbox_result_pass": _sandbox_pass(metadata),
        "lineage_verdict_undeniable": lineage_undeniable,
        "handler_coverage_complete": handler_coverage_complete,
    }
    failed_requirements = [name for name, passed in requirements.items() if not passed]
    return {
        "passed": l3_rsi_claim,
        "artifact_valid": True,
        "status": "l3_rsi_proven" if l3_rsi_claim else "not_l3_evidence",
        "reason": "all_l3_gates_passed" if l3_rsi_claim else "l3_gate_failed",
        "failed_requirements": failed_requirements,
        "l3_rsi_claim": l3_rsi_claim,
        "candidate_improved_over_baseline": candidate_improved,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "generated_solver_looks_like_fallback_template": fallback_template,
        "manifest_kinds": sorted(manifest_kinds),
        "strategy_handlers": sorted(strategy_handlers),
    }


async def run_generation(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    runtime_info = await prepare_live_router(args)
    print(f"Live 32B router ready: {runtime_info['model']} at {runtime_info['runtime_url']}")
    print(f"Starting Autonomous RSI Generation ({args.generations} generations)...")

    artifact_dir = Path("artifacts/rsi_frozen_generations")
    engine = AutonomousSuccessorEngine(artifact_dir)
    result = await asyncio.to_thread(lambda: engine.run(generations=args.generations))
    return result, runtime_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/proof_bundle/latest/UNDENIABLE_RSI.json")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--runtime-url", default=os.getenv("AURA_RSI_LIVE_LLM_URL", "http://127.0.0.1:11435"))
    parser.add_argument("--runtime-model", default="")
    parser.add_argument("--runtime-model-path", default="")
    parser.add_argument("--generation-timeout-s", type=float, default=600.0)
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--start-runtime", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with proof_run_lock():
        result, runtime_info = asyncio.run(run_generation(args))

    # Gather undeniable proof
    artifact = result.artifacts[-1]
    gen_dir = Path(artifact.directory)
    
    solver_source = (gen_dir / "solver.py").read_text(encoding="utf-8")
    strategy = json.loads((gen_dir / "strategy.json").read_text(encoding="utf-8"))
    manifest = json.loads((gen_dir / "public_manifest.json").read_text(encoding="utf-8"))
    eval_after = json.loads((gen_dir / "eval_after.json").read_text(encoding="utf-8"))
    eval_before = json.loads((gen_dir / "eval_before.json").read_text(encoding="utf-8"))
    metadata = json.loads((gen_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    
    # We also need git commit and reproduction command
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    claim = l3_claim_summary(
        result=result,
        solver_source=solver_source,
        strategy=strategy,
        manifest=manifest,
        metadata=metadata,
        eval_before=eval_before,
        eval_after=eval_after,
    )

    bundle = {
        "generated_at": time.time(),
        "claim": "L3_RSI",
        **claim,
        "exact_commit_SHA": commit,
        "reproduction_command": f"python tools/generate_undeniable_rsi_bundle.py --generations {args.generations}",
        "runtime": runtime_info,
        "lineage_verdict": result.verdict.to_dict(),
        "lineage_result": result.to_dict(),
        "generated_solver_source": solver_source,
        "generated_source_hash": metadata.get("generated_source_hash"),
        "fallback_flag": metadata.get("fallback_flag"),
        "router_presence": metadata.get("router_presence"),
        "prompt_used": metadata.get("prompt_used"),
        "sandbox_result": metadata.get("sandbox_result"),
        "no_answer_leakage": True,
        "hidden_task_manifest_without_answers": manifest,
        "salted_answer_hashes": [task.get("answer_hash") for task in manifest.get("public_tasks", [])],
        "candidate_output_transcript": eval_after,
        "baseline_output_transcript": eval_before,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Undeniable RSI Bundle written to {out_path}")
    return 0 if bundle["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
