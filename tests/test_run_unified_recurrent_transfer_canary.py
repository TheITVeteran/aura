"""Runner contracts for the source-bound broad transfer canary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS
from tools import run_unified_recurrent_transfer_canary as runner


class _Controller:
    def __init__(self, digest: str) -> None:
        self.digest = digest

    def parameter_sha256(self) -> str:
        return self.digest


class _Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 2, 3]

    def decode(self, tokens, **_kwargs):
        return '{"choice":1}' if tokens else ""


class _Task:
    def __init__(self, domain: str) -> None:
        self.task_id = f"task-{domain}"
        self.domain = domain
        self.public = SimpleNamespace(prompt=f"prompt-{domain}")

    def score(self, _text: str):
        return SimpleNamespace(
            correct=True,
            parsed=True,
            to_dict=lambda: {"correct": True, "parsed": True},
        )


def test_runner_forwards_bootstrap_transport_to_layout_and_loader(
    tmp_path,
    monkeypatch,
) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    observed: list[tuple[str, object]] = []

    monkeypatch.setattr(runner, "_ensure_private_directory", lambda path: path)
    monkeypatch.setattr(runner, "_issuer", lambda *_args: object())
    monkeypatch.setattr(runner, "_tasks", lambda _issuer: ())

    def layout(_campaign_dir, *, bootstrap_output_dir=None):
        observed.append(("layout", bootstrap_output_dir))
        return SimpleNamespace(checkpoint_dir=tmp_path)

    def context(_campaign_dir, **kwargs):
        observed.append(("context", kwargs["bootstrap_output_dir"]))
        raise RuntimeError("captured canary context")

    monkeypatch.setattr(runner, "_evaluation_layout", layout)
    monkeypatch.setattr(
        runner,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(receipt={}),
    )
    monkeypatch.setattr(runner, "unified_evaluation_context", context)

    with pytest.raises(RuntimeError, match="captured canary context"):
        runner.run_canary(
            campaign_dir=tmp_path,
            stem="checkpoint_latest",
            output_dir=tmp_path / "output",
            seeds=(3,),
            difficulty=1,
            recurrence_depth=4,
            max_tokens=16,
            source_commit="a" * 40,
            memory_limit_gb=4.0,
            cache_limit_gb=1.0,
            wired_limit_gb=8.0,
            bootstrap_output_dir=bootstrap,
        )

    assert observed == [("layout", bootstrap), ("context", bootstrap)]


def test_runner_executes_parent_treatment_and_real_action_lesion(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _Controller("1" * 64)
    treatment = _Controller("2" * 64)
    tasks = [_Task(domain) for domain in FRONTIER_DOMAINS]
    plan = {
        "plan_sha256": "3" * 64,
        "parent_controller_sha256": parent.parameter_sha256(),
        "treatment_controller_sha256": treatment.parameter_sha256(),
        "max_tokens": 32,
        "recurrence_depth": 4,
        "tasks": [{"task_id": task.task_id} for task in tasks],
    }
    typed_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(runner, "_contract_complete", lambda *_args: True)
    monkeypatch.setattr(
        runner,
        "decode_base_greedy_tokens",
        lambda *_args, **_kwargs: ((7,), True, 1),
    )

    def typed(_model, controller, _spec, _tokens, **kwargs):
        typed_calls.append(
            (controller.parameter_sha256(), kwargs["typed_action_lesion"])
        )
        return (8,), True, 2

    monkeypatch.setattr(runner, "decode_typed_process_tokens", typed)
    monkeypatch.setattr(
        runner,
        "seal_transfer_canary_result",
        lambda _plan, candidates: {"supported": True, "rows": len(candidates)},
    )
    candidates, verdict = runner._run_loaded(  # noqa: SLF001
        output_dir=tmp_path,
        plan=plan,
        tasks=tasks,
        model=object(),
        tokenizer=_Tokenizer(),
        spec=object(),
        parent_controller=parent,
        treatment_controller=treatment,
    )

    assert len(candidates) == len(FRONTIER_DOMAINS) * 4
    assert verdict == {"supported": True, "rows": len(candidates)}
    assert typed_calls.count((parent.parameter_sha256(), False)) == len(FRONTIER_DOMAINS)
    assert typed_calls.count((treatment.parameter_sha256(), False)) == len(FRONTIER_DOMAINS)
    assert typed_calls.count((treatment.parameter_sha256(), True)) == len(FRONTIER_DOMAINS)
