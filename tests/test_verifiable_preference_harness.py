"""Tests for the verifiable-reward preference harness (RLVR/DPO data engine).

Contract: a problem with a verified-correct AND a verified-wrong candidate yields a
preference pair (chosen=correct, rejected=wrong); soundness is absolute — a pair is
emitted ONLY when the verifier actually CHECKED both sides; pairs dedup and persist;
and the export format is what a DPO trainer consumes.
"""
from __future__ import annotations

from core.learning.verifiable_preference_harness import (
    Attempt,
    VerifiablePreferenceHarness,
    get_verifiable_preference_harness,
)


def _h(tmp_path, name="p.jsonl"):
    return VerifiablePreferenceHarness(store_path=tmp_path / name)


def test_verified_vs_refuted_makes_a_preference_pair(tmp_path):
    h = _h(tmp_path)
    n = h.ingest(
        "compute 17 mod 5",
        [
            Attempt(candidate="17 mod 5 = 2", verified=True, checked=True, confidence=0.9),
            Attempt(candidate="17 mod 5 = 3", verified=False, checked=True, confidence=0.1),
        ],
        domain="math",
    )
    assert n == 1
    rows = h.export_dpo_rows()
    assert len(rows) == 1
    assert rows[0]["chosen"] == "17 mod 5 = 2"
    assert rows[0]["rejected"] == "17 mod 5 = 3"
    assert rows[0]["prompt"] == "compute 17 mod 5"


def test_no_pair_without_contrast(tmp_path):
    h = _h(tmp_path)
    # Only correct candidates → no negative half → no preference signal.
    n = h.ingest("q", [
        Attempt("a", verified=True, checked=True),
        Attempt("b", verified=True, checked=True),
    ])
    assert n == 0


def test_soundness_unchecked_attempts_never_produce_pairs(tmp_path):
    """The cardinal rule: a vacuous pass (checked=False) is not a verified signal."""
    h = _h(tmp_path)
    n = h.ingest("q", [
        Attempt("looks right", verified=True, checked=False),   # not actually checked
        Attempt("looks wrong", verified=False, checked=False),  # not actually checked
    ])
    assert n == 0
    assert h.export_dpo_rows() == []


def test_pairs_dedup(tmp_path):
    h = _h(tmp_path)
    attempts = [
        Attempt("good", verified=True, checked=True),
        Attempt("bad", verified=False, checked=True),
    ]
    assert h.ingest("same q", attempts) == 1
    assert h.ingest("same q", attempts) == 0  # identical pair not re-emitted


def test_persistence_uses_narrow_file_write_governance(tmp_path, monkeypatch):
    observed_domains = []

    class Gateway:
        def append_text(self, path, text, **_kwargs):
            from core.governance_context import get_active_governance

            token = get_active_governance()
            observed_domains.append(token.domain if token else "")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)

    import core.runtime.file_write_gateway as gateway_module

    monkeypatch.setattr(gateway_module, "get_file_write_gateway", lambda: Gateway())
    h = _h(tmp_path)
    assert h.ingest(
        "q",
        [
            Attempt("right", verified=True, checked=True),
            Attempt("wrong", verified=False, checked=True),
        ],
    ) == 1
    assert observed_domains == ["file_write"]


def test_failed_governed_persistence_remains_retryable(tmp_path, monkeypatch):
    class FailingGateway:
        def append_text(self, *_args, **_kwargs):
            raise RuntimeError("durability unavailable")

    import core.runtime.file_write_gateway as gateway_module

    monkeypatch.setattr(
        gateway_module, "get_file_write_gateway", lambda: FailingGateway()
    )
    h = _h(tmp_path)
    attempts = [
        Attempt("right", verified=True, checked=True),
        Attempt("wrong", verified=False, checked=True),
    ]
    assert h.ingest("retry me", attempts) == 0
    assert not (tmp_path / "p.jsonl").exists()

    class WorkingGateway:
        def append_text(self, path, text, **_kwargs):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)

    monkeypatch.setattr(
        gateway_module,
        "get_file_write_gateway",
        lambda: WorkingGateway(),
    )
    assert h.ingest("retry me", attempts) == 1
    assert (tmp_path / "p.jsonl").exists()


def test_persists_across_instances(tmp_path):
    path = tmp_path / "p.jsonl"
    h = VerifiablePreferenceHarness(store_path=path)
    h.ingest("q", [
        Attempt("right", verified=True, checked=True),
        Attempt("wrong", verified=False, checked=True),
    ])
    assert path.exists()
    reborn = VerifiablePreferenceHarness(store_path=path)
    assert len(reborn.export_dpo_rows()) == 1
    # The reloaded harness knows the pair already exists (dedup survives restart).
    assert reborn.ingest("q", [
        Attempt("right", verified=True, checked=True),
        Attempt("wrong", verified=False, checked=True),
    ]) == 0


def test_singleton_and_registration():
    eng = get_verifiable_preference_harness()
    assert get_verifiable_preference_harness() is eng
    from core.container import ServiceContainer

    assert ServiceContainer.has(VerifiablePreferenceHarness.SERVICE_NAME)
