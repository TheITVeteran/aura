"""A malformed or oversized receipt is a contract failure, not a crash.

CP126 94593618: int() on a worker-supplied field sat outside any protective
conversion, and deep_reason called the whole validator with no envelope — so
a malformed receipt escaped as an exception instead of the promised ok=false
with a reason. A validator that can be crashed by the thing it validates is
not a validator.

CP126 09f2fbcf: decision lists, trails, timings and identity blocks had no
count, size or depth budget before the facade copied and iterated them, so
the validator was a denial-of-service surface for the process it protects.
"""
from __future__ import annotations

import pytest

from core.brain import latent_cortex_service as mod
from core.brain.latent_cortex_service import LatentCortexService


@pytest.fixture()
def service():
    return LatentCortexService()


def _errors(service, receipt):
    return service._safe_receipt_contract_errors(
        receipt, {}, None, None, ..., "general"
    )


# --- 94593618: fail honest, never raise ----------------------------------


@pytest.mark.parametrize(
    "evaluations", ["lots", [1, 2], {"n": 1}, None, object(), float("nan")]
)
def test_a_malformed_evaluation_count_does_not_raise(service, evaluations):
    errors = _errors(service, {"verifier_guidance": {"evaluations": evaluations}})

    assert isinstance(errors, list)
    assert not any("validator_error" in e for e in errors)


def test_a_receipt_that_explodes_on_access_returns_a_contract_error(service):
    class Hostile(dict):
        def get(self, *args, **kwargs):
            raise RecursionError("deep")

    errors = _errors(service, Hostile())

    assert any(e.startswith("receipt_contract_validator_error") for e in errors)


@pytest.mark.parametrize(
    "exc", [TypeError("t"), ValueError("v"), KeyError("k"), ArithmeticError("a")]
)
def test_every_conversion_failure_becomes_a_contract_error(service, exc, monkeypatch):
    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(service, "_receipt_contract_errors", _boom)

    errors = service._safe_receipt_contract_errors({}, {})

    assert errors and errors[0].startswith("receipt_contract_validator_error")


def test_a_non_mapping_receipt_is_still_a_contract_error(service):
    assert "receipt_not_mapping" in _errors(service, "not a receipt")


def test_the_facade_uses_the_protected_wrapper():
    import inspect

    source = inspect.getsource(LatentCortexService.deep_reason)
    assert "_safe_receipt_contract_errors(" in source


# --- 09f2fbcf: bounded before it is walked -------------------------------


def test_an_oversized_receipt_is_refused(service):
    huge = {"trail": [{"x": i} for i in range(mod._MAX_RECEIPT_ITEMS + 10)]}

    assert "receipt_size_exceeds_budget" in _errors(service, huge)


def test_a_deeply_nested_receipt_is_refused(service):
    deep = current = {}
    for _ in range(mod._MAX_RECEIPT_DEPTH + 10):
        current["n"] = {}
        current = current["n"]

    assert "receipt_nesting_exceeds_budget" in _errors(service, deep)


def test_a_receipt_with_too_many_keys_is_refused(service):
    wide = {f"k{i}": i for i in range(mod._MAX_RECEIPT_KEYS + 10)}

    assert "receipt_key_count_exceeds_budget" in _errors(service, wide)


def test_the_size_check_runs_before_the_validator(service, monkeypatch):
    """A bound applied after the walk protects nothing."""
    called = []
    monkeypatch.setattr(
        service, "_receipt_contract_errors",
        lambda *a, **k: called.append(1) or [],
    )
    huge = {"trail": [{"x": i} for i in range(mod._MAX_RECEIPT_ITEMS + 10)]}

    service._safe_receipt_contract_errors(huge, {})

    assert called == []


def test_an_ordinary_receipt_is_not_refused_for_size(service):
    ordinary = {"episode_id": "x", "budget": {"spent_layer_apps": 10}}

    errors = _errors(service, ordinary)

    assert not any("exceeds_budget" in e for e in errors)


def test_the_bounds_are_generous_enough_for_real_evidence():
    assert mod._MAX_RECEIPT_ITEMS >= 100_000
    assert mod._MAX_RECEIPT_DEPTH >= 16
    assert mod._MAX_RECEIPT_KEYS >= 256
