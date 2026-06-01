from __future__ import annotations
from math import exp, tanh as _tanh
from typing import Dict, Iterable

Vector = Dict[str, float]

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def clamp_signed(x: float) -> float:
    return max(-1.0, min(1.0, x))

def sigmoid(x: float, gain: float = 1.0, bias: float = 0.0) -> float:
    z = max(-60.0, min(60.0, gain * (x - bias)))
    return 1.0 / (1.0 + exp(-z))

def tanh(x: float) -> float:
    return _tanh(max(-20.0, min(20.0, x)))

def l1(v: Vector) -> float:
    return sum(abs(x) for x in v.values())

def l2(v: Vector) -> float:
    return sum(x * x for x in v.values()) ** 0.5

def add(a: Vector, b: Vector) -> Vector:
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in keys}

def sub(a: Vector, b: Vector) -> Vector:
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in keys}

def mul(a: Vector, scalar: float) -> Vector:
    return {k: v * scalar for k, v in a.items()}

def mix(old: Vector, new: Vector, rate: float) -> Vector:
    keys = set(old) | set(new)
    return {k: old.get(k, 0.0) * (1 - rate) + new.get(k, 0.0) * rate for k in keys}

def bound01(v: Vector) -> Vector:
    return {k: clamp(x, 0.0, 1.0) for k, x in v.items()}

def bound_signed(v: Vector) -> Vector:
    return {k: clamp_signed(x) for k, x in v.items()}

def weighted_error(predicted: Vector, observed: Vector, precision: Vector) -> Vector:
    keys = set(predicted) | set(observed) | set(precision)
    return {
        k: (observed.get(k, 0.0) - predicted.get(k, 0.0)) * precision.get(k, 1.0)
        for k in keys
    }

def normalize_sum(v: Vector) -> Vector:
    total = sum(max(0.0, x) for x in v.values())
    if total <= 1e-9:
        n = len(v) or 1
        return {k: 1.0 / n for k in v}
    return {k: max(0.0, x) / total for k, x in v.items()}
