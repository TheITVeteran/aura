"""The no-go: what it costs to move an ambient constant, computed.

This module exists to be RIGHT rather than encouraging. Its whole job is to
return the number that says a request is impossible, and to refuse to return a
finite number when the request is not merely expensive but unreachable.

The physics
-----------
A canonical light scalar ``phi`` shifts log-constants ``l_A = ln lambda_A``
with coupling ``B_{Aa} = Mbar_P * d l_A / d phi^a``. Holding a spherical
region of radius ``R`` at an offset ``Delta phi`` costs, at minimum, the
gradient energy of the profile that sustains it::

    E_out = 2 pi R (Delta phi)^2
    E_min = 2 pi R (Delta phi)^2 * F(mR),   F(x) = 1 + x + x^2 / 3

and the same region's black-hole energy is::

    E_BH = R c^4 / 2G = 4 pi Mbar_P^2 R

so the ratio that decides feasibility is::

    E_min / E_BH = (1/2) F(mR) (Delta phi / Mbar_P)^2

For several observables at once, with metric ``G`` on field space::

    C = B G^-1 B^T
    D_min^2 / Mbar_P^2 = dl^T C^+ dl
    E_min / E_BH  >=  (1/2) dl^T C^+ dl

The pseudo-inverse is the trap
------------------------------
``C^+`` is a Moore-Penrose pseudo-inverse, and a pseudo-inverse ANSWERS EVERY
QUESTION. Hand it a target that no combination of fields can produce and it
silently returns the least-squares projection — a finite, plausible, entirely
fictitious energy for a shift that cannot happen at any budget.

So every result here carries the residual ``(I - C C^+) dl``. When that is
nonzero the verdict is UNREACHABLE and the energy figure is reported as what
it actually is: the cost of the reachable part only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Reduced Planck mass in GeV. E_BH = 4 pi Mbar_P^2 R in natural units.
MBAR_PLANCK_GEV = 2.435323e18

#: Anything at or above this fraction of a black hole's energy for the same
#: region is not an engineering problem. The value is not a safety margin —
#: it is the point past which the region collapses instead of holding the
#: field offset you asked for.
BLACK_HOLE_FRACTION_HARD_LIMIT = 1.0

#: Relative tolerance for calling a residual component zero. Scaled by the
#: norm of the target, so it means "this direction is unreachable" rather
#: than "this number is small".
_REACHABILITY_RTOL = 1e-9


def yukawa_form_factor(m_r: float) -> float:
    """``F(mR) = 1 + mR + (mR)^2 / 3`` — the mass penalty on holding an offset.

    A massless field costs only the gradient energy. A massive one must also
    pay for fighting its own potential back to zero outside the region, and
    that penalty grows quadratically in the region size measured in Compton
    wavelengths.
    """
    if not math.isfinite(m_r):
        raise ValueError("m*R must be finite")
    if m_r < 0.0:
        raise ValueError("m*R must be non-negative")
    return 1.0 + m_r + (m_r * m_r) / 3.0


@dataclass(frozen=True)
class SingleFieldCost:
    """What one canonical scalar costs over one region."""

    delta_phi_over_mbar: float
    m_r: float
    form_factor: float
    energy_fraction_of_black_hole: float
    verdict: str
    #: Present only when the caller supplied a coupling and a target shift.
    coupling_b: float | None = None
    delta_log_constant: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lawc05.single_field_cost.v1",
            "delta_phi_over_mbar": self.delta_phi_over_mbar,
            "m_r": self.m_r,
            "form_factor": self.form_factor,
            "energy_fraction_of_black_hole": self.energy_fraction_of_black_hole,
            "verdict": self.verdict,
            "coupling_b": self.coupling_b,
            "delta_log_constant": self.delta_log_constant,
        }


def single_field_cost(
    *,
    delta_phi_over_mbar: float | None = None,
    m_r: float = 0.0,
    delta_log_constant: float | None = None,
    coupling_b: float | None = None,
) -> SingleFieldCost:
    """Cost of holding one region at a field offset, as a fraction of E_BH.

    Give it either the field offset directly, or a target shift in a log
    constant plus the coupling that shift rides on — in which case
    ``Delta phi / Mbar_P = delta_l / B``, which is where the brutal numbers
    come from: a coupling of 1e-6 turns a 1 ppm target into an order-unity
    field excursion.
    """
    if delta_phi_over_mbar is None:
        if delta_log_constant is None or coupling_b is None:
            raise ValueError(
                "supply delta_phi_over_mbar, or both delta_log_constant and coupling_b"
            )
        if coupling_b == 0.0:
            raise ValueError(
                "coupling_b is zero: this observable does not respond to this "
                "field at any energy"
            )
        delta_phi_over_mbar = float(delta_log_constant) / float(coupling_b)

    offset = float(delta_phi_over_mbar)
    if not math.isfinite(offset):
        raise ValueError("delta_phi_over_mbar must be finite")

    form = yukawa_form_factor(float(m_r))
    fraction = 0.5 * form * offset * offset
    return SingleFieldCost(
        delta_phi_over_mbar=offset,
        m_r=float(m_r),
        form_factor=form,
        energy_fraction_of_black_hole=fraction,
        verdict=_verdict_for(fraction),
        coupling_b=None if coupling_b is None else float(coupling_b),
        delta_log_constant=(
            None if delta_log_constant is None else float(delta_log_constant)
        ),
    )


def _verdict_for(fraction: float) -> str:
    if fraction >= BLACK_HOLE_FRACTION_HARD_LIMIT:
        return "COLLAPSES_INSTEAD"
    if fraction >= 1e-3:
        return "GRAVITATIONALLY_PROHIBITIVE"
    if fraction >= 1e-12:
        return "ENERGETICALLY_EXTREME"
    return "ENERGETICALLY_PLAUSIBLE"


@dataclass(frozen=True)
class MultiObservableCost:
    """The cost, and the part of the request that has no cost because it
    cannot be done at all."""

    distance_squared_over_mbar2: float
    energy_fraction_of_black_hole: float
    verdict: str
    reachable: bool
    residual: np.ndarray
    residual_norm: float
    target_norm: float
    reachable_rank: int
    observable_count: int
    field_count: int
    condition_number: float
    #: Field-space offset that achieves the reachable part, in units of Mbar_P.
    phi_solution_over_mbar: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lawc05.multi_observable_cost.v1",
            "distance_squared_over_mbar2": self.distance_squared_over_mbar2,
            "energy_fraction_of_black_hole": self.energy_fraction_of_black_hole,
            "verdict": self.verdict,
            "reachable": self.reachable,
            "residual": self.residual.tolist(),
            "residual_norm": self.residual_norm,
            "target_norm": self.target_norm,
            "reachable_rank": self.reachable_rank,
            "observable_count": self.observable_count,
            "field_count": self.field_count,
            "condition_number": self.condition_number,
            "phi_solution_over_mbar": self.phi_solution_over_mbar.tolist(),
        }


def multi_observable_cost(
    coupling_matrix: Any,
    delta_log_constants: Any,
    *,
    field_metric: Any | None = None,
    m_r: float = 0.0,
) -> MultiObservableCost:
    """``D_min^2 / Mbar_P^2 = dl^T C^+ dl`` with the reachability check kept.

    ``coupling_matrix`` is ``B`` with shape (observables, fields).
    ``field_metric`` is ``G`` on field space, shape (fields, fields); the
    identity when omitted. ``G`` must be symmetric positive definite — a
    non-invertible metric is a degenerate field space, not a cheaper one.
    """
    b_matrix = np.asarray(coupling_matrix, dtype=float)
    if b_matrix.ndim != 2:
        raise ValueError("coupling_matrix must be 2-D (observables x fields)")
    n_obs, n_fields = b_matrix.shape
    if n_obs == 0 or n_fields == 0:
        raise ValueError("coupling_matrix must be non-empty")
    if not np.all(np.isfinite(b_matrix)):
        raise ValueError("coupling_matrix must be finite")

    target = np.asarray(delta_log_constants, dtype=float).reshape(-1)
    if target.shape[0] != n_obs:
        raise ValueError(
            f"delta_log_constants has {target.shape[0]} entries, "
            f"coupling_matrix has {n_obs} observables"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("delta_log_constants must be finite")

    if field_metric is None:
        g_matrix = np.eye(n_fields)
    else:
        g_matrix = np.asarray(field_metric, dtype=float)
        if g_matrix.shape != (n_fields, n_fields):
            raise ValueError("field_metric must be (fields x fields)")
        if not np.allclose(g_matrix, g_matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("field_metric must be symmetric")
        eigenvalues = np.linalg.eigvalsh(g_matrix)
        if float(np.min(eigenvalues)) <= 0.0:
            raise ValueError(
                "field_metric must be positive definite; a degenerate metric is a "
                "degenerate field space, not a cheaper one"
            )

    g_inverse = np.linalg.inv(g_matrix)
    c_matrix = b_matrix @ g_inverse @ b_matrix.T
    c_pinv = np.linalg.pinv(c_matrix)

    # The projector onto what C can actually produce. Everything outside it is
    # unreachable at ANY energy, and the pseudo-inverse will not say so.
    projector = c_matrix @ c_pinv
    residual = target - projector @ target
    target_norm = float(np.linalg.norm(target))
    residual_norm = float(np.linalg.norm(residual))
    tolerance = _REACHABILITY_RTOL * max(target_norm, 1.0)
    reachable = residual_norm <= tolerance

    distance_squared = float(target @ c_pinv @ target)
    # Numerical noise can make a quadratic form slightly negative near zero.
    distance_squared = max(0.0, distance_squared)

    form = yukawa_form_factor(float(m_r))
    fraction = 0.5 * form * distance_squared

    singular_values = np.linalg.svd(c_matrix, compute_uv=False)
    positive = singular_values[singular_values > 0.0]
    condition = (
        float(positive[0] / positive[-1]) if positive.size else float("inf")
    )

    phi_solution = g_inverse @ b_matrix.T @ c_pinv @ target

    if not reachable:
        verdict = "UNREACHABLE_AT_ANY_ENERGY"
    else:
        verdict = _verdict_for(fraction)

    return MultiObservableCost(
        distance_squared_over_mbar2=distance_squared,
        energy_fraction_of_black_hole=fraction,
        verdict=verdict,
        reachable=reachable,
        residual=residual,
        residual_norm=residual_norm,
        target_norm=target_norm,
        reachable_rank=int(np.linalg.matrix_rank(c_matrix)),
        observable_count=n_obs,
        field_count=n_fields,
        condition_number=condition,
        phi_solution_over_mbar=phi_solution,
    )


def black_hole_energy_gev(radius_gev_inverse: float) -> float:
    """``E_BH = 4 pi Mbar_P^2 R`` in natural units (GeV, with R in 1/GeV)."""
    if radius_gev_inverse <= 0.0:
        raise ValueError("radius must be positive")
    return 4.0 * math.pi * MBAR_PLANCK_GEV**2 * float(radius_gev_inverse)


def microscope_style_report(
    *,
    coupling_b: float = 1e-6,
    delta_log_constant: float = 1e-6,
    m_r: float = 0.0,
) -> dict[str, Any]:
    """The worked example, computed rather than quoted.

    MICROSCOPE-scale couplings put ``B ~ 1e-6`` on a typical scalar. Asking
    for a part-per-million shift in alpha then needs an order-unity excursion
    in units of the reduced Planck mass, and half a black hole to hold it.
    """
    cost = single_field_cost(
        delta_log_constant=delta_log_constant,
        coupling_b=coupling_b,
        m_r=m_r,
    )
    return {
        "schema": "aura.lawc05.microscope_report.v1",
        "inputs": {
            "coupling_b": coupling_b,
            "delta_log_constant": delta_log_constant,
            "m_r": m_r,
        },
        "delta_phi_over_mbar": cost.delta_phi_over_mbar,
        "energy_fraction_of_black_hole": cost.energy_fraction_of_black_hole,
        "verdict": cost.verdict,
        "conclusion": (
            "No pure-software command changes ambient constants. Holding the "
            "offset costs a finite fraction of the black-hole energy for the "
            "same region, and at order unity the region collapses instead."
        ),
    }


__all__ = [
    "BLACK_HOLE_FRACTION_HARD_LIMIT",
    "MBAR_PLANCK_GEV",
    "MultiObservableCost",
    "SingleFieldCost",
    "black_hole_energy_gev",
    "microscope_style_report",
    "multi_observable_cost",
    "single_field_cost",
    "yukawa_form_factor",
]
