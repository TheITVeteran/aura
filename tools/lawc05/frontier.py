"""Where the no-go binds, and where it does not.

The energy bound in :mod:`nogo` is correct — the form factor is verified to
machine precision against a direct solution of the radial problem. What it is
NOT is the binding constraint over most of parameter space, and reading it as
though it were leads to the wrong conclusion in the one regime that matters.

The two bounds
--------------
**Bound A (energy).** Holding a region at an offset costs
``E = 2 pi R (Delta phi)^2 F(mR)``. Evaluated at a ppm target with a
MICROSCOPE-scale coupling this is half a black hole, which is the headline
result and is right.

**Bound B (sourcing).** You cannot deposit that energy as a coherent field
offset by wishing. A scalar is sourced by matter it couples to, and the static
field deep inside a source much larger than the field's range is

    phi = B rho / (m^2 Mbar_P)   =>   delta_l = B phi / Mbar_P
                                              = B^2 rho / (m^2 Mbar_P^2)

which depends on the coupling SQUARED and on the density of real material.

Which one binds
---------------
At a ppm target, Bound A binds and the answer is "no, ever".

At a 1e-18 target in a sub-millimetre region — the regime an optical clock can
actually resolve — Bound A gives about thirty joules, which is nothing. Bound B
then asks for a coupling of ``alpha ~ 1e11`` at 100 micron range. That is at
the EDGE of what short-range gravity experiments have excluded, not far above
it, and it is an EMPIRICAL bound rather than a theorem.

So the honest statement is narrower and more useful than the slogan:

    Changing an ambient constant by a humanly noticeable amount is forbidden
    by energy. Changing one by a metrologically detectable amount is not
    forbidden by energy at all — it is bounded by a coupling limit that
    experiments are still tightening, and that limit is the real frontier.

Everything here is classical, static, tree-level, with no backreaction, and
the coupling envelope is an order-of-magnitude summary of the short-range
gravity literature that a serious user should REPLACE with their own
compilation. It is labelled as such at the point of use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .nogo import MBAR_PLANCK_GEV, yukawa_form_factor

# ── Units. Natural units (GeV) inside, SI at the boundary. ────────────────
GEV_INVERSE_PER_METRE = 5.067731e15
JOULES_PER_GEV = 1.602177e-10
GEV_PER_KILOGRAM = 5.609589e26

#: Densities that exist as bulk laboratory material.
DENSITY_TUNGSTEN = 19300.0
DENSITY_GOLD = 19300.0
DENSITY_OSMIUM = 22590.0
DENSITY_LEAD = 11340.0


def metres_to_inverse_gev(x_m: float) -> float:
    return float(x_m) * GEV_INVERSE_PER_METRE


def inverse_gev_to_metres(x_gev_inv: float) -> float:
    return float(x_gev_inv) / GEV_INVERSE_PER_METRE


def density_to_natural(kg_per_m3: float) -> float:
    """kg/m^3 to GeV^4."""
    return float(kg_per_m3) * GEV_PER_KILOGRAM / (GEV_INVERSE_PER_METRE**3)


def range_to_mass_gev(range_m: float) -> float:
    """A Yukawa range in metres to the mass in GeV that produces it."""
    if range_m <= 0.0:
        raise ValueError("range must be positive")
    return 1.0 / metres_to_inverse_gev(range_m)


def mass_gev_to_range(mass_gev: float) -> float:
    if mass_gev <= 0.0:
        return math.inf
    return inverse_gev_to_metres(1.0 / mass_gev)


# ── Bound A: energy to hold an offset ─────────────────────────────────────
def energy_to_hold_joules(
    *,
    delta_log_constant: float,
    coupling_b: float,
    region_radius_m: float,
    field_mass_gev: float = 0.0,
) -> float:
    """``E = 2 pi R (Delta phi)^2 F(mR)`` in joules.

    This is a NECESSARY condition, not a sufficient one. Satisfying it says
    only that the field energy is affordable; see :func:`shift_from_source`
    for whether anything can actually put the field there.
    """
    if coupling_b == 0.0:
        raise ValueError("coupling_b is zero: the observable does not respond")
    if region_radius_m <= 0.0:
        raise ValueError("region_radius_m must be positive")

    delta_phi = (float(delta_log_constant) / float(coupling_b)) * MBAR_PLANCK_GEV
    radius = metres_to_inverse_gev(region_radius_m)
    form = yukawa_form_factor(float(field_mass_gev) * radius)
    energy_gev = 2.0 * math.pi * radius * delta_phi * delta_phi * form
    return energy_gev * JOULES_PER_GEV


# ── Bound B: what a real source produces ──────────────────────────────────
def shift_from_source(
    *,
    coupling_b: float,
    density_kg_per_m3: float,
    field_mass_gev: float,
) -> float:
    """``delta_l = B^2 rho / (m^2 Mbar_P^2)`` — the deep-interior static field.

    Valid when the source is much larger than the field's range, and the
    sensor sits inside it. A sensor outside the source sees less, so this is
    an upper bound on what that configuration delivers.

    The coupling enters SQUARED: once to source the field, once to convert the
    field into a shift of the observable. That is why a weak coupling is
    doubly punishing here and only singly punishing in the energy bound.
    """
    if field_mass_gev <= 0.0:
        raise ValueError(
            "a massless field has infinite range; use a finite mass, or the "
            "source integral does not converge to a local value"
        )
    rho = density_to_natural(density_kg_per_m3)
    return (
        float(coupling_b) ** 2
        * rho
        / (float(field_mass_gev) ** 2 * MBAR_PLANCK_GEV**2)
    )


def coupling_needed_for_shift(
    *,
    delta_log_constant: float,
    density_kg_per_m3: float,
    field_mass_gev: float,
) -> float:
    """The coupling ``B`` a source of this density must have to deliver the shift."""
    unit = shift_from_source(
        coupling_b=1.0,
        density_kg_per_m3=density_kg_per_m3,
        field_mass_gev=field_mass_gev,
    )
    if unit <= 0.0:
        return math.inf
    return math.sqrt(float(delta_log_constant) / unit)


# ── The empirical coupling envelope ───────────────────────────────────────
def default_alpha_envelope(range_m: float) -> float:
    """Order-of-magnitude ceiling on ``alpha`` (Yukawa strength / gravity).

    REPLACE THIS for any serious use. It is a coarse reading of the
    short-range gravity and Casimir literature, good to an order of magnitude
    at best, and it is the single number that decides every conclusion below.
    It is a function so that a caller can substitute their own compilation
    without touching anything else.

    Shape: essentially unconstrained below a micron, tightening steeply
    through the tens-of-microns range where Casimir-background subtraction
    becomes possible, and reaching the sub-1e-5 regime at laboratory and
    planetary distances where equivalence-principle tests dominate.
    """
    r = float(range_m)
    if r <= 0.0:
        raise ValueError("range must be positive")
    knots_m = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e3, 1e7]
    knots_alpha = [1e18, 1e15, 1e11, 1e6, 1e2, 1e0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-9]
    log_r = math.log10(r)
    xs = [math.log10(k) for k in knots_m]
    ys = [math.log10(a) for a in knots_alpha]
    if log_r <= xs[0]:
        return 10.0**ys[0]
    if log_r >= xs[-1]:
        return 10.0**ys[-1]
    return 10.0 ** float(np.interp(log_r, xs, ys))


def max_coupling_b(range_m: float, envelope: Callable[[float], float] | None = None) -> float:
    """``B_max = sqrt(alpha_max)`` at this range."""
    alpha = (envelope or default_alpha_envelope)(range_m)
    return math.sqrt(max(alpha, 0.0))


# ── Which bound binds ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class FrontierVerdict:
    """A target, and the reason it is or is not reachable."""

    delta_log_constant: float
    region_radius_m: float
    range_m: float
    field_mass_gev: float
    density_kg_per_m3: float
    energy_joules: float
    coupling_used: float
    coupling_needed: float
    coupling_allowed: float
    coupling_headroom_decades: float
    binding_constraint: str
    reachable: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lawc05.frontier_verdict.v1",
            "delta_log_constant": self.delta_log_constant,
            "region_radius_m": self.region_radius_m,
            "range_m": self.range_m,
            "field_mass_gev": self.field_mass_gev,
            "density_kg_per_m3": self.density_kg_per_m3,
            "energy_joules": self.energy_joules,
            "coupling_used": self.coupling_used,
            "coupling_needed": self.coupling_needed,
            "coupling_allowed": self.coupling_allowed,
            "coupling_headroom_decades": self.coupling_headroom_decades,
            "binding_constraint": self.binding_constraint,
            "reachable": self.reachable,
            "notes": list(self.notes),
        }


#: Energy a laboratory can plausibly put into a static field configuration.
LAB_ENERGY_BUDGET_JOULES = 1.0e6


def evaluate_target(
    *,
    delta_log_constant: float,
    region_radius_m: float,
    range_m: float,
    density_kg_per_m3: float = DENSITY_TUNGSTEN,
    energy_budget_joules: float = LAB_ENERGY_BUDGET_JOULES,
    envelope: Callable[[float], float] | None = None,
) -> FrontierVerdict:
    """Report which constraint actually stops this target, and by how far.

    The point of returning BOTH bounds is that they fail in different
    directions and a single number hides that. A target can be energetically
    free and still impossible; that is the normal case, and the slogan version
    of the no-go does not say so.
    """
    mass_gev = range_to_mass_gev(range_m)
    b_allowed = max_coupling_b(range_m, envelope)
    b_needed = coupling_needed_for_shift(
        delta_log_constant=delta_log_constant,
        density_kg_per_m3=density_kg_per_m3,
        field_mass_gev=mass_gev,
    )
    # Energy is evaluated at the coupling that would actually be used: the
    # largest allowed, because a weaker one only makes the energy worse.
    b_used = min(b_allowed, b_needed) if b_needed > 0 else b_allowed
    b_used = max(b_used, 1e-300)
    energy = energy_to_hold_joules(
        delta_log_constant=delta_log_constant,
        coupling_b=b_used,
        region_radius_m=region_radius_m,
        field_mass_gev=mass_gev,
    )

    headroom = (
        math.log10(b_allowed / b_needed)
        if (b_allowed > 0 and b_needed > 0 and math.isfinite(b_needed))
        else -math.inf
    )
    energy_ok = energy <= energy_budget_joules
    coupling_ok = b_needed <= b_allowed

    notes: list[str] = []
    if coupling_ok and energy_ok:
        binding = "NONE"
        reachable = True
        notes.append(
            "Both bounds clear. This is an experiment, not a thought experiment."
        )
    elif not coupling_ok and not energy_ok:
        binding = "BOTH"
        reachable = False
    elif not coupling_ok:
        binding = "COUPLING"
        reachable = False
        notes.append(
            "Energy is NOT the obstacle here — the field energy is affordable. "
            "The obstacle is that no admissible source can put the field there. "
            "This bound is empirical and is being tightened, not a theorem."
        )
    else:
        binding = "ENERGY"
        reachable = False
        notes.append(
            "The source could deliver it; holding it costs more than the budget."
        )
    if range_m > region_radius_m:
        notes.append(
            "range exceeds the region radius: the deep-interior formula "
            "overestimates what a source of this size delivers"
        )
    return FrontierVerdict(
        delta_log_constant=float(delta_log_constant),
        region_radius_m=float(region_radius_m),
        range_m=float(range_m),
        field_mass_gev=mass_gev,
        density_kg_per_m3=float(density_kg_per_m3),
        energy_joules=energy,
        coupling_used=b_used,
        coupling_needed=b_needed,
        coupling_allowed=b_allowed,
        coupling_headroom_decades=headroom,
        binding_constraint=binding,
        reachable=reachable,
        notes=tuple(notes),
    )


def best_reachable_shift(
    *,
    region_radius_m: float,
    range_m: float,
    density_kg_per_m3: float = DENSITY_TUNGSTEN,
    energy_budget_joules: float = LAB_ENERGY_BUDGET_JOULES,
    envelope: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """The largest shift this configuration can produce without breaking a bound.

    The coupling ceiling gives the shift directly; the energy budget gives a
    second ceiling. The answer is the smaller, and which one it was is the
    interesting part.
    """
    mass_gev = range_to_mass_gev(range_m)
    b_allowed = max_coupling_b(range_m, envelope)
    from_source = shift_from_source(
        coupling_b=b_allowed,
        density_kg_per_m3=density_kg_per_m3,
        field_mass_gev=mass_gev,
    )
    radius = metres_to_inverse_gev(region_radius_m)
    form = yukawa_form_factor(mass_gev * radius)
    energy_gev = energy_budget_joules / JOULES_PER_GEV
    delta_phi_max = math.sqrt(energy_gev / (2.0 * math.pi * radius * form))
    from_energy = b_allowed * delta_phi_max / MBAR_PLANCK_GEV

    limited_by = "COUPLING" if from_source <= from_energy else "ENERGY"
    return {
        "schema": "aura.lawc05.best_reachable_shift.v1",
        "region_radius_m": region_radius_m,
        "range_m": range_m,
        "coupling_allowed": b_allowed,
        "shift_from_source": from_source,
        "shift_from_energy_budget": from_energy,
        "best_shift": min(from_source, from_energy),
        "limited_by": limited_by,
    }


#: Fractional-frequency resolution of a state-of-the-art optical clock
#: comparison. The number a shift has to beat to be seen at all.
OPTICAL_CLOCK_RESOLUTION = 1e-18


def detectability_scan(
    *,
    ranges_m: Any = None,
    region_radius_m: float = 1e-3,
    density_kg_per_m3: float = DENSITY_TUNGSTEN,
    energy_budget_joules: float = LAB_ENERGY_BUDGET_JOULES,
    threshold: float = OPTICAL_CLOCK_RESOLUTION,
    envelope: Callable[[float], float] | None = None,
) -> list[dict[str, Any]]:
    """Scan range and report where a detectable shift is not excluded.

    This is the atlas: the map of where energy is not the problem, so that
    effort goes to the bound that is.
    """
    if ranges_m is None:
        ranges_m = np.logspace(-8, 0, 17)
    rows: list[dict[str, Any]] = []
    for range_m in np.asarray(ranges_m, dtype=float):
        best = best_reachable_shift(
            region_radius_m=region_radius_m,
            range_m=float(range_m),
            density_kg_per_m3=density_kg_per_m3,
            energy_budget_joules=energy_budget_joules,
            envelope=envelope,
        )
        best["threshold"] = threshold
        best["detectable"] = bool(best["best_shift"] >= threshold)
        best["decades_short"] = (
            math.log10(best["best_shift"] / threshold)
            if best["best_shift"] > 0
            else -math.inf
        )
        rows.append(best)
    return rows


__all__ = [
    "DENSITY_GOLD",
    "DENSITY_LEAD",
    "DENSITY_OSMIUM",
    "DENSITY_TUNGSTEN",
    "FrontierVerdict",
    "LAB_ENERGY_BUDGET_JOULES",
    "OPTICAL_CLOCK_RESOLUTION",
    "best_reachable_shift",
    "coupling_needed_for_shift",
    "default_alpha_envelope",
    "density_to_natural",
    "detectability_scan",
    "energy_to_hold_joules",
    "evaluate_target",
    "inverse_gev_to_metres",
    "mass_gev_to_range",
    "max_coupling_b",
    "metres_to_inverse_gev",
    "range_to_mass_gev",
    "shift_from_source",
]
