"""The channel that actually works: effective constants in engineered vacuum.

:mod:`frontier` closes the ambient door twice over. This module opens the one
that is not closed, and the comparison is the whole argument.

An AMBIENT constant is a property of the vacuum everywhere. Moving it needs a
scalar coupled to matter at a strength that short-range gravity experiments
have excluded by roughly nine decades in ``alpha``.

An EFFECTIVE constant is a property of an excitation in a medium — a
quasiparticle mass, a gap, a Landé factor, a fine-structure constant as seen
by an intersubband transition. Those are set by the electromagnetic
environment, and the electromagnetic environment is something a fabricated
structure changes for free, because the vacuum supplies the field.

Concretely: confine a mode to volume ``V`` and its zero-point field is

    E_vac = sqrt(hbar omega / (2 eps0 V))

which grows as ``V^{-1/2}``. Squeeze the mode into a sub-wavelength resonator
and a dipole ``d`` sees a vacuum Rabi coupling ``g = d E_vac`` that can reach a
sizeable fraction of ``omega``. In that ultrastrong regime the matter
properties themselves shift — this is measured, not proposed; split-ring
resonators have moved fractional quantum Hall gaps by tens of percent.

The energy ledger is the point. The zero-point energy in the mode is
``hbar omega / 2``, of order 1e-24 J. Nothing has to be paid to hold it,
because it is the vacuum's own energy. The design cost is fabrication, and
the design itself is code.

Everything here is a single-mode dipole estimate: order-of-magnitude, useful
for deciding what to simulate properly, not a substitute for solving Maxwell
in the actual geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── SI constants ──────────────────────────────────────────────────────────
HBAR = 1.054571817e-34
EPS0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 299792458.0

#: Above this normalised coupling the mode and the matter no longer have
#: separate identities; perturbative shift formulas stop meaning anything.
ULTRASTRONG_THRESHOLD = 0.1
DEEP_STRONG_THRESHOLD = 1.0


def vacuum_field_amplitude(omega_rad_s: float, mode_volume_m3: float) -> float:
    """``E_vac = sqrt(hbar omega / (2 eps0 V))`` in V/m."""
    if omega_rad_s <= 0.0:
        raise ValueError("omega must be positive")
    if mode_volume_m3 <= 0.0:
        raise ValueError("mode volume must be positive")
    return math.sqrt(HBAR * omega_rad_s / (2.0 * EPS0 * mode_volume_m3))


def vacuum_rabi_coupling(
    *, dipole_cm: float, omega_rad_s: float, mode_volume_m3: float
) -> float:
    """``g = d E_vac / hbar`` in rad/s, for a dipole in C*m."""
    if dipole_cm <= 0.0:
        raise ValueError("dipole must be positive")
    return dipole_cm * vacuum_field_amplitude(omega_rad_s, mode_volume_m3) / HBAR


def dipole_from_length(length_m: float, charges: float = 1.0) -> float:
    """A transition dipole of ``q * a`` — the usual back-of-envelope form."""
    return charges * ELEMENTARY_CHARGE * float(length_m)


def diffraction_limited_volume(omega_rad_s: float, refractive_index: float = 1.0) -> float:
    """``(lambda / 2n)^3`` — the smallest volume without sub-wavelength tricks.

    Quoted so a caller can see how far below it a resonator design has gone;
    that ratio is the entire source of the coupling enhancement.
    """
    if refractive_index <= 0.0:
        raise ValueError("refractive index must be positive")
    wavelength = 2.0 * math.pi * SPEED_OF_LIGHT / omega_rad_s
    return (wavelength / (2.0 * refractive_index)) ** 3


@dataclass(frozen=True)
class EffectiveCouplingEstimate:
    """A single-mode estimate of how much a fabricated vacuum moves matter."""

    omega_rad_s: float
    mode_volume_m3: float
    vacuum_field_v_per_m: float
    coupling_rad_s: float
    normalized_coupling: float
    regime: str
    fractional_shift_estimate: float
    zero_point_energy_joules: float
    volume_compression: float
    perturbative: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lawc05.effective_coupling_estimate.v1",
            "omega_rad_s": self.omega_rad_s,
            "mode_volume_m3": self.mode_volume_m3,
            "vacuum_field_v_per_m": self.vacuum_field_v_per_m,
            "coupling_rad_s": self.coupling_rad_s,
            "normalized_coupling": self.normalized_coupling,
            "regime": self.regime,
            "fractional_shift_estimate": self.fractional_shift_estimate,
            "zero_point_energy_joules": self.zero_point_energy_joules,
            "volume_compression": self.volume_compression,
            "perturbative": self.perturbative,
            "notes": list(self.notes),
        }


def estimate_effective_shift(
    *,
    omega_rad_s: float,
    mode_volume_m3: float,
    dipole_cm: float,
    refractive_index: float = 1.0,
) -> EffectiveCouplingEstimate:
    """How far a confined vacuum mode moves an effective matter constant.

    The dispersive shift of a matter level by a detuned vacuum mode goes as
    ``eta^2`` for small ``eta = g / omega``. That estimate is reported, and it
    is reported as NOT PERTURBATIVE once ``eta`` passes the ultrastrong
    threshold, because at that point the number is no longer an approximation
    of anything — the correct treatment diagonalises the coupled system, and
    saying "20%" from a quadratic formula would be inventing precision.
    """
    field = vacuum_field_amplitude(omega_rad_s, mode_volume_m3)
    g = vacuum_rabi_coupling(
        dipole_cm=dipole_cm, omega_rad_s=omega_rad_s, mode_volume_m3=mode_volume_m3
    )
    eta = g / omega_rad_s
    diffraction_volume = diffraction_limited_volume(omega_rad_s, refractive_index)
    compression = diffraction_volume / mode_volume_m3

    notes: list[str] = []
    if eta >= DEEP_STRONG_THRESHOLD:
        regime = "DEEP_STRONG"
        perturbative = False
        notes.append(
            "eta >= 1: the mode and the matter are one system. Diagonalise it; "
            "no perturbative shift is meaningful."
        )
    elif eta >= ULTRASTRONG_THRESHOLD:
        regime = "ULTRASTRONG"
        perturbative = False
        notes.append(
            "eta >= 0.1: this is the regime the measured quantum-Hall gap shifts "
            "live in. The quadratic estimate is an order-of-magnitude marker only."
        )
    else:
        regime = "PERTURBATIVE"
        perturbative = True

    if compression < 1.0:
        notes.append(
            "mode volume exceeds the diffraction limit: no sub-wavelength "
            "enhancement is present in this design"
        )

    return EffectiveCouplingEstimate(
        omega_rad_s=omega_rad_s,
        mode_volume_m3=mode_volume_m3,
        vacuum_field_v_per_m=field,
        coupling_rad_s=g,
        normalized_coupling=eta,
        regime=regime,
        fractional_shift_estimate=eta * eta,
        zero_point_energy_joules=0.5 * HBAR * omega_rad_s,
        volume_compression=compression,
        perturbative=perturbative,
        notes=tuple(notes),
    )


def collective_coupling(single_coupling_rad_s: float, n_emitters: float) -> float:
    """``g_N = g sqrt(N)`` — the Dicke enhancement.

    This is not a detail, it is how the measured experiments get where they
    get. One 10 nm dipole in a squeezed THz mode reaches eta ~ 1e-2; the 2DEG
    under a split-ring resonator has 1e5 or more electrons sharing the mode,
    and sqrt(N) carries it into the ultrastrong regime.

    N is also the dominant uncertainty in any estimate built on this: it is
    set by carrier density times the mode's actual overlap with the electron
    gas, and the overlap is exactly what a full solve computes and this does
    not.
    """
    if n_emitters < 0.0:
        raise ValueError("n_emitters must be non-negative")
    return float(single_coupling_rad_s) * math.sqrt(float(n_emitters))


def emitters_in_2deg(
    *, sheet_density_per_cm2: float, mode_area_m2: float, overlap: float = 1.0
) -> float:
    """How many carriers actually share the mode.

    ``overlap`` is the fraction of the mode's field that lies in the gas. It
    defaults to 1.0, which is optimistic and stated as such: a real resonator
    puts a good deal of its field in the substrate and the metal.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1]")
    per_m2 = float(sheet_density_per_cm2) * 1e4
    return per_m2 * float(mode_area_m2) * float(overlap)


def mode_volume_for_target_eta(
    *, target_eta: float, omega_rad_s: float, dipole_cm: float
) -> float:
    """The mode volume a design must reach for a given normalised coupling.

    This is the compiler's actual objective: everything else in the structure
    exists to hit this number. ``eta = d sqrt(hbar omega / 2 eps0 V) / (hbar omega)``
    inverts to ``V = d^2 / (2 eps0 hbar omega eta^2)``.
    """
    if target_eta <= 0.0:
        raise ValueError("target_eta must be positive")
    return dipole_cm**2 / (2.0 * EPS0 * HBAR * omega_rad_s * target_eta**2)


def channel_comparison(
    *,
    ambient_shift: float,
    ambient_energy_joules: float,
    ambient_coupling_shortfall_decades: float,
    effective: EffectiveCouplingEstimate,
) -> dict[str, Any]:
    """Put the two channels side by side, which is the entire argument.

    The ambient channel is quoted with its coupling shortfall because its
    energy figure alone is misleading — it is affordable and still impossible.
    """
    return {
        "schema": "aura.lawc05.channel_comparison.v1",
        "ambient": {
            "best_shift": ambient_shift,
            "energy_joules": ambient_energy_joules,
            "coupling_shortfall_decades": ambient_coupling_shortfall_decades,
            "blocked_by": "coupling bound (empirical), not energy",
        },
        "effective": {
            "fractional_shift": effective.fractional_shift_estimate,
            "regime": effective.regime,
            "zero_point_energy_joules": effective.zero_point_energy_joules,
            "blocked_by": "nothing — this regime is measured",
        },
        "shift_ratio_effective_over_ambient": (
            effective.fractional_shift_estimate / ambient_shift
            if ambient_shift > 0
            else math.inf
        ),
        "energy_ratio_ambient_over_effective": (
            ambient_energy_joules / effective.zero_point_energy_joules
            if effective.zero_point_energy_joules > 0
            else math.inf
        ),
    }


__all__ = [
    "DEEP_STRONG_THRESHOLD",
    "ELEMENTARY_CHARGE",
    "EPS0",
    "HBAR",
    "ULTRASTRONG_THRESHOLD",
    "EffectiveCouplingEstimate",
    "channel_comparison",
    "collective_coupling",
    "emitters_in_2deg",
    "diffraction_limited_volume",
    "dipole_from_length",
    "estimate_effective_shift",
    "mode_volume_for_target_eta",
    "vacuum_field_amplitude",
    "vacuum_rabi_coupling",
]
