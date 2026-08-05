"""A copy universe whose laws are a writable field, and are measurable from inside.

Channel 3. The first two channels are about this universe; this one is about
making a world where the constants are state, and then checking that the claim
means something.

The bar for "means something" is deliberately high, because a simulation will
happily display any law you assert. So nothing here reads a parameter to
decide what the world does:

* the matter fields evolve under ``phi_tt = div(c(q)^2 grad phi) - m(q)^2 phi``,
  discretised so that the operator stays self-adjoint and the energy is a real
  conserved quantity rather than a number printed next to the animation;
* :func:`recover_dispersion` measures ``omega(k)`` from the simulated field by
  Fourier transform alone and fits ``omega^2 = c^2 k^2 + m^2``. It is handed
  the recording, never the parameters. If the recovered constants match the
  ones that were painted, the world's laws are discoverable by an inhabitant.
  If they do not, this module is wrong and the test says so.

The law field
-------------
``q(x)`` is written by whoever is running the world::

    c(q) = c0 / (1 + beta q^2)        always <= c0, so the global cone holds
    m(q) = m0 + lambda q              stability flips where m_A = 2 m_B
    V(q) = (lambda_q / 4)(q^2 - v^2)^2

The causality bound is structural, not enforced: ``c(q) <= c0`` for every real
``q`` because ``beta >= 0``. An inhabitant can change what light does locally
and can never outrun the global cone, which is the one law they are not given
write access to.

What this is not
----------------
It is a classical field theory on a grid. Painting ``q`` costs work and the
ledger tracks it; erasing the record costs ``k T ln 2`` per bit and the ledger
tracks that too. Neither of those makes the simulated world physically special.
The honest claim is narrow and still interesting: inside it, law is state, the
inhabitants could measure it, and the accounting is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: How far a spectral peak must stand above the median of its column before it
#: counts as a mode rather than as noise. A flat spectrum has no frequency, and
#: fitting one produces a straight line through fabricated points.
_PEAK_PROMINENCE = 3.0

#: Boltzmann constant, J/K. Only used for the Landauer ledger.
BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class LawParameters:
    """How the law field maps to the constants the matter fields obey."""

    c0: float = 1.0
    beta: float = 0.5
    mass_a0: float = 1.0
    mass_a_slope: float = 0.6
    mass_b0: float = 0.3
    mass_b_slope: float = 0.05
    lambda_q: float = 1.0
    vacuum_q: float = 1.0

    def speed(self, q: np.ndarray | float) -> np.ndarray | float:
        """``c(q) = c0 / (1 + beta q^2)``. Bounded above by ``c0`` by construction."""
        return self.c0 / (1.0 + self.beta * np.asarray(q) ** 2)

    def mass_a(self, q: np.ndarray | float) -> np.ndarray | float:
        return np.abs(self.mass_a0 + self.mass_a_slope * np.asarray(q))

    def mass_b(self, q: np.ndarray | float) -> np.ndarray | float:
        return np.abs(self.mass_b0 + self.mass_b_slope * np.asarray(q))

    def decay_allowed(self, q: np.ndarray | float) -> np.ndarray | bool:
        """``A -> B + B`` is open where ``m_A(q) > 2 m_B(q)``.

        This is the whole point of a writable law: the same particle is stable
        in one painted region and unstable in another, and nothing about the
        particle changed.
        """
        return self.mass_a(q) > 2.0 * self.mass_b(q)

    def potential(self, q: np.ndarray | float) -> np.ndarray | float:
        arr = np.asarray(q)
        return 0.25 * self.lambda_q * (arr**2 - self.vacuum_q**2) ** 2


def _face_average(field_2d: np.ndarray, axis: int) -> np.ndarray:
    """Coefficient on the cell face, so the operator stays self-adjoint.

    Using the cell-centred value instead turns ``div(c^2 grad phi)`` into
    ``c^2 lap phi``, which is a different operator whenever ``c`` varies, and
    it does not conserve energy. The drift that produces looks exactly like
    physics until you measure it.
    """
    return 0.5 * (field_2d + np.roll(field_2d, -1, axis=axis))


def divergence_of_gradient(
    phi: np.ndarray, coefficient: np.ndarray, dx: float
) -> np.ndarray:
    """``div(coefficient * grad phi)`` on a periodic grid, self-adjoint form."""
    result = np.zeros_like(phi)
    for axis in (0, 1):
        face = _face_average(coefficient, axis)
        forward = np.roll(phi, -1, axis=axis) - phi
        flux = face * forward
        result += (flux - np.roll(flux, 1, axis=axis)) / (dx * dx)
    return result


@dataclass
class LedgerEntry:
    """One accounted transaction against the world."""

    kind: str
    detail: str
    work_joules: float = 0.0
    bits_erased: float = 0.0


@dataclass
class CopyUniverse:
    """A 2-D world with a writable law field and an honest energy ledger."""

    size: int = 64
    dx: float = 1.0
    params: LawParameters = field(default_factory=LawParameters)
    temperature_k: float = 300.0
    _q: np.ndarray = field(init=False)
    _phi_a: np.ndarray = field(init=False)
    _pi_a: np.ndarray = field(init=False)
    _phi_b: np.ndarray = field(init=False)
    _pi_b: np.ndarray = field(init=False)
    ledger: list[LedgerEntry] = field(default_factory=list)
    steps_taken: int = 0

    def __post_init__(self) -> None:
        shape = (self.size, self.size)
        self._q = np.full(shape, self.params.vacuum_q, dtype=float)
        self._phi_a = np.zeros(shape)
        self._pi_a = np.zeros(shape)
        self._phi_b = np.zeros(shape)
        self._pi_b = np.zeros(shape)

    # ── the law field ────────────────────────────────────────────────────
    @property
    def q(self) -> np.ndarray:
        return self._q

    def max_speed(self) -> float:
        """The fastest signal anywhere. Must never exceed ``c0``."""
        return float(np.max(self.params.speed(self._q)))

    def causality_holds(self) -> bool:
        return self.max_speed() <= self.params.c0 * (1.0 + 1e-12)

    def free_energy(self) -> float:
        """``F = integral( (1/2)|grad q|^2 + V(q) )`` — the cost of the law itself."""
        grad_x = (np.roll(self._q, -1, axis=0) - self._q) / self.dx
        grad_y = (np.roll(self._q, -1, axis=1) - self._q) / self.dx
        density = 0.5 * (grad_x**2 + grad_y**2) + self.params.potential(self._q)
        return float(np.sum(density) * self.dx**2)

    def paint(self, mask: np.ndarray, value: float) -> float:
        """Write the law inside ``mask``. Returns the work done.

        The work is measured as the change in free energy, not asserted. A
        painter who lowers the free energy is given a negative number, because
        that is what happened.
        """
        before = self.free_energy()
        self._q = np.where(mask, float(value), self._q)
        work = self.free_energy() - before
        self.ledger.append(
            LedgerEntry(kind="paint", detail=f"q<-{value:+.3f}", work_joules=work)
        )
        return work

    def nucleate_bubble(
        self, centre: tuple[int, int], radius: float, value: float
    ) -> dict[str, Any]:
        """A finite domain of different law, with its measured cost.

        The thin-wall expectation is ``E ~ 2 pi R (Delta q)^2`` in 2-D. The
        number returned is the measured free-energy change, and the expectation
        is returned beside it so the two can be compared rather than conflated.
        """
        rows, cols = np.ogrid[: self.size, : self.size]
        distance = np.sqrt((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)
        mask = distance <= radius
        delta_q = float(value) - float(np.mean(self._q[mask])) if mask.any() else 0.0
        work = self.paint(mask, value)
        self.ledger[-1].kind = "nucleate"
        return {
            "radius": float(radius),
            "delta_q": delta_q,
            "measured_work": work,
            "thin_wall_expectation": 2.0 * np.pi * float(radius) * delta_q**2,
            "cells": int(mask.sum()),
        }

    def erase(self, bits: float) -> float:
        """Landauer: erasing information dissipates at least ``k T ln 2`` per bit."""
        if bits < 0.0:
            raise ValueError("cannot erase a negative number of bits")
        heat = float(bits) * BOLTZMANN * self.temperature_k * np.log(2.0)
        self.ledger.append(
            LedgerEntry(
                kind="erase",
                detail=f"{bits:g} bits at {self.temperature_k:g} K",
                work_joules=heat,
                bits_erased=float(bits),
            )
        )
        return heat

    def landauer_floor_joules(self) -> float:
        return sum(entry.bits_erased for entry in self.ledger) * (
            BOLTZMANN * self.temperature_k * np.log(2.0)
        )

    # ── matter ───────────────────────────────────────────────────────────
    def seed_noise(self, amplitude: float = 1e-3, seed: int = 0) -> None:
        """Excite every mode a little, so the dispersion relation is visible."""
        rng = np.random.default_rng(seed)
        self._phi_a = amplitude * rng.standard_normal((self.size, self.size))
        self._phi_b = amplitude * rng.standard_normal((self.size, self.size))
        self._pi_a = np.zeros_like(self._phi_a)
        self._pi_b = np.zeros_like(self._phi_b)

    def max_stable_dt(self, safety: float = 0.4) -> float:
        """CFL for the 2-D wave operator at the fastest local speed."""
        return safety * self.dx / (self.max_speed() * np.sqrt(2.0))

    def _acceleration(self, phi: np.ndarray, mass: np.ndarray) -> np.ndarray:
        speed_squared = np.asarray(self.params.speed(self._q)) ** 2
        return divergence_of_gradient(phi, speed_squared, self.dx) - mass**2 * phi

    def step(self, dt: float, record: list[np.ndarray] | None = None) -> None:
        """One velocity-Verlet step: symplectic AND synchronised.

        Plain leapfrog keeps ``pi`` half a step ahead of ``phi``, which is fine
        for the trajectory and wrong for the energy: any quantity built from
        both at once picks up an O(dt) offset and oscillates by tens of percent.
        That oscillation is an artefact of when the variables are sampled, and
        reading it as physics — or worse, tuning the display until it looks
        flat — is how a simulation starts lying about its own conservation law.

        Verlet costs one cached force evaluation and puts both variables on the
        same instant, so the reported energy is the energy.
        """
        mass_a = np.asarray(self.params.mass_a(self._q))
        mass_b = np.asarray(self.params.mass_b(self._q))

        accel_a = self._acceleration(self._phi_a, mass_a)
        accel_b = self._acceleration(self._phi_b, mass_b)

        self._pi_a += 0.5 * dt * accel_a
        self._pi_b += 0.5 * dt * accel_b
        self._phi_a += dt * self._pi_a
        self._phi_b += dt * self._pi_b
        self._pi_a += 0.5 * dt * self._acceleration(self._phi_a, mass_a)
        self._pi_b += 0.5 * dt * self._acceleration(self._phi_b, mass_b)

        self.steps_taken += 1
        if record is not None:
            record.append(self._phi_a.copy())

    def run(self, steps: int, dt: float, record: bool = False) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for _ in range(steps):
            self.step(dt, frames if record else None)
        return frames

    def matter_energy(self) -> float:
        """``E = sum( pi^2/2 + c^2|grad phi|^2/2 + m^2 phi^2/2 )`` for both fields."""
        speed_squared = np.asarray(self.params.speed(self._q)) ** 2
        total = 0.0
        for phi, momentum, mass in (
            (self._phi_a, self._pi_a, np.asarray(self.params.mass_a(self._q))),
            (self._phi_b, self._pi_b, np.asarray(self.params.mass_b(self._q))),
        ):
            grad_x = (np.roll(phi, -1, axis=0) - phi) / self.dx
            grad_y = (np.roll(phi, -1, axis=1) - phi) / self.dx
            face_x = _face_average(speed_squared, 0)
            face_y = _face_average(speed_squared, 1)
            density = (
                0.5 * momentum**2
                + 0.5 * (face_x * grad_x**2 + face_y * grad_y**2)
                + 0.5 * mass**2 * phi**2
            )
            total += float(np.sum(density) * self.dx**2)
        return total


@dataclass(frozen=True)
class RecoveredLaw:
    """What an inhabitant would measure, WITH how well they could measure it.

    The uncertainties are not decoration. ``omega^2 = c^2 k^2 + m^2`` is a
    straight line whose slope is the speed and whose intercept is the mass, and
    when ``m >> c k`` over the available modes that line is nearly flat: the
    intercept is pinned and the slope is barely constrained. A recovery that
    reported ``c`` as a bare number there would be stating a fitted artefact
    with the same confidence as a measurement.

    :attr:`speed_well_determined` is the flag to check before believing
    :attr:`speed`.
    """

    speed: float
    mass: float
    speed_stderr: float
    mass_stderr: float
    residual: float
    modes_used: int
    #: c k_max / m over the fitted modes. Below ~1 the speed is weakly
    #: constrained no matter how clean the data is.
    dispersion_leverage: float

    @property
    def speed_well_determined(self) -> bool:
        return self.speed > 0.0 and self.speed_stderr / self.speed < 0.1

    @property
    def mass_well_determined(self) -> bool:
        return self.mass > 0.0 and self.mass_stderr / self.mass < 0.1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lawc05.recovered_law.v2",
            "speed": self.speed,
            "speed_stderr": self.speed_stderr,
            "speed_well_determined": self.speed_well_determined,
            "mass": self.mass,
            "mass_stderr": self.mass_stderr,
            "mass_well_determined": self.mass_well_determined,
            "residual": self.residual,
            "modes_used": self.modes_used,
            "dispersion_leverage": self.dispersion_leverage,
        }


def recover_dispersion(
    frames: list[np.ndarray] | np.ndarray,
    dt: float,
    dx: float,
    *,
    max_modes: int = 24,
) -> RecoveredLaw:
    """Measure ``c`` and ``m`` from a recording. Blind: no parameters are read.

    Method: transform in space and time, take the dominant temporal frequency
    for each spatial mode, undo the two discretisation artefacts that a naive
    fit would absorb into the answer, and regress ``omega^2`` on ``k^2``.

    The two corrections matter and are the difference between recovering the
    law and recovering the grid:

    * the spatial operator's eigenvalue is ``(4/dx^2) sin^2(k dx / 2)``, not
      ``k^2``, so the fit uses the former;
    * leapfrog's numerical frequency satisfies ``sin(w dt/2) = Omega dt/2``,
      so the measured peak is inverted back to ``Omega`` before fitting.

    Skipping either one still produces a confident-looking straight line with
    the wrong slope, which is the failure mode this function exists to avoid.
    """
    stack = np.asarray(frames)
    if stack.ndim != 3 or stack.shape[0] < 8:
        raise ValueError("need at least 8 recorded frames of a 2-D field")
    n_t, n_x, n_y = stack.shape

    # A Hann window in time. Without it a mode whose period does not divide the
    # recording leaks across neighbouring bins and drags the peak, which shows
    # up as a systematic error in the recovered mass rather than as noise.
    window = np.hanning(n_t)[:, None, None]
    spectrum = np.fft.fftn(stack * window, axes=(0, 1, 2))
    power = np.abs(spectrum) ** 2

    freqs = np.fft.fftfreq(n_t, d=dt) * 2.0 * np.pi
    kx = np.fft.fftfreq(n_x, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(n_y, d=dy_of(dx)) * 2.0 * np.pi

    lattice_k2: list[float] = []
    omega2: list[float] = []
    half_t = n_t // 2

    # Walk the lowest spatial modes along each axis: high-k modes are where
    # the lattice dispersion is most curved and the peak least clean.
    candidates: list[tuple[int, int]] = []
    limit = max(1, min(max_modes, n_x // 4))
    candidates += [(i, 0) for i in range(1, limit)]
    candidates += [(0, j) for j in range(1, limit)]

    for i, j in candidates:
        column = power[1:half_t, i, j]
        if column.size == 0:
            continue
        peak = int(np.argmax(column)) + 1
        # A mode with no signal in it has no frequency. argmax on a flat or
        # empty spectrum returns bin zero and the arithmetic below would turn
        # that into a confident-looking data point, so require the peak to
        # actually stand out before treating it as a measurement.
        peak_power = float(column[peak - 1])
        median_power = float(np.median(column))
        if peak_power <= 0.0 or peak_power < _PEAK_PROMINENCE * max(
            median_power, 1e-300
        ):
            continue
        # Sub-bin peak location. The FFT resolves omega only to 2 pi / (T dt),
        # and for a light field the mass sits inside one bin — so without this
        # the recovered mass is quantised by the recording length rather than
        # measured. Quadratic interpolation on the log-power neighbourhood.
        offset = _parabolic_offset(power[1:half_t, i, j], peak - 1)
        measured_omega = abs(freqs[1] - freqs[0]) * (peak + offset)
        measured_omega = abs(measured_omega)
        if measured_omega <= 0.0:
            continue
        # Undo the leapfrog time discretisation.
        argument = measured_omega * dt / 2.0
        if argument >= np.pi / 2.0:
            continue
        true_omega = (2.0 / dt) * np.sin(argument)
        # The spatial operator's own eigenvalue.
        k2_lattice = (4.0 / dx**2) * (
            np.sin(kx[i] * dx / 2.0) ** 2 + np.sin(ky[j] * dx / 2.0) ** 2
        )
        lattice_k2.append(k2_lattice)
        omega2.append(true_omega**2)

    if len(lattice_k2) < 3:
        raise ValueError("not enough clean modes to fit a dispersion relation")

    x = np.asarray(lattice_k2)
    y = np.asarray(omega2)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = float(residuals[0]) if len(residuals) else 0.0

    # Standard errors on the fit, propagated through the square roots.
    n_points = x.size
    dof = max(n_points - 2, 1)
    prediction = design @ np.array([slope, intercept])
    sigma_squared = float(np.sum((y - prediction) ** 2)) / dof
    try:
        covariance = sigma_squared * np.linalg.inv(design.T @ design)
        slope_err = float(np.sqrt(max(covariance[0, 0], 0.0)))
        intercept_err = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        slope_err = float("inf")
        intercept_err = float("inf")

    speed = float(np.sqrt(max(slope, 0.0)))
    mass = float(np.sqrt(max(intercept, 0.0)))
    speed_err = slope_err / (2.0 * speed) if speed > 0.0 else float("inf")
    mass_err = intercept_err / (2.0 * mass) if mass > 0.0 else float("inf")

    # How much of the observed omega^2 the k-dependent term is responsible for.
    # This is the geometric reason the slope can be unmeasurable even when the
    # data are clean, so it is reported rather than inferred from the error.
    leverage = (
        speed * float(np.sqrt(np.max(x))) / mass if mass > 0.0 else float("inf")
    )

    return RecoveredLaw(
        speed=speed,
        mass=mass,
        speed_stderr=speed_err,
        mass_stderr=mass_err,
        residual=residual,
        modes_used=int(n_points),
        dispersion_leverage=float(leverage),
    )


def _parabolic_offset(column: np.ndarray, index: int) -> float:
    """Sub-bin correction to a spectral peak, in bins.

    Fits a parabola through the log-power at the peak and its two neighbours.
    Returns 0.0 at an edge or when the neighbourhood is not concave, because a
    correction derived from a non-peak is a fabrication, not a refinement.
    """
    if index <= 0 or index >= column.size - 1:
        return 0.0
    with np.errstate(divide="ignore"):
        left, middle, right = (
            float(np.log(max(column[index - 1], 1e-300))),
            float(np.log(max(column[index], 1e-300))),
            float(np.log(max(column[index + 1], 1e-300))),
        )
    denominator = left - 2.0 * middle + right
    if denominator >= 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(offset) if abs(offset) <= 0.5 else 0.0


def dy_of(dx: float) -> float:
    """The grid is square; named so the axis asymmetry is impossible to miss."""
    return dx


__all__ = [
    "BOLTZMANN",
    "CopyUniverse",
    "LawParameters",
    "LedgerEntry",
    "RecoveredLaw",
    "divergence_of_gradient",
    "recover_dispersion",
]
