"""Standalone (no-STK) astrodynamics core for the Earth -> 2024 YR4 porkchop.

Why this exists
---------------
``asteroid_porkchop_yr4.py`` is the deliverable that runs inside STK.  STK
Engine is Windows/Linux-only, so this module reproduces the same computation
with nothing but NumPy, which lets the porkchop be generated (and the STK
numbers cross-checked) anywhere.

It makes exactly the same modelling assumptions as the STK script:

* restricted two-body problem, Sun point mass;
* zero-revolution Lambert transfer between the two body positions;
* prograde transfer, so the short/long way is picked from the sign of the
  z-component of ``r1 x r2`` (identical rule to the STK Lambert profile);
* ``C3_launch = |v1 - v_Earth|^2`` and ``dv_arrival = |v_asteroid - v2|``
  (a rendezvous, i.e. the arrival impulse is actually burned).

Ephemerides are the JPL Horizons heliocentric ICRF cartesian state vectors in
``horizons_results.txt`` (2024 YR4) and ``horizons_earth.txt`` (Earth).  Times
are JD TDB throughout.
"""

from __future__ import annotations

import numpy as np

from horizons_to_stk import parse_horizons_vectors

#: Sun gravitational parameter, km^3/s^2 (DE440/441).
MU_SUN = 1.32712440041279419e11

#: Seconds in a day.
DAY = 86400.0

_FOUR_PI2 = 4.0 * np.pi**2


# ---------------------------------------------------------------------------
# Stumpff functions
# ---------------------------------------------------------------------------
def stumpff_c2_c3(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Stumpff functions ``C(psi)`` and ``S(psi)``.

    Series expansions are used near ``psi = 0`` where the closed forms are
    numerically ill-conditioned.
    """
    psi = np.asarray(psi, dtype=float)
    c2 = np.empty_like(psi)
    c3 = np.empty_like(psi)

    pos = psi > 1e-6
    neg = psi < -1e-6
    small = ~(pos | neg)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if np.any(pos):
            s = np.sqrt(np.where(pos, psi, 1.0))
            c2 = np.where(pos, (1.0 - np.cos(s)) / np.where(pos, psi, 1.0), c2)
            c3 = np.where(pos, (s - np.sin(s)) / np.where(pos, psi, 1.0) ** 1.5, c3)
        if np.any(neg):
            s = np.sqrt(np.where(neg, -psi, 1.0))
            c2 = np.where(neg, (np.cosh(s) - 1.0) / np.where(neg, -psi, 1.0), c2)
            c3 = np.where(
                neg, (np.sinh(s) - s) / np.where(neg, -psi, 1.0) ** 1.5, c3
            )

    if np.any(small):
        p = np.where(small, psi, 0.0)
        c2 = np.where(small, 0.5 - p / 24.0 + p**2 / 720.0, c2)
        c3 = np.where(small, 1.0 / 6.0 - p / 120.0 + p**2 / 5040.0, c3)

    return c2, c3


# ---------------------------------------------------------------------------
# Two-body (Kepler) propagation, universal variables
# ---------------------------------------------------------------------------
def kepler_propagate(
    r0: np.ndarray, v0: np.ndarray, dt: np.ndarray, mu: float = MU_SUN, maxiter: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state vector under a point-mass central body.

    Parameters
    ----------
    r0, v0 : ndarray, shape (..., 3)
        Initial position (km) and velocity (km/s).
    dt : ndarray, shape (...)
        Propagation time (s); may be negative.

    Returns
    -------
    tuple(ndarray, ndarray)
        Propagated position and velocity, same shape as the inputs.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    dt = np.asarray(dt, dtype=float)

    sqrt_mu = np.sqrt(mu)
    r0n = np.linalg.norm(r0, axis=-1)
    v0n2 = np.sum(v0 * v0, axis=-1)
    alpha = 2.0 / r0n - v0n2 / mu  # 1/a
    rv0 = np.sum(r0 * v0, axis=-1) / sqrt_mu

    # Elliptical initial guess; every body here is on a bound orbit.
    chi = sqrt_mu * dt * alpha

    r = r0n.copy()
    for _ in range(maxiter):
        psi = chi**2 * alpha
        c2, c3 = stumpff_c2_c3(psi)
        r = chi**2 * c2 + rv0 * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2)
        f = (
            chi**3 * c3
            + rv0 * chi**2 * c2
            + r0n * chi * (1.0 - psi * c3)
            - sqrt_mu * dt
        )
        step = f / r
        chi = chi - step
        if np.all(np.abs(step) < 1e-9 * np.maximum(1.0, np.abs(chi))):
            break

    psi = chi**2 * alpha
    c2, c3 = stumpff_c2_c3(psi)
    r = chi**2 * c2 + rv0 * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2)

    f = 1.0 - chi**2 / r0n * c2
    g = dt - chi**3 / sqrt_mu * c3
    fdot = sqrt_mu / (r * r0n) * chi * (psi * c3 - 1.0)
    gdot = 1.0 - chi**2 / r * c2

    r_new = f[..., None] * r0 + g[..., None] * v0
    v_new = fdot[..., None] * r0 + gdot[..., None] * v0
    return r_new, v_new


# ---------------------------------------------------------------------------
# Ephemeris
# ---------------------------------------------------------------------------
class SunCentredEphemeris:
    """Tabulated heliocentric states with two-body interpolation.

    Horizons is sampled at a finite step (10 days for 2024 YR4).  Rather than
    interpolating the samples polynomially, each query is Kepler-propagated
    from the two bracketing samples and the two arcs are blended with a
    smoothstep weight.  Over half a sample step the neglected planetary
    perturbations are worth a few km, i.e. far below anything that matters for
    a porkchop, and the result is continuous and derivative-continuous.
    """

    def __init__(self, path: str, name: str = ""):
        points = sorted(parse_horizons_vectors(path), key=lambda p: p.jd_tdb)
        self.name = name or path
        self.jd = np.array([p.jd_tdb for p in points])
        self.r = np.array([[p.x, p.y, p.z] for p in points])
        self.v = np.array([[p.vx, p.vy, p.vz] for p in points])
        self.jd_start = float(self.jd[0])
        self.jd_stop = float(self.jd[-1])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<SunCentredEphemeris {self.name!r}: {len(self.jd)} pts, "
            f"JD {self.jd_start} .. {self.jd_stop}>"
        )

    def state(self, jd) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position, velocity)`` in km and km/s at the given JD TDB."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        if np.any(jd < self.jd_start - 1e-9) or np.any(jd > self.jd_stop + 1e-9):
            raise ValueError(
                f"{self.name}: requested epoch outside ephemeris coverage "
                f"(JD {self.jd_start} .. {self.jd_stop})"
            )

        i = np.clip(np.searchsorted(self.jd, jd) - 1, 0, len(self.jd) - 2)
        jd_a, jd_b = self.jd[i], self.jd[i + 1]

        r_a, v_a = kepler_propagate(self.r[i], self.v[i], (jd - jd_a) * DAY)
        r_b, v_b = kepler_propagate(self.r[i + 1], self.v[i + 1], (jd - jd_b) * DAY)

        tau = (jd - jd_a) / (jd_b - jd_a)
        w = (tau**2 * (3.0 - 2.0 * tau))[..., None]
        return (1.0 - w) * r_a + w * r_b, (1.0 - w) * v_a + w * v_b


# ---------------------------------------------------------------------------
# Lambert solver, universal variables (Bate/Mueller/White, Vallado alg. 58)
# ---------------------------------------------------------------------------
def lambert_universal(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: np.ndarray,
    mu: float = MU_SUN,
    prograde: bool = True,
    iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the zero-revolution Lambert problem, vectorised over a grid.

    The transfer direction follows the same rule as the STK Lambert profile in
    ``asteroid_porkchop_yr4.py``: a prograde transfer takes the short way when
    ``(r1 x r2)_z > 0`` and the long way otherwise.

    Parameters
    ----------
    r1, r2 : ndarray, shape (..., 3)
        Departure and arrival position vectors (km).
    tof : ndarray, shape (...)
        Time of flight (s).  Non-positive entries yield NaN.

    Returns
    -------
    tuple(ndarray, ndarray)
        Departure and arrival velocity on the transfer arc (km/s).  Entries
        that have no solution are NaN.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    tof = np.asarray(tof, dtype=float)

    sqrt_mu = np.sqrt(mu)
    r1n = np.linalg.norm(r1, axis=-1)
    r2n = np.linalg.norm(r2, axis=-1)

    cos_dnu = np.clip(np.sum(r1 * r2, axis=-1) / (r1n * r2n), -1.0, 1.0)
    hz = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    short_way = (hz > 0.0) if prograde else (hz < 0.0)
    dm = np.where(short_way, 1.0, -1.0)

    A = dm * np.sqrt(r1n * r2n * (1.0 + cos_dnu))
    feasible = (np.abs(A) > 1e-9 * (r1n + r2n)) & (tof > 0.0) & np.isfinite(tof)

    def time_of_flight(psi):
        """TOF for a trial psi; NaN where the geometry is invalid."""
        c2, c3 = stumpff_c2_c3(psi)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y = r1n + r2n + A * (psi * c3 - 1.0) / np.sqrt(c2)
            ok = np.isfinite(y) & (y > 0.0)
            chi = np.sqrt(np.where(ok, y, 1.0) / c2)
            t = (chi**3 * c3 + A * np.sqrt(np.where(ok, y, 1.0))) / sqrt_mu
        return np.where(ok, t, np.nan), y

    # TOF is monotonically increasing in psi, so plain bisection converges.
    # psi -> 4*pi^2 is the one-revolution limit (TOF -> infinity); the lower
    # bracket is walked down until it undershoots the requested TOF.  Where the
    # geometry is invalid (y < 0, only possible on strongly hyperbolic short-way
    # arcs) the trial is treated as "too fast", which pushes the bracket up.
    psi_hi = np.full(r1n.shape, _FOUR_PI2 - 1e-4)
    psi_lo = np.full(r1n.shape, -_FOUR_PI2)
    for _ in range(5):
        t_lo, _ = time_of_flight(psi_lo)
        too_slow = np.isfinite(t_lo) & (t_lo > tof)
        if not np.any(too_slow):
            break
        psi_lo = np.where(too_slow, psi_lo * 4.0, psi_lo)

    psi = 0.5 * (psi_lo + psi_hi)
    for _ in range(iterations):
        psi = 0.5 * (psi_lo + psi_hi)
        t_psi, _ = time_of_flight(psi)
        # NaN (invalid geometry) counts as "too fast" -> raise the lower bound.
        too_fast = ~(t_psi > tof)
        psi_lo = np.where(too_fast, psi, psi_lo)
        psi_hi = np.where(too_fast, psi_hi, psi)

    c2, c3 = stumpff_c2_c3(psi)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = r1n + r2n + A * (psi * c3 - 1.0) / np.sqrt(c2)
        t_psi, _ = time_of_flight(psi)

        converged = (
            feasible
            & np.isfinite(y)
            & (y > 0.0)
            & np.isfinite(t_psi)
            & (np.abs(t_psi - tof) < 1e-4 * np.maximum(tof, 1.0))
        )

        y_safe = np.where(converged, y, 1.0)
        f = 1.0 - y_safe / r1n
        g = A * np.sqrt(y_safe / mu)
        gdot = 1.0 - y_safe / r2n

        v1 = (r2 - f[..., None] * r1) / g[..., None]
        v2 = (gdot[..., None] * r2 - r1) / g[..., None]

    bad = ~converged
    v1 = np.where(bad[..., None], np.nan, v1)
    v2 = np.where(bad[..., None], np.nan, v2)
    return v1, v2


# ---------------------------------------------------------------------------
# Porkchop grid
# ---------------------------------------------------------------------------
def solve_porkchop(
    departure_ephem: SunCentredEphemeris,
    arrival_ephem: SunCentredEphemeris,
    launch_jd: np.ndarray,
    arrival_jd: np.ndarray,
    mu: float = MU_SUN,
    prograde: bool = True,
) -> dict:
    """Compute the porkchop grids for every launch/arrival combination.

    Returns
    -------
    dict
        ``c3_launch`` (km^2/s^2), ``dv_arrival`` (km/s), ``dv_departure``
        (km/s), ``tof_days``, each of shape ``(len(arrival_jd), len(launch_jd))``
        so that rows index arrival date and columns index launch date -- the
        orientation Matplotlib's contour functions expect.
    """
    launch_jd = np.asarray(launch_jd, dtype=float)
    arrival_jd = np.asarray(arrival_jd, dtype=float)

    r_dep, v_dep = departure_ephem.state(launch_jd)  # (nl, 3)
    r_arr, v_arr = arrival_ephem.state(arrival_jd)  # (na, 3)

    # Broadcast to (n_arrival, n_launch, 3).
    r1 = np.broadcast_to(r_dep[None, :, :], (len(arrival_jd), len(launch_jd), 3))
    v1_body = np.broadcast_to(v_dep[None, :, :], r1.shape)
    r2 = np.broadcast_to(r_arr[:, None, :], r1.shape)
    v2_body = np.broadcast_to(v_arr[:, None, :], r1.shape)

    tof = (arrival_jd[:, None] - launch_jd[None, :]) * DAY

    v1, v2 = lambert_universal(r1, r2, tof, mu=mu, prograde=prograde)

    dv_departure = np.linalg.norm(v1 - v1_body, axis=-1)
    dv_arrival = np.linalg.norm(v2_body - v2, axis=-1)

    return {
        "c3_launch": dv_departure**2,
        "dv_departure": dv_departure,
        "dv_arrival": dv_arrival,
        "dv_total": dv_departure + dv_arrival,
        "tof_days": tof / DAY,
        "launch_jd": launch_jd,
        "arrival_jd": arrival_jd,
    }


# ---------------------------------------------------------------------------
# Calendar helpers (JD TDB <-> datetime)
# ---------------------------------------------------------------------------
def jd_to_datetime(jd):
    """Convert Julian Date to a ``datetime`` (proleptic Gregorian)."""
    from datetime import datetime, timedelta

    jd = np.asarray(jd, dtype=float)
    # JD 2451545.0 == 2000-01-01 12:00 TT
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    out = [j2000 + timedelta(days=float(d) - 2451545.0) for d in np.atleast_1d(jd)]
    return out[0] if jd.ndim == 0 else out


def datetime_to_jd(dt) -> float:
    """Convert a ``datetime`` to a Julian Date (proleptic Gregorian)."""
    from datetime import datetime

    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    return 2451545.0 + (dt - j2000).total_seconds() / DAY
