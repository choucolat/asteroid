"""Screen many candidate asteroids at once (no STK).

Run:
    python batch_porkchop.py
        Auto-discovers every ``horizons_*.txt`` file in the current
        directory (other than horizons_earth.txt) and treats each as an
        asteroid target.

    python batch_porkchop.py apophis.txt bennu.txt didymos.txt
        Analyse exactly the listed files instead of auto-discovering.

For every asteroid ephemeris supplied (same JPL Horizons vector-table format
as horizons_results.txt), this computes, over the full time span where the
asteroid's and Earth's ephemerides both have coverage:

* best intercept  -- min launch C3, Lambert targeted exactly on the asteroid
* best flyby      -- min arrival v_infinity, Lambert biased FLYBY_STANDOFF_KM
                     off the asteroid in the B-plane (see plot_porkchop.py /
                     porkchop_core.solve_flyby_porkchop for why this matters
                     -- targeting the exact position is a collision course,
                     not a flyby)
* best rendezvous -- min total mission dv (launch + arrival braking burn)

and collects the results into one comparison table plus summary bar charts,
so many bodies can be screened without hand-tuning a launch/arrival window
for each one individually. This intentionally uses a coarser, auto-derived
grid than plot_porkchop.py (which hand-picks a tight window around 2024
YR4's known encounter) -- the goal here is fast relative screening across
many targets, not a publication-quality porkchop for any single one. Once a
target looks promising here, go build a dedicated tight-window porkchop for
it the way plot_porkchop.py does for 2024 YR4.

Produces
--------
asteroid_comparison.csv
    One row per successfully analysed asteroid, every number above plus the
    launch/arrival dates and times of flight that achieve it.
asteroid_comparison.png
    Four bar charts (best C3, best flyby v_infinity, best rendezvous total
    dv, and time of flight for all three mission types side by side)
    comparing all analysed asteroids.

Time-of-flight statistics (min/median/mean/max across the analysed batch,
for each of the three mission types) are printed to the console after the
per-asteroid table.

"Reachable by next period" statistic
-------------------------------------
For the objects in PERIOD_DATA (below), also checks a discovery-latency
question: if this object were discovered now, would a mission be able to
reach it soon enough? Concretely: "period 1" runs from the object's
discovery (or from whenever our ephemeris coverage begins, if that's
later -- we can't launch before our data starts anyway) to the next real
Earth close approach after that; "period 2" runs from there to the
following close approach. This checks whether a launch-after-reference /
arrival-by-end-of-period-2 transfer exists with C3 under --max-c3. Close
approach dates come from JPL's CAD tool, not from the Lambert grid --
they're real physical encounters, not a mission-design choice. If an
object has fewer than two close approaches inside its downloaded
ephemeris span, period 2 is capped at the ephemeris's own end instead of
a real second approach (flagged in the output); if it has none at all,
the statistic is reported as unavailable rather than guessed at.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from horizons_to_stk import parse_horizons_target_name
from porkchop_core import (
    SunCentredEphemeris,
    datetime_to_jd,
    flyby_offset_closest_approach,
    injection_dv,
    jd_to_datetime,
    launch_asymptote_radec,
    launch_reachable,
    solve_flyby_porkchop,
    solve_porkchop,
)

DAY = 86400.0

# Named parking-orbit shortcuts: (altitude_km, inclination_deg). LEO only --
# escape injection is always cheaper from a low orbit (Oberth effect), so
# MEO/GEO parking orbits aren't offered here; see the parking-orbit
# conversation for the reasoning. --altitude-km/--inclination-deg override
# these directly if you want a specific combination not listed.
PARKING_ORBITS: dict[str, tuple[float, float]] = {
    "leo-equatorial": (200.0, 28.5),   # Cape Canaveral-like, no plane change
    "leo-iss": (400.0, 51.6),          # ISS-like inclination
    "sun-sync": (700.0, 98.0),         # polar / sun-synchronous
}

# Real discovery dates (SBDB "discovery.date", or "orbit.first_obs" where no
# formal discovery-credit record exists) and real Earth close-approach dates
# (JPL CAD tool, dist-max 0.5-1 AU) for the 10 candidates fetched earlier in
# this conversation. Keyed by bare designation -- see designation_key().
# Add an entry here for any new candidate you want this statistic for.
PERIOD_DATA: dict[str, dict] = {
    "2024 YR4":  {"discovery": "2024-12-25", "approaches": ["2024-12-25", "2028-12-17"]},
    "99942":     {"discovery": "2004-06-19", "approaches": ["2027-12-29", "2028-09-12", "2029-04-13", "2029-11-26"]},
    "101955":    {"discovery": "1999-09-11", "approaches": ["2030-06-21", "2031-02-18"]},
    "162173":    {"discovery": "1999-05-10", "approaches": ["2033-12-21"]},
    "25143":     {"discovery": "1998-09-26", "approaches": ["2033-03-23"]},
    "65803":     {"discovery": "1996-04-11", "approaches": []},
    "2011 AA37": {"discovery": "2011-01-13", "approaches": ["2026-08-10"]},
    "2012 EC":   {"discovery": "2012-03-01", "approaches": ["2028-01-10"]},
    "2013 WA44": {"discovery": "2006-05-07", "approaches": ["2029-01-14", "2029-06-13"]},
    "2001 CQ36": {"discovery": "2001-02-13", "approaches": ["2031-01-31", "2031-09-18"]},
}


def designation_key(fullname: str) -> str:
    """Extract a bare designation from a Horizons target string, to look it
    up in PERIOD_DATA regardless of exact formatting, e.g.
    "99942 Apophis (2004 MN4)" -> "99942", "(2024 YR4)" -> "2024 YR4"."""
    s = fullname.strip()
    m = re.match(r"^\(([^)]+)\)$", s)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(\d+)\b", s)
    if m:
        return m.group(1)
    return s


def _parse_iso_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d")


def next_period_reachability(
    name: str,
    launch_jd: np.ndarray,
    arrival_jd: np.ndarray,
    c3: np.ndarray,
    tof: np.ndarray,
    jd_lo: float,
    jd_hi: float,
    max_c3: float,
) -> dict:
    """See the "Reachable by next period" note in the module docstring."""
    meta = PERIOD_DATA.get(designation_key(name))
    if meta is None:
        return {"available": False, "note": "no discovery/close-approach data on file"}

    discovery_jd = datetime_to_jd(_parse_iso_date(meta["discovery"]))
    reference_jd = max(discovery_jd, jd_lo)
    approaches_jd = sorted(
        datetime_to_jd(_parse_iso_date(d)) for d in meta["approaches"]
    )
    approaches_jd = [a for a in approaches_jd if a > reference_jd]

    if not approaches_jd:
        return {
            "available": False,
            "note": "no Earth close approach found within this ephemeris's coverage",
            "reference_date": jd_to_datetime(reference_jd),
        }

    period1_end_jd = approaches_jd[0]
    if len(approaches_jd) >= 2:
        period2_end_jd = approaches_jd[1]
        capped = False
    else:
        period2_end_jd = jd_hi
        capped = True

    mask = (
        np.isfinite(c3)
        & (tof > 0)
        & (launch_jd[None, :] >= reference_jd)
        & (arrival_jd[:, None] <= period2_end_jd)
    )
    reachable = bool(np.any(mask & (c3 <= max_c3)))
    best_c3 = float(np.min(np.where(mask, c3, np.inf))) if np.any(mask) else float("nan")

    return {
        "available": True,
        "reference_date": jd_to_datetime(reference_jd),
        "period1_end": jd_to_datetime(period1_end_jd),
        "period2_end": jd_to_datetime(period2_end_jd),
        "period2_capped_by_data": capped,
        "reachable": reachable,
        "best_c3_in_window": best_c3,
    }

def parking_orbit_analysis(
    v_inf_departure: np.ndarray,
    c3: np.ndarray,
    solved: np.ndarray,
    launch_jd: np.ndarray,
    arrival_jd: np.ndarray,
    i_int: tuple,
    altitude_km: float,
    inclination_deg: float,
) -> dict:
    """Injection delta-v and launch-site reachability for a given LEO
    parking orbit, both at the unconstrained min-C3 point (``i_int``, the
    same point the "best intercept" figures elsewhere report) and at the
    best point that's actually reachable from this parking-orbit
    inclination without an extra plane-change burn -- these can differ,
    since the cheapest C3 solution isn't necessarily launchable from a given
    site/inclination on that date. See porkchop_core.injection_dv /
    launch_asymptote_radec for the underlying physics.
    """
    _, dla = launch_asymptote_radec(v_inf_departure)
    reachable = launch_reachable(dla, inclination_deg)
    inj_dv = injection_dv(c3, altitude_km)

    out = {
        "altitude_km": altitude_km,
        "inclination_deg": inclination_deg,
        "injection_dv_at_best_c3": float(inj_dv[i_int]),
        "dla_at_best_c3": float(dla[i_int]),
        "reachable_at_best_c3": bool(reachable[i_int]),
        "best_reachable_available": False,
    }

    mask = solved & reachable
    if np.any(mask):
        i_park = np.unravel_index(np.argmin(np.where(mask, inj_dv, np.inf)), inj_dv.shape)
        out.update(
            best_reachable_available=True,
            best_reachable_launch=jd_to_datetime(launch_jd[i_park[1]]),
            best_reachable_arrival=jd_to_datetime(arrival_jd[i_park[0]]),
            best_reachable_c3=float(c3[i_park]),
            best_reachable_injection_dv=float(inj_dv[i_park]),
        )

    return out


#: Bar colours, reused from plot_porkchop.py's palette choices.
COL_C3 = "#1f2937"
COL_VINF = "#c2610c"
COL_DVT = "#0f766e"
COL_TOF_INT = "#1f2937"
COL_TOF_FLY = "#c2610c"
COL_TOF_RDV = "#0f766e"


def find_asteroid_files(explicit: list[str], earth_name: str) -> list[str]:
    """Explicit file list if given, else auto-discover ``horizons_*.txt``."""
    if explicit:
        return explicit
    candidates = sorted(glob.glob("horizons_*.txt")) + sorted(glob.glob("*.txt"))
    seen: dict[str, None] = {}
    for path in candidates:
        base = os.path.basename(path)
        if base == earth_name or base in seen:
            continue
        seen[base] = None
    return list(seen.keys())


def analyze_asteroid(
    name: str,
    ast: SunCentredEphemeris,
    earth: SunCentredEphemeris,
    standoff_km: float,
    step_days: float,
    min_tof_days: float,
    max_tof_days: float,
    max_c3_next_period: float,
    park_altitude_km: float,
    park_inclination_deg: float,
) -> dict | None:
    """Run intercept / flyby / rendezvous solves over an auto-derived grid.

    Returns ``None`` (with a printed reason) if the ephemeris overlap is too
    short to fit a single transfer, or if nothing in the grid converges.
    """
    jd_lo = max(earth.jd_start, ast.jd_start)
    jd_hi = min(earth.jd_stop, ast.jd_stop)
    span_days = jd_hi - jd_lo
    if span_days < min_tof_days + 2 * step_days:
        print(
            f"  skip {name}: ephemeris overlap with Earth is only "
            f"{span_days:.0f} d, too short for a {min_tof_days:.0f} d "
            "minimum transfer"
        )
        return None

    launch_jd = np.arange(jd_lo, jd_hi - min_tof_days, step_days)
    arrival_jd = np.arange(jd_lo + min_tof_days, jd_hi, step_days)
    if len(launch_jd) < 2 or len(arrival_jd) < 2:
        print(f"  skip {name}: grid too coarse for the available overlap")
        return None

    res = solve_porkchop(earth, ast, launch_jd, arrival_jd)
    c3, dva, tof, dvt = (
        res["c3_launch"],
        res["dv_arrival"],
        res["tof_days"],
        res["dv_total"],
    )
    solved = np.isfinite(c3) & (tof > 0) & (tof <= max_tof_days)
    if not np.any(solved):
        print(f"  skip {name}: no feasible transfer converged in the grid")
        return None

    i_int = np.unravel_index(np.argmin(np.where(solved, c3, np.inf)), c3.shape)
    i_rdv = np.unravel_index(np.argmin(np.where(solved, dvt, np.inf)), dvt.shape)

    fres = solve_flyby_porkchop(earth, ast, launch_jd, arrival_jd, standoff_km)
    fc3, fdva, ftof = fres["c3_launch"], fres["dv_arrival"], fres["tof_days"]
    fsolved = np.isfinite(fc3) & (ftof > 0) & (ftof <= max_tof_days)

    out = {
        "name": name,
        "n_ephem_points": len(ast.jd),
        "overlap_days": span_days,
        "int_launch": jd_to_datetime(launch_jd[i_int[1]]),
        "int_arrival": jd_to_datetime(arrival_jd[i_int[0]]),
        "int_c3": float(c3[i_int]),
        "int_tof_days": float(tof[i_int]),
        "int_vinf_arrival": float(dva[i_int]),
        "rdv_launch": jd_to_datetime(launch_jd[i_rdv[1]]),
        "rdv_arrival": jd_to_datetime(arrival_jd[i_rdv[0]]),
        "rdv_total_dv": float(dvt[i_rdv]),
        "rdv_tof_days": float(tof[i_rdv]),
        "fly_launch": None,
        "fly_arrival": None,
        "fly_vinf": np.nan,
        "fly_tof_days": np.nan,
        "fly_standoff_km": standoff_km,
        "fly_achieved_km": np.nan,
    }

    if np.any(fsolved):
        i_fly = np.unravel_index(np.argmin(np.where(fsolved, fdva, np.inf)), fdva.shape)
        achieved = flyby_offset_closest_approach(
            earth, ast, launch_jd[i_fly[1]], arrival_jd[i_fly[0]], standoff_km
        )
        out.update(
            fly_launch=jd_to_datetime(launch_jd[i_fly[1]]),
            fly_arrival=jd_to_datetime(arrival_jd[i_fly[0]]),
            fly_vinf=float(fdva[i_fly]),
            fly_tof_days=float(ftof[i_fly]),
            fly_achieved_km=achieved,
        )
    else:
        print(f"  note {name}: no flyby transfer converged (intercept/rendezvous still valid)")

    period = next_period_reachability(
        name, launch_jd, arrival_jd, c3, tof, jd_lo, jd_hi, max_c3_next_period
    )
    out["next_period"] = period

    out["parking_orbit"] = parking_orbit_analysis(
        res["v_inf_departure"], c3, solved, launch_jd, arrival_jd, i_int,
        park_altitude_km, park_inclination_deg,
    )

    return out


def write_csv(results: list[dict], path: str) -> None:
    fields = [
        "name", "n_ephem_points", "overlap_days",
        "int_launch", "int_arrival", "int_c3", "int_tof_days", "int_vinf_arrival",
        "fly_launch", "fly_arrival", "fly_vinf", "fly_tof_days",
        "fly_standoff_km", "fly_achieved_km",
        "rdv_launch", "rdv_arrival", "rdv_total_dv", "rdv_tof_days",
        "next_period_available", "next_period_reference", "next_period1_end",
        "next_period2_end", "next_period2_capped_by_data",
        "next_period_reachable", "next_period_best_c3",
        "park_altitude_km", "park_inclination_deg",
        "park_injection_dv_at_best_c3", "park_dla_at_best_c3", "park_reachable_at_best_c3",
        "park_best_reachable_available", "park_best_reachable_launch",
        "park_best_reachable_arrival", "park_best_reachable_c3",
        "park_best_reachable_injection_dv",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            p = r.get("next_period", {})
            po = r.get("parking_orbit", {})
            flat = dict(r)
            flat["next_period_available"] = p.get("available", False)
            flat["next_period_reference"] = p.get("reference_date")
            flat["next_period1_end"] = p.get("period1_end")
            flat["next_period2_end"] = p.get("period2_end")
            flat["next_period2_capped_by_data"] = p.get("period2_capped_by_data", "")
            flat["next_period_reachable"] = p.get("reachable", "")
            flat["next_period_best_c3"] = p.get("best_c3_in_window", np.nan)
            flat["park_altitude_km"] = po.get("altitude_km")
            flat["park_inclination_deg"] = po.get("inclination_deg")
            flat["park_injection_dv_at_best_c3"] = po.get("injection_dv_at_best_c3", np.nan)
            flat["park_dla_at_best_c3"] = po.get("dla_at_best_c3", np.nan)
            flat["park_reachable_at_best_c3"] = po.get("reachable_at_best_c3", "")
            flat["park_best_reachable_available"] = po.get("best_reachable_available", False)
            flat["park_best_reachable_launch"] = po.get("best_reachable_launch")
            flat["park_best_reachable_arrival"] = po.get("best_reachable_arrival")
            flat["park_best_reachable_c3"] = po.get("best_reachable_c3", np.nan)
            flat["park_best_reachable_injection_dv"] = po.get("best_reachable_injection_dv", np.nan)
            row = []
            for k in fields:
                v = flat.get(k)
                if isinstance(v, dt.datetime):
                    v = v.strftime("%Y-%m-%d")
                elif isinstance(v, float):
                    v = f"{v:.4f}" if np.isfinite(v) else ""
                row.append(v)
            w.writerow(row)


def print_table(results: list[dict]) -> None:
    hdr = (
        f"{'asteroid':<20}{'C3 [km2/s2]':>13}{'int TOF':>9}"
        f"{'flyby v_inf':>13}{'fly TOF':>9}"
        f"{'rdv dv':>10}{'rdv TOF':>9}{'span [d]':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        vinf = f"{r['fly_vinf']:.2f}" if np.isfinite(r["fly_vinf"]) else "n/a"
        fly_tof = f"{r['fly_tof_days']:.0f}" if np.isfinite(r["fly_tof_days"]) else "n/a"
        print(
            f"{r['name']:<20}{r['int_c3']:>13.3f}{r['int_tof_days']:>9.0f}"
            f"{vinf:>13}{fly_tof:>9}"
            f"{r['rdv_total_dv']:>10.2f}{r['rdv_tof_days']:>9.0f}"
            f"{r['overlap_days']:>10.0f}"
        )


def print_tof_stats(results: list[dict]) -> None:
    """Print min/median/mean/max time-of-flight across the analysed batch,
    for each of the three mission types, so a single number in the table
    can be put in context of the spread across all candidates."""
    groups = [
        ("intercept", [r["int_tof_days"] for r in results]),
        ("flyby", [r["fly_tof_days"] for r in results if np.isfinite(r["fly_tof_days"])]),
        ("rendezvous", [r["rdv_tof_days"] for r in results]),
    ]
    print("\nTime-of-flight statistics across the batch (days):")
    hdr = f"{'mission type':<14}{'n':>4}{'min':>8}{'median':>8}{'mean':>8}{'max':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, values in groups:
        if not values:
            print(f"{label:<14}{'0':>4}{'--':>8}{'--':>8}{'--':>8}{'--':>8}")
            continue
        arr = np.asarray(values, dtype=float)
        print(
            f"{label:<14}{len(arr):>4}{arr.min():>8.0f}{np.median(arr):>8.0f}"
            f"{arr.mean():>8.0f}{arr.max():>8.0f}"
        )


def print_next_period_stats(results: list[dict], max_c3: float) -> None:
    print(
        f"\nReachable by next period (launch after discovery, arrive by the "
        f"second Earth close approach after that, C3 <= {max_c3:g} km^2/s^2):"
    )
    hdr = f"{'asteroid':<20}{'reference':>12}{'period 1 end':>14}{'period 2 end':>14}{'reachable':>11}"
    print(hdr)
    print("-" * len(hdr))
    n_available = n_reachable = 0
    for r in results:
        p = r["next_period"]
        if not p.get("available"):
            note = p.get("note", "n/a")
            print(f"{r['name']:<20}{'--':>12}{'--':>14}{'--':>14}  {note}")
            continue
        n_available += 1
        reachable = p["reachable"]
        n_reachable += int(reachable)
        p2_str = p["period2_end"].strftime("%Y-%m-%d") + ("*" if p["period2_capped_by_data"] else "")
        print(
            f"{r['name']:<20}"
            f"{p['reference_date'].strftime('%Y-%m-%d'):>12}"
            f"{p['period1_end'].strftime('%Y-%m-%d'):>14}"
            f"{p2_str:>14}"
            f"{'yes' if reachable else 'no':>11}"
        )
    if n_available:
        print(f"\n{n_reachable} of {n_available} evaluable candidates reachable by their next period.")
    print("* period 2 end is capped at the ephemeris's coverage end -- no real second close approach on file within it.")


def print_parking_orbit_stats(results: list[dict]) -> None:
    if not results:
        return
    po0 = results[0]["parking_orbit"]
    print(
        f"\nParking orbit: {po0['altitude_km']:g} km altitude, "
        f"{po0['inclination_deg']:g} deg inclination "
        "(injection dv for a single tangential burn onto each transfer's "
        "C3; 'direct?' = whether that C3 point's departure asymptote "
        "declination is within the parking orbit's inclination, i.e. "
        "reachable without a separate plane-change burn):"
    )
    hdr = (
        f"{'asteroid':<20}{'inj dv @ best C3':>17}{'DLA':>8}{'direct?':>9}"
        f"{'best direct inj dv':>20}"
    )
    print(hdr)
    print("-" * len(hdr))
    n_direct = 0
    for r in results:
        po = r["parking_orbit"]
        n_direct += int(po["reachable_at_best_c3"])
        if po["best_reachable_available"]:
            best = f"{po['best_reachable_injection_dv']:.3f} km/s"
        else:
            best = "none in grid"
        print(
            f"{r['name']:<20}{po['injection_dv_at_best_c3']:>14.3f} km/s"
            f"{po['dla_at_best_c3']:>7.1f}°"
            f"{'yes' if po['reachable_at_best_c3'] else 'no':>9}"
            f"{best:>20}"
        )
    print(
        f"\n{n_direct} of {len(results)} candidates' cheapest-C3 transfer is "
        "directly launchable from this parking orbit without a plane-change "
        "burn; for the rest, see the 'best direct inj dv' column for the "
        "cheapest transfer that actually is."
    )


def plot_comparison(results: list[dict], out_path: str) -> None:
    names = [r["name"] for r in results]
    c3 = [r["int_c3"] for r in results]
    vinf = [r["fly_vinf"] for r in results]
    dvt = [r["rdv_total_dv"] for r in results]

    width = 4.4 * max(len(names), 1) ** 0.4 + 6
    fig, axes = plt.subplots(2, 2, figsize=(width, 10.5))
    bar_specs = [
        (axes[0, 0], c3, "Best launch $C_3$  [km$^2$/s$^2$]", COL_C3),
        (axes[0, 1], vinf, "Best flyby arrival $v_\\infty$  [km/s]", COL_VINF),
        (axes[1, 0], dvt, "Best rendezvous total $\\Delta v$  [km/s]", COL_DVT),
    ]
    for ax, values, label, color in bar_specs:
        order = np.argsort([v if np.isfinite(v) else np.inf for v in values])
        sorted_names = [names[i] for i in order]
        sorted_values = [values[i] for i in order]
        finite = [v if np.isfinite(v) else 0.0 for v in sorted_values]
        ax.bar(sorted_names, finite, color=color)
        for i, v in enumerate(sorted_values):
            label_text = f"{v:.2f}" if np.isfinite(v) else "n/a"
            ax.text(i, (v if np.isfinite(v) else 0.0), label_text, ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Fourth panel: grouped bars of time of flight for all three mission
    # types, at each asteroid's own optimum point for that mission type
    # (so these TOFs correspond to the same launch/arrival dates shown in
    # the other three panels, not to a shared or "best-TOF" trajectory).
    ax = axes[1, 1]
    int_tof = [r["int_tof_days"] for r in results]
    fly_tof = [r["fly_tof_days"] for r in results]
    rdv_tof = [r["rdv_tof_days"] for r in results]
    x = np.arange(len(names))
    bw = 0.26
    ax.bar(x - bw, int_tof, width=bw, color=COL_TOF_INT, label="Intercept")
    fly_tof_plot = [v if np.isfinite(v) else 0.0 for v in fly_tof]
    ax.bar(x, fly_tof_plot, width=bw, color=COL_TOF_FLY, label="Flyby")
    ax.bar(x + bw, rdv_tof, width=bw, color=COL_TOF_RDV, label="Rendezvous")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, fontsize=8)
    ax.set_ylabel("Time of flight  [days]", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Earth → candidate asteroids — mission-design screening", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mpl.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asteroids", nargs="*", help="Horizons vector-table .txt files to analyse")
    parser.add_argument("--earth", default="horizons_earth.txt")
    parser.add_argument("--standoff-km", type=float, default=500.0)
    parser.add_argument("--step-days", type=float, default=5.0)
    parser.add_argument("--min-tof-days", type=float, default=60.0)
    parser.add_argument("--max-tof-days", type=float, default=900.0)
    parser.add_argument(
        "--max-c3-next-period", type=float, default=30.0,
        help="C3 threshold (km^2/s^2) for the 'reachable by next period' statistic",
    )
    parser.add_argument(
        "--parking-orbit", choices=sorted(PARKING_ORBITS), default=None,
        help=f"named parking orbit preset: {', '.join(sorted(PARKING_ORBITS))} "
             "(overrides --altitude-km/--inclination-deg)",
    )
    parser.add_argument("--altitude-km", type=float, default=400.0)
    parser.add_argument("--inclination-deg", type=float, default=28.5)
    parser.add_argument("--out-prefix", default="asteroid_comparison")
    args = parser.parse_args()

    if args.parking_orbit:
        args.altitude_km, args.inclination_deg = PARKING_ORBITS[args.parking_orbit]

    earth = SunCentredEphemeris(args.earth, "Earth")
    print(earth)

    files = find_asteroid_files(args.asteroids, os.path.basename(args.earth))
    if not files:
        raise SystemExit(
            "No asteroid ephemeris files found. Pass file paths explicitly, "
            "or drop horizons_*.txt files into the working directory."
        )
    print(f"Found {len(files)} candidate file(s): {', '.join(files)}\n")

    results = []
    for path in files:
        label = parse_horizons_target_name(path) or os.path.splitext(os.path.basename(path))[0]
        try:
            ast = SunCentredEphemeris(path, label)
        except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the batch
            print(f"  skip {path}: could not parse as a Horizons vector table ({exc})")
            continue
        print(f"Analysing {label} ({path})...")
        r = analyze_asteroid(
            label, ast, earth,
            standoff_km=args.standoff_km,
            step_days=args.step_days,
            min_tof_days=args.min_tof_days,
            max_tof_days=args.max_tof_days,
            max_c3_next_period=args.max_c3_next_period,
            park_altitude_km=args.altitude_km,
            park_inclination_deg=args.inclination_deg,
        )
        if r is not None:
            results.append(r)

    if not results:
        raise SystemExit("\nNo asteroid produced a feasible transfer -- nothing to report.")

    print()
    print_table(results)
    print_tof_stats(results)
    print_next_period_stats(results, args.max_c3_next_period)
    print_parking_orbit_stats(results)

    csv_path = f"{args.out_prefix}.csv"
    png_path = f"{args.out_prefix}.png"
    write_csv(results, csv_path)
    plot_comparison(results, png_path)
    print(f"\nWrote {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
