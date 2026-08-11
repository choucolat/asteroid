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
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from horizons_to_stk import parse_horizons_target_name
from porkchop_core import (
    SunCentredEphemeris,
    flyby_offset_closest_approach,
    jd_to_datetime,
    solve_flyby_porkchop,
    solve_porkchop,
)

DAY = 86400.0

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

    return out


def write_csv(results: list[dict], path: str) -> None:
    fields = [
        "name", "n_ephem_points", "overlap_days",
        "int_launch", "int_arrival", "int_c3", "int_tof_days", "int_vinf_arrival",
        "fly_launch", "fly_arrival", "fly_vinf", "fly_tof_days",
        "fly_standoff_km", "fly_achieved_km",
        "rdv_launch", "rdv_arrival", "rdv_total_dv", "rdv_tof_days",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            row = []
            for k in fields:
                v = r[k]
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
    parser.add_argument("--out-prefix", default="asteroid_comparison")
    args = parser.parse_args()

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
        )
        if r is not None:
            results.append(r)

    if not results:
        raise SystemExit("\nNo asteroid produced a feasible transfer -- nothing to report.")

    print()
    print_table(results)
    print_tof_stats(results)

    csv_path = f"{args.out_prefix}.csv"
    png_path = f"{args.out_prefix}.png"
    write_csv(results, csv_path)
    plot_comparison(results, png_path)
    print(f"\nWrote {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
