"""Generate the Earth -> 2024 YR4 porkchop plots without STK.

Run:
    python plot_porkchop.py

Produces
--------
porkchop_yr4_2028_intercept.png
    The mission-relevant window: launch mid-2027 -> end-2028, arrival up to the
    end of the supplied ephemeris.  Filled contours are launch characteristic
    energy C3; overlaid are arrival dv (rendezvous burn) and time of flight.
    Framed as a kinetic-impactor case: the Lambert targets the asteroid
    exactly, and the optimum minimises launch C3.
porkchop_yr4_2028_flyby.png
    Same launch/arrival window as the intercept figure, but the Lambert
    target point is biased FLYBY_STANDOFF_KM (500 km) off the asteroid in the
    B-plane, so the trajectory actually passes the asteroid at that distance
    instead of hitting it; no braking burn is performed. Filled contours are
    arrival v_infinity -- the relative encounter speed, which is the figure
    of merit for a reconnaissance flyby -- with launch C3 and time of flight
    overlaid. The optimum marked is the minimum v_infinity, not the minimum
    C3, and the achieved closest-approach distance is verified numerically
    (flyby_offset_closest_approach) rather than assumed.
porkchop_yr4_2028_rendezvous.png
    Same window, but the filled contours are total mission dv
    (dv_departure + dv_arrival) -- the figure of merit if the spacecraft has to
    stop at the asteroid rather than fly past it.
porkchop_yr4_overview.png
    The full span covered by horizons_results.txt, showing both intercept
    opportunities (Dec 2024, already past, and Dec 2028).
porkchop_yr4_grids.npz
    The raw grids, for further analysis or for cross-checking the STK run.

All modelling assumptions match asteroid_porkchop_yr4.py -- see porkchop_core.
"""

from __future__ import annotations

import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex

from porkchop_core import (
    SunCentredEphemeris,
    datetime_to_jd,
    flyby_closest_approach,
    flyby_offset_closest_approach,
    jd_to_datetime,
    solve_flyby_porkchop,
    solve_porkchop,
)

#: Target flyby standoff distance -- see the flyby figure below.
FLYBY_STANDOFF_KM = 500.0

# --- palette -----------------------------------------------------------------
# Sequential single-hue ramp for the magnitude field, resampled to equal OKLCH
# lightness steps so adjacent bands stay distinguishable; categorical inks for
# the three overlay families, checked for CVD separation.
INK_C3 = "#1f2937"  # slate  - characteristic energy contour lines
INK_DV = "#c2610c"  # orange - arrival dv
INK_TOF = "#a21caf"  # magenta - time of flight
HALO = [patheffects.withStroke(linewidth=2.6, foreground="white")]


def equal_lightness_ramp(name: str, n: int, lo: float = 0.30, hi: float = 1.0):
    """Sample a Matplotlib colormap at equally spaced perceptual lightness."""

    def oklab_l(rgb):
        def s2lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = (s2lin(c) for c in rgb[:3])
        l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_

    cmap = mpl.colormaps[name]
    xs = np.linspace(lo, hi, 512)
    ls = np.array([oklab_l(cmap(x)) for x in xs])
    targets = np.linspace(ls[0], ls[-1], n)
    # ls is monotonically decreasing, so interpolate on the reversed arrays.
    picks = np.interp(targets[::-1], ls[::-1], xs[::-1])[::-1]
    return [to_hex(cmap(x)) for x in picks]


# --- shared plotting ---------------------------------------------------------
def draw_porkchop(
    ax,
    launch_dates,
    arrival_dates,
    field,
    field_levels,
    field_label,
    dv_arrival,
    dv_levels,
    tof_days,
    tof_levels,
    field_line_fmt="%g",
    dv_label="Arrival $\\Delta v$",
    dv_fmt="%g km/s",
    field_label_levels=None,
):
    """Draw one porkchop panel and return the filled-contour artist."""
    colors = equal_lightness_ramp("Blues", len(field_levels) - 1)
    cmap = ListedColormap(colors)
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(field_levels, cmap.N)

    filled = ax.contourf(
        launch_dates,
        arrival_dates,
        field,
        levels=field_levels,
        cmap=cmap,
        norm=norm,
        extend="max",
    )

    lines = ax.contour(
        launch_dates,
        arrival_dates,
        field,
        levels=field_levels,
        colors=INK_C3,
        linewidths=0.9,
    )
    labels = ax.clabel(
        lines,
        levels=field_label_levels,
        inline=True,
        fmt=field_line_fmt,
        fontsize=8,
        colors=INK_C3,
        inline_spacing=6,
    )
    plt.setp(labels, path_effects=HALO)

    if dv_levels is not None and len(dv_levels):
        dv_lines = ax.contour(
            launch_dates,
            arrival_dates,
            dv_arrival,
            levels=dv_levels,
            colors=INK_DV,
            linewidths=1.8,
        )
        dv_labels = ax.clabel(
            dv_lines, inline=True, fmt=dv_fmt, fontsize=9, colors=INK_DV
        )
        plt.setp(dv_labels, path_effects=HALO)
        dv_lines.set(
            path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")]
        )

    if tof_levels is not None and len(tof_levels):
        tof_lines = ax.contour(
            launch_dates,
            arrival_dates,
            tof_days,
            levels=tof_levels,
            colors=INK_TOF,
            linewidths=1.8,
            linestyles="dashed",
        )
        tof_labels = ax.clabel(
            tof_lines, inline=True, fmt="%g d", fontsize=9, colors=INK_TOF,
            use_clabeltext=True,
        )
        plt.setp(tof_labels, path_effects=HALO)
        tof_lines.set(
            path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")]
        )

    ax.set_xlabel("LAUNCH DATE")
    ax.set_ylabel("ARRIVAL DATE")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=90)

    handles = [mlines.Line2D([], [], color=INK_C3, lw=0.9, label=field_label)]
    if dv_levels is not None and len(dv_levels):
        handles.append(mlines.Line2D([], [], color=INK_DV, lw=1.8, label=dv_label))
    if tof_levels is not None and len(tof_levels):
        handles.append(
            mlines.Line2D([], [], color=INK_TOF, lw=1.8, ls="--", label="Time of flight")
        )
    ax.legend(handles=handles, loc="upper left", framealpha=0.92, fontsize=9)
    return filled


def mark(ax, x, y):
    """Put a marker on the optimum."""
    ax.plot(
        [x], [y], marker="o", ms=9, mfc="white", mec=INK_C3, mew=2.0, zorder=8,
        clip_on=False,
    )


def callout(ax, text, xy=(0.985, 0.03), ha="right", va="bottom"):
    """Drop a summary box into the infeasible (arrival before launch) corner."""
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9.5,
        color=INK_C3,
        zorder=9,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor=INK_C3,
            linewidth=1.0,
            alpha=0.95,
        ),
    )


def main() -> None:
    # Only when run as a script -- importing the helpers (as
    # asteroid_porkchop_yr4.py does) must not force a headless backend.
    mpl.use("Agg")

    earth = SunCentredEphemeris("horizons_earth.txt", "Earth")
    yr4 = SunCentredEphemeris("horizons_results.txt", "2024 YR4")
    print(earth)
    print(yr4)

    jd = lambda *a: datetime_to_jd(dt.datetime(*a))  # noqa: E731

    c3_levels = np.linspace(0, 45, 10)

    # ------------------------------------------------- figure 1: intercept ----
    # 2024 YR4 passes 0.054 AU from Earth on 2028-12-18; that encounter is the
    # only accessible opportunity inside the supplied ephemeris, and it sets the
    # framing of this panel.
    launch_jd = np.arange(jd(2027, 8, 1), jd(2028, 11, 25), 1.0)
    arrival_jd = np.arange(jd(2028, 7, 15), jd(2029, 3, 15), 1.0)
    res = solve_porkchop(earth, yr4, launch_jd, arrival_jd)
    launch_dates, arrival_dates = jd_to_datetime(launch_jd), jd_to_datetime(arrival_jd)
    c3, dva, tof, tot = (
        res["c3_launch"],
        res["dv_arrival"],
        res["tof_days"],
        res["dv_total"],
    )
    solved = np.isfinite(c3) & (tof > 0)
    print(f"intercept window grid {c3.shape}, {solved.sum()} solved transfers")
    i_int = np.unravel_index(
        np.argmin(np.where(solved & (tof > 60), c3, np.inf)), c3.shape
    )
    intercept_closest_km = flyby_closest_approach(
        earth,
        yr4,
        launch_jd[i_int[1]],
        arrival_jd[i_int[0]],
    )

    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    ax.set_title(
        "EARTH → (2024 YR4)   BALLISTIC TRANSFER TRAJECTORY\n"
        "Launch characteristic energy $C_3$ — restricted two-body, "
        "zero-revolution Lambert transfer",
        fontsize=13,
        pad=14,
    )
    filled = draw_porkchop(
        ax,
        launch_dates,
        arrival_dates,
        c3,
        c3_levels,
        "$C_3$ launch",
        dva,
        [10, 12, 14, 16, 20, 25],
        tof,
        [50, 100, 150, 200, 250, 300, 350, 400, 450],
        dv_label="Arrival $v_\\infty$",
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.015, fraction=0.04, extend="max")
    cbar.set_label("Launch characteristic energy $C_3$  [km$^2$/s$^2$]")
    mark(ax, launch_dates[i_int[1]], arrival_dates[i_int[0]])
    callout(
        ax,
        f"minimum $C_3$ = {c3[i_int]:.2f} km$^2$/s$^2$\n"
        f"launch  {launch_dates[i_int[1]]:%d %b %Y}\n"
        f"arrive  {arrival_dates[i_int[0]]:%d %b %Y}\n"
        f"time of flight  {tof[i_int]:.0f} d\n"
        f"arrival $v_\\infty$  {dva[i_int]:.2f} km/s",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    fig.savefig("porkchop_yr4_2028_intercept.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------- figure 2: rendezvous -----
    rlaunch_jd = np.arange(jd(2028, 3, 1), jd(2029, 1, 20), 1.0)
    rarrival_jd = np.arange(jd(2028, 9, 1), min(jd(2029, 12, 20), yr4.jd_stop), 2.0)
    rres = solve_porkchop(earth, yr4, rlaunch_jd, rarrival_jd)
    rl, ra = jd_to_datetime(rlaunch_jd), jd_to_datetime(rarrival_jd)
    rc3, rdva, rtof, rtot = (
        rres["c3_launch"],
        rres["dv_arrival"],
        rres["tof_days"],
        rres["dv_total"],
    )
    rsolved = np.isfinite(rc3) & (rtof > 0)
    i_rdv = np.unravel_index(np.argmin(np.where(rsolved, rtot, np.inf)), rtot.shape)

    fig, ax = plt.subplots(figsize=(13.5, 10))
    ax.set_title(
        "EARTH → (2024 YR4)   RENDEZVOUS TRANSFER\n"
        "Total mission $\\Delta v = \\Delta v_{launch} + \\Delta v_{arrival}$ "
        "(the spacecraft matches the asteroid's orbit)",
        fontsize=13,
        pad=14,
    )
    tot_levels = np.array([9, 11, 13, 15, 18, 21, 25, 30, 35, 40])
    filled = draw_porkchop(
        ax,
        rl,
        ra,
        rtot,
        tot_levels,
        "Total $\\Delta v$",
        rdva,
        [1, 2, 5, 10],
        rtof,
        [100, 200, 300, 400, 500],
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.015, fraction=0.04, extend="max")
    cbar.set_label("Total mission $\\Delta v$  [km/s]")
    mark(ax, rl[i_rdv[1]], ra[i_rdv[0]])
    callout(
        ax,
        f"minimum total $\\Delta v$ = {rtot[i_rdv]:.2f} km/s\n"
        f"launch  {rl[i_rdv[1]]:%d %b %Y}\n"
        f"arrive  {ra[i_rdv[0]]:%d %b %Y}\n"
        f"$C_3$  {rc3[i_rdv]:.1f} km$^2$/s$^2$  "
        f"($\\Delta v_{{launch}}$ {rres['dv_departure'][i_rdv]:.2f} km/s)\n"
        f"arrival $\\Delta v$  {rdva[i_rdv]:.2f} km/s\n"
        f"time of flight  {rtof[i_rdv]:.0f} d",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.savefig("porkchop_yr4_2028_rendezvous.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------- figure 3: flyby -----
    # A real flyby does not target the asteroid's exact position -- that
    # trajectory is a collision course (see the previous exchange). Instead
    # the Lambert target point is biased off the asteroid by FLYBY_STANDOFF_KM
    # in the B-plane (solve_flyby_porkchop / b_plane_offset), giving a
    # trajectory that actually passes the asteroid at ~that distance rather
    # than through it. Because 2024 YR4's own gravity is negligible for a
    # body this size, the offset Lambert target IS ~the achieved closest
    # approach; flyby_offset_closest_approach numerically confirms this at
    # the optimum rather than just asserting it.
    fres = solve_flyby_porkchop(
        earth, yr4, launch_jd, arrival_jd, FLYBY_STANDOFF_KM
    )
    fc3, fdva, ftof = fres["c3_launch"], fres["dv_arrival"], fres["tof_days"]
    fsolved = np.isfinite(fc3) & (ftof > 0)
    i_fly = np.unravel_index(np.argmin(np.where(fsolved, fdva, np.inf)), fdva.shape)
    fly_closest_km = flyby_offset_closest_approach(
        earth,
        yr4,
        launch_jd[i_fly[1]],
        arrival_jd[i_fly[0]],
        FLYBY_STANDOFF_KM,
    )

    vinf_levels = np.linspace(2, 20, 10)

    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    ax.set_title(
        f"EARTH → (2024 YR4)   FLYBY TRAJECTORY ({FLYBY_STANDOFF_KM:.0f} km "
        "standoff, no braking burn)\n"
        "Arrival $v_\\infty$ — restricted two-body, zero-revolution Lambert "
        "transfer, B-plane-biased off the asteroid",
        fontsize=13,
        pad=14,
    )
    filled = draw_porkchop(
        ax,
        launch_dates,
        arrival_dates,
        fdva,
        vinf_levels,
        "Arrival $v_\\infty$",
        fc3,
        [10, 15, 20, 25, 30, 35, 40],
        ftof,
        [50, 100, 150, 200, 250, 300, 350, 400, 450],
        dv_label="Launch $C_3$",
        dv_fmt="%g km$^2$/s$^2$",
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.015, fraction=0.04, extend="max")
    cbar.set_label("Arrival $v_\\infty$ (flyby encounter speed)  [km/s]")
    mark(ax, launch_dates[i_fly[1]], arrival_dates[i_fly[0]])
    callout(
        ax,
        f"minimum flyby $v_\\infty$ = {fdva[i_fly]:.2f} km/s\n"
        f"launch  {launch_dates[i_fly[1]]:%d %b %Y}\n"
        f"arrive  {arrival_dates[i_fly[0]]:%d %b %Y}\n"
        f"time of flight  {ftof[i_fly]:.0f} d\n"
        f"launch $C_3$  {fc3[i_fly]:.2f} km$^2$/s$^2$\n"
        f"target standoff  {FLYBY_STANDOFF_KM:.0f} km "
        f"(achieved: {fly_closest_km:.0f} km)\n"
        f"mission $\\Delta v$ (launch only, no braking burn)  "
        f"{np.sqrt(fc3[i_fly]):.2f} km/s",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    fig.savefig("porkchop_yr4_2028_flyby.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------- figure 4: overview ----
    olaunch_jd = np.arange(yr4.jd_start, yr4.jd_stop - 20, 5.0)
    oarrival_jd = np.arange(yr4.jd_start + 20, yr4.jd_stop, 5.0)
    ores = solve_porkchop(earth, yr4, olaunch_jd, oarrival_jd)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ax.set_title(
        "EARTH → (2024 YR4)   full span of horizons_results.txt\n"
        "Launch $C_3$ — the two low-energy islands are the Dec 2024 and "
        "Dec 2028 Earth close approaches",
        fontsize=13,
        pad=14,
    )
    # A quasi-logarithmic level set: the C3 field spans four orders of magnitude
    # over the full span, so uniform levels would show nothing but the islands.
    overview_levels = np.array([0, 5, 10, 20, 40, 80, 160, 320, 640, 1280])
    filled = draw_porkchop(
        ax,
        jd_to_datetime(olaunch_jd),
        jd_to_datetime(oarrival_jd),
        ores["c3_launch"],
        overview_levels,
        "$C_3$ launch",
        None,
        None,
        ores["tof_days"],
        [400, 800, 1200],
        field_label_levels=[5, 20, 80, 320],
    )
    cbar = fig.colorbar(
        filled, ax=ax, pad=0.02, fraction=0.045, extend="max", spacing="uniform"
    )
    cbar.set_label("Launch characteristic energy $C_3$  [km$^2$/s$^2$]")

    for jd_l, jd_a, text in [
        (jd(2024, 3, 1), jd(2024, 12, 26), "Dec 2024 encounter\n(0.0058 AU — already past)"),
        (jd(2028, 4, 7), jd(2028, 12, 21), "Dec 2028 encounter\n(0.054 AU)"),
    ]:
        x, y = jd_to_datetime(jd_l), jd_to_datetime(jd_a)
        ax.plot([x], [y], marker="o", ms=9, mfc="white", mec=INK_C3, mew=2.0, zorder=8)
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(16, -34),
            fontsize=9,
            color=INK_C3,
            zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=INK_C3,
                linewidth=1.0,
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="-", color=INK_C3, lw=1.0),
        )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.savefig("porkchop_yr4_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- export -----
    np.savez_compressed(
        "porkchop_yr4_grids.npz",
        intercept_launch_jd=launch_jd,
        intercept_arrival_jd=arrival_jd,
        intercept_c3_launch=c3,
        intercept_dv_departure=res["dv_departure"],
        intercept_dv_arrival=dva,
        intercept_dv_total=tot,
        intercept_tof_days=tof,
        rendezvous_launch_jd=rlaunch_jd,
        rendezvous_arrival_jd=rarrival_jd,
        rendezvous_c3_launch=rc3,
        rendezvous_dv_departure=rres["dv_departure"],
        rendezvous_dv_arrival=rdva,
        rendezvous_dv_total=rtot,
        rendezvous_tof_days=rtof,
        flyby_standoff_km=FLYBY_STANDOFF_KM,
        flyby_launch_jd=launch_jd,
        flyby_arrival_jd=arrival_jd,
        flyby_c3_launch=fc3,
        flyby_dv_arrival_vinf=fdva,
        flyby_tof_days=ftof,
    )

    print("\nBest kinetic-impactor intercept (targets the asteroid exactly, min C3):")
    print(
        f"  launch {launch_dates[i_int[1]]:%Y-%m-%d}  arrive {arrival_dates[i_int[0]]:%Y-%m-%d}"
        f"  C3 {c3[i_int]:.3f} km^2/s^2  TOF {tof[i_int]:.0f} d"
        f"  arrival v_inf {dva[i_int]:.2f} km/s"
        f"  min heliocentric separation along arc {intercept_closest_km:.0f} km"
    )
    print(
        f"Best flyby ({FLYBY_STANDOFF_KM:.0f} km B-plane standoff, "
        "no braking burn, min v_inf):"
    )
    print(
        f"  launch {launch_dates[i_fly[1]]:%Y-%m-%d}  arrive {arrival_dates[i_fly[0]]:%Y-%m-%d}"
        f"  C3 {fc3[i_fly]:.3f} km^2/s^2  TOF {ftof[i_fly]:.0f} d"
        f"  arrival v_inf {fdva[i_fly]:.2f} km/s"
        f"  achieved closest approach {fly_closest_km:.0f} km"
    )
    print("Best rendezvous (stop at the asteroid):")
    print(
        f"  launch {rl[i_rdv[1]]:%Y-%m-%d}  arrive {ra[i_rdv[0]]:%Y-%m-%d}"
        f"  C3 {rc3[i_rdv]:.1f} km^2/s^2  arrival dv {rdva[i_rdv]:.2f} km/s"
        f"  total dv {rtot[i_rdv]:.2f} km/s  TOF {rtof[i_rdv]:.0f} d"
    )
    print(
        "\nWrote porkchop_yr4_2028_intercept.png, porkchop_yr4_2028_flyby.png, "
        "porkchop_yr4_2028_rendezvous.png, porkchop_yr4_overview.png, "
        "porkchop_yr4_grids.npz"
    )


if __name__ == "__main__":
    main()
