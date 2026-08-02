"""Convert a JPL Horizons vector-table ephemeris (Type 3: position, velocity,
LT, range, range-rate; heliocentric; ICRF) into an STK .e ephemeris file.

Why this exists
----------------
STK's built-in planet ephemerides don't include minor bodies like
2024 YR4. Rather than hand-rolling a separate position/velocity
interpolator outside of STK, this converts the downloaded Horizons
vectors into STK's native .e format so 2024 YR4 can be added to the
scenario as a Satellite (central body = Sun) using the StkExternal
propagator -- i.e. it behaves exactly like Earth/Mars from the rest
of the porkchop script (same get_object_pos_vel_at_epoch() calls,
same data providers).

Reference for the .e format:
https://help.agi.com/stk/Content/stk/importfiles-01.htm
Reference for loading it via Connect (used from PySTK via
root.execute_command):
https://help.agi.com/stk/Subsystems/connectCmds/Content/cmd_SetStateFromFile.htm
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class EphemPoint:
    jd_tdb: float
    calendar: str  # as given by Horizons, e.g. "2023-Jan-01 00:00:00.0000"
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


def parse_horizons_vectors(path: str) -> list[EphemPoint]:
    """Parse the $$SOE / $$EOE vector block of a Horizons Type-3 text file.

    Assumes Output units = KM-S and Output type = GEOMETRIC cartesian
    states, which is what the uploaded horizons_results.txt uses.
    """
    with open(path, "r") as f:
        text = f.read()

    soe = text.index("$$SOE")
    eoe = text.index("$$EOE")
    block = text[soe + len("$$SOE"):eoe]

    # Each record is 4 lines:
    #   <JD> = A.D. <calendar date/time> TDB
    #    X =... Y =... Z =...
    #   VX=... VY=... VZ=...
    #   LT=... RG=... RR=...
    record_re = re.compile(
        r"(?P<jd>\d+\.\d+)\s*=\s*A\.D\.\s*(?P<cal>.+?)\s*TDB\s*\n"
        r"\s*X\s*=\s*(?P<x>[-\dE.+]+)\s*Y\s*=\s*(?P<y>[-\dE.+]+)\s*Z\s*=\s*(?P<z>[-\dE.+]+)\s*\n"
        r"\s*VX=\s*(?P<vx>[-\dE.+]+)\s*VY=\s*(?P<vy>[-\dE.+]+)\s*VZ=\s*(?P<vz>[-\dE.+]+)\s*\n"
    )

    points = []
    for m in record_re.finditer(block):
        points.append(
            EphemPoint(
                jd_tdb=float(m.group("jd")),
                calendar=m.group("cal").strip(),
                x=float(m.group("x")),
                y=float(m.group("y")),
                z=float(m.group("z")),
                vx=float(m.group("vx")),
                vy=float(m.group("vy")),
                vz=float(m.group("vz")),
            )
        )
    if not points:
        raise ValueError(f"No vector records parsed from {path}")
    return points


_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def horizons_calendar_to_datetime(cal: str) -> datetime:
    """'2023-Jan-01 00:00:00.0000' -> datetime (naive; treated as TDB,
    approximated as UTC -- see write_stk_ephemeris_file docstring)."""
    date_part, time_part = cal.split()
    year, mon, day = date_part.split("-")
    h, mi, s = time_part.split(":")
    sec = float(s)
    whole_sec = int(sec)
    micro = round((sec - whole_sec) * 1e6)
    return datetime(int(year), _MONTHS[mon], int(day), int(h), int(mi), whole_sec, micro)


def format_utcg(dt: datetime) -> str:
    """Format a datetime as STK's UTCG string, e.g. '1 Jan 2023 00:00:00.000000000'."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return (
        f"{dt.day} {months[dt.month - 1]} {dt.year} "
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond * 1000:09d}"
    )


def write_stk_ephemeris_file(
    points: list[EphemPoint],
    out_path: str,
    central_body: str = "Sun",
    coord_system: str = "J2000",
    interpolation_order: int = 5,
) -> None:
    """Write an STK .e (EphemerisTimePosVel) file from parsed Horizons points.

    Note on time system: Horizons vectors here are tagged TDB. STK's .e
    ScenarioEpoch/date fields are UTCG. TDB-UTC differs by ~69 seconds
    (as of the 2020s) and drifts extremely slowly -- negligible next to
    the 10-day sampling used for this porkchop, so the TDB calendar
    string is written directly as if it were UTCG. If sub-minute
    accuracy in the epoch mattered for your application you'd want to
    apply that ~69 s correction explicitly.
    """
    points = sorted(points, key=lambda p: p.jd_tdb)
    epoch_dt = horizons_calendar_to_datetime(points[0].calendar)

    lines = []
    lines.append("stk.v.11.0")
    lines.append("BEGIN Ephemeris")
    lines.append(f"NumberOfEphemerisPoints {len(points)}")
    lines.append(f"ScenarioEpoch {format_utcg(epoch_dt)}")
    lines.append(f"CentralBody {central_body}")
    lines.append(f"CoordinateSystem {coord_system}")
    lines.append("InterpolationMethod Lagrange")
    lines.append(f"InterpolationOrder {interpolation_order}")
    lines.append("DistanceUnit Kilometers")
    lines.append("EphemerisTimePosVel")
    for p in points:
        dt = horizons_calendar_to_datetime(p.calendar)
        t_sec = (dt - epoch_dt).total_seconds()
        lines.append(
            f"{t_sec:.6f} {p.x:.9E} {p.y:.9E} {p.z:.9E} "
            f"{p.vx:.9E} {p.vy:.9E} {p.vz:.9E}"
        )
    lines.append("END Ephemeris")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    pts = parse_horizons_vectors("horizons_results.txt")
    print(f"Parsed {len(pts)} points")
    print("First:", pts[0])
    print("Last:", pts[-1])
    write_stk_ephemeris_file(pts, "yr4.e")
    print("Wrote yr4.e")
