"""Download JPL Horizons vector ephemerides for Earth and a table of
candidate asteroids, in the exact format porkchop_core.SunCentredEphemeris /
horizons_to_stk.parse_horizons_vectors expect -- so you don't have to use
the Horizons web app by hand for every target.

Run:
    python fetch_horizons.py
        Re-downloads Earth (with extended coverage -- see EARTH below) plus
        every candidate in TARGETS, writing horizons_<name>.txt files into
        the current directory.

    python fetch_horizons.py --only apophis bennu
        Downloads just the named target(s) (plus Earth, unless --no-earth
        is also given).

    python fetch_horizons.py --no-earth
        Skip re-downloading Earth (e.g. if your existing horizons_earth.txt
        already covers everything you need).

Add more asteroids by adding a row to TARGETS -- see the comment above it
for the COMMAND syntax, and cneos.jpl.nasa.gov/nhats/ or
cneos.jpl.nasa.gov/ca/ for how to pick a target and a sensible date bracket
(covered earlier in this conversation).

Settings match horizons_results.txt / horizons_earth.txt exactly: Vectors,
Sun body-center (500@10), ICRF frame (REF_PLANE=FRAME), KM-S units,
geometric states (VEC_CORR defaults to NONE), table format 3. Uses the
public Horizons API -- https://ssd-api.jpl.nasa.gov/doc/horizons.html --
no API key needed. Only the standard library is used (urllib), so no pip
install is required.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# name -> (COMMAND body, start, stop, step). COMMAND syntax:
#   numbered/named small body -> "<number>;"       e.g. "99942;"       (Apophis)
#   provisional designation   -> "DES=<desig>;"    e.g. "DES=2011 AA37;"
#   major body (Earth, etc.)  -> "<naif id>"        e.g. "399"
# Dates are picked to bracket each object's best-known near-term Earth
# approach / accessibility window with margin, per the selection discussion
# earlier -- see cneos.jpl.nasa.gov/nhats/ and cneos.jpl.nasa.gov/ca/ to
# pick your own.
TARGETS: dict[str, tuple[str, str, str, str]] = {
    "apophis":   ("99942;",           "2026-01-01", "2030-06-01", "10d"),
    "bennu":     ("101955;",          "2026-01-01", "2032-01-01", "10d"),
    "ryugu":     ("162173;",          "2030-01-01", "2035-01-01", "10d"),
    "itokawa":   ("25143;",           "2030-01-01", "2034-01-01", "10d"),
    "didymos":   ("65803;",           "2026-01-01", "2034-01-01", "10d"),
    "2011_aa37": ("DES=2011 AA37;",   "2025-07-01", "2028-01-01", "10d"),
    "2012_ec":   ("DES=2012 EC;",     "2026-07-01", "2029-01-01", "10d"),
    "2013_wa44": ("DES=2013 WA44;",   "2027-07-01", "2030-07-01", "10d"),
    "2001_cq36": ("DES=2001 CQ36;",   "2029-07-01", "2032-01-01", "10d"),
}

#: Earth needs to cover every asteroid window above -- extend past the
#: existing horizons_earth.txt (which only reaches 2030-Feb-01) out to
#: Ryugu's 2033 window with margin.
EARTH: tuple[str, str, str, str] = ("399", "2025-01-01", "2035-06-01", "1d")


def build_url(command: str, start: str, stop: str, step: str) -> str:
    """Build the Horizons API query string.

    IMPORTANT: the '=' inside a DES=... COMMAND value must stay literal
    (not percent-encoded) or Horizons silently drops the command content --
    this was confirmed empirically against the live API, not just from the
    docs. Every other character, including the literal quotes Horizons'
    argument parser expects around COMMAND, gets percent-encoded normally.
    """
    params = {
        "format": "text",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "500@10",
        "START_TIME": start,
        "STOP_TIME": stop,
        "STEP_SIZE": step,
        "VEC_TABLE": "3",
        "REF_PLANE": "FRAME",
        "TIME_TYPE": "TDB",
    }
    parts = [f"{k}={urllib.parse.quote(str(v), safe='@')}" for k, v in params.items()]
    parts.append("COMMAND=" + urllib.parse.quote(f"'{command}'", safe="="))
    return API_URL + "?" + "&".join(parts)


def fetch(command: str, start: str, stop: str, step: str) -> str:
    url = build_url(command, start, stop, step)
    with urllib.request.urlopen(url, timeout=90) as resp:
        return resp.read().decode("utf-8", errors="replace")


def save(name: str, text: str) -> bool:
    if "$$SOE" not in text or "$$EOE" not in text:
        print(f"  FAILED {name}: no vector data in response -- first lines:")
        for line in text.splitlines()[:8]:
            print("    " + line)
        return False
    path = f"horizons_{name}.txt"
    with open(path, "w") as f:
        f.write(text)
    n_points = text.count(" = A.D. ")
    print(f"  wrote {path} ({n_points} points)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", help="subset of TARGETS keys to fetch (default: all)"
    )
    parser.add_argument(
        "--no-earth", action="store_true", help="skip re-downloading Earth"
    )
    args = parser.parse_args()

    ok, failed = [], []

    if not args.no_earth:
        print("Fetching Earth...")
        cmd, start, stop, step = EARTH
        try:
            if save("earth", fetch(cmd, start, stop, step)):
                ok.append("earth")
            else:
                failed.append("earth")
        except urllib.error.URLError as exc:
            print(f"  FAILED earth: {exc}")
            failed.append("earth")
        time.sleep(1)

    names = args.only if args.only else list(TARGETS)
    for name in names:
        if name not in TARGETS:
            print(f"  skip {name}: not in TARGETS")
            continue
        cmd, start, stop, step = TARGETS[name]
        print(f"Fetching {name} ({cmd})...")
        try:
            if save(name, fetch(cmd, start, stop, step)):
                ok.append(name)
            else:
                failed.append(name)
        except urllib.error.URLError as exc:
            print(f"  FAILED {name}: {exc}")
            failed.append(name)
        time.sleep(1)  # be polite to the API

    print(f"\n{len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))
    print("Next: python batch_porkchop.py")


if __name__ == "__main__":
    sys.exit(main())
