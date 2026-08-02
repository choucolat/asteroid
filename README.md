# Earth → (2024 YR4) porkchop plot

Porkchop analysis for a mission from Earth to the near-Earth asteroid
**(2024 YR4)**, adapted from the PySTK Earth → Mars porkchop tutorial.

## The mission window

2024 YR4 has a ~4-year orbit (a = 2.52 AU, e = 0.66, i = 3.4°) that reaches
0.85 AU at perihelion. Inside the span covered by `horizons_results.txt`
(2023-01-01 → 2029-12-25) it makes two close approaches to Earth:

| Encounter | Miss distance | Status |
|---|---|---|
| 2024-12-25 | 0.0058 AU | already past |
| **2028-12-18** | **0.054 AU** | the accessible opportunity |

Every low-energy launch window is tied to one of these. The analysis is
therefore centred on the December 2028 encounter.

Two mission types come out very differently:

* **Intercept / flyby** (only the launch burn counts) — reachable at
  essentially **C3 ≈ 0 km²/s²**, e.g. launch 2028-04-07, arrive 2028-12-21,
  258 days, arriving at 13.5 km/s relative velocity.
* **Rendezvous** (also stop at the asteroid) — the cheapest is **9.65 km/s
  total Δv**: launch 2028-11-16, arrive 2029-07-28, C3 = 73.8 km²/s², arrival
  Δv = 1.06 km/s, 254 days.

## Files

| File | What it is |
|---|---|
| `asteroid_porkchop_yr4.py` | **The STK script.** Builds the scenario, adds Earth and 2024 YR4, sweeps the launch/arrival grid with Astrogator's Lambert profile, and plots. Needs STK Engine + a licence. |
| `horizons_to_stk.py` | Converts the JPL Horizons vector table into an STK `.e` ephemeris so 2024 YR4 can exist as an STK object. Run directly to regenerate `yr4.e`. |
| `porkchop_core.py` | Standalone NumPy astrodynamics core — Kepler propagation, universal-variables Lambert solver, porkchop grid. No STK. |
| `plot_porkchop.py` | Generates the figures from `porkchop_core`; also supplies the shared plotting helpers the STK script imports. |
| `horizons_results.txt` | JPL Horizons heliocentric ICRF state vectors for 2024 YR4, 10-day steps. |
| `horizons_earth.txt` | The same for Earth, 1-day steps (used only by the standalone path; STK uses its own planetary ephemeris). |
| `yr4.e` | Generated STK ephemeris file. |

## Running

The standalone path needs only NumPy and Matplotlib and runs anywhere:

```bash
python plot_porkchop.py
```

It writes `porkchop_yr4_2028_intercept.png`,
`porkchop_yr4_2028_rendezvous.png`, `porkchop_yr4_overview.png` and
`porkchop_yr4_grids.npz`.

The STK path needs STK Engine (Windows or Linux — **STK does not run on
macOS**):

```bash
python asteroid_porkchop_yr4.py
```

Each grid point is a full mission control sequence run, so start with
`GRID_STEP_DAYS = 10` at the top of the file for a quick pass before dropping
to 5.

## Modelling assumptions

Both paths make the same ones, so their outputs are directly comparable:
restricted two-body problem with the Sun as a point mass, zero-revolution
Lambert transfer, prograde (short-way vs long-way chosen from the sign of
(r₁ × r₂)·ẑ), C3<sub>launch</sub> = |v₁ − v<sub>Earth</sub>|² and
Δv<sub>arrival</sub> = |v<sub>asteroid</sub> − v₂|. No launch-vehicle,
departure-parking-orbit, or gravity-assist modelling.

## Accuracy notes

* The Lambert solver reproduces Vallado's Example 7-5 benchmark to 7
  significant figures.
* `horizons_results.txt` is sampled every 10 days. Interpolating it against an
  independently downloaded 1-day Horizons ephemeris over 2028-01 → 2029-06
  gives a median position error of 6 km (worst 432 km, at the December 2028
  close approach), which moves the C3 grid by at most 0.02 km²/s². The 10-day
  sampling is fine for this analysis.
* Horizons tags its vectors TDB; the `.e` file's ScenarioEpoch is converted to
  UTC (−69.184 s).
