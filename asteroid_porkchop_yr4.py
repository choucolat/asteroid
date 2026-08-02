# # Porkchop plot: Earth -> (2024 YR4)
#
# Adapted from the PySTK "Porkchop plots" tutorial (Earth -> Mars) to target the
# near-Earth asteroid 2024 YR4.
#
# ## What is a porkchop plot?
# A porkchop plot is a graphical tool used in astrodynamics to visualize the fuel or energy requirements for space missions, especially interplanetary travel. It shows contours of mission parameters, typically launch and arrival dates, with color-coded regions indicating different energy costs. The plot's shape often resembles a porkchop, hence the name.
# Porkchop plots are usually generated under the assumption of the restricted two-body problem. The restricted two-body problem assumes that each body is treated as a single point mass, focusing solely on the gravitational influence of the Sun. The trajectory of the spacecraft is computed by solving a Lambert transfer.
#
# ## What changed relative to the Earth -> Mars tutorial
#
# 1. **The target body now exists.** The original loop called
#    `lambert_solver(satellite, earth, yr4, ...)` but `yr4` was never created --
#    only Earth and Mars were added to the scenario, so the script died with
#    `NameError: name 'yr4' is not defined`.  STK's built-in planet ephemerides
#    do not include minor bodies, so 2024 YR4 is added as a **Satellite** driven
#    by an STK external ephemeris (`yr4.e`) generated from the JPL Horizons
#    vectors in `horizons_results.txt` (see `horizons_to_stk.py`).  It is then
#    queried through exactly the same data providers as the planets.
# 2. **The scenario analysis interval is set.** A new scenario defaults to a
#    ~1 day interval starting at the current date.  Every
#    `execute_single_elements(...)` call at a 2027/2028 epoch would have been
#    outside that interval, and the external ephemeris would have been clipped
#    to it.  The interval is now set to cover the ephemeris before any object is
#    added.
# 3. **Mars is gone** -- it was still being created and never used.
# 4. **Failures no longer kill the run.** `lambert_solver` returned
#    `(None, None, None)` for non-positive times of flight, which then raised
#    `TypeError` on assignment into a float array.  Non-convergent Lambert arcs
#    and MCS failures now yield `NaN` and the sweep carries on.
# 5. **The date span targets the real opportunity.** The inherited 2023-2028
#    launch / 2023-2029 arrival span at 10-day steps is ~40,000 MCS runs, and
#    almost all of it is unreachable.  2024 YR4 passes 0.054 AU from Earth on
#    2028-12-18, and that encounter is the only accessible window inside the
#    supplied ephemeris, so the sweep is centred on it.
# 6. **Plot fixes**: contour levels matched to an asteroid target (the inherited
#    0-5 km/s arrival and 0-400 day time-of-flight levels never intersected the
#    data), a colorbar that is inside the figure, readable date ticks, and the
#    figure is saved rather than only shown -- STK Engine here runs headless.
#
# The results are cross-checked by `plot_porkchop.py`, a standalone NumPy
# implementation of the same model that needs no STK licence.

# ## Configuration
#
# Coarsen `GRID_STEP_DAYS` to 10 for a quick first pass; each grid point is a
# full Astrogator mission control sequence run.

# +
from pathlib import Path


HERE = Path(__file__).resolve().parent
HORIZONS_FILE = HERE / "horizons_results.txt"
EPHEMERIS_FILE = HERE / "yr4.e"

# Scenario analysis interval -- must span every epoch queried below and stay
# inside the ephemeris written by horizons_to_stk.py.
SCENARIO_START = "2 Jan 2023 00:00:00.000"
SCENARIO_STOP = "20 Dec 2029 00:00:00.000"

# Sweep centred on the December 2028 Earth close approach.
FIRST_LAUNCH, LAST_LAUNCH = "01 Aug 2027", "25 Nov 2028"
FIRST_ARRIVAL, LAST_ARRIVAL = "15 Jul 2028", "15 Mar 2029"
GRID_STEP_DAYS = 5
# -

# ## Build the external ephemeris for 2024 YR4
#
# STK has no ephemeris for minor bodies, so convert the Horizons vector table
# into STK's native `.e` format if that has not already been done.

# +
from horizons_to_stk import parse_horizons_vectors, write_stk_ephemeris_file


if not EPHEMERIS_FILE.exists():
    points = parse_horizons_vectors(str(HORIZONS_FILE))
    write_stk_ephemeris_file(points, str(EPHEMERIS_FILE))
    print(f"Wrote {EPHEMERIS_FILE.name} ({len(points)} ephemeris points)")
# -

# ## Launch STK
# Start a new STK instance in headless mode (no GUI):

# +
from ansys.stk.core.stkengine import STKEngine


stk = STKEngine.start_application(no_graphics=True)
print(f"Using {stk.version}")
# -

# ## Create a new scenario
# The central body for this scenario must be the Sun.

# +
from ansys.stk.core.stkobjects import STKObjectType


root = stk.new_object_root()
scenario = root.children.new_on_central_body(
    STKObjectType.SCENARIO, "PorkchopPlot", "Sun"
)

# Set the analysis interval BEFORE adding objects: the external ephemeris is
# clipped to this interval, and data providers cannot be evaluated outside it.
scenario.set_time_period(SCENARIO_START, SCENARIO_STOP)
root.rewind()
# -


# ## Generating a span of launch and arrival dates
#
# Generating a span of launch and arrival dates allows to compute, in the future, for any combination of dates. When manipulating dates and times in PySTK, it is important to always use the API. Do not rely on the Python <inv:datetime> module since it does not account for leap-seconds.


def linspace_dates(start_date, end_date, interval, unit):
    """Generate a list of linearly spaced dates.

    Parameters
    ----------
    start_date : ~ansys.stk.core.stkutil.Date
        Date to start the generation.
    end_date : ~ansys.stk.core.stkutil.Date
        Date to end the generation.
    interval : float
        Desired time interval between generated dates.
    unit : str
        Name of the units associated to the time interval.

    Returns
    -------
    dates : list[~ansys.stk.core.stkutil.Date]
        List of dates linearly spaced by the desired interval.

    """
    dates = []
    next_date = start_date
    while next_date.whole_days <= end_date.whole_days:
        dates.append(next_date)
        next_date = next_date.add(unit, interval)
    return dates


# Declare the launch and arrival boundary dates:

first_launch, last_launch = [
    root.conversion_utility.new_date("UTCG", date)
    for date in [FIRST_LAUNCH, LAST_LAUNCH]
]
first_arrival, last_arrival = [
    root.conversion_utility.new_date("UTCG", date)
    for date in [FIRST_ARRIVAL, LAST_ARRIVAL]
]

# Then, generate the launch and arrival date spans:

launch_span = linspace_dates(first_launch, last_launch, GRID_STEP_DAYS, "day")
arrival_span = linspace_dates(first_arrival, last_arrival, GRID_STEP_DAYS, "day")

print(
    f"Launch span:  {launch_span[0].format('UTCG')} .. "
    f"{launch_span[-1].format('UTCG')} ({len(launch_span)} dates)"
)
print(
    f"Arrival span: {arrival_span[0].format('UTCG')} .. "
    f"{arrival_span[-1].format('UTCG')} ({len(arrival_span)} dates)"
)


# ## Solve the ephemerides of the bodies
#
# With launch and arrival dates generated, it is time to generate a routine for computing the state vector of the bodies. This function was presented in the Lambert transfer example. Therefore, it is reproduced in here:


# +
def from_data_result_to_dict(data_result: "DataProviderResult") -> dict:
    """Convert a data provider result to a dictionary.

    Parameters
    ----------
    data_result : DataProviderResult
        Data result instance to be converted.

    Returns
    -------
    dict
        Dictionary representing the elements and values of the data provider.

    """
    return {
        key: data_result.data_sets.item(key_id).get_values()
        for key_id, key in enumerate(data_result.data_sets.element_names)
    }


def get_object_pos_vel_at_epoch(
    stk_object: "STKObject", epoch: str, frame_name: str
) -> tuple:
    """Compute the position and velocity vectors of an object in the desired reference frame.

    Parameters
    ----------
    stk_object : ~ansys.stkcore.stkobjects.STKObject
        Name of the object.
    epoch : ~ansys.stk.core.stkutil.Date
        Epoch date.
    frame_name : str
        Reference frame name.

    Returns
    -------
    tuple(list[float, float, float], list[float, float, float])
        Tuple containing the position and velocity vectors as a list.

    """
    state = {"Position": None, "Velocity": None}
    for path in state:
        data_provider = (
            stk_object.data_providers.get_data_provider_time_varying_from_path(
                f"Cartesian {path}/{frame_name}"
            )
        )
        data = from_data_result_to_dict(
            data_provider.execute_single_elements(epoch.format("UTCG"), ["x", "y", "z"])
        )
        state[path] = [coord[0] for coord in data.values()]
    return tuple(state.values())


# -

# ## Add the departure and arrival bodies
#
# Earth comes from STK's built-in planetary ephemeris. 2024 YR4 does not exist
# in STK, so it is added as a Satellite propagated from the external `.e` file.
# Both then answer to the same `Cartesian Position/ICRF` data provider, which is
# the whole point of going through the `.e` file rather than interpolating the
# Horizons table by hand.

# +
from ansys.stk.core.stkobjects import (
    EphemSourceType,
    PlanetPositionSourceType,
    PropagatorType,
)


def resolve_enum(enum_type, *candidate_names):
    """Return the first member of ``enum_type`` that exists.

    PySTK's enum spellings have moved around between releases; failing here with
    the list of what *is* available beats an opaque ``AttributeError`` deep in
    the sweep.
    """
    for name in candidate_names:
        member = getattr(enum_type, name, None)
        if member is not None:
            return member
    available = [n for n in dir(enum_type) if n.isupper()]
    raise AttributeError(
        f"None of {candidate_names} found on {enum_type.__name__}. Available: {available}"
    )


earth = scenario.children.new_on_central_body(STKObjectType.PLANET, "Earth", "Sun")
earth.common_tasks.set_position_source_central_body("Earth", EphemSourceType.DEFAULT)

yr4 = scenario.children.new_on_central_body(STKObjectType.SATELLITE, "YR4", "Sun")
yr4.set_propagator_type(
    resolve_enum(PropagatorType, "STK_EXTERNAL", "STKEXTERNAL", "STK_EXTERNAL_FILE")
)
yr4.propagator.filename = str(EPHEMERIS_FILE)
yr4.propagator.propagate()
# -

# Print a few states to verify the two bodies answer the same way. Earth should
# sit ~1 AU (1.5e8 km) from the Sun; 2024 YR4 ranges between 0.85 and 4.2 AU.

# +
import numpy as np


print("\nEphemerides check")
print("=================\n")
print(f"{'Body':<10} {'Date':<25} {'Range [km]':>14}  {'Speed [km/s]':>12}")
print(f"{'----':<10} {'----':<25} {14 * '-':>14}  {12 * '-':>12}")
for body, name in [(earth, "Earth"), (yr4, "2024 YR4")]:
    for date in launch_span[:2]:
        r, v = get_object_pos_vel_at_epoch(body, date, "ICRF")
        print(
            f"{name:<10} {date.format('UTCG'):<25} "
            f"{np.linalg.norm(r):>14,.0f}  {np.linalg.norm(v):>12.3f}"
        )
# -

# ## Add a satellite
#
# A satellite object is used to solve for the Lambert transfer between Earth and the asteroid. Astrogator is used for its propagation. Make sure to clean the main sequence.

# +
satellite = scenario.children.new_on_central_body(
    STKObjectType.SATELLITE, "Satellite", "Sun"
)
satellite.set_propagator_type(PropagatorType.ASTROGATOR)
satellite.propagator.main_sequence.remove_all()
# -

# ### Initial state
#
# The initial state of the satellite must be changed for every launch date. However, the initial state segment instance remains in the main sequence. Therefore, it is possible to configure here only the constant parameters of the initial state of the satellite:

# +
from ansys.stk.core.stkobjects.astrogator import ElementSetType, SegmentType


initial_state = satellite.propagator.main_sequence.insert(
    SegmentType.INITIAL_STATE, "Initial State", "-"
)
initial_state.coord_system_name = "CentralBody/Sun Inertial"
initial_state.set_element_type(ElementSetType.CARTESIAN)
# -

# ### Interplanetary transfer
#
# For the interplanetary transfer, a similar situation occurs. The different segments of the Lambert trajectory remain no matter the launch and arrival dates. Therefore, it is possible configure here only the constant parameters.

# Start by declaring the different segments of the Lambert transfer:

transfer = satellite.propagator.main_sequence.insert(
    SegmentType.TARGET_SEQUENCE, "Lambert Transfer", "-"
)
first_impulse = transfer.segments.insert(SegmentType.MANEUVER, "First Impulse", "-")
propagate = transfer.segments.insert(SegmentType.PROPAGATE, "Propagate", "-")
last_impulse = transfer.segments.insert(SegmentType.MANEUVER, "Last Impulse", "-")

# Configure the type of segments:

# +
from ansys.stk.core.stkobjects.astrogator import ManeuverType


first_impulse.set_maneuver_type(ManeuverType.IMPULSIVE)
propagate.propagator_name = "Sun Point Mass"
last_impulse.set_maneuver_type(ManeuverType.IMPULSIVE)
# -

# Add a Lambert profile to the interplanetary transfer:

transfer.profiles.remove_all()
lambert = transfer.profiles.add("Lambert Profile")

# Configure the profile. Do not configure any parameters related with the target:

# +
from ansys.stk.core.stkobjects.astrogator import (
    LambertSolutionOptionType,
    LambertTargetCoordinateType,
)


lambert.coord_system_name = "CentralBody/Sun Inertial"
lambert.set_target_coord_type(LambertTargetCoordinateType.CARTESIAN)
lambert.enable_second_maneuver = True

lambert.solution_option = LambertSolutionOptionType.FIXED_TIME
lambert.revolutions = 0
lambert.central_body_collision_altitude_padding = 0
# -

# Allow the profile to write its results to the transfer segments:

# +
lambert.enable_write_to_first_maneuver = True
lambert.first_maneuver_segment = first_impulse.name

lambert.enable_write_duration_to_propagate = True
lambert.disable_non_lambert_propagate_stop_conditions = True
lambert.propagate_segment = propagate.name

lambert.enable_write_to_second_maneuver = True
lambert.second_maneuver_segment = last_impulse.name
# -

# Activate the profile and configure its behavior when solving for the results:

# +
from ansys.stk.core.stkobjects.astrogator import (
    ProfileMode,
    ProfilesFinish,
    TargetSequenceAction,
)


lambert.mode = ProfileMode.ACTIVE
transfer.action = TargetSequenceAction.RUN_ACTIVE_PROFILES
transfer.when_profiles_finish = ProfilesFinish.RUN_TO_RETURN_AND_CONTINUE
transfer.continue_on_failure = False
transfer.reset_inner_targeters = False
# -

# ## Solve the transfer
#
# Only the constant parameters have been configured so far. Now, it is time to create a routine capable of modifying the state of the satellite and the Lambert profile so that the required impulses can be solved for any launch and arrival date.
#
# **Note:** the porkchop plots presented in NASA's manual assume prograde transfers. However, the Lambert profile only differentiates between long and short solution transfers. The relation between prograde/retrograde and long/short transfers is set by the angular momentum. Although its magnitude can not be found unless solving the problem, its direction can be retrieved from the initial and final position vectors. Therefore, depending on the angular momentum, a long or short transfer is imposed.
#
# **Note on units:** data providers return km and km/s (the default unit
# preferences), whereas the Astrogator Lambert profile target state and the
# maneuver magnitude are in internal units (m and m/s). Hence the factors of
# 1000 below.

# +
from ansys.stk.core.stkobjects.astrogator import LambertDirectionOfMotionType


def lambert_solver(
    satellite, departure_body, arrival_body, launch_date, arrival_date, is_prograde=True
):
    """Solve the Lambert transfer between two bodies for a given launch and arrival date.

    Returns
    -------
    tuple(float, float, float)
        Departure impulse (km/s), arrival impulse (km/s) and time of flight (s).
        The impulses are ``nan`` when no transfer exists or STK fails to solve.
    """
    # Retrieve the segments and lambert profile from the satellite
    main_sequence = satellite.propagator.main_sequence
    initial_state = main_sequence["Initial State"]
    transfer = main_sequence["Lambert Transfer"]
    lambert = transfer.profiles["Lambert Profile"]
    first_impulse = transfer.segments["First Impulse"]
    last_impulse = transfer.segments["Last Impulse"]

    # Compute the time of flight
    time_of_flight = arrival_date.span(launch_date).value
    if time_of_flight <= 0:
        return np.nan, np.nan, time_of_flight
    lambert.time_of_flight = time_of_flight

    # Compute the departure and arrival state vectors
    departure_position, departure_velocity = get_object_pos_vel_at_epoch(
        departure_body, launch_date, "ICRF"
    )
    arrival_position, arrival_velocity = get_object_pos_vel_at_epoch(
        arrival_body, arrival_date, "ICRF"
    )

    # Impose the direction of motion according to the angular momentum of the orbit
    r1_cross_r2 = np.cross(departure_position, arrival_position)
    r1_times_r2 = np.linalg.norm(departure_position) * np.linalg.norm(arrival_position)
    h0_z = (r1_cross_r2 / r1_times_r2)[-1]

    if is_prograde:
        path = (
            LambertDirectionOfMotionType.LONG
            if h0_z < 0
            else LambertDirectionOfMotionType.SHORT
        )
    else:
        path = (
            LambertDirectionOfMotionType.SHORT
            if h0_z < 0
            else LambertDirectionOfMotionType.LONG
        )
    lambert.direction_of_motion = path

    # Update the initial state of the satellite
    initial_state.orbit_epoch = launch_date.format("UTCG")
    initial_state.element.x = departure_position[0]
    initial_state.element.y = departure_position[1]
    initial_state.element.z = departure_position[2]
    initial_state.element.vx = departure_velocity[0]
    initial_state.element.vy = departure_velocity[1]
    initial_state.element.vz = departure_velocity[2]

    # Update final state of satellite (internal units: m and m/s)
    lambert.target_position_x = arrival_position[0] * 1000
    lambert.target_position_y = arrival_position[1] * 1000
    lambert.target_position_z = arrival_position[2] * 1000
    lambert.target_velocity_x = arrival_velocity[0] * 1000
    lambert.target_velocity_y = arrival_velocity[1] * 1000
    lambert.target_velocity_z = arrival_velocity[2] * 1000

    # Run the mission control sequence. A geometry the Lambert profile cannot
    # solve must not abort the whole sweep.
    try:
        satellite.propagator.run_mcs()
        satellite.propagator.apply_all_profile_changes()
        delta_v1 = first_impulse.maneuver.attitude_control.magnitude / 1000
        delta_v2 = last_impulse.maneuver.attitude_control.magnitude / 1000
    except Exception:
        return np.nan, np.nan, time_of_flight

    return delta_v1, delta_v2, time_of_flight


# -

# Finally, solve the transfer for every launch and arrival date combination:

# +
import time


dv_arrival_values = np.full((len(arrival_span), len(launch_span)), np.nan)
c3_launch_values = np.full((len(arrival_span), len(launch_span)), np.nan)
tof_values = np.full((len(arrival_span), len(launch_span)), np.nan)

total = len(launch_span) * len(arrival_span)
started = time.time()
print(f"\nSolving {total:,} launch/arrival combinations...")

for i, launch_date in enumerate(launch_span):
    for j, arrival_date in enumerate(arrival_span):
        dv_launch, dv_arrival, tof = lambert_solver(
            satellite, earth, yr4, launch_date, arrival_date
        )

        dv_arrival_values[j, i] = dv_arrival
        c3_launch_values[j, i] = dv_launch**2
        tof_values[j, i] = tof / 3600 / 24

    done = (i + 1) * len(arrival_span)
    elapsed = time.time() - started
    eta = elapsed / done * (total - done)
    print(
        f"  launch {launch_date.format('UTCG')[:11]}  "
        f"{done:>7,}/{total:,}  elapsed {elapsed / 60:5.1f} min  ETA {eta / 60:5.1f} min",
        flush=True,
    )

solved = np.isfinite(c3_launch_values)
print(f"Solved {solved.sum():,} of {total:,} combinations")
if solved.any():
    k = np.unravel_index(
        np.nanargmin(np.where(solved & (tof_values > 60), c3_launch_values, np.inf)),
        c3_launch_values.shape,
    )
    print(
        f"Minimum C3 = {c3_launch_values[k]:.3f} km^2/s^2 for launch "
        f"{launch_span[k[1]].format('UTCG')[:11]} / arrival "
        f"{arrival_span[k[0]].format('UTCG')[:11]} (TOF {tof_values[k]:.0f} d, "
        f"arrival v_inf {dv_arrival_values[k]:.2f} km/s)"
    )
# -

# ## Plot the porkchop
#
# With the values for $C_{3_{\text{launch}}}$, $\Delta v_{\text{arrival}}$, and the time of flight, it is possible to generate the porkchop plot. However, before proceeding, we need to convert the `Date` objects to `datetime.datetime` objects so that Matplotlib can correctly interpret them as time values.

# +
from datetime import datetime


def as_datetime(date):
    """Convert a :ref:`~Date` object into a :ref:`~datetime.datetime` instance.

    Warns
    -----
    If a leap second is casted, one second is subtracted.

    Note
    ----
    Casting as Python dates introduces a loss of precision. Avoid using casted
    date types in future computations.

    """
    utcg_format_str = "%d %b %Y %H:%M:%S.%f"
    try:
        return datetime.strptime(date.format("UTCG"), utcg_format_str)
    except ValueError as LeapSecondsError:
        import warnings

        warnings.warn(f"Date {date.format('UTCG')} is a leap second.")
        adjusted_date = date.subtract("sec", 1)
        return datetime.strptime(adjusted_date.format("UTCG"), utcg_format_str)


# -

# Cast dates to ensure Matplotlib representation:

launch_dates = [as_datetime(date) for date in launch_span]
arrival_dates = [as_datetime(date) for date in arrival_span]

# Save the grids so the figure can be re-drawn without re-running STK:

np.savez_compressed(
    HERE / "porkchop_yr4_stk.npz",
    launch_iso=np.array([d.isoformat() for d in launch_dates]),
    arrival_iso=np.array([d.isoformat() for d in arrival_dates]),
    c3_launch=c3_launch_values,
    dv_arrival=dv_arrival_values,
    tof_days=tof_values,
)

# Contour levels appropriate for this target. 2024 YR4 is reached at essentially
# zero C3 near its close approach, but the arrival relative velocity is ~13 km/s
# -- an order of magnitude above the 0-5 km/s levels inherited from the Mars
# tutorial, which would have drawn no lines at all.

# +
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from plot_porkchop import callout, draw_porkchop, mark


c3_launch_levels = np.linspace(0, 45, 10)
dv_arrival_levels = [10, 12, 14, 16, 20, 25]
tof_levels = [50, 100, 150, 200, 250, 300, 350, 400, 450]

fig, ax = plt.subplots(figsize=(13.5, 9.5))
ax.set_title(
    "EARTH → (2024 YR4)   BALLISTIC TRANSFER TRAJECTORY\n"
    "Solved in STK / Astrogator — Lambert profile, Sun point mass",
    fontsize=13,
    pad=14,
)

filled = draw_porkchop(
    ax,
    launch_dates,
    arrival_dates,
    c3_launch_values,
    c3_launch_levels,
    "$C_3$ launch",
    dv_arrival_values,
    dv_arrival_levels,
    tof_values,
    tof_levels,
    dv_label="Arrival $v_\\infty$",
)
cbar = fig.colorbar(filled, ax=ax, pad=0.015, fraction=0.04, extend="max")
cbar.set_label("Launch characteristic energy $C_3$  [km$^2$/s$^2$]")

if solved.any():
    mark(ax, launch_dates[k[1]], arrival_dates[k[0]])
    callout(
        ax,
        f"minimum $C_3$ = {c3_launch_values[k]:.2f} km$^2$/s$^2$\n"
        f"launch  {launch_dates[k[1]]:%d %b %Y}\n"
        f"arrive  {arrival_dates[k[0]]:%d %b %Y}\n"
        f"time of flight  {tof_values[k]:.0f} d\n"
        f"arrival $v_\\infty$  {dv_arrival_values[k]:.2f} km/s",
    )

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.yaxis.set_major_locator(mdates.MonthLocator())

fig.savefig(HERE / "porkchop_yr4_stk.png", dpi=160, bbox_inches="tight")
print(f"\nWrote {HERE / 'porkchop_yr4_stk.png'}")
plt.show()
# -
