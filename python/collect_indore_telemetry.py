import csv
import os
import sys
import math
from collections import defaultdict

import traci


# ============================================================
# INDORE-01 CONFIGURATION
# ============================================================

SUMO_CONFIG = (
    r"C:\Users\jinda\Sumo\2026-08-24-00-28-36"
    r"\osm.sumocfg"
)

OUTPUT_DIR = (
    r"C:\C-V2X-OODA\datasets\observe_ds"
)

RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "indore_01_telemetry_v5.csv"
)

STATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "indore_01_scenario_states_v5.csv"
)


# ============================================================
# SIMULATION SETTINGS
# ============================================================

# SUMO simulation resolution
STEP_LENGTH = 0.1

# Scenario aggregation window
WINDOW_SIZE = 5.0


# ============================================================
# TRAFFIC THRESHOLDS
# ============================================================

# Vehicle below this speed is considered queued.
# Unit: m/s
QUEUE_SPEED_THRESHOLD = 2.0


# Edge mean speed below this is considered congested.
# Unit: m/s
CONGESTION_SPEED_THRESHOLD = 5.0


# ------------------------------------------------------------
# Density filtering
# ------------------------------------------------------------

# Ignore very short OSM connector / geometry edges.
#
# Example:
#
# 1 vehicle on a 0.2 m edge gives:
#
# 1 / 0.0002 = 5000 veh/km
#
# Such a value is mathematically correct but not useful
# as a traffic-density measurement.
#
# Therefore we only calculate density for edges >= 20 m.
MIN_DENSITY_EDGE_LENGTH_M = 20.0


# ------------------------------------------------------------
# High-density threshold
# ------------------------------------------------------------

# Unit:
# vehicles / km / lane
#
# This is a configurable research threshold rather than
# a universal traffic-engineering threshold.
HIGH_DENSITY_THRESHOLD = 25.0


# ------------------------------------------------------------
# Robust density statistic
# ------------------------------------------------------------

DENSITY_PERCENTILE = 95


# ============================================================
# CHECK SUMO CONFIGURATION
# ============================================================

if not os.path.exists(SUMO_CONFIG):

    print()
    print("=" * 75)
    print("ERROR: SUMO CONFIGURATION NOT FOUND")
    print("=" * 75)

    print()
    print(SUMO_CONFIG)

    print()

    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RAW TELEMETRY FIELDS
# ============================================================

RAW_FIELDS = [

    # Time
    "time",

    # Vehicle
    "vehicle_id",
    "vehicle_type",

    # Position
    "x",
    "y",

    # Kinematics
    "speed",
    "acceleration",
    "angle",

    # Road
    "edge_id",
    "lane_id",
    "lane_position",

    # Route
    "route_id",
    "route_index",

    # Waiting
    "waiting_time",
    "accumulated_waiting_time",

    # Distance
    "distance_travelled",
]


# ============================================================
# SCENARIO STATE FIELDS
# ============================================================

STATE_FIELDS = [

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "window_start",
    "window_end",

    # --------------------------------------------------------
    # Traffic volume
    # --------------------------------------------------------

    "vehicle_count",
    "vehicle_observations",

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    "mean_speed",
    "speed_std",
    "min_speed",
    "max_speed",

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    "mean_acceleration",

    # --------------------------------------------------------
    # Queue
    # --------------------------------------------------------

    "queue_length",
    "queue_ratio",

    # --------------------------------------------------------
    # Congestion
    # --------------------------------------------------------

    "congested_edges",

    # --------------------------------------------------------
    # Edge density
    # --------------------------------------------------------

    "mean_edge_density_veh_per_km",
    "p95_edge_density_veh_per_km",
    "max_edge_density_veh_per_km",

    # --------------------------------------------------------
    # Lane-normalized density
    # --------------------------------------------------------

    "mean_lane_density_veh_per_km",
    "p95_lane_density_veh_per_km",
    "max_lane_density_veh_per_km",

    # --------------------------------------------------------
    # High-density edges
    # --------------------------------------------------------

    "high_density_edges",

    # --------------------------------------------------------
    # Waiting
    # --------------------------------------------------------

    "mean_waiting_time",
    "max_waiting_time",

    # --------------------------------------------------------
    # Traffic lights
    # --------------------------------------------------------

    "active_traffic_lights",
    "green_signal_links",
    "yellow_signal_links",
    "red_signal_links",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_mean(values):

    """
    Calculate arithmetic mean.

    Returns 0 when there are no values.
    """

    if not values:

        return 0.0

    return (
        sum(values)
        /
        len(values)
    )


def calculate_std(values):

    """
    Calculate population standard deviation.

    Returns 0 when there are fewer than
    two observations.
    """

    if len(values) <= 1:

        return 0.0

    average = calculate_mean(
        values
    )

    variance = (

        sum(
            (value - average) ** 2
            for value in values
        )
        /
        len(values)
    )

    return math.sqrt(
        variance
    )


def calculate_percentile(
    values,
    percentile_value
):

    """
    Calculate percentile using linear interpolation.

    Does not require NumPy.
    """

    if not values:

        return 0.0

    sorted_values = sorted(
        values
    )

    if len(sorted_values) == 1:

        return sorted_values[0]

    position = (
        (len(sorted_values) - 1)
        *
        (
            percentile_value
            /
            100.0
        )
    )

    lower_index = int(
        math.floor(position)
    )

    upper_index = int(
        math.ceil(position)
    )

    if (
        lower_index
        ==
        upper_index
    ):

        return sorted_values[
            lower_index
        ]

    weight = (
        position
        -
        lower_index
    )

    return (

        sorted_values[
            lower_index
        ]

        +

        (
            sorted_values[
                upper_index
            ]

            -
            sorted_values[
                lower_index
            ]
        )
        *
        weight
    )


# ============================================================
# SUMO EDGE CACHE
# ============================================================

edge_length_cache = {}

edge_lane_count_cache = {}

edge_speed_cache = {}


# ============================================================
# GET EDGE GEOMETRY
# ============================================================

def get_edge_geometry(edge_id):

    """
    Get:

        edge length in meters
        number of lanes

    SUMO 1.27.1 TraCI does not expose edge.getLength().

    Therefore:

        edge.getLaneNumber()

    is used for lane count and:

        lane.getLength()

    is used for physical lane length.
    """

    if edge_id in edge_length_cache:

        return (

            edge_length_cache[
                edge_id
            ],

            edge_lane_count_cache[
                edge_id
            ]
        )


    try:

        # ----------------------------------------------------
        # Number of lanes
        # ----------------------------------------------------

        lane_count = (
            traci.edge.getLaneNumber(
                edge_id
            )
        )

        lane_count = max(
            1,
            lane_count
        )


        # ----------------------------------------------------
        # Obtain lane lengths
        # ----------------------------------------------------

        lane_lengths = []


        for lane_index in range(
            lane_count
        ):

            lane_id = (
                f"{edge_id}_{lane_index}"
            )

            try:

                lane_length = (
                    traci.lane.getLength(
                        lane_id
                    )
                )

                if lane_length > 0:

                    lane_lengths.append(
                        lane_length
                    )

            except traci.TraCIException:

                continue


        # ----------------------------------------------------
        # Representative edge length
        # ----------------------------------------------------

        if lane_lengths:

            edge_length = (

                sum(lane_lengths)
                /
                len(lane_lengths)
            )

        else:

            edge_length = 0.0


        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        edge_length_cache[
            edge_id
        ] = edge_length

        edge_lane_count_cache[
            edge_id
        ] = lane_count


        return (
            edge_length,
            lane_count
        )


    except traci.TraCIException:

        return (
            0.0,
            1
        )


# ============================================================
# GET EDGE SPEED LIMIT
# ============================================================

def get_edge_speed(edge_id):

    """
    Get representative maximum speed for an edge.

    SUMO exposes getMaxSpeed() at the lane level.

    We calculate the average maximum speed across
    all lanes belonging to the edge.
    """

    if edge_id in edge_speed_cache:

        return edge_speed_cache[
            edge_id
        ]


    try:

        lane_count = (
            traci.edge.getLaneNumber(
                edge_id
            )
        )

        lane_count = max(
            1,
            lane_count
        )


        lane_speeds = []


        for lane_index in range(
            lane_count
        ):

            lane_id = (
                f"{edge_id}_{lane_index}"
            )

            try:

                lane_speed = (
                    traci.lane.getMaxSpeed(
                        lane_id
                    )
                )

                if lane_speed > 0:

                    lane_speeds.append(
                        lane_speed
                    )

            except traci.TraCIException:

                continue


        if lane_speeds:

            edge_speed = (

                sum(lane_speeds)
                /
                len(lane_speeds)
            )

        else:

            # Fallback:
            # 50 km/h
            edge_speed = 13.89


        edge_speed_cache[
            edge_id
        ] = edge_speed


        return edge_speed


    except traci.TraCIException:

        return 13.89


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("=" * 75)
print("INDORE-01 SUMO + TraCI DATA COLLECTION")
print("=" * 75)

print()

print("SUMO configuration:")
print(SUMO_CONFIG)

print()

print("Simulation timestep:")
print(
    f"{STEP_LENGTH} seconds"
)

print()

print("Scenario aggregation window:")
print(
    f"{WINDOW_SIZE} seconds"
)

print()

print("Queue threshold:")
print(
    f"{QUEUE_SPEED_THRESHOLD} m/s"
)

print()

print("Congestion threshold:")
print(
    f"{CONGESTION_SPEED_THRESHOLD} m/s"
)

print()

print("Minimum density edge length:")
print(
    f"{MIN_DENSITY_EDGE_LENGTH_M} m"
)

print()

print("High-density threshold:")
print(
    f"{HIGH_DENSITY_THRESHOLD} "
    "vehicles/km/lane"
)

print()

print("Density percentile:")
print(
    f"P{DENSITY_PERCENTILE}"
)

print()

print("Raw telemetry:")
print(
    RAW_OUTPUT
)

print()

print("Scenario states:")
print(
    STATE_OUTPUT
)

print()


# ============================================================
# START SUMO
# ============================================================

sumo_command = [

    "sumo",

    "-c",
    SUMO_CONFIG,

    "--step-length",
    str(STEP_LENGTH),

    "--no-warnings",
]


try:

    traci.start(
        sumo_command
    )

except Exception as error:

    print()
    print("=" * 75)
    print("ERROR STARTING SUMO / TRACI")
    print("=" * 75)

    print()
    print(error)

    sys.exit(1)


print(
    "SUMO + TraCI started."
)

print()


# ============================================================
# OPEN OUTPUT FILES
# ============================================================

try:

    raw_file = open(

        RAW_OUTPUT,

        "w",

        newline="",

        encoding="utf-8"
    )

    state_file = open(

        STATE_OUTPUT,

        "w",

        newline="",

        encoding="utf-8"
    )


except PermissionError:

    print()
    print("=" * 75)
    print("PERMISSION ERROR")
    print("=" * 75)

    print()
    print(
        "Close the CSV files in Excel "
        "or another application."
    )

    try:

        traci.close()

    except:

        pass

    sys.exit(1)


# ============================================================
# CREATE CSV WRITERS
# ============================================================

raw_writer = csv.DictWriter(

    raw_file,

    fieldnames=RAW_FIELDS
)

state_writer = csv.DictWriter(

    state_file,

    fieldnames=STATE_FIELDS
)


raw_writer.writeheader()

state_writer.writeheader()


# ============================================================
# AGGREGATION WINDOW VARIABLES
# ============================================================

window_start = 0.0


# ------------------------------------------------------------
# Vehicle observations
# ------------------------------------------------------------

window_vehicle_samples = []

window_vehicle_ids = set()


# ------------------------------------------------------------
# Speed
# ------------------------------------------------------------

window_speeds = []


# ------------------------------------------------------------
# Acceleration
# ------------------------------------------------------------

window_accelerations = []


# ------------------------------------------------------------
# Waiting
# ------------------------------------------------------------

window_waiting_times = []


# ------------------------------------------------------------
# Queue
# ------------------------------------------------------------

window_queue_samples = []


# ------------------------------------------------------------
# Edge speed history
# ------------------------------------------------------------

window_edges = defaultdict(
    list
)


# ------------------------------------------------------------
# Edge geometry
# ------------------------------------------------------------

window_edge_lengths = {}

window_edge_lanes = {}


# ------------------------------------------------------------
# Edge density
# ------------------------------------------------------------

window_edge_density_samples = defaultdict(
    list
)

window_lane_density_samples = defaultdict(
    list
)


# ------------------------------------------------------------
# Congestion
# ------------------------------------------------------------

window_congested_edges = set()


# ------------------------------------------------------------
# High density
# ------------------------------------------------------------

window_high_density_edges = set()


# ============================================================
# WRITE SCENARIO WINDOW
# ============================================================

def write_scenario_window(
    start_time,
    end_time
):

    """
    Convert the current 5-second telemetry window
    into a scenario-state record.
    """

    if not window_vehicle_samples:

        return


    # ========================================================
    # VEHICLE COUNT
    # ========================================================

    vehicle_count = len(
        window_vehicle_ids
    )

    vehicle_observations = len(
        window_vehicle_samples
    )


    # ========================================================
    # SPEED
    # ========================================================

    mean_speed = calculate_mean(
        window_speeds
    )

    speed_std = calculate_std(
        window_speeds
    )

    min_speed = (

        min(window_speeds)

        if window_speeds

        else 0.0
    )

    max_speed = (

        max(window_speeds)

        if window_speeds

        else 0.0
    )


    # ========================================================
    # ACCELERATION
    # ========================================================

    mean_acceleration = calculate_mean(
        window_accelerations
    )


    # ========================================================
    # QUEUE
    # ========================================================

    queue_length = (

        sum(
            window_queue_samples
        )

        /

        len(
            window_queue_samples
        )

        if window_queue_samples

        else 0.0
    )


    queue_ratio = (

        queue_length
        /
        vehicle_count

        if vehicle_count > 0

        else 0.0
    )


    # ========================================================
    # WAITING
    # ========================================================

    mean_waiting_time = calculate_mean(
        window_waiting_times
    )

    max_waiting_time = (

        max(
            window_waiting_times
        )

        if window_waiting_times

        else 0.0
    )


    # ========================================================
    # DENSITY
    # ========================================================

    all_edge_density_values = []

    all_lane_density_values = []


    # --------------------------------------------------------
    # Collect edge density observations
    # --------------------------------------------------------

    for (

        edge_id,

        values

    ) in window_edge_density_samples.items():

        if not values:

            continue

        all_edge_density_values.extend(
            values
        )


    # --------------------------------------------------------
    # Collect lane density observations
    # --------------------------------------------------------

    for (

        edge_id,

        values

    ) in window_lane_density_samples.items():

        if not values:

            continue

        all_lane_density_values.extend(
            values
        )


    # --------------------------------------------------------
    # Mean density
    # --------------------------------------------------------

    mean_edge_density = calculate_mean(
        all_edge_density_values
    )

    mean_lane_density = calculate_mean(
        all_lane_density_values
    )


    # --------------------------------------------------------
    # P95 density
    # --------------------------------------------------------

    p95_edge_density = (
        calculate_percentile(
            all_edge_density_values,
            DENSITY_PERCENTILE
        )
    )

    p95_lane_density = (
        calculate_percentile(
            all_lane_density_values,
            DENSITY_PERCENTILE
        )
    )


    # --------------------------------------------------------
    # Maximum density
    # --------------------------------------------------------

    max_edge_density = (

        max(
            all_edge_density_values
        )

        if all_edge_density_values

        else 0.0
    )

    max_lane_density = (

        max(
            all_lane_density_values
        )

        if all_lane_density_values

        else 0.0
    )


    # ========================================================
    # TRAFFIC LIGHTS
    # ========================================================

    active_traffic_lights = 0

    green_signal_links = 0

    yellow_signal_links = 0

    red_signal_links = 0


    try:

        traffic_light_ids = (
            traci.trafficlight
            .getIDList()
        )

        active_traffic_lights = len(
            traffic_light_ids
        )


        for traffic_light_id in (
            traffic_light_ids
        ):

            try:

                signal_state = (
                    traci.trafficlight
                    .getRedYellowGreenState(
                        traffic_light_id
                    )
                )


                green_signal_links += (

                    signal_state.count(
                        "G"
                    )

                    +

                    signal_state.count(
                        "g"
                    )
                )


                yellow_signal_links += (

                    signal_state.count(
                        "Y"
                    )

                    +

                    signal_state.count(
                        "y"
                    )
                )


                red_signal_links += (

                    signal_state.count(
                        "R"
                    )

                    +

                    signal_state.count(
                        "r"
                    )
                )


            except traci.TraCIException:

                continue


    except traci.TraCIException:

        pass


    # ========================================================
    # WRITE SCENARIO STATE
    # ========================================================

    state_writer.writerow({

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        "window_start":
            round(
                start_time,
                2
            ),

        "window_end":
            round(
                end_time,
                2
            ),


        # ----------------------------------------------------
        # Traffic
        # ----------------------------------------------------

        "vehicle_count":
            vehicle_count,

        "vehicle_observations":
            vehicle_observations,


        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------

        "mean_speed":
            round(
                mean_speed,
                3
            ),

        "speed_std":
            round(
                speed_std,
                3
            ),

        "min_speed":
            round(
                min_speed,
                3
            ),

        "max_speed":
            round(
                max_speed,
                3
            ),


        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        "mean_acceleration":
            round(
                mean_acceleration,
                3
            ),


        # ----------------------------------------------------
        # Queue
        # ----------------------------------------------------

        "queue_length":
            round(
                queue_length,
                3
            ),

        "queue_ratio":
            round(
                queue_ratio,
                3
            ),


        # ----------------------------------------------------
        # Congestion
        # ----------------------------------------------------

        "congested_edges":
            len(
                window_congested_edges
            ),


        # ----------------------------------------------------
        # Density
        # ----------------------------------------------------

        "mean_edge_density_veh_per_km":
            round(
                mean_edge_density,
                3
            ),

        "p95_edge_density_veh_per_km":
            round(
                p95_edge_density,
                3
            ),

        "max_edge_density_veh_per_km":
            round(
                max_edge_density,
                3
            ),


        # ----------------------------------------------------
        # Lane density
        # ----------------------------------------------------

        "mean_lane_density_veh_per_km":
            round(
                mean_lane_density,
                3
            ),

        "p95_lane_density_veh_per_km":
            round(
                p95_lane_density,
                3
            ),

        "max_lane_density_veh_per_km":
            round(
                max_lane_density,
                3
            ),


        # ----------------------------------------------------
        # High density
        # ----------------------------------------------------

        "high_density_edges":
            len(
                window_high_density_edges
            ),


        # ----------------------------------------------------
        # Waiting
        # ----------------------------------------------------

        "mean_waiting_time":
            round(
                mean_waiting_time,
                3
            ),

        "max_waiting_time":
            round(
                max_waiting_time,
                3
            ),


        # ----------------------------------------------------
        # Traffic lights
        # ----------------------------------------------------

        "active_traffic_lights":
            active_traffic_lights,

        "green_signal_links":
            green_signal_links,

        "yellow_signal_links":
            yellow_signal_links,

        "red_signal_links":
            red_signal_links,
    })


# ============================================================
# MAIN SIMULATION
# ============================================================

step = 0


try:

    while (

        traci.simulation
        .getMinExpectedNumber()

        >

        0
    ):


        # ====================================================
        # ADVANCE SUMO
        # ====================================================

        traci.simulationStep()


        current_time = (
            traci.simulation.getTime()
        )


        # ====================================================
        # ACTIVE VEHICLES
        # ====================================================

        vehicle_ids = (
            traci.vehicle.getIDList()
        )


        # ====================================================
        # VEHICLE TELEMETRY
        # ====================================================

        for vehicle_id in vehicle_ids:

            try:

                # ------------------------------------------------
                # Position
                # ------------------------------------------------

                position = (
                    traci.vehicle
                    .getPosition(
                        vehicle_id
                    )
                )

                x = position[0]

                y = position[1]


                # ------------------------------------------------
                # Kinematics
                # ------------------------------------------------

                speed = (
                    traci.vehicle
                    .getSpeed(
                        vehicle_id
                    )
                )

                acceleration = (
                    traci.vehicle
                    .getAcceleration(
                        vehicle_id
                    )
                )

                angle = (
                    traci.vehicle
                    .getAngle(
                        vehicle_id
                    )
                )


                # ------------------------------------------------
                # Road
                # ------------------------------------------------

                edge_id = (
                    traci.vehicle
                    .getRoadID(
                        vehicle_id
                    )
                )

                lane_id = (
                    traci.vehicle
                    .getLaneID(
                        vehicle_id
                    )
                )

                lane_position = (
                    traci.vehicle
                    .getLanePosition(
                        vehicle_id
                    )
                )


                # ------------------------------------------------
                # Vehicle
                # ------------------------------------------------

                vehicle_type = (
                    traci.vehicle
                    .getTypeID(
                        vehicle_id
                    )
                )


                # ------------------------------------------------
                # Route
                # ------------------------------------------------

                route_id = (
                    traci.vehicle
                    .getRouteID(
                        vehicle_id
                    )
                )

                route_index = (
                    traci.vehicle
                    .getRouteIndex(
                        vehicle_id
                    )
                )


                # ------------------------------------------------
                # Waiting
                # ------------------------------------------------

                waiting_time = (
                    traci.vehicle
                    .getWaitingTime(
                        vehicle_id
                    )
                )

                accumulated_waiting_time = (
                    traci.vehicle
                    .getAccumulatedWaitingTime(
                        vehicle_id
                    )
                )


                # ------------------------------------------------
                # Distance
                # ------------------------------------------------

                distance_travelled = (
                    traci.vehicle
                    .getDistance(
                        vehicle_id
                    )
                )


                # ====================================================
                # RAW TELEMETRY
                # ====================================================

                raw_writer.writerow({

                    "time":
                        round(
                            current_time,
                            2
                        ),

                    "vehicle_id":
                        vehicle_id,

                    "vehicle_type":
                        vehicle_type,

                    "x":
                        round(
                            x,
                            3
                        ),

                    "y":
                        round(
                            y,
                            3
                        ),

                    "speed":
                        round(
                            speed,
                            3
                        ),

                    "acceleration":
                        round(
                            acceleration,
                            3
                        ),

                    "angle":
                        round(
                            angle,
                            3
                        ),

                    "edge_id":
                        edge_id,

                    "lane_id":
                        lane_id,

                    "lane_position":
                        round(
                            lane_position,
                            3
                        ),

                    "route_id":
                        route_id,

                    "route_index":
                        route_index,

                    "waiting_time":
                        round(
                            waiting_time,
                            3
                        ),

                    "accumulated_waiting_time":
                        round(
                            accumulated_waiting_time,
                            3
                        ),

                    "distance_travelled":
                        round(
                            distance_travelled,
                            3
                        ),
                })


                # ====================================================
                # WINDOW AGGREGATION
                # ====================================================

                window_vehicle_samples.append(
                    vehicle_id
                )

                window_vehicle_ids.add(
                    vehicle_id
                )

                window_speeds.append(
                    speed
                )

                window_accelerations.append(
                    acceleration
                )

                window_waiting_times.append(
                    waiting_time
                )


                # ====================================================
                # QUEUE
                # ====================================================

                if (
                    speed
                    <
                    QUEUE_SPEED_THRESHOLD
                ):

                    window_queue_samples.append(
                        1
                    )

                else:

                    window_queue_samples.append(
                        0
                    )


                # ====================================================
                # EDGE PROCESSING
                # ====================================================

                # Ignore SUMO internal junction edges.
                if (

                    edge_id

                    and

                    not edge_id.startswith(
                        ":"
                    )
                ):


                    # ------------------------------------------------
                    # Speed history
                    # ------------------------------------------------

                    window_edges[
                        edge_id
                    ].append(
                        speed
                    )


                    # ------------------------------------------------
                    # Geometry
                    # ------------------------------------------------

                    if (
                        edge_id
                        not in
                        window_edge_lengths
                    ):

                        (
                            edge_length,
                            lane_count
                        ) = get_edge_geometry(
                            edge_id
                        )

                        window_edge_lengths[
                            edge_id
                        ] = edge_length

                        window_edge_lanes[
                            edge_id
                        ] = lane_count


                    # ------------------------------------------------
                    # Congestion
                    # ------------------------------------------------

                    edge_speed_limit = (
                        get_edge_speed(
                            edge_id
                        )
                    )

                    edge_mean_speed = (
                        calculate_mean(
                            window_edges[
                                edge_id
                            ]
                        )
                    )


                    # ------------------------------------------------
                    # Absolute congestion
                    # ------------------------------------------------

                    if (

                        edge_mean_speed

                        <

                        CONGESTION_SPEED_THRESHOLD
                    ):

                        window_congested_edges.add(
                            edge_id
                        )


                    # ------------------------------------------------
                    # Relative congestion
                    # ------------------------------------------------

                    elif (

                        edge_speed_limit

                        >

                        0

                        and

                        edge_mean_speed

                        <

                        (
                            0.5
                            *
                            edge_speed_limit
                        )
                    ):

                        window_congested_edges.add(
                            edge_id
                        )


            except traci.TraCIException as error:

                print(

                    f"WARNING: vehicle "
                    f"{vehicle_id}: "
                    f"{error}"
                )


        # ====================================================
        # COUNT VEHICLES ON EACH EDGE
        # ====================================================

        edge_vehicle_counts = defaultdict(
            int
        )


        for vehicle_id in vehicle_ids:

            try:

                current_edge = (
                    traci.vehicle
                    .getRoadID(
                        vehicle_id
                    )
                )


                if (

                    current_edge

                    and

                    not current_edge.startswith(
                        ":"
                    )
                ):

                    edge_vehicle_counts[
                        current_edge
                    ] += 1


            except traci.TraCIException:

                continue


        # ====================================================
        # EDGE DENSITY
        # ====================================================

        for (

            current_edge,

            vehicles_on_edge

        ) in edge_vehicle_counts.items():


            # ------------------------------------------------
            # Get geometry
            # ------------------------------------------------

            edge_length = (
                window_edge_lengths.get(
                    current_edge
                )
            )

            lane_count = (
                window_edge_lanes.get(
                    current_edge,
                    1
                )
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Ignore short OSM connector edges.
            # ------------------------------------------------

            if (

                edge_length is None

                or

                edge_length
                <
                MIN_DENSITY_EDGE_LENGTH_M
            ):

                continue


            # ====================================================
            # VEHICLES / KM
            # ====================================================

            density_veh_per_km = (

                vehicles_on_edge

                /

                (
                    edge_length
                    /
                    1000.0
                )
            )


            # ====================================================
            # VEHICLES / KM / LANE
            # ====================================================

            lane_density = (

                density_veh_per_km

                /

                max(
                    1,
                    lane_count
                )
            )


            # ====================================================
            # STORE
            # ====================================================

            window_edge_density_samples[
                current_edge
            ].append(
                density_veh_per_km
            )

            window_lane_density_samples[
                current_edge
            ].append(
                lane_density
            )


            # ====================================================
            # HIGH DENSITY EDGE
            # ====================================================

            if (

                lane_density

                >=

                HIGH_DENSITY_THRESHOLD
            ):

                window_high_density_edges.add(
                    current_edge
                )


        # ====================================================
        # STEP COUNTER
        # ====================================================

        step += 1


        # ====================================================
        # WRITE 5-SECOND WINDOW
        # ====================================================

        if (

            current_time

            >=

            (
                window_start
                +
                WINDOW_SIZE
            )
        ):


            write_scenario_window(

                window_start,

                current_time
            )


            # ------------------------------------------------
            # RESET
            # ------------------------------------------------

            window_start = current_time

            window_vehicle_samples = []

            window_vehicle_ids = set()

            window_speeds = []

            window_accelerations = []

            window_waiting_times = []

            window_queue_samples = []

            window_edges = defaultdict(
                list
            )

            window_edge_lengths = {}

            window_edge_lanes = {}

            window_edge_density_samples = defaultdict(
                list
            )

            window_lane_density_samples = defaultdict(
                list
            )

            window_congested_edges = set()

            window_high_density_edges = set()


        # ====================================================
        # PROGRESS
        # ====================================================

        if step % 1000 == 0:

            print(

                f"Time: "
                f"{current_time:8.1f} s | "

                f"Active vehicles: "
                f"{len(vehicle_ids):4d} | "

                f"Steps: "
                f"{step:6d}"
            )


    # ========================================================
    # FINAL PARTIAL WINDOW
    # ========================================================

    if window_vehicle_samples:

        final_time = (
            traci.simulation.getTime()
        )

        write_scenario_window(

            window_start,

            final_time
        )


finally:

    # ========================================================
    # CLOSE FILES
    # ========================================================

    try:

        raw_file.close()

    except:

        pass


    try:

        state_file.close()

    except:

        pass


    # ========================================================
    # CLOSE TRACI
    # ========================================================

    try:

        traci.close()

    except:

        pass


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 75)
print("INDORE-01 DATA COLLECTION COMPLETE")
print("=" * 75)

print()

print("Raw telemetry:")
print(
    RAW_OUTPUT
)

print()

print("Scenario states:")
print(
    STATE_OUTPUT
)

print()

print("Simulation steps:")
print(
    step
)

print()

print("Simulation timestep:")
print(
    f"{STEP_LENGTH} seconds"
)

print()

print("Scenario aggregation:")
print(
    f"{WINDOW_SIZE} seconds"
)

print()

print("Minimum density edge length:")
print(
    f"{MIN_DENSITY_EDGE_LENGTH_M} m"
)

print()

print("Density statistics:")
print(
    "Mean + P95 + Maximum"
)

print()

print("Done.")