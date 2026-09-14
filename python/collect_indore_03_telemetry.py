import csv
import os
import sys
import math
from collections import defaultdict

import traci


# ============================================================
# INDORE-03 — HEAVY CONGESTION
# ============================================================

SCENARIO_ID = "INDORE_03"
SCENARIO_NAME = "HEAVY_CONGESTION"


# ============================================================
# SUMO CONFIGURATION
# ============================================================

SUMO_CONFIG = (
    r"C:\Users\jinda\Sumo\Indore-03"
    r"\osm.sumocfg"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    r"C:\C-V2X-OODA\datasets\observe_ds"
)


RAW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "indore_03_telemetry_v1.csv"
)


STATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "indore_03_scenario_states_v1.csv"
)


# ============================================================
# SIMULATION SETTINGS
# ============================================================

STEP_LENGTH = 0.1

WINDOW_SIZE = 5.0


# ============================================================
# TRAFFIC THRESHOLDS
# ============================================================

QUEUE_SPEED_THRESHOLD = 2.0

CONGESTION_SPEED_THRESHOLD = 5.0


# ============================================================
# DENSITY SETTINGS
# ============================================================

# Ignore very short OSM connector/geometry edges.
#
# This prevents artificial values such as:
#
# 5000 veh/km
#
# from extremely short edges.

MIN_DENSITY_EDGE_LENGTH_M = 20.0


# High-density threshold:
# vehicles / km / lane

HIGH_DENSITY_THRESHOLD = 25.0


# Robust density statistic

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
# RAW TELEMETRY CSV FIELDS
# ============================================================

RAW_FIELDS = [

    "time",

    "scenario_id",
    "scenario_name",

    "vehicle_id",
    "vehicle_type",

    "x",
    "y",

    "speed",
    "acceleration",
    "angle",

    "edge_id",
    "lane_id",
    "lane_position",

    "route_id",
    "route_index",

    "waiting_time",
    "accumulated_waiting_time",

    "distance_travelled",
]


# ============================================================
# SCENARIO STATE CSV FIELDS
# ============================================================

STATE_FIELDS = [

    "scenario_id",
    "scenario_name",

    "window_start",
    "window_end",

    "vehicle_count",
    "vehicle_observations",

    "mean_speed",
    "speed_std",
    "min_speed",
    "max_speed",

    "mean_acceleration",

    "queue_length",
    "queue_ratio",

    "congested_edges",

    "mean_edge_density_veh_per_km",
    "p95_edge_density_veh_per_km",
    "max_edge_density_veh_per_km",

    "mean_lane_density_veh_per_km",
    "p95_lane_density_veh_per_km",
    "max_lane_density_veh_per_km",

    "high_density_edges",

    "mean_waiting_time",
    "max_waiting_time",

    "active_traffic_lights",
    "green_signal_links",
    "yellow_signal_links",
    "red_signal_links",
]


# ============================================================
# STATISTICS
# ============================================================

def calculate_mean(values):

    if not values:
        return 0.0

    return (
        sum(values)
        /
        len(values)
    )


def calculate_std(values):

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
    p
):

    if not values:
        return 0.0

    values = sorted(
        values
    )

    if len(values) == 1:
        return values[0]

    position = (

        (len(values) - 1)

        *

        (
            p
            /
            100.0
        )
    )

    lower = int(
        math.floor(position)
    )

    upper = int(
        math.ceil(position)
    )

    if lower == upper:
        return values[lower]

    fraction = (
        position
        -
        lower
    )

    return (

        values[lower]

        +

        (
            values[upper]
            -
            values[lower]
        )

        *

        fraction
    )


# ============================================================
# SUMO CACHE
# ============================================================

edge_geometry_cache = {}

edge_speed_cache = {}


# ============================================================
# GET EDGE LENGTH + LANE COUNT
# ============================================================

def get_edge_geometry(
    edge_id
):

    """
    SUMO 1.27.1 does not provide:

        traci.edge.getLength()

    Therefore the length is obtained from
    the individual lanes.
    """

    if edge_id in edge_geometry_cache:

        return edge_geometry_cache[
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


        lane_lengths = []


        for lane_index in range(
            lane_count
        ):

            lane_id = (
                f"{edge_id}_{lane_index}"
            )

            try:

                length = (
                    traci.lane.getLength(
                        lane_id
                    )
                )

                if length > 0:

                    lane_lengths.append(
                        length
                    )

            except traci.TraCIException:

                continue


        if lane_lengths:

            edge_length = (

                sum(lane_lengths)
                /
                len(lane_lengths)
            )

        else:

            edge_length = 0.0


        result = (
            edge_length,
            lane_count
        )


        edge_geometry_cache[
            edge_id
        ] = result


        return result


    except traci.TraCIException:

        return (
            0.0,
            1
        )


# ============================================================
# GET EDGE SPEED LIMIT
# ============================================================

def get_edge_speed(
    edge_id
):

    """
    SUMO 1.27.1 does not provide:

        traci.edge.getMaxSpeed()

    Therefore we use:

        traci.lane.getMaxSpeed()
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


        speeds = []


        for lane_index in range(
            lane_count
        ):

            lane_id = (
                f"{edge_id}_{lane_index}"
            )

            try:

                speed = (
                    traci.lane.getMaxSpeed(
                        lane_id
                    )
                )

                if speed > 0:

                    speeds.append(
                        speed
                    )

            except traci.TraCIException:

                continue


        if speeds:

            result = calculate_mean(
                speeds
            )

        else:

            # Fallback: 50 km/h
            result = 13.89


        edge_speed_cache[
            edge_id
        ] = result


        return result


    except traci.TraCIException:

        return 13.89


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 75)
print("INDORE-03 — HEAVY CONGESTION")
print("SUMO + TraCI TELEMETRY COLLECTION")
print("=" * 75)

print()

print("Scenario ID:")
print(SCENARIO_ID)

print()

print("Scenario:")
print(SCENARIO_NAME)

print()

print("SUMO configuration:")
print(SUMO_CONFIG)

print()

print("Raw telemetry output:")
print(RAW_OUTPUT)

print()

print("Scenario-state output:")
print(STATE_OUTPUT)

print()

print("Simulation timestep:")
print(
    f"{STEP_LENGTH} seconds"
)

print()

print("Aggregation window:")
print(
    f"{WINDOW_SIZE} seconds"
)

print()

print("Minimum density edge length:")
print(
    f"{MIN_DENSITY_EDGE_LENGTH_M} m"
)

print()

print("P95 density:")
print(
    f"P{DENSITY_PERCENTILE}"
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
    print("ERROR STARTING SUMO")
    print("=" * 75)
    print()

    print(error)

    sys.exit(1)


print(
    "SUMO + TraCI started successfully."
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
        "Close the CSV files if they are open "
        "in Excel or another application."
    )

    try:
        traci.close()
    except:
        pass

    sys.exit(1)


# ============================================================
# CSV WRITERS
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
# WINDOW STATE
# ============================================================

window_start = 0.0


window_vehicle_ids = set()

window_vehicle_samples = []

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


# ============================================================
# WRITE ONE SCENARIO WINDOW
# ============================================================

def write_scenario_window(
    start_time,
    end_time
):

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

        calculate_mean(
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

    edge_density_values = []

    lane_density_values = []


    for values in (
        window_edge_density_samples.values()
    ):

        edge_density_values.extend(
            values
        )


    for values in (
        window_lane_density_samples.values()
    ):

        lane_density_values.extend(
            values
        )


    mean_edge_density = (
        calculate_mean(
            edge_density_values
        )
    )


    p95_edge_density = (
        calculate_percentile(
            edge_density_values,
            DENSITY_PERCENTILE
        )
    )


    max_edge_density = (

        max(
            edge_density_values
        )

        if edge_density_values

        else 0.0
    )


    mean_lane_density = (
        calculate_mean(
            lane_density_values
        )
    )


    p95_lane_density = (
        calculate_percentile(
            lane_density_values,
            DENSITY_PERCENTILE
        )
    )


    max_lane_density = (

        max(
            lane_density_values
        )

        if lane_density_values

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

        traffic_lights = (
            traci.trafficlight
            .getIDList()
        )


        active_traffic_lights = len(
            traffic_lights
        )


        for tl_id in traffic_lights:

            try:

                state = (
                    traci.trafficlight
                    .getRedYellowGreenState(
                        tl_id
                    )
                )


                green_signal_links += (

                    state.count("G")
                    +
                    state.count("g")
                )


                yellow_signal_links += (

                    state.count("Y")
                    +
                    state.count("y")
                )


                red_signal_links += (

                    state.count("R")
                    +
                    state.count("r")
                )


            except traci.TraCIException:

                continue


    except traci.TraCIException:

        pass


    # ========================================================
    # WRITE STATE
    # ========================================================

    state_writer.writerow({

        "scenario_id":
            SCENARIO_ID,

        "scenario_name":
            SCENARIO_NAME,

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

        "vehicle_count":
            vehicle_count,

        "vehicle_observations":
            vehicle_observations,

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

        "mean_acceleration":
            round(
                mean_acceleration,
                3
            ),

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

        "congested_edges":
            len(
                window_congested_edges
            ),

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

        "high_density_edges":
            len(
                window_high_density_edges
            ),

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
# MAIN SIMULATION LOOP
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
            traci.simulation
            .getTime()
        )


        # ====================================================
        # ACTIVE VEHICLES
        # ====================================================

        vehicle_ids = (
            traci.vehicle
            .getIDList()
        )


        # ====================================================
        # VEHICLE TELEMETRY
        # ====================================================

        for vehicle_id in vehicle_ids:

            try:

                # ------------------------------------------------
                # Position
                # ------------------------------------------------

                x, y = (
                    traci.vehicle
                    .getPosition(
                        vehicle_id
                    )
                )


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
                # RAW CSV
                # ====================================================

                raw_writer.writerow({

                    "time":
                        round(
                            current_time,
                            2
                        ),

                    "scenario_id":
                        SCENARIO_ID,

                    "scenario_name":
                        SCENARIO_NAME,

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
                # AGGREGATION
                # ====================================================

                window_vehicle_ids.add(
                    vehicle_id
                )


                window_vehicle_samples.append(
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


                # ------------------------------------------------
                # Queue
                # ------------------------------------------------

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
                # EDGE
                # ====================================================

                if (

                    edge_id

                    and

                    not edge_id.startswith(
                        ":"
                    )
                ):


                    window_edges[
                        edge_id
                    ].append(
                        speed
                    )


                    # ------------------------------------------------
                    # Geometry
                    # ------------------------------------------------

                    if edge_id not in (
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

                    edge_mean_speed = (
                        calculate_mean(
                            window_edges[
                                edge_id
                            ]
                        )
                    )


                    edge_speed_limit = (
                        get_edge_speed(
                            edge_id
                        )
                    )


                    if (

                        edge_mean_speed
                        <
                        CONGESTION_SPEED_THRESHOLD
                    ):

                        window_congested_edges.add(
                            edge_id
                        )


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
        # VEHICLES PER EDGE
        # ====================================================

        edge_vehicle_counts = defaultdict(
            int
        )


        for vehicle_id in vehicle_ids:

            try:

                edge_id = (
                    traci.vehicle
                    .getRoadID(
                        vehicle_id
                    )
                )


                if (

                    edge_id

                    and

                    not edge_id.startswith(
                        ":"
                    )
                ):

                    edge_vehicle_counts[
                        edge_id
                    ] += 1


            except traci.TraCIException:

                continue


        # ====================================================
        # DENSITY
        # ====================================================

        for (

            edge_id,
            vehicles_on_edge

        ) in edge_vehicle_counts.items():


            edge_length = (
                window_edge_lengths.get(
                    edge_id
                )
            )


            lane_count = (
                window_edge_lanes.get(
                    edge_id,
                    1
                )
            )


            # ------------------------------------------------
            # Density filter
            # ------------------------------------------------

            if (

                edge_length is None

                or

                edge_length
                <
                MIN_DENSITY_EDGE_LENGTH_M
            ):

                continue


            # ------------------------------------------------
            # Vehicles/km
            # ------------------------------------------------

            density = (

                vehicles_on_edge

                /

                (
                    edge_length
                    /
                    1000.0
                )
            )


            # ------------------------------------------------
            # Vehicles/km/lane
            # ------------------------------------------------

            lane_density = (

                density

                /

                max(
                    1,
                    lane_count
                )
            )


            window_edge_density_samples[
                edge_id
            ].append(
                density
            )


            window_lane_density_samples[
                edge_id
            ].append(
                lane_density
            )


            # ------------------------------------------------
            # High-density edge
            # ------------------------------------------------

            if (

                lane_density

                >=

                HIGH_DENSITY_THRESHOLD
            ):

                window_high_density_edges.add(
                    edge_id
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
            # RESET WINDOW
            # ------------------------------------------------

            window_start = current_time


            window_vehicle_ids = set()

            window_vehicle_samples = []

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
                f"{step:7d}"
            )


    # ========================================================
    # FINAL PARTIAL WINDOW
    # ========================================================

    if window_vehicle_samples:

        final_time = (
            traci.simulation
            .getTime()
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
# FINAL MESSAGE
# ============================================================

print()
print("=" * 75)
print("INDORE-03 DATA COLLECTION COMPLETE")
print("=" * 75)

print()

print("Scenario:")
print(
    f"{SCENARIO_ID} — {SCENARIO_NAME}"
)

print()

print("Simulation steps:")
print(step)

print()

print("Raw telemetry:")
print(RAW_OUTPUT)

print()

print("Scenario states:")
print(STATE_OUTPUT)

print()

print("Density filter:")
print(
    f"Edges < {MIN_DENSITY_EDGE_LENGTH_M} m ignored"
)

print()

print(
    f"P{DENSITY_PERCENTILE} density calculated"
)

print()

print("Done.")