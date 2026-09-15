import csv
import math
import os
import traci

SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"

TELEMETRY_OUTPUT = "indore_04_telemetry_v1.csv"
SCENARIO_OUTPUT = "indore_04_scenario_states_v1.csv"

STEP_LENGTH = 0.1
WINDOW_LENGTH = 5.0

QUEUE_SPEED = 2.0
CONGESTION_SPEED = 5.0
MIN_EDGE_LENGTH = 20.0

AMBULANCE_ID = "AMBULANCE_01"


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():

    telemetry_fields = [
        "time",
        "scenario_id",
        "scenario_name",
        "vehicle_id",
        "vehicle_type",
        "is_ambulance",
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

    state_fields = [
        "scenario_id",
        "scenario_name",
        "window_start",
        "window_end",
        "vehicle_count",
        "vehicle_observations",
        "ambulance_present",
        "ambulance_speed",
        "ambulance_waiting_time",
        "ambulance_distance_travelled",
        "ambulance_x",
        "ambulance_y",
        "ambulance_edge_id",
        "ambulance_lane_id",
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
        "mean_waiting_time",
        "max_waiting_time",
        "active_traffic_lights",
        "green_signal_links",
        "yellow_signal_links",
        "red_signal_links",
    ]

    with open(TELEMETRY_OUTPUT, "w", newline="", encoding="utf-8") as tf, \
         open(SCENARIO_OUTPUT, "w", newline="", encoding="utf-8") as sf:

        telemetry_writer = csv.DictWriter(
            tf,
            fieldnames=telemetry_fields
        )

        state_writer = csv.DictWriter(
            sf,
            fieldnames=state_fields
        )

        telemetry_writer.writeheader()
        state_writer.writeheader()

        traci.start([
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG,
            "--step-length",
            str(STEP_LENGTH),
            "--time-to-teleport",
            "-1",
            "--no-step-log",
            "true",
        ])

        window_start = 0.0
        window_records = []

        try:

            while traci.simulation.getMinExpectedNumber() > 0:

                traci.simulationStep()

                sim_time = finite(traci.simulation.getTime())
                active_ids = traci.vehicle.getIDList()

                current_records = []

                for vid in active_ids:

                    try:

                        x, y = traci.vehicle.getPosition(vid)

                        record = {
                            "time": sim_time,
                            "scenario_id": "INDORE_04",
                            "scenario_name": "AMBULANCE_EMERGENCY",
                            "vehicle_id": vid,
                            "vehicle_type": traci.vehicle.getTypeID(vid),
                            "is_ambulance": int(vid == AMBULANCE_ID),
                            "x": finite(x),
                            "y": finite(y),
                            "speed": finite(
                                traci.vehicle.getSpeed(vid)
                            ),
                            "acceleration": finite(
                                traci.vehicle.getAcceleration(vid)
                            ),
                            "angle": finite(
                                traci.vehicle.getAngle(vid)
                            ),
                            "edge_id": traci.vehicle.getRoadID(vid),
                            "lane_id": traci.vehicle.getLaneID(vid),
                            "lane_position": finite(
                                traci.vehicle.getLanePosition(vid)
                            ),
                            "route_id": traci.vehicle.getRouteID(vid),
                            "route_index": traci.vehicle.getRouteIndex(vid),
                            "waiting_time": finite(
                                traci.vehicle.getWaitingTime(vid)
                            ),
                            "accumulated_waiting_time": finite(
                                traci.vehicle.getAccumulatedWaitingTime(vid)
                            ),
                            "distance_travelled": finite(
                                traci.vehicle.getDistance(vid)
                            ),
                        }

                        telemetry_writer.writerow(record)
                        current_records.append(record)
                        window_records.append(record)

                    except traci.TraCIException:
                        continue

                if sim_time - window_start >= WINDOW_LENGTH:

                    vehicle_ids = {
                        r["vehicle_id"]
                        for r in window_records
                    }

                    speeds = [
                        r["speed"]
                        for r in window_records
                    ]

                    accelerations = [
                        r["acceleration"]
                        for r in window_records
                    ]

                    waits = [
                        r["waiting_time"]
                        for r in window_records
                    ]

                    # Queue observations
                    queued = sum(
                        1
                        for r in window_records
                        if r["speed"] < QUEUE_SPEED
                    )

                    queue_ratio = (
                        queued / len(window_records)
                        if window_records else 0.0
                    )

                    # Per-edge observations
                    edge_vehicle_ids = {}
                    edge_speeds = {}

                    for r in current_records:

                        edge = r["edge_id"]

                        if not edge or edge.startswith(":"):
                            continue

                        edge_vehicle_ids.setdefault(
                            edge, set()
                        ).add(r["vehicle_id"])

                        edge_speeds.setdefault(
                            edge, []
                        ).append(r["speed"])

                    densities = []

                    for edge, ids in edge_vehicle_ids.items():

                        try:
                            lane_ids = traci.edge.getLaneNumber(edge)

                            if lane_ids <= 0:
                                continue

                            lane_id = f"{edge}_0"
                            length = traci.lane.getLength(lane_id)

                            if length >= MIN_EDGE_LENGTH:

                                density = (
                                    len(ids)
                                    / (length / 1000.0)
                                )

                                densities.append(density)

                        except traci.TraCIException:
                            continue

                    congested_edges = sum(
                        1
                        for values in edge_speeds.values()
                        if values
                        and sum(values) / len(values)
                        < CONGESTION_SPEED
                    )

                    # Traffic signals
                    active_tls = 0
                    green_links = 0
                    yellow_links = 0
                    red_links = 0

                    for tls_id in traci.trafficlight.getIDList():

                        try:

                            state = (
                                traci.trafficlight
                                .getRedYellowGreenState(tls_id)
                            )

                            if not state:
                                continue

                            active_tls += 1

                            green_links += (
                                state.count("G")
                                + state.count("g")
                            )

                            yellow_links += (
                                state.count("Y")
                                + state.count("y")
                            )

                            red_links += (
                                state.count("R")
                                + state.count("r")
                            )

                        except traci.TraCIException:
                            continue

                    # Ambulance state
                    ambulance_records = [
                        r
                        for r in window_records
                        if r["vehicle_id"] == AMBULANCE_ID
                    ]

                    if ambulance_records:

                        a = ambulance_records[-1]

                        ambulance_present = 1
                        ambulance_speed = a["speed"]
                        ambulance_waiting = a["waiting_time"]
                        ambulance_distance = a[
                            "distance_travelled"
                        ]
                        ambulance_x = a["x"]
                        ambulance_y = a["y"]
                        ambulance_edge = a["edge_id"]
                        ambulance_lane = a["lane_id"]

                    else:

                        ambulance_present = 0
                        ambulance_speed = 0.0
                        ambulance_waiting = 0.0
                        ambulance_distance = 0.0
                        ambulance_x = 0.0
                        ambulance_y = 0.0
                        ambulance_edge = ""
                        ambulance_lane = ""

                    # Aggregate mobility
                    if speeds:

                        mean_speed = sum(speeds) / len(speeds)

                        variance = sum(
                            (s - mean_speed) ** 2
                            for s in speeds
                        ) / len(speeds)

                        speed_std = math.sqrt(variance)

                        min_speed = min(speeds)
                        max_speed = max(speeds)

                    else:

                        mean_speed = 0.0
                        speed_std = 0.0
                        min_speed = 0.0
                        max_speed = 0.0

                    mean_acceleration = (
                        sum(accelerations)
                        / len(accelerations)
                        if accelerations else 0.0
                    )

                    mean_waiting = (
                        sum(waits)
                        / len(waits)
                        if waits else 0.0
                    )

                    max_waiting = (
                        max(waits)
                        if waits else 0.0
                    )

                    state_writer.writerow({
                        "scenario_id": "INDORE_04",
                        "scenario_name": "AMBULANCE_EMERGENCY",
                        "window_start": window_start,
                        "window_end": sim_time,
                        "vehicle_count": len(vehicle_ids),
                        "vehicle_observations": len(window_records),
                        "ambulance_present": ambulance_present,
                        "ambulance_speed": ambulance_speed,
                        "ambulance_waiting_time": ambulance_waiting,
                        "ambulance_distance_travelled": ambulance_distance,
                        "ambulance_x": ambulance_x,
                        "ambulance_y": ambulance_y,
                        "ambulance_edge_id": ambulance_edge,
                        "ambulance_lane_id": ambulance_lane,
                        "mean_speed": mean_speed,
                        "speed_std": speed_std,
                        "min_speed": min_speed,
                        "max_speed": max_speed,
                        "mean_acceleration": mean_acceleration,
                        "queue_length": queued,
                        "queue_ratio": queue_ratio,
                        "congested_edges": congested_edges,
                        "mean_edge_density_veh_per_km": (
                            sum(densities) / len(densities)
                            if densities else 0.0
                        ),
                        "p95_edge_density_veh_per_km": percentile(
                            densities, 0.95
                        ),
                        "max_edge_density_veh_per_km": (
                            max(densities)
                            if densities else 0.0
                        ),
                        "mean_waiting_time": mean_waiting,
                        "max_waiting_time": max_waiting,
                        "active_traffic_lights": active_tls,
                        "green_signal_links": green_links,
                        "yellow_signal_links": yellow_links,
                        "red_signal_links": red_links,
                    })

                    window_start = sim_time
                    window_records = []

        finally:
            traci.close()

    print()
    print("==========================================")
    print("SITUATION 4 TELEMETRY COMPLETE")
    print("==========================================")
    print(f"Telemetry : {os.path.abspath(TELEMETRY_OUTPUT)}")
    print(f"Context   : {os.path.abspath(SCENARIO_OUTPUT)}")
    print("==========================================")


if __name__ == "__main__":
    main()
