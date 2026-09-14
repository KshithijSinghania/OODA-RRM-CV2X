import csv
import os
import sys
import time

import traci


# ============================================================
# CONFIGURATION
# ============================================================

SUMO_BINARY = "sumo"

SUMO_CONFIG = (
    r"C:\C-V2X-OODA\sumo\topology_01_intersection"
    r"\test_network.net.xml"
)

ROUTE_FILE = (
    r"C:\C-V2X-OODA\sumo\topology_01_intersection"
    r"\test.rou.xml"
)

OUTPUT_FILE = (
    r"C:\C-V2X-OODA\datasets\observe_ds"
    r"\vehicle_telemetry.csv"
)

SIMULATION_END = 60.0
STEP_LENGTH = 0.1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

output_directory = os.path.dirname(OUTPUT_FILE)

os.makedirs(output_directory, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(SUMO_CONFIG):
    print("ERROR: Network file not found:")
    print(SUMO_CONFIG)
    sys.exit(1)

if not os.path.exists(ROUTE_FILE):
    print("ERROR: Route file not found:")
    print(ROUTE_FILE)
    sys.exit(1)


# ============================================================
# START SUMO
# ============================================================

sumo_cmd = [
    SUMO_BINARY,
    "-n",
    SUMO_CONFIG,
    "-r",
    ROUTE_FILE,
    "--begin",
    "0",
    "--end",
    str(SIMULATION_END),
    "--step-length",
    str(STEP_LENGTH),
]

print("Starting SUMO...")
print("Network :", SUMO_CONFIG)
print("Routes  :", ROUTE_FILE)
print("Output  :", OUTPUT_FILE)

traci.start(sumo_cmd)

print("SUMO started successfully.")
print()


# ============================================================
# CSV OUTPUT
# ============================================================

fieldnames = [
    "time",
    "vehicle_id",
    "vehicle_type",

    # Position
    "x",
    "y",

    # Kinematics
    "speed",
    "acceleration",
    "angle",

    # Road information
    "edge_id",
    "lane_id",
    "lane_position",

    # Route information
    "route_id",
    "route_index",

    # Vehicle state
    "waiting_time",
    "accumulated_waiting_time",
    "distance_travelled",
]


with open(
    OUTPUT_FILE,
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    step = 0

    while traci.simulation.getTime() < SIMULATION_END:

        # Advance SUMO by one simulation step
        traci.simulationStep()

        current_time = traci.simulation.getTime()

        vehicle_ids = traci.vehicle.getIDList()

        # ----------------------------------------------------
        # Collect telemetry for every active vehicle
        # ----------------------------------------------------

        for vehicle_id in vehicle_ids:

            try:

                position = traci.vehicle.getPosition(
                    vehicle_id
                )

                x = position[0]
                y = position[1]

                speed = traci.vehicle.getSpeed(
                    vehicle_id
                )

                acceleration = traci.vehicle.getAcceleration(
                    vehicle_id
                )

                angle = traci.vehicle.getAngle(
                    vehicle_id
                )

                edge_id = traci.vehicle.getRoadID(
                    vehicle_id
                )

                lane_id = traci.vehicle.getLaneID(
                    vehicle_id
                )

                lane_position = traci.vehicle.getLanePosition(
                    vehicle_id
                )

                vehicle_type = traci.vehicle.getTypeID(
                    vehicle_id
                )

                route_id = traci.vehicle.getRouteID(
                    vehicle_id
                )

                route_index = traci.vehicle.getRouteIndex(
                    vehicle_id
                )

                waiting_time = traci.vehicle.getWaitingTime(
                    vehicle_id
                )

                accumulated_waiting_time = (
                    traci.vehicle.getAccumulatedWaitingTime(
                        vehicle_id
                    )
                )

                distance_travelled = (
                    traci.vehicle.getDistance(
                        vehicle_id
                    )
                )

                # ------------------------------------------------
                # Write telemetry
                # ------------------------------------------------

                writer.writerow(
                    {
                        "time": round(current_time, 2),
                        "vehicle_id": vehicle_id,
                        "vehicle_type": vehicle_type,

                        "x": round(x, 3),
                        "y": round(y, 3),

                        "speed": round(speed, 3),
                        "acceleration": round(
                            acceleration,
                            3,
                        ),
                        "angle": round(angle, 3),

                        "edge_id": edge_id,
                        "lane_id": lane_id,
                        "lane_position": round(
                            lane_position,
                            3,
                        ),

                        "route_id": route_id,
                        "route_index": route_index,

                        "waiting_time": round(
                            waiting_time,
                            3,
                        ),

                        "accumulated_waiting_time": round(
                            accumulated_waiting_time,
                            3,
                        ),

                        "distance_travelled": round(
                            distance_travelled,
                            3,
                        ),
                    }
                )

            except traci.TraCIException as error:

                print(
                    f"WARNING: Could not read "
                    f"{vehicle_id}: {error}"
                )

        step += 1

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if step % 10 == 0:

            print(
                f"Time: {current_time:6.1f} s | "
                f"Active vehicles: {len(vehicle_ids):2d}"
            )


# ============================================================
# CLOSE SUMO
# ============================================================

traci.close()

print()
print("=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
print(f"Telemetry written to:")
print(OUTPUT_FILE)
print()