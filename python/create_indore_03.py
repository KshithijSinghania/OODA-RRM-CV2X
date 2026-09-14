import xml.etree.ElementTree as ET
import random
import os
import copy


# ============================================================
# INDORE-03 — HEAVY CONGESTION
# ============================================================

BASE_DIR = r"C:\Users\jinda\Sumo\2026-08-24-00-28-36"

OUTPUT_DIR = r"C:\Users\jinda\Sumo\Indore-03"

INPUT_TRIPS = os.path.join(
    BASE_DIR,
    "osm.passenger.trips.xml"
)

OUTPUT_TRIPS = os.path.join(
    OUTPUT_DIR,
    "osm.passenger.trips.xml"
)


# ============================================================
# TRAFFIC SETTINGS
# ============================================================

# 3x baseline demand
TRAFFIC_MULTIPLIER = 3.0

RANDOM_SEED = 303


# ============================================================
# PREPARE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


if not os.path.exists(
    INPUT_TRIPS
):

    print("ERROR:")
    print(
        "Baseline trip file not found:"
    )
    print(INPUT_TRIPS)

    raise SystemExit(1)


# ============================================================
# LOAD
# ============================================================

tree = ET.parse(
    INPUT_TRIPS
)

root = tree.getroot()


random.seed(
    RANDOM_SEED
)


# ============================================================
# FIND BASELINE TRIPS
# ============================================================

trips = [

    element

    for element in root.iter()

    if element.tag == "trip"
]


baseline_count = len(
    trips
)


target_count = int(
    baseline_count
    *
    TRAFFIC_MULTIPLIER
)


extra_count = (
    target_count
    -
    baseline_count
)


print()
print("=" * 70)
print("INDORE-03 HEAVY CONGESTION GENERATOR")
print("=" * 70)
print()

print(
    "Baseline trips:",
    baseline_count
)

print(
    "Target trips:",
    target_count
)

print(
    "Additional trips:",
    extra_count
)

print(
    "Multiplier:",
    TRAFFIC_MULTIPLIER
)

print(
    "Random seed:",
    RANDOM_SEED
)

print()


# ============================================================
# CREATE ADDITIONAL TRIPS
# ============================================================

for index in range(
    extra_count
):

    source_trip = random.choice(
        trips
    )

    new_trip = copy.deepcopy(
        source_trip
    )


    # --------------------------------------------------------
    # Unique ID
    # --------------------------------------------------------

    new_trip.set(
        "id",
        f"INDORE03_{index:06d}"
    )


    # --------------------------------------------------------
    # Random departure shift
    # --------------------------------------------------------

    original_depart = (
        source_trip.get(
            "depart"
        )
    )


    if original_depart is not None:

        try:

            depart = float(
                original_depart
            )

            shift = random.uniform(
                -30.0,
                30.0
            )

            new_depart = max(
                0.0,
                depart + shift
            )

            new_trip.set(
                "depart",
                f"{new_depart:.2f}"
            )

        except ValueError:

            pass


    root.append(
        new_trip
    )


# ============================================================
# SORT TRIPS BY DEPARTURE
# ============================================================

children = list(
    root
)

non_trip_children = [

    child

    for child in children

    if child.tag != "trip"
]

trip_children = [

    child

    for child in children

    if child.tag == "trip"
]


def get_departure(element):

    value = element.get(
        "depart"
    )

    if value is None:

        return float("inf")

    try:

        return float(value)

    except ValueError:

        return float("inf")


trip_children.sort(
    key=get_departure
)


root[:] = (
    non_trip_children
    +
    trip_children
)


# ============================================================
# WRITE
# ============================================================

tree.write(
    OUTPUT_TRIPS,
    encoding="UTF-8",
    xml_declaration=True
)


# ============================================================
# VERIFY
# ============================================================

check_tree = ET.parse(
    OUTPUT_TRIPS
)

check_root = (
    check_tree.getroot()
)


final_count = sum(

    1

    for element
    in check_root.iter()

    if element.tag == "trip"
)


print("=" * 70)
print("INDORE-03 GENERATION COMPLETE")
print("=" * 70)
print()

print(
    "Baseline:",
    baseline_count
)

print(
    "Final:",
    final_count
)

print(
    "Increase:",
    final_count - baseline_count
)

print()

print(
    "Output:"
)

print(
    OUTPUT_TRIPS
)

print()
print("Done.")