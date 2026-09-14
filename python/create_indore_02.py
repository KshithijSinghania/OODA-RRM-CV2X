import xml.etree.ElementTree as ET
import random
import os
import copy


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\jinda\Sumo\2026-08-24-00-28-36"

OUTPUT_DIR = r"C:\Users\jinda\Sumo\Indore-02"

INPUT_TRIPS = os.path.join(
    BASE_DIR,
    "osm.passenger.trips.xml"
)

OUTPUT_TRIPS = os.path.join(
    OUTPUT_DIR,
    "osm.passenger.trips.xml"
)

# High-traffic multiplier
TRAFFIC_MULTIPLIER = 2.0

# Reproducible randomness
RANDOM_SEED = 202


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_TRIPS):

    print("ERROR:")
    print("Baseline trip file not found:")
    print(INPUT_TRIPS)

    raise SystemExit(1)


# ============================================================
# LOAD XML
# ============================================================

print()
print("=" * 70)
print("INDORE-02 HIGH TRAFFIC GENERATOR")
print("=" * 70)
print()

print("Input:")
print(INPUT_TRIPS)

print()

print("Output:")
print(OUTPUT_TRIPS)

print()

print(
    "Traffic multiplier:",
    TRAFFIC_MULTIPLIER
)

print(
    "Random seed:",
    RANDOM_SEED
)

print()


tree = ET.parse(
    INPUT_TRIPS
)

root = tree.getroot()


# ============================================================
# RANDOM GENERATOR
# ============================================================

random.seed(
    RANDOM_SEED
)


# ============================================================
# FIND TRIPS
# ============================================================

trip_elements = []

for element in root.iter():

    if element.tag == "trip":

        trip_elements.append(
            element
        )


# ============================================================
# COUNT BASELINE
# ============================================================

baseline_count = len(
    trip_elements
)


print(
    "Baseline trips:",
    baseline_count
)


# ============================================================
# DETERMINE EXTRA TRIPS
# ============================================================

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


print(
    "Target trips:",
    target_count
)

print(
    "Additional trips:",
    extra_count
)

print()


# ============================================================
# FIND MAX DEPARTURE TIME
# ============================================================

max_depart = 0.0


for trip in trip_elements:

    depart = trip.get(
        "depart"
    )

    if depart is None:

        continue

    try:

        depart_value = float(
            depart
        )

        max_depart = max(
            max_depart,
            depart_value
        )

    except ValueError:

        continue


# ============================================================
# CREATE EXTRA TRIPS
# ============================================================

for index in range(
    extra_count
):

    # Select an existing trip as template.
    source_trip = random.choice(
        trip_elements
    )


    # Deep copy
    new_trip = copy.deepcopy(
        source_trip
    )


    # --------------------------------------------------------
    # New unique ID
    # --------------------------------------------------------

    original_id = (
        source_trip.get(
            "id",
            f"trip_{index}"
        )
    )

    new_id = (
        f"INDORE02_{index:06d}"
    )

    new_trip.set(
        "id",
        new_id
    )


    # --------------------------------------------------------
    # Randomize departure time
    #
    # Keep the new traffic within the same
    # simulation demand period.
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

            # Small random shift:
            # +/- 30 seconds
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


    # --------------------------------------------------------
    # Add new trip
    # --------------------------------------------------------

    root.append(
        new_trip
    )


# ============================================================
# SORT TRIPS BY DEPARTURE
# ============================================================

# SUMO expects departure ordering for clean loading.

all_children = list(
    root
)


def departure_time(element):

    depart = element.get(
        "depart"
    )

    if depart is None:

        return float("inf")

    try:

        return float(
            depart
        )

    except ValueError:

        return float("inf")


# Keep non-trip elements in place as much as possible.
#
# The normal SUMO passenger file generally contains
# vType/trip elements. We sort trip elements only.

trip_children = [

    child

    for child in all_children

    if child.tag == "trip"
]


other_children = [

    child

    for child in all_children

    if child.tag != "trip"
]


trip_children.sort(
    key=departure_time
)


# Rebuild root
root[:] = (
    other_children
    +
    trip_children
)


# ============================================================
# WRITE FILE
# ============================================================

tree.write(
    OUTPUT_TRIPS,
    encoding="UTF-8",
    xml_declaration=True
)


# ============================================================
# VALIDATE
# ============================================================

output_tree = ET.parse(
    OUTPUT_TRIPS
)

output_root = (
    output_tree.getroot()
)


final_trip_count = sum(

    1

    for element
    in output_root.iter()

    if element.tag == "trip"
)


print()
print("=" * 70)
print("HIGH TRAFFIC GENERATION COMPLETE")
print("=" * 70)

print()

print(
    "Baseline trips:",
    baseline_count
)

print(
    "Final trips:",
    final_trip_count
)

print(
    "Increase:",
    final_trip_count - baseline_count
)

print()

print(
    "Generated:"
)

print(
    OUTPUT_TRIPS
)

print()
print("Done.")