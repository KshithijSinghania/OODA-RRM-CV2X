import os
import glob
import json
import math
from collections import defaultdict

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = r"datasets\\observe_ds\Situation_3\indore_03_telemetry_v1.csv"

# Use the full Situation 3 telemetry file directly.
# No split CSVs are available for this dataset.
INPUT_PATTERN = INPUT_DIR

OUTPUT_DIR = os.path.join(
    r"datasets\\observe_ds\Situation_3",
    "situation3_context"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_CSV = os.path.join(
    OUTPUT_DIR,
    "situation3_vehicle_context.csv"
)

CONTEXT_JSON = os.path.join(
    OUTPUT_DIR,
    "situation3_context.json"
)

CONTEXT_TXT = os.path.join(
    OUTPUT_DIR,
    "situation3_context.txt"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentile(series, p):
    """Safe percentile calculation."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return None

    return float(np.percentile(s, p))


def stats(series):
    """Return compact statistical summary."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None
        }

    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p05": percentile(s, 5),
        "p25": percentile(s, 25),
        "p75": percentile(s, 75),
        "p95": percentile(s, 95),
        "max": float(s.max())
    }


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


def safe_max(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.max()) if len(s) else 0.0


# ============================================================
# FIND INPUT FILES
# ============================================================

files = sorted(glob.glob(INPUT_PATTERN))

if not files:
    raise FileNotFoundError(
        "No telemetry_part_*.csv files found in:\n"
        + INPUT_DIR
    )

print("=" * 70)
print("SITUATION-1 TELEMETRY CONTEXT EXTRACTION")
print("=" * 70)

print("\nInput files found:", len(files))

for f in files:
    print("  ", os.path.basename(f))


# ============================================================
# PASS 1
# Read files individually and collect compact information.
# This avoids depending on one large CSV.
# ============================================================

vehicle_data = defaultdict(lambda: {
    "records": 0,
    "times": [],
    "speed": [],
    "acceleration": [],
    "waiting_time": [],
    "accumulated_waiting_time": [],
    "distance_travelled": [],
    "x": [],
    "y": [],
    "edges": set(),
    "lanes": set()
})

total_rows = 0

global_min_time = math.inf
global_max_time = -math.inf

global_min_x = math.inf
global_max_x = -math.inf
global_min_y = math.inf
global_max_y = -math.inf


# Keep compact time-bucket information
time_buckets = defaultdict(lambda: {
    "records": 0,
    "vehicles": set(),
    "speed": [],
    "waiting": [],
    "x": [],
    "y": []
})


for file_index, filepath in enumerate(files, start=1):

    print(
        f"\nProcessing {file_index}/{len(files)}: "
        f"{os.path.basename(filepath)}"
    )

    df = pd.read_csv(filepath)

    total_rows += len(df)

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "time",
        "x",
        "y",
        "speed",
        "acceleration",
        "lane_position",
        "route_index",
        "waiting_time",
        "accumulated_waiting_time",
        "distance_travelled"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Global ranges
    # --------------------------------------------------------

    if "time" in df:
        t = df["time"].dropna()

        if len(t):
            global_min_time = min(
                global_min_time,
                float(t.min())
            )

            global_max_time = max(
                global_max_time,
                float(t.max())
            )

    if "x" in df:
        x = df["x"].dropna()

        if len(x):
            global_min_x = min(
                global_min_x,
                float(x.min())
            )

            global_max_x = max(
                global_max_x,
                float(x.max())
            )

    if "y" in df:
        y = df["y"].dropna()

        if len(y):
            global_min_y = min(
                global_min_y,
                float(y.min())
            )

            global_max_y = max(
                global_max_y,
                float(y.max())
            )

    # --------------------------------------------------------
    # Vehicle-level processing
    # --------------------------------------------------------

    for vehicle_id, group in df.groupby(
        "vehicle_id",
        sort=False
    ):

        v = vehicle_data[vehicle_id]

        v["records"] += len(group)

        for col, key in [
            ("time", "times"),
            ("speed", "speed"),
            ("acceleration", "acceleration"),
            ("waiting_time", "waiting_time"),
            (
                "accumulated_waiting_time",
                "accumulated_waiting_time"
            ),
            (
                "distance_travelled",
                "distance_travelled"
            ),
            ("x", "x"),
            ("y", "y")
        ]:
            if col in group:
                values = group[col].dropna().tolist()
                v[key].extend(values)

        if "edge_id" in group:
            v["edges"].update(
                group["edge_id"]
                .dropna()
                .astype(str)
                .unique()
            )

        if "lane_id" in group:
            v["lanes"].update(
                group["lane_id"]
                .dropna()
                .astype(str)
                .unique()
            )

    # --------------------------------------------------------
    # Time-bucket processing
    # 10-second buckets
    # --------------------------------------------------------

    if "time" in df:

        df["_time_bucket"] = (
            np.floor(df["time"] / 10) * 10
        )

        for bucket, group in df.groupby(
            "_time_bucket"
        ):

            b = time_buckets[int(bucket)]

            b["records"] += len(group)

            if "vehicle_id" in group:
                b["vehicles"].update(
                    group["vehicle_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

            if "speed" in group:
                b["speed"].extend(
                    group["speed"]
                    .dropna()
                    .tolist()
                )

            if "waiting_time" in group:
                b["waiting"].extend(
                    group["waiting_time"]
                    .dropna()
                    .tolist()
                )

            if "x" in group:
                b["x"].extend(
                    group["x"]
                    .dropna()
                    .tolist()
                )

            if "y" in group:
                b["y"].extend(
                    group["y"]
                    .dropna()
                    .tolist()
                )


# ============================================================
# BASIC SCENARIO INFORMATION
# ============================================================

vehicle_ids = sorted(vehicle_data.keys())

vehicle_count = len(vehicle_ids)

if global_min_time == math.inf:
    global_min_time = None

if global_max_time == -math.inf:
    global_max_time = None

duration = None

if global_min_time is not None and global_max_time is not None:
    duration = global_max_time - global_min_time


# ============================================================
# VEHICLE-LEVEL CONTEXT TABLE
# ============================================================

vehicle_rows = []

for vehicle_id in vehicle_ids:

    v = vehicle_data[vehicle_id]

    times = pd.Series(v["times"])
    speeds = pd.Series(v["speed"])
    accels = pd.Series(v["acceleration"])
    waits = pd.Series(v["waiting_time"])
    accumulated_waits = pd.Series(
        v["accumulated_waiting_time"]
    )
    distances = pd.Series(
        v["distance_travelled"]
    )
    xs = pd.Series(v["x"])
    ys = pd.Series(v["y"])

    row = {
        "vehicle_id": vehicle_id,

        "records": v["records"],

        "start_time": (
            float(times.min())
            if len(times)
            else None
        ),

        "end_time": (
            float(times.max())
            if len(times)
            else None
        ),

        "trajectory_duration_s": (
            float(times.max() - times.min())
            if len(times)
            else None
        ),

        # Speed
        "mean_speed_mps": safe_mean(speeds),
        "median_speed_mps": (
            float(speeds.median())
            if len(speeds)
            else None
        ),
        "speed_std_mps": (
            float(speeds.std())
            if len(speeds)
            else None
        ),
        "min_speed_mps": (
            float(speeds.min())
            if len(speeds)
            else None
        ),
        "max_speed_mps": safe_max(speeds),

        "speed_p05_mps": percentile(speeds, 5),
        "speed_p25_mps": percentile(speeds, 25),
        "speed_p75_mps": percentile(speeds, 75),
        "speed_p95_mps": percentile(speeds, 95),

        # Acceleration
        "mean_acceleration_mps2": safe_mean(accels),
        "acceleration_std_mps2": (
            float(accels.std())
            if len(accels)
            else None
        ),
        "min_acceleration_mps2": (
            float(accels.min())
            if len(accels)
            else None
        ),
        "max_acceleration_mps2": safe_max(accels),

        "acceleration_p05_mps2":
            percentile(accels, 5),

        "acceleration_p95_mps2":
            percentile(accels, 95),

        # Waiting
        "mean_waiting_time_s": safe_mean(waits),
        "max_waiting_time_s": safe_max(waits),

        "total_accumulated_waiting_time_s":
            safe_max(accumulated_waits),

        # Distance
        "max_distance_travelled_m":
            safe_max(distances),

        # Position
        "min_x": (
            float(xs.min())
            if len(xs)
            else None
        ),

        "max_x": (
            float(xs.max())
            if len(xs)
            else None
        ),

        "min_y": (
            float(ys.min())
            if len(ys)
            else None
        ),

        "max_y": (
            float(ys.max())
            if len(ys)
            else None
        ),

        "unique_edges":
            len(v["edges"]),

        "unique_lanes":
            len(v["lanes"])
    }

    # --------------------------------------------------------
    # Derived mobility indicators
    # --------------------------------------------------------

    row["speed_variability"] = (
        row["speed_std_mps"] /
        max(row["mean_speed_mps"], 0.1)
    )

    row["acceleration_activity"] = (
        row["acceleration_std_mps2"]
        if row["acceleration_std_mps2"] is not None
        else 0
    )

    row["waiting_ratio"] = (
        row["mean_waiting_time_s"] /
        max(row["trajectory_duration_s"], 0.1)
    )

    # Mobility stress score.
    # This is intentionally a mobility-derived indicator,
    # NOT a network measurement.
    speed_stress = max(
        0,
        1 -
        min(row["mean_speed_mps"] / 15.0, 1)
    )

    waiting_stress = min(
        row["waiting_ratio"] * 5,
        1
    )

    variability_stress = min(
        row["speed_variability"],
        1
    )

    acceleration_stress = min(
        row["acceleration_activity"] / 3.0,
        1
    )

    row["mobility_stress"] = round(
        0.35 * speed_stress +
        0.30 * waiting_stress +
        0.20 * variability_stress +
        0.15 * acceleration_stress,
        6
    )

    vehicle_rows.append(row)


vehicle_context = pd.DataFrame(vehicle_rows)

vehicle_context = vehicle_context.sort_values(
    "vehicle_id"
).reset_index(drop=True)


# ============================================================
# SCENARIO-LEVEL STATISTICS
# ============================================================

scenario_stats = {}

for col in [
    "mean_speed_mps",
    "speed_std_mps",
    "mean_acceleration_mps2",
    "acceleration_std_mps2",
    "mean_waiting_time_s",
    "max_waiting_time_s",
    "total_accumulated_waiting_time_s",
    "max_distance_travelled_m",
    "mobility_stress"
]:

    scenario_stats[col] = stats(
        vehicle_context[col]
    )


# ============================================================
# TRAFFIC / TIME EVOLUTION
# ============================================================

time_rows = []

for bucket in sorted(time_buckets):

    b = time_buckets[bucket]

    speeds = pd.Series(b["speed"])
    waits = pd.Series(b["waiting"])

    time_rows.append({
        "time_start_s": bucket,
        "time_end_s": bucket + 10,

        "telemetry_records": b["records"],

        "active_vehicles":
            len(b["vehicles"]),

        "mean_speed_mps":
            safe_mean(speeds),

        "median_speed_mps":
            (
                float(speeds.median())
                if len(speeds)
                else None
            ),

        "mean_waiting_time_s":
            safe_mean(waits)
    })

time_evolution = pd.DataFrame(time_rows)


# ============================================================
# GLOBAL CORRELATIONS
# ============================================================

correlation_columns = [
    "mean_speed_mps",
    "speed_std_mps",
    "mean_acceleration_mps2",
    "acceleration_std_mps2",
    "mean_waiting_time_s",
    "max_waiting_time_s",
    "max_distance_travelled_m",
    "mobility_stress"
]

correlations = (
    vehicle_context[
        [
            c for c in correlation_columns
            if c in vehicle_context.columns
        ]
    ]
    .corr()
    .round(4)
    .to_dict()
)


# ============================================================
# VEHICLE MOBILITY-STRESS GROUPS
# ============================================================

vehicle_context["mobility_condition"] = pd.cut(
    vehicle_context["mobility_stress"],
    bins=[
        -np.inf,
        0.20,
        0.40,
        0.60,
        0.80,
        np.inf
    ],
    labels=[
        "very_low",
        "low",
        "moderate",
        "high",
        "very_high"
    ]
)


stress_distribution = (
    vehicle_context[
        "mobility_condition"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


# ============================================================
# WRITE VEHICLE CONTEXT CSV
# ============================================================

vehicle_context.to_csv(
    CONTEXT_CSV,
    index=False
)


# ============================================================
# CREATE JSON CONTEXT
# ============================================================

context = {
    "scenario": "Indore-01",
    "purpose":
        "Compact SUMO telemetry context for "
        "ns-3-calibrated C-V2X network modeling",

    "input_files": [
        os.path.basename(f)
        for f in files
    ],

    "input_file_count": len(files),

    "telemetry": {
        "total_records": total_rows,
        "vehicle_count": vehicle_count,
        "simulation_start_s": global_min_time,
        "simulation_end_s": global_max_time,
        "duration_s": duration,

        "spatial_extent": {
            "min_x": (
                None
                if global_min_x == math.inf
                else global_min_x
            ),
            "max_x": (
                None
                if global_max_x == -math.inf
                else global_max_x
            ),
            "min_y": (
                None
                if global_min_y == math.inf
                else global_min_y
            ),
            "max_y": (
                None
                if global_max_y == -math.inf
                else global_max_y
            )
        }
    },

    "vehicle_statistics": scenario_stats,

    "mobility_stress_distribution":
        stress_distribution,

    "correlations":
        correlations,

    "time_evolution":
        time_evolution.to_dict(
            orient="records"
        )
}


with open(
    CONTEXT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        context,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# HUMAN-READABLE TXT REPORT
# ============================================================

with open(
    CONTEXT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "============================================================\n"
    )

    f.write(
        "INDORE-01 TELEMETRY CONTEXT REPORT\n"
    )

    f.write(
        "============================================================\n\n"
    )

    f.write("INPUT FILES\n")
    f.write("-----------\n")

    for file in files:
        f.write(
            f"{os.path.basename(file)}\n"
        )

    f.write("\n")

    f.write("SCENARIO\n")
    f.write("--------\n")

    f.write(
        f"Total telemetry records: {total_rows}\n"
    )

    f.write(
        f"Unique vehicles: {vehicle_count}\n"
    )

    f.write(
        f"Start time: {global_min_time} s\n"
    )

    f.write(
        f"End time: {global_max_time} s\n"
    )

    f.write(
        f"Duration: {duration} s\n\n"
    )

    f.write("SPATIAL EXTENT\n")
    f.write("--------------\n")

    f.write(
        f"X: {global_min_x} to {global_max_x}\n"
    )

    f.write(
        f"Y: {global_min_y} to {global_max_y}\n\n"
    )

    f.write("SCENARIO-LEVEL VEHICLE STATISTICS\n")
    f.write("---------------------------------\n")

    for metric, values in scenario_stats.items():

        f.write(f"\n{metric}\n")

        for key, value in values.items():
            f.write(
                f"  {key}: {value}\n"
            )

    f.write("\n\nMOBILITY STRESS DISTRIBUTION\n")
    f.write("----------------------------\n")

    for key, value in stress_distribution.items():

        f.write(
            f"{key}: {value} vehicles\n"
        )

    f.write("\n\nCORRELATIONS\n")
    f.write("------------\n")

    corr_df = (
        vehicle_context[
            [
                c for c in correlation_columns
                if c in vehicle_context.columns
            ]
        ]
        .corr()
        .round(4)
    )

    f.write(
        corr_df.to_string()
    )

    f.write("\n\n\nTIME EVOLUTION (10-second buckets)\n")
    f.write("----------------------------------\n")

    f.write(
        time_evolution.to_string(
            index=False
        )
    )

    f.write("\n\n\nVEHICLE CONTEXT\n")
    f.write("---------------\n")

    f.write(
        vehicle_context.to_string(
            index=False
        )
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print("\nCreated:")

print("\n1. Vehicle-level context CSV:")
print(CONTEXT_CSV)

print("\n2. Compact JSON context:")
print(CONTEXT_JSON)

print("\n3. Human-readable report:")
print(CONTEXT_TXT)

print("\nSummary:")
print("  Vehicles :", vehicle_count)
print("  Records  :", total_rows)
print("  Duration :", duration, "seconds")

print("\nOriginal telemetry files were NOT modified.")