# Observe-DS — Indore Traffic Dataset

## Overview

Observe-DS is a simulation-based dataset being constructed to train and evaluate the **Observe agent** in the C-V2X OODA architecture.

The dataset captures different urban traffic conditions in **Indore, India** using Eclipse SUMO simulations. Raw vehicle-level telemetry is collected and then aggregated into short temporal traffic-state windows.

```text
SUMO Traffic Simulation
        ↓
Vehicle Telemetry
        ↓
5-second Aggregation
        ↓
Scenario-State Dataset
        ↓
Observe Agent Training
```

## Simulation Environment

- **Simulator:** Eclipse SUMO 1.27.1
- **Location:** Indore, India
- **Telemetry interface:** SUMO TraCI
- **Raw telemetry resolution:** 0.1 seconds
- **Scenario-state aggregation:** 5 seconds
- **Output format:** CSV

## Scenarios

| Scenario | Condition | Approx. Trips |
|---|---|---:|
| Indore-01 | Normal Traffic | 313 |
| Indore-02 | High Traffic | 626 |
| Indore-03 | Heavy Congestion | ~939 |
| Indore-04 | Ambulance / Emergency | Planned |
| Indore-05 | Road Incident | Planned |
| Indore-06 | Signal Variation | Planned |

The first three scenarios provide a controlled progression:

```text
Normal Traffic
      ↓
High Traffic
      ↓
Heavy Congestion
```

For these initial scenarios, the road network is kept constant so that changes in the resulting telemetry primarily reflect changes in traffic demand.

## Raw Telemetry

Vehicle-level telemetry is collected every **0.1 seconds**.

The raw dataset contains:

- Simulation time
- Scenario ID and condition
- Vehicle ID and type
- Vehicle position (`x`, `y`)
- Speed
- Acceleration
- Heading/angle
- Road edge
- Lane
- Lane position
- Route information
- Waiting time
- Accumulated waiting time
- Distance travelled

### Raw telemetry schema

```text
time
scenario_id
scenario_name
vehicle_id
vehicle_type
x
y
speed
acceleration
angle
edge_id
lane_id
lane_position
route_id
route_index
waiting_time
accumulated_waiting_time
distance_travelled
```

## Scenario-State Dataset

Raw telemetry is aggregated into **5-second windows** to create a higher-level representation of the traffic situation.

The scenario-state dataset includes features such as:

- Vehicle count
- Vehicle observations
- Mean speed
- Speed standard deviation
- Minimum and maximum speed
- Mean acceleration
- Queue length
- Queue ratio
- Number of congested edges
- Mean edge density
- P95 edge density
- Maximum edge density
- Mean lane density
- P95 lane density
- Maximum lane density
- Number of high-density edges
- Mean waiting time
- Maximum waiting time
- Active traffic lights
- Green signal links
- Yellow signal links
- Red signal links

## Dataset Files

Each completed scenario produces two primary datasets:

```text
indore_01_telemetry.csv
indore_01_scenario_states.csv

indore_02_telemetry_v1.csv
indore_02_scenario_states_v1.csv

indore_03_telemetry_v1.csv
indore_03_scenario_states_v1.csv
```

### Raw telemetry files

These contain detailed vehicle-level observations from SUMO and preserve the underlying simulation information.

### Scenario-state files

These contain aggregated traffic conditions and provide a more compact representation of the traffic situation for the Observe agent.

## Density Processing

Traffic density is calculated using SUMO road/lane information.

Very short road segments are excluded from density calculations to avoid unrealistic density values caused by extremely short OSM connector edges.

Current minimum edge length:

```text
20 meters
```

The dataset records:

```text
Mean density
P95 density
Maximum density
```

P95 density is retained as a robust high-density indicator rather than relying only on the maximum value.

## Dataset Objective

The eventual training samples are intended to follow:

```text
Traffic telemetry / scenario state
              ↓
        Observe Agent
              ↓
     Situational Summary
```

For example:

```text
INPUT
--------------------------------
High vehicle density
Low average speed
Large queue ratio
Multiple congested edges
High waiting time
--------------------------------
              ↓
        Observe Agent
              ↓
OUTPUT
--------------------------------
Heavy congestion is present across
multiple road segments, with reduced
vehicle speeds and significant queue
formation.
--------------------------------
```

The planned **Observe-DS contains approximately 15,000 samples** covering different traffic conditions and simulation variations.

## Current Status

### Completed

- SUMO 1.27.1 simulation environment
- Indore road network
- Baseline traffic scenario
- High-traffic scenario
- Heavy-congestion scenario
- TraCI-based telemetry collection
- Raw vehicle telemetry generation
- 5-second scenario-state aggregation
- Traffic density calculation
- Congestion and queue metrics

### Planned

- Ambulance/emergency scenario
- Road-incident scenario
- Signal-variation scenario
- Additional simulation seeds and traffic variations
- Final Observe-DS construction
- Situational-summary annotation
- Observe-agent fine-tuning

## Summary

The dataset provides a simulation-based foundation for the Observe agent by converting detailed SUMO vehicle telemetry into structured representations of urban traffic conditions.

The initial Indore scenarios establish a controlled progression from:

```text
Normal Traffic
      ↓
High Traffic
      ↓
Heavy Congestion
```

Additional scenarios will introduce emergency vehicles, incidents, and signal variations to increase the diversity and robustness of the final Observe-DS.
