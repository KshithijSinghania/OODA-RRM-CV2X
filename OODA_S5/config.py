"""Shared configuration for the OODA-RRM-CV2X dataset pipeline."""

# --- the four situations -------------------------------------------------
SITUATIONS = {
    "S5": {"n_vehicles": 313, "ambulance": False, "expected_tag": "baseline"},
    "S1": {"n_vehicles": 44,  "ambulance": False, "expected_tag": "baseline"},
    "S2": {"n_vehicles": 626, "ambulance": False, "expected_tag": "intersection_dense"},
    "S3": {"n_vehicles": 313, "ambulance": False, "expected_tag": "intersection_dense"},
    "S4": {"n_vehicles": 313, "ambulance": True,  "expected_tag": "emergency_preempt"},
}

# --- QoS thresholds (tune after looking at your data) ---------------------
LOSS_VIOLATION_PCT = 10.0      # per-vehicle loss above this = violation
DELAY_VIOLATION_MS = 30.0      # sim-calibrated: ~p96 of free-flow S1/S3/S4 delay (20 ms was inside baseline noise)
DELAY_TAIL_MS = 100.0
LOSS_SPIKE_PCT = 50.0
STATIONARY_SPEED_MPS = 0.5
HIGH_JITTER_MS = 10.0         # S3 baseline jitter p90 is ~15 ms
DELIVERY_RATIO_LOW = 0.80    # rx_bytes / tx_bytes below this = throughput_drop
ALERT_RADIUS_M = 300.0         # ambulance alert zone

# severity = f(fraction of vehicles violating loss OR delay threshold)
SEVERITY_BINS = [(0.05, "nominal"), (0.20, "elevated"), (0.50, "degraded"), (1.01, "critical")]

# load level from mean local_density_300m (vehicles within 300 m; recomputed from positions for every situation)
LOAD_BINS = [(10, "low"), (30, "moderate"), (60, "high"), (1e9, "very_high")]

# --- spatial zones = nearest of the 12 fixed cells in the ns-3 'fixed network' script -------
# (index 0 = macro gNB at 25 m, 1..11 = RSUs at 6 m). Nearest site is a PROXY for the serving cell:
# the script does not log the serving cell, and A3 handover hysteresis is ignored here.
CELL_SITES = [(1824.7, 395.1), (1492.5, 394.0), (1725.5, 722.1), (2196.9, 702.4), (1882.2, 608.9), (1404.6, 666.8),
              (353.4, 767.4), (1026.1, 687.0), (1092.5, 323.6), (687.7, 283.3), (234.0, 259.0), (2006.1, 1299.7)]
ZONE_SIZE_M = 600   # legacy grid; unused when CELL_SITES is set
MIN_VEHICLES_PER_ZONE = 3

# --- trend ------------------------------------------------------------------
TREND_LOOKBACK = 3             # previous windows used as baseline
TREND_LOSS_PP = 3.0
TREND_LOAD_PCT = 10.0

# --- splits: contiguous blocks of windows keep train/test from sharing time --
SPLIT_BLOCK_WINDOWS = 20
SPLIT_FRACS = (0.8, 0.1, 0.1)  # train / val / test
SEED = 42

OBSERVE_SYSTEM_PROMPT = (
    "You are the OBSERVE agent of a C-V2X radio resource management system. "
    "Given aggregated telemetry for one RSU cell (or the whole network) over a 5 s window, "
    "return ONLY a JSON ObserveSummary: load level, QoS severity, trend, mobility, V2X message rates, "
    "estimated resource state (flagged as modelled), hotspots, anomalies, special events "
    "(e.g. an active emergency vehicle) and a short narrative (<60 words)."
)
MSG_RATE_NOMINAL_HZ = 100.0     # CAM/BSM-like message rate of a regular vehicle (100 pkt/s in the ns-3 script)
ALERT_RATE_FACTOR = 1.25        # >= 1.25x nominal = alert-rate mode (data: 150 Hz inside the ambulance alert zone)

# =========================== ORIENT =========================================
ORIENT_MIN_N = 5   # need >=5 vehicles before QoS degradation / standstill can define 'intersection_dense'
ORIENT_VARIANTS = {"emergency_preempt": 3, "intersection_dense": 2, "baseline": 1}   # context variants per Observe sample

INTENT_PRIORITY = ["regulatory_compliance", "safety_first", "emergency_priority",
                   "oem_qos_contract", "throughput_fairness", "energy_saving"]

ORIENT_SYSTEM_PROMPT = (
    "You are the ORIENT agent of a C-V2X radio resource management system. Input: an Observe summary for one RSU cell, an intent "
    "registry (operator SLOs, operator mode, OEM contract, regulatory context) and retrieved knowledge snippets. "
    "Classify the scenario (baseline / intersection_dense / emergency_preempt / platoon_stable / platoon_merge / "
    "handover_imminent), resolve intent conflicts using the priority order "
    + " > ".join(INTENT_PRIORITY) +
    ", and return ONLY a JSON ContextObject: scenario_tag, confidence, qos_weights (urllc/embb/mmtc summing to 1), "
    "constraint_overrides, active_intents, intent_resolution, rag_refs (only snippets you actually used), rationale (<60 words)."
)

# Knowledge-base snippets. Paraphrased, generic summaries written for this pipeline -- REPLACE with verbatim chunks
# (and real clause numbers) from your FAISS corpus before the final dataset.
RAG_KB = {
    "TS22.186": "3GPP TS 22.186: stage-1 requirements for enhanced V2X use cases (advanced driving, platooning, extended sensors, remote driving) with per-use-case latency, reliability and data-rate targets.",
    "TR36.885": "3GPP TR 36.885: study of LTE-based V2X; defines V2X evaluation scenarios (urban, freeway) and channel models used for link-level and system-level analysis.",
    "TS28.312": "3GPP TS 28.312: intent-driven management for mobile networks; intent expressions, handling, conflict reporting and fulfilment feedback.",
    "NRV2X-MODE1": "NR-V2X sidelink Mode 1: the gNB schedules sidelink resources, enabling deterministic, centrally controlled allocation and preemption.",
    "NRV2X-MODE2": "NR-V2X sidelink Mode 2: UEs autonomously select sidelink resources by sensing; contention and collisions rise with density.",
    "ROAD-EVENT-ONTOLOGY": "Project road-event ontology: emergency vehicle classes (ambulance, fire, police) hold right-of-way and trigger communication priority for nearby vehicles.",
    "TMC-REG": "Project local-regulation store: traffic-management-centre mandates (emergency corridors, safety zones, regional power limits) override operator preferences.",
}
RAG_RELEVANT = {
    "emergency_preempt": ["TS22.186", "NRV2X-MODE1", "ROAD-EVENT-ONTOLOGY"],
    "intersection_dense": ["TS22.186", "TR36.885", "NRV2X-MODE1"],
    "baseline": ["TS22.186", "TS28.312"],
}

# =========================== DECIDE (proposal system model; ALL numbers are PLACEHOLDERS) =============
# Proposal Sec. 2.1: N_V vehicles, M RSUs (each with a MEC server), one macro gNB, 5.9 GHz ITS band, K PRBs shared by
# V2V sidelink (PC5), V2I uplink and V2I downlink; per-vehicle power budgets; MEC capacity; URLLC reliability and
# interference limits. Decide outputs class-level parameters only (PRB priority, per-class power budget, MEC
# offloading policy, interference coordination).
SYSTEM_MODEL = {"links": "V2V PC5 sidelink + V2I uplink + V2I downlink share K PRBs per RSU", "K_prb": 106, "prb_khz": 180,
                "carrier": "5.9 GHz ITS band", "max_vehicle_power_dbm": 23, "mec_capacity_tasks_per_s": 150}
DECIDE_SYSTEM_PROMPT = (
    "You are the DECIDE agent of a C-V2X radio resource management system. Input: the Orient ContextObject, the state of "
    "ONE RSU cell (with its co-located MEC server), the cell's current allocation snapshot and its system model. Return "
    "ONLY a JSON class-level policy: integer PRB shares per class (urllc/embb/mmtc, sum 100, honouring "
    "constraint_overrides), per-class maximum vehicle power (dBm, never above max_vehicle_power_dbm or tx_power_cap_dbm), "
    "MEC offloading policy with integer compute shares (sum 100), an interference-coordination strategy, and preemption. "
    "The mean transmit power across PRBs must respect the interference limit implied by cell load and strategy."
)
OPT = {
    # --- traffic (rates from the CSVs; class mix INVENTED: the simulator has a single traffic type) ---
    "ue_rate_mbps": 0.432, "alert_rate_mbps": 0.648, "ambulance_rate_mbps": 0.864,
    "class_mix_regular": {"urllc": 0.20, "embb": 0.60, "mmtc": 0.20},
    "urllc_target": 0.99, "shortfall_weight": 10.0, "tie_eps": 0.001, "power_penalty": 0.002,
    # --- radio (link-budget shape; absolute scale calibrated to the ns-3 hot-cell capacity) ---
    "cal_capacity_mbps": 9.0,           # capacity of a cell with no co-channel interference at 23 dBm (anchor, placeholder)
    "levels_dbm": [10, 14, 17, 20, 23], "d_ref_m": 150.0, "path_exp": 2.8, "fc_ghz": 5.9, "noise_figure_db": 9.0,
    "iot_db_per_vehicle": 0.3, "iot_db_max": 12.0,                    # interference-over-thermal grows with vehicles in the cell
    "strategy": {   # (IoT factor, usable-PRB factor, interference-cap relief dB, reuse factor)  -- all invented
        "none": (1.0, 1.0, 0.0, 1), "power_backoff_edge": (0.8, 1.0, 0.0, 1),
        "sidelink_mode1_scheduling": (0.6, 0.9, 3.0, 1), "frequency_reuse": (0.5, 0.5, 3.0, 2)},
    "cap_dbm_base": 23.0, "cap_db_per_vehicle": 0.25, "cap_db_max_drop": 10.0,   # PRB-averaged power limit shrinks with load
    # --- MEC (placeholders; measure on the Jetson Orin) ---
    "task_rate_per_vehicle": {"urllc": 1.0, "embb": 2.0, "mmtc": 0.5}, "uplink_mbps_per_task_rate": 0.05,
    "local_value": 0.3, "mec_weight": 0.5, "offload_levels": [0.0, 0.5, 1.0],
}

# =========================== ACT (rule teacher; ALL O-RAN names and thresholds are PLACEHOLDERS) =============
ACT_SYSTEM_PROMPT = (
    "You are the ACT agent of a C-V2X radio resource management system. Input: the Decide policy for ONE RSU cell, the "
    "Orient constraints, the QoS Observe saw before the policy, and the enforcer's execution feedback (or null). "
    "1) Type-check the policy against the Command DSL and the constraints; if it is malformed or violates a constraint, "
    "reject it (a1_policy null, no E2 commands) and escalate to decide. 2) Otherwise translate it into one A1 policy and "
    "E2 control messages. 3) Check the feedback: wrong policy id, failed authentication, stale window, impossible or "
    "self-contradictory values mean spoofed feedback (do not trust it, escalate to decide). 4) Otherwise escalate "
    "failures to decide, a new emergency event under a non-preempting policy to orient, and persistent or new QoS "
    "degradation to decide. Return ONLY a JSON Act output."
)
ACT_ADVERSARIAL = {"malformed": 1000, "spoofed": 1000}      # extra samples on top of the one-per-Decide-sample base
ACT_FAULT_P = {"applied_false": 0.04, "violation_flags": 0.04}   # synthetic enforcer faults injected into clean feedback
ACT_NOMINAL_LOSS_MAX = 8.0      # severity 'nominal' with mean loss above this is self-contradictory (real data max 3.6)
ACT_NOMINAL_DELAY_MAX = 40.0    # same for delay in ms (real data max 20.3)
