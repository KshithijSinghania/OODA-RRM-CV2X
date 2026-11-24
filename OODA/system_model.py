"""Shared PLACEHOLDER system model (proposal Sec. 2.1) used by BOTH Observe (estimated resource state) and Decide (optimizer).
One definition, so the state Decide receives is exactly what Observe reported. All numbers live in config.OPT / config.SYSTEM_MODEL."""
import math
import config as C

CL = ("urllc", "embb", "mmtc")
MIN_SHARE = 5
O = C.OPT
STRATS = ["none", "power_backoff_edge", "sidelink_mode1_scheduling", "frequency_reuse"]   # 'none' first: wins ties

PL_DB = 32.4 + 20 * math.log10(O["fc_ghz"]) + 10 * O["path_exp"] * math.log10(O["d_ref_m"])
NOISE_DBM = -174 + 10 * math.log10(C.SYSTEM_MODEL["prb_khz"] * 1e3) + O["noise_figure_db"]

def eff(level, s, N):
    iot = min(O["iot_db_max"], O["iot_db_per_vehicle"] * N) * O["strategy"][s][0]
    return math.log2(1 + 10 ** ((level - PL_DB - NOISE_DBM - iot) / 10))
REF = eff(23, "none", 0)

def rho(level, s, N):
    """Mbps per 1 % of the PRBs at a given power level, strategy and cell load."""
    return O["cal_capacity_mbps"] / 100 * eff(level, s, N) / REF * O["strategy"][s][1]

def cap_dbm(N, s):
    """PRB-averaged transmit-power (interference) limit, shrinking with cell load."""
    return O["cap_dbm_base"] - min(O["cap_db_max_drop"], O["cap_db_per_vehicle"] * N) + O["strategy"][s][2]

def cell_load(o):
    """Per-class demand (Mbps), per-class MEC task rate and vehicle count of a cell. o needs n_active_vehicles, special_events."""
    N = o["n_active_vehicles"]; ev = o["special_events"][0] if o["special_events"] else None
    n_alert = min(N, ev["n_vehicles_within_300m"]) if ev else 0
    amb = bool(ev and ev["in_scope"])
    reg = N - n_alert
    d0 = {c: reg * O["ue_rate_mbps"] * O["class_mix_regular"][c] for c in CL}
    d0["urllc"] += n_alert * O["alert_rate_mbps"] + (O["ambulance_rate_mbps"] if amb else 0.0)
    nveh = {c: reg * O["class_mix_regular"][c] for c in CL}; nveh["urllc"] += n_alert + (1 if amb else 0)
    T = {c: nveh[c] * O["task_rate_per_vehicle"][c] for c in CL}
    return ({c: round(max(v, 0.05), 3) for c, v in d0.items()}, {c: round(max(v, 0.1), 3) for c, v in T.items()}, N)

def resource_state(o):
    """MODELLED stand-ins for the proposal's PRB-occupancy / interference / MEC-load telemetry (not in the traces) and the
    'current PRB/power state' Decide needs: the uncontrolled baseline (demand-proportional PRBs, maximum power)."""
    d0, T, N = cell_load(o); tot = sum(d0.values())
    p = {c: max(MIN_SHARE, round(100 * d0[c] / tot)) for c in CL}; p["urllc"] = 100 - p["embb"] - p["mmtc"]
    return {"estimated": True,
            "offered_load_ratio": round(tot / (rho(23, "none", N) * 100), 2),
            "interference_iot_db": round(min(O["iot_db_max"], O["iot_db_per_vehicle"] * N), 1),
            "mec_load_ratio": round(sum(T.values()) / C.SYSTEM_MODEL["mec_capacity_tasks_per_s"], 2),
            "current_allocation": {"prb_pct": {c: int(p[c]) for c in CL}, "power_dbm": {c: 23 for c in CL}}}

__all__ = ["CL", "MIN_SHARE", "O", "STRATS", "PL_DB", "NOISE_DBM", "eff", "REF", "rho", "cap_dbm", "cell_load", "resource_state"]
