"""Deterministic Observe: telemetry window (the Observe agent's input) -> ObserveSummary, no LLM (microseconds).
Re-implements the summary rules of build_observe.make_summary from the INPUT dict only (no pandas, no hidden state), so it can run in deployment."""
import config as C
import system_model as SM

def r2(x): return None if x is None else round(float(x) + 0.0, 2)
def level(v, bins):
    for hi, name in bins:
        if v < hi: return name
    return bins[-1][1]

def observe_rules(inp, f_viol=None, ov=None):
    """f_viol (fraction of vehicles violating QoS) is not in the telemetry input; it is approximated by max(frac_loss_gt_10pct, frac_delay_gt_thr)."""
    n = inp["n_active_vehicles"]; scope = inp["scope"]["type"]; zone = inp["scope"]["cell_id"]; ws, we = inp["window_s"]
    lm, lp = inp["loss_pct"]["mean"], inp["loss_pct"]["p95"]; dm = inp["delay_ms"]["mean"]
    fl10, fd20, fd100 = inp["frac_loss_gt_10pct"], inp["frac_delay_gt_thr"], inp["frac_delay_gt_100ms"]
    if f_viol is None: f_viol = max(fl10, fd20)
    prev = inp["prev_windows_mean"]
    if prev is None: t_qos, t_load, d_loss, d_veh = "unknown", "unknown", None, None
    else:
        d_loss = lm - prev["loss_pct"]; d_veh = 100.0 * (n - prev["n_vehicles"]) / max(prev["n_vehicles"], 1)
        t_qos = "worsening" if d_loss > C.TREND_LOSS_PP else "improving" if d_loss < -C.TREND_LOSS_PP else "stable"
        t_load = "ramping_up" if d_veh > C.TREND_LOAD_PCT else "draining" if d_veh < -C.TREND_LOAD_PCT else "steady"
    sev = level(f_viol, C.SEVERITY_BINS); load = level(inp["density_300m"]["mean"], C.LOAD_BINS)
    an = []
    if inp["n_loss_gt_50pct"] >= 1 and inp["n_loss_gt_50pct"] / n >= 0.10: an.append("loss_spike")
    if fd100 >= 0.05: an.append("high_delay_tail")
    if inp["frac_speed_lt_0.5"] >= 0.5 and inp["waiting_s_mean"] >= 5: an.append("standstill_cluster")
    if inp["jitter_ms_mean"] >= C.HIGH_JITTER_MS: an.append("high_jitter")
    if inp["delivery_ratio"] < C.DELIVERY_RATIO_LOW: an.append("throughput_drop")
    if ov is not None: load, sev, t_qos, t_load, an = ov        # categorical fields predicted by the Observe LLM (compact2.decode_observe2); everything else stays rule-computed
    emerg = inp["emergency_vehicles"]
    where = "network" if scope == "network" else f"cell {zone}"
    narr = (f"{(where[0].upper() + where[1:])}: {n} vehicles, {load.replace('_', ' ')} load, QoS {sev} "
            f"(mean loss {lm:.1f}%, delay {dm:.1f} ms, {f_viol*100:.0f}% vehicles violating). Trend: QoS {t_qos}, load {t_load.replace('_', ' ')}.")
    if an: narr += " Anomalies: " + ", ".join(an).replace("_", " ") + "."
    if emerg:
        e = emerg[0]
        narr += (f" Emergency vehicle {e['vehicle_id']} " + ("in this cell" if e["in_scope"] else "nearby") + f"; {e['n_vehicles_within_300m']} vehicles within 300 m"
                 + (f", nearest {e['nearest_vehicle_dist_m']:.0f} m." if e["nearest_vehicle_dist_m"] is not None else "."))
    return {"window_s": [ws, we], "scope": {"type": scope, "cell_id": zone}, "n_active_vehicles": n, "load_level": load,
            "qos_status": {"severity": sev, "frac_loss_gt_10pct": fl10, "frac_delay_gt_thr": fd20, "frac_delay_gt_100ms": fd100, "mean_loss_pct": lm,
                           "p95_loss_pct": lp, "mean_delay_ms": dm, "mean_jitter_ms": inp["jitter_ms_mean"], "mean_throughput_mbps": inp["throughput_mbps_mean"]},
            "trend": {"qos": t_qos, "load": t_load, "delta_loss_pp": r2(d_loss), "delta_vehicles_pct": r2(d_veh)},
            "mobility": {"mean_speed_mps": inp["speed_mps_mean"], "frac_stationary": inp["frac_speed_lt_0.5"], "mean_wait_s": inp["waiting_s_mean"]},
            "v2x_messages": inp["v2x_messages"], "resource_state": inp["resource_state"],
            "hotspots": {"edges": [e["edge"] for e in inp["worst_edges"]], "vehicles_at_risk": [v["id"] for v in inp["worst_vehicles"] if v["loss_pct"] > C.LOSS_VIOLATION_PCT]},
            "anomalies": an, "special_events": [{"type": "emergency_vehicle", **e} for e in emerg], "narrative": narr}
