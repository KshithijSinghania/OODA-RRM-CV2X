"""Build Observe-DS (telemetry -> situational summary) from one situation CSV.

Each window yields one 'network' sample (Non-RT RIC view) plus one sample per RSU cell (zones stand in
for RSU coverage cells). Works for all four situations; columns that only exist in S4
(ambulance_active, min_dist_to_ambulance_m) are used when present.

Usage: python build_observe.py --csv indore_04_combined_5s.csv --situation S4
"""
import argparse, json, hashlib
import numpy as np, pandas as pd
import config as C
import system_model as SM

r2 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)

def anon_cell(situation, zone_id, ws):
    """Per-sample pseudonym for a cell. Consistent inside one sample (so Act can still address the zone) but unrelated
    across samples, so a model cannot memorise 'cell X is congested'. True cell is kept in record metadata only."""
    h = int(hashlib.md5(f"{C.SEED}-anon-{situation}-{zone_id}-{ws}".encode()).hexdigest()[:8], 16)
    return "R%03d" % (100 + h % 900)

def level(value, bins):
    for ub, name in bins:
        if value < ub:
            return name
    return bins[-1][1]

def is_emergency(vt: pd.Series) -> pd.Series:
    return vt.astype(str).str.contains("emerg|ambul", case=False, regex=True)

def add_density(df: pd.DataFrame) -> pd.DataFrame:
    """#other vehicles within 300 m of end position in the same window (matches S2's local_density_300m exactly)."""
    dens = np.zeros(len(df))
    xy = df[["end_x", "end_y"]].to_numpy()
    for idx in df.groupby("window_start").indices.values():
        p = xy[idx]
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        dens[idx] = (d2 <= 300.0 ** 2).sum(1) - 1
    df["density_300m"] = dens
    return df

def aggregate(cars: pd.DataFrame, dur: float) -> dict:
    """Telemetry-DB style aggregation over regular vehicles (ambulance excluded) of one scope/window."""
    viol = (cars.packet_loss_percent > C.LOSS_VIOLATION_PCT) | (cars.mean_delay_ms > C.DELAY_VIOLATION_MS)
    worst_edges = (cars.groupby("end_edge_id")
                   .agg(n=("vehicle_id", "size"), loss=("packet_loss_percent", "mean"), delay=("mean_delay_ms", "mean"))
                   .query("n >= 2").sort_values("loss", ascending=False).head(3))
    worst_veh = cars.sort_values("packet_loss_percent", ascending=False).head(3)
    return {
        "n": int(len(cars)),
        "msg_rate_mean": float((cars.tx_packets / dur).mean()),
        "f_alert_rate": float(((cars.tx_packets / dur) >= C.MSG_RATE_NOMINAL_HZ * C.ALERT_RATE_FACTOR).mean()),
        "loss_mean": cars.packet_loss_percent.mean(), "loss_p95": cars.packet_loss_percent.quantile(.95),
        "delay_mean": cars.mean_delay_ms.mean(), "delay_p95": cars.mean_delay_ms.quantile(.95),
        "jitter_mean": cars.mean_jitter_ms.mean(), "thr_mean": cars.throughput_mbps.mean(),
        "delivery": float(cars.rx_bytes.sum() / max(cars.tx_bytes.sum(), 1)),
        "dens_mean": cars.density_300m.mean(), "dens_max": cars.density_300m.max(),
        "speed_mean": cars.mean_speed.mean(),
        "frac_stationary": float((cars.mean_speed < C.STATIONARY_SPEED_MPS).mean()),
        "wait_mean": cars.mean_waiting_time.mean(),
        "f_loss10": float((cars.packet_loss_percent > C.LOSS_VIOLATION_PCT).mean()),
        "f_delay20": float((cars.mean_delay_ms > C.DELAY_VIOLATION_MS).mean()),
        "f_delay100": float((cars.mean_delay_ms > C.DELAY_TAIL_MS).mean()),
        "f_viol": float(viol.mean()),
        "n_loss_spike": int((cars.packet_loss_percent > C.LOSS_SPIKE_PCT).sum()),
        "worst_edges": [(i, int(r.n), r2(r.loss), r2(r.delay)) for i, r in worst_edges.iterrows()],
        "worst_veh": [(r.vehicle_id, r2(r.packet_loss_percent), r2(r.mean_delay_ms), r2(r.mean_speed)) for r in worst_veh.itertuples()],
    }

def emergency_info(cars: pd.DataFrame, amb: pd.DataFrame, has_dist: bool) -> list:
    """Emergency events relevant to this scope: ambulance inside it, or cars inside its alert zone."""
    out = []
    near = None; n_in = 0
    if has_dist:
        d = cars.loc[cars.ambulance_active == 1, "min_dist_to_ambulance_m"].dropna()
        if len(d):
            near = float(d.min()); n_in = int((d <= C.ALERT_RADIUS_M).sum())
    if len(amb) or n_in > 0:
        vid = amb.vehicle_id.iloc[0] if len(amb) else "AMBULANCE_01"
        out.append({"vehicle_id": vid, "in_scope": bool(len(amb)),
                    "n_vehicles_within_300m": n_in, "nearest_vehicle_dist_m": r2(near),
                    "link_loss_pct": r2(amb.packet_loss_percent.mean()) if len(amb) else None,
                    "link_delay_ms": r2(amb.mean_delay_ms.mean()) if len(amb) else None})
    return out

def _rs(scope, a, emerg):
    return SM.resource_state({"n_active_vehicles": a["n"], "special_events": emerg}) if scope == "cell" else None

def telemetry_input(ws, we, scope, zone_id, a, prev, emerg):
    """What the Observe agent is given (the 'DB query result'). No situation id: it must not leak downstream."""
    return {
        "window_s": [ws, we], "scope": {"type": scope, "cell_id": zone_id},
        "n_active_vehicles": a["n"],
        "loss_pct": {"mean": r2(a["loss_mean"]), "p95": r2(a["loss_p95"])},
        "delay_ms": {"mean": r2(a["delay_mean"]), "p95": r2(a["delay_p95"])},
        "jitter_ms_mean": r2(a["jitter_mean"]), "throughput_mbps_mean": r2(a["thr_mean"]),
        "delivery_ratio": r2(a["delivery"]),
        "v2x_messages": {"mean_rate_hz": r2(a["msg_rate_mean"]), "frac_alert_rate": r2(a["f_alert_rate"])},
        "resource_state": _rs(scope, a, emerg),
        "density_300m": {"mean": r2(a["dens_mean"]), "max": r2(a["dens_max"])},
        "speed_mps_mean": r2(a["speed_mean"]), "frac_speed_lt_0.5": r2(a["frac_stationary"]),
        "waiting_s_mean": r2(a["wait_mean"]),
        "frac_loss_gt_10pct": r2(a["f_loss10"]), "frac_delay_gt_thr": r2(a["f_delay20"]),
        "frac_delay_gt_100ms": r2(a["f_delay100"]), "n_loss_gt_50pct": a["n_loss_spike"],
        "worst_edges": [{"edge": e, "n": n, "loss_pct": l, "delay_ms": d} for e, n, l, d in a["worst_edges"]],
        "worst_vehicles": [{"id": v, "loss_pct": l, "delay_ms": d, "speed_mps": s} for v, l, d, s in a["worst_veh"]],
        "prev_windows_mean": ({"loss_pct": r2(prev["loss_mean"]), "n_vehicles": r2(prev["n"])} if prev else None),
        "emergency_vehicles": emerg,
    }

def make_summary(ws, we, scope, zone_id, a, prev, emerg):
    if prev is None:
        t_qos, t_load, d_loss, d_veh = "unknown", "unknown", None, None
    else:
        d_loss = a["loss_mean"] - prev["loss_mean"]
        d_veh = 100.0 * (a["n"] - prev["n"]) / max(prev["n"], 1)
        t_qos = "worsening" if d_loss > C.TREND_LOSS_PP else "improving" if d_loss < -C.TREND_LOSS_PP else "stable"
        t_load = "ramping_up" if d_veh > C.TREND_LOAD_PCT else "draining" if d_veh < -C.TREND_LOAD_PCT else "steady"
    sev = level(a["f_viol"], C.SEVERITY_BINS)
    load = level(a["dens_mean"], C.LOAD_BINS)
    an = []
    if a["n_loss_spike"] >= 1 and a["n_loss_spike"] / a["n"] >= 0.10: an.append("loss_spike")
    if a["f_delay100"] >= 0.05: an.append("high_delay_tail")
    if a["frac_stationary"] >= 0.5 and a["wait_mean"] >= 5: an.append("standstill_cluster")
    if a["jitter_mean"] >= C.HIGH_JITTER_MS: an.append("high_jitter")
    if a["delivery"] < C.DELIVERY_RATIO_LOW: an.append("throughput_drop")
    where = "network" if scope == "network" else f"cell {zone_id}"
    narr = (f"{(where[0].upper() + where[1:])}: {a['n']} vehicles, {load.replace('_', ' ')} load, QoS {sev} "
            f"(mean loss {a['loss_mean']:.1f}%, delay {a['delay_mean']:.1f} ms, {a['f_viol']*100:.0f}% vehicles violating). "
            f"Trend: QoS {t_qos}, load {t_load.replace('_', ' ')}.")
    if an: narr += " Anomalies: " + ", ".join(an).replace("_", " ") + "."
    if emerg:
        e = emerg[0]
        narr += (f" Emergency vehicle {e['vehicle_id']} " + ("in this cell" if e["in_scope"] else "nearby")
                 + f"; {e['n_vehicles_within_300m']} vehicles within 300 m"
                 + (f", nearest {e['nearest_vehicle_dist_m']:.0f} m." if e["nearest_vehicle_dist_m"] is not None else "."))
    return {
        "window_s": [ws, we], "scope": {"type": scope, "cell_id": zone_id},
        "n_active_vehicles": a["n"], "load_level": load,
        "qos_status": {"severity": sev, "frac_loss_gt_10pct": r2(a["f_loss10"]), "frac_delay_gt_thr": r2(a["f_delay20"]),
                       "frac_delay_gt_100ms": r2(a["f_delay100"]), "mean_loss_pct": r2(a["loss_mean"]), "p95_loss_pct": r2(a["loss_p95"]),
                       "mean_delay_ms": r2(a["delay_mean"]), "mean_jitter_ms": r2(a["jitter_mean"]), "mean_throughput_mbps": r2(a["thr_mean"])},
        "trend": {"qos": t_qos, "load": t_load, "delta_loss_pp": r2(d_loss), "delta_vehicles_pct": r2(d_veh)},
        "mobility": {"mean_speed_mps": r2(a["speed_mean"]), "frac_stationary": r2(a["frac_stationary"]), "mean_wait_s": r2(a["wait_mean"])},
        "v2x_messages": {"mean_rate_hz": r2(a["msg_rate_mean"]), "frac_alert_rate": r2(a["f_alert_rate"])},
        "resource_state": _rs(scope, a, emerg),
        "hotspots": {"edges": [e for e, *_ in a["worst_edges"]],
                     "vehicles_at_risk": [v for v, l, d, s in a["worst_veh"] if l > C.LOSS_VIOLATION_PCT]},
        "anomalies": an,
        "special_events": [{"type": "emergency_vehicle", **e} for e in emerg],
        "narrative": narr,
    }

def assign_splits(df: pd.DataFrame, windows: list) -> dict:
    """Window -> split. Contiguous blocks of windows, stratified by (congested, ambulance-present)
    so val/test always contain every kind of period, not only the quiet tail."""
    widx = {w: i for i, w in enumerate(windows)}
    bsz = min(C.SPLIT_BLOCK_WINDOWS, max(3, len(windows) // 28))
    blk = df.window_start.map(lambda w: widx[w] // bsz)
    loss = df.groupby(blk).packet_loss_percent.mean()
    amb = df.groupby(blk).emerg.any()
    rng = np.random.default_rng(C.SEED)
    block_split = {}
    for key in sorted({(bool(loss[b] > C.LOSS_VIOLATION_PCT), bool(amb[b])) for b in loss.index}):
        ids = [b for b in loss.index if (bool(loss[b] > C.LOSS_VIOLATION_PCT), bool(amb[b])) == key]
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * C.SPLIT_FRACS[2])) if len(ids) >= 3 else 0
        n_val = max(1, round(len(ids) * C.SPLIT_FRACS[1])) if len(ids) >= 3 else 0
        for i, b in enumerate(ids):
            block_split[b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return {w: block_split[widx[w] // bsz] for w in windows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--situation", required=True, choices=C.SITUATIONS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    # S1/S3 CSVs carry placeholder rows for vehicles not (yet) in the simulation: no type, no position, 0 packets.
    active = df.vehicle_type.notna() & (df.tx_packets > 0)
    print(f"dropping {int((~active).sum())} inactive/placeholder rows of {len(df)}")
    df = df[active].reset_index(drop=True)
    df["emerg"] = is_emergency(df.vehicle_type)
    df = add_density(df)
    if "local_density_300m" in df:
        print("density check vs CSV column: max abs diff =", float((df.density_300m - df.local_density_300m).abs().max()))
    has_dist = "min_dist_to_ambulance_m" in df and "ambulance_active" in df
    sites = np.array(C.CELL_SITES)
    d2 = ((df[["end_x", "end_y"]].to_numpy()[:, None, :] - sites[None, :, :]) ** 2).sum(-1)
    df["cell"] = ["C%02d" % i for i in d2.argmin(1)]          # C00 = macro gNB, C01..C11 = RSUs
    n_cars = df.loc[~df.emerg, "vehicle_id"].nunique()
    print(f"{args.situation}: {len(df)} rows, {n_cars} regular vehicles (expected {C.SITUATIONS[args.situation]['n_vehicles']}), "
          f"{int(df.emerg.sum())} emergency-vehicle rows")

    windows = sorted(df.window_start.unique())
    win_split = assign_splits(df, windows)
    win_split = {w: "test" for w in windows}      # HELD-OUT: every window is test
    scopes = [("network", "ALL", df)] + [("cell", z, g) for z, g in df.groupby("cell")]
    out = args.out or f"data/observe_{args.situation}.jsonl"
    count = 0
    with open(out, "w") as f:
        for scope, zone_id, sdf in scopes:
            hist = []
            for ws, wdf in sdf.groupby("window_start"):
                cars, amb = wdf[~wdf.emerg], wdf[wdf.emerg]
                if len(cars) == 0 or (scope == "cell" and len(cars) < C.MIN_VEHICLES_PER_ZONE):
                    continue
                we = float(wdf.window_end.max())
                a = aggregate(cars, we - float(ws))
                emerg = emergency_info(cars, amb, has_dist)
                prev = None
                if hist:
                    last = hist[-C.TREND_LOOKBACK:]
                    prev = {"loss_mean": np.mean([h["loss_mean"] for h in last]), "n": np.mean([h["n"] for h in last])}
                hist.append(a)
                zlabel = "ALL" if scope == "network" else anon_cell(args.situation, zone_id, int(ws))
                inp = telemetry_input(float(ws), we, scope, zlabel, a, prev, emerg)
                outp = make_summary(float(ws), we, scope, zlabel, a, prev, emerg)
                rec = {
                    "id": f"{args.situation}-{scope}-{zone_id}-{int(ws)}", "agent": "observe",
                    "situation": args.situation, "split": win_split[ws], "layer": "rsu_edge" if scope == "cell" else "non_rt_ric", "cell": zone_id,   # metadata only: NOT part of the training text
                    
                    "messages": [
                        {"role": "system", "content": C.OBSERVE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(inp, separators=(",", ":"))},
                        {"role": "assistant", "content": json.dumps(outp, separators=(",", ":"))},
                    ],
                }
                f.write(json.dumps(rec) + "\n"); count += 1
    print(f"wrote {count} samples -> {out}")

if __name__ == "__main__":
    main()
