"""Build Decide-DS from Orient samples (RSU-cell level) following the PROPOSAL's system model.

Teacher 'cplex': per cell and window, for each interference strategy solve a small MILP with IBM CPLEX
(PRB share per class, power level per class, MEC offload level + compute share per class) and keep the best.
Teacher 'rule': rule baseline. ALL model parameters are placeholders (config.OPT / SYSTEM_MODEL, ASSUMPTIONS.md 6b).

Usage: python build_decide.py [--teacher cplex|rule]
"""
import argparse, json, glob, os, hashlib, time
import config as C

from system_model import *   # CL, MIN_SHARE, O, STRATS, rho, cap_dbm, cell_load, resource_state (single shared model)

# ------------------------------------------------------------------ the MILP (one strategy)
def solve_strategy(s, w, cons, d0, T, N, levels):
    from docplex.mp.model import Model
    m = Model("decide")
    K = range(len(O["offload_levels"])); L = range(len(levels))
    p = {c: m.integer_var(0, 100, f"p_{c}") for c in CL}
    g = {c: m.integer_var(0, 100, f"g_{c}") for c in CL}
    y = {(c, l): m.binary_var(f"y_{c}_{l}") for c in CL for l in L}
    wv = {(c, l): m.continuous_var(0, 100, f"w_{c}_{l}") for c in CL for l in L}
    b = {(c, k): m.binary_var(f"b_{c}_{k}") for c in CL for k in K}
    sv = {c: m.continuous_var(0, name=f"s_{c}") for c in CL}
    e = {c: m.continuous_var(0, name=f"e_{c}") for c in CL}
    dev = {c: m.continuous_var(0, 100, f"dev_{c}") for c in CL}
    short = m.continuous_var(0, name="short")
    m.add_constraint(m.sum(p.values()) == 100); m.add_constraint(m.sum(g.values()) == 100)
    m.add_constraint(p["urllc"] >= cons["urllc_min_prb_share_pct"]); m.add_constraint(p["embb"] <= cons["embb_max_prb_share_pct"])
    for c in CL:
        m.add_constraint(p[c] >= MIN_SHARE); m.add_constraint(g[c] >= MIN_SHARE)
        m.add_constraint(m.sum(y[c, l] for l in L) == 1); m.add_constraint(m.sum(b[c, k] for k in K) == 1)
        for l in L:                                           # wv = p * y (linearised)
            m.add_constraint(wv[c, l] <= 100 * y[c, l]); m.add_constraint(wv[c, l] <= p[c])
            m.add_constraint(wv[c, l] >= p[c] - 100 * (1 - y[c, l]))
        q = m.sum(O["offload_levels"][k] * b[c, k] for k in K)
        dem = d0[c] + O["uplink_mbps_per_task_rate"] * T[c] * q          # offloading costs uplink capacity
        m.add_constraint(sv[c] <= dem)
        m.add_constraint(sv[c] <= m.sum(rho(levels[l], s, N) * wv[c, l] for l in L))
        m.add_constraint(e[c] <= q * T[c]); m.add_constraint(e[c] <= C.SYSTEM_MODEL["mec_capacity_tasks_per_s"] * g[c] / 100.0)
        m.add_constraint(dev[c] >= p[c] - 100 * w[c]); m.add_constraint(dev[c] >= 100 * w[c] - p[c])
    m.add_constraint(sv["urllc"] + short >= O["urllc_target"] * d0["urllc"])
    # interference limit: PRB-averaged linear transmit power must not exceed the load/strategy-dependent cap
    m.add_constraint(m.sum(wv[c, l] / 100.0 * 10 ** (levels[l] / 10) for c in CL for l in L) <= 10 ** (cap_dbm(N, s) / 10))
    radio = m.sum(w[c] * sv[c] / d0[c] for c in CL)
    mec = m.sum(w[c] * (O["local_value"] * (T[c] - m.sum(O["offload_levels"][k] * b[c, k] for k in K) * T[c]) + e[c]) / T[c] for c in CL)
    m.maximize(radio + O["mec_weight"] * mec - O["shortfall_weight"] * short / d0["urllc"]
               - O["tie_eps"] * m.sum(dev.values()) - O["power_penalty"] * m.sum(levels[l] * y[c, l] for c in CL for l in L) / 23.0)
    m.parameters.mip.tolerances.mipgap = 0.0; m.parameters.mip.tolerances.absmipgap = 0.0   # exact optimum, no gap tolerance
    sol = m.solve(log_output=False)
    if sol is None or "OPTIMAL" not in str(m.solve_details.status).upper(): return None
    pick = lambda c: levels[[l for l in L if y[c, l].solution_value > .5][0]]
    return {"obj": m.objective_value, "s": s,
            "p": {c: int(round(p[c].solution_value)) for c in CL}, "g": {c: int(round(g[c].solution_value)) for c in CL},
            "lvl": {c: pick(c) for c in CL},
            "q": {c: O["offload_levels"][[k for k in K if b[c, k].solution_value > .5][0]] for c in CL}}

_cache = {}; _stats = {"problems": 0, "milps": 0, "hits": 0, "nonoptimal": 0}

def optimize(w, cons, o):
    d0, T, N = cell_load(o)
    cap = cons.get("tx_power_cap_dbm", 99)
    levels = [l for l in O["levels_dbm"] if l <= cap] or [min(O["levels_dbm"])]
    key = (tuple(round(w[c], 2) for c in CL), cons["urllc_min_prb_share_pct"], cons["embb_max_prb_share_pct"], tuple(levels),
           tuple(d0[c] for c in CL), tuple(T[c] for c in CL), N)
    if key in _cache: _stats["hits"] += 1; return _cache[key]
    _stats["problems"] += 1; best = None
    for s in STRATS:
        r = solve_strategy(s, w, cons, d0, T, N, levels); _stats["milps"] += 1
        if r is None: _stats["nonoptimal"] += 1; continue
        if best is None or r["obj"] > best["obj"] + 1e-6: best = r
    _cache[key] = best
    return best

# ------------------------------------------------------------------ rule baseline
def prb_shares_rule(w, cons):
    umin, emax = cons["urllc_min_prb_share_pct"], cons["embb_max_prb_share_pct"]
    u = max(round(100 * w["urllc"]), umin); rest = 100 - u
    e = round(min(rest * w["embb"] / max(w["embb"] + w["mmtc"], 1e-9), emax)); m = rest - e
    if m < MIN_SHARE: e -= MIN_SHARE - m; m = MIN_SHARE
    e = max(e, MIN_SHARE); u = 100 - e - m
    return {"urllc": int(u), "embb": int(e), "mmtc": int(m)}

def rule(w, cons, tag, sev):
    p = prb_shares_rule(w, cons); cap = cons.get("tx_power_cap_dbm", 23)
    s = {"emergency_preempt": "sidelink_mode1_scheduling"}.get(tag, "frequency_reuse" if tag == "intersection_dense" and sev == "critical"
         else "power_backoff_edge" if tag == "intersection_dense" and sev == "degraded" else "none")
    g = {c: max(MIN_SHARE, round(100 * w[c])) for c in CL}; g["urllc"] = 100 - g["embb"] - g["mmtc"]
    q = {"emergency_preempt": 0.5, "intersection_dense": 0.5}.get(tag, 0.0)
    return {"p": p, "g": g, "lvl": {"urllc": min(23, cap), "embb": min(17, cap), "mmtc": min(14, cap)}, "q": {c: q for c in CL}, "s": s}

# ------------------------------------------------------------------ policy object
def to_policy(r, cons, key):
    p = r["p"]; order = sorted(CL, key=lambda k: (-p[k], CL.index(k)))
    ids = cons.get("preempt_vehicle_ids", [])
    qs = list(r["q"].values()); mode = "none" if max(qs) == 0 else "full" if min(qs) == 1 else "selective"
    return {"policy_id": "P-" + hashlib.md5(key.encode()).hexdigest()[:8],
            "valid_for_ms": int(min(5000, max(100, 5 * cons["max_policy_latency_ms"]))),
            "prb_priority": {"urllc_pct": p["urllc"], "embb_pct": p["embb"], "mmtc_pct": p["mmtc"], "priority_order": order},
            "power_budget_dbm": {c: float(r["lvl"][c]) for c in CL},
            "mec_offloading": {"mode": mode, "offload_ratio": round(sum(qs) / 3, 2),
                               "priority_class": max(CL, key=lambda c: (r["g"][c], -CL.index(c))),
                               "compute_share_pct": {c: int(r["g"][c]) for c in CL}},
            "interference_coordination": {"strategy": r["s"], "reuse_factor": O["strategy"][r["s"]][3]},
            "preemption": {"enabled": bool(ids), "vehicle_ids": ids}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", default="cplex", choices=["cplex", "rule"]); a = ap.parse_args()
    prefix = "decide" if a.teacher == "cplex" else "decide_rule"; t0 = time.time()
    for path in sorted(glob.glob("data/orient_S*.jsonl")):
        s = os.path.basename(path).split("_")[1].split(".")[0]; n = 0
        with open(f"data/{prefix}_{s}.jsonl", "w") as f:
            for line in open(path):
                r = json.loads(line); inp = json.loads(r["messages"][1]["content"]); ctx = json.loads(r["messages"][2]["content"])
                o = inp["observe_summary"]
                if o["scope"]["type"] != "cell": continue
                w, cons = ctx["qos_weights"], ctx["constraint_overrides"]
                res = optimize(w, cons, o) if a.teacher == "cplex" else None
                if res is None: res = rule(w, cons, ctx["scenario_tag"], o["qos_status"]["severity"])
                state = {k: o[k] for k in ("window_s", "n_active_vehicles", "load_level", "trend", "special_events", "anomalies")}
                state["qos"] = {k: o["qos_status"][k] for k in ("severity", "mean_loss_pct", "p95_loss_pct", "mean_delay_ms", "frac_loss_gt_10pct")}
                state["cell_id"] = o["scope"]["cell_id"]
                state["v2x_messages"] = o["v2x_messages"]
                state["resource_state"] = {k: o["resource_state"][k] for k in ("estimated", "offered_load_ratio", "interference_iot_db", "mec_load_ratio")}
                din = {"context": ctx, "cell_state": state, "current_allocation": o["resource_state"]["current_allocation"], "system_model": C.SYSTEM_MODEL}
                rec = {"id": r["id"] + "-d", "agent": "decide", "teacher": a.teacher, "situation": r["situation"], "split": r["split"],
                       "obs_id": r["obs_id"], "scenario_tag": r["scenario_tag"],
                       "messages": [{"role": "system", "content": C.DECIDE_SYSTEM_PROMPT},
                                    {"role": "user", "content": json.dumps(din, separators=(",", ":"))},
                                    {"role": "assistant", "content": json.dumps(to_policy(res, cons, r["id"]), separators=(",", ":"))}]}
                f.write(json.dumps(rec) + "\n"); n += 1
        print(f"{s}: {n} decide samples ({a.teacher})")
    print(f"done in {time.time()-t0:.0f}s; solver stats: {_stats}")

if __name__ == "__main__":
    main()
