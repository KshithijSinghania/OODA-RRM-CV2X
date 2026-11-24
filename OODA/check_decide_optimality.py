"""Independent check: recompute the Decide objective in plain Python and try to beat CPLEX's solution by random + local search."""
import json, glob, random, math
import build_decide as B
CL, O = B.CL, B.O

def evaluate(sol, w, cons, o, levels):
    d0, T, N = B.cell_load(o); s = sol["s"]
    p, g, lvl, q = sol["p"], sol["g"], sol["lvl"], sol["q"]
    if sum(p.values()) != 100 or sum(g.values()) != 100: return None
    if p["urllc"] < cons["urllc_min_prb_share_pct"] or p["embb"] > cons["embb_max_prb_share_pct"]: return None
    if min(p.values()) < B.MIN_SHARE or min(g.values()) < B.MIN_SHARE or any(lvl[c] not in levels for c in CL): return None
    if sum(p[c] / 100 * 10 ** (lvl[c] / 10) for c in CL) > 10 ** (B.cap_dbm(N, s) / 10) * (1 + 1e-9): return None
    obj = 0; sv = {}
    for c in CL:
        dem = d0[c] + O["uplink_mbps_per_task_rate"] * T[c] * q[c]
        sv[c] = min(dem, B.rho(lvl[c], s, N) * p[c])
        e = min(q[c] * T[c], B.C.SYSTEM_MODEL["mec_capacity_tasks_per_s"] * g[c] / 100)
        obj += w[c] * sv[c] / d0[c] + O["mec_weight"] * w[c] * (O["local_value"] * (T[c] - q[c] * T[c]) + e) / T[c]
    short = max(0, O["urllc_target"] * d0["urllc"] - sv["urllc"])
    obj += -O["shortfall_weight"] * short / d0["urllc"] - O["tie_eps"] * sum(abs(p[c] - 100 * w[c]) for c in CL) \
           - O["power_penalty"] * sum(lvl.values()) / 23.0
    return obj

def rand_split(rng, lo=B.MIN_SHARE):
    a = sorted(rng.sample(range(lo, 100 - 2 * lo + 1), 2)); a = [lo + a[0] - lo, a[1]]
    x = rng.randint(lo, 100 - 2 * lo); y = rng.randint(lo, 100 - x - lo); return {"urllc": x, "embb": y, "mmtc": 100 - x - y}

recs = [json.loads(l) for f in sorted(glob.glob("data/decide_S*.jsonl")) for l in open(f)]
rng = random.Random(7); worse = checked = 0; maxgap = -1e9; lowest = 1e9
for r in rng.sample(recs, 120):
    din = json.loads(r["messages"][1]["content"]); pol = json.loads(r["messages"][2]["content"]); cons = din["context"]["constraint_overrides"]
    w = din["context"]["qos_weights"]
    # rebuild an observe-like object for cell_load (needs only these fields)
    o = {"n_active_vehicles": din["cell_state"]["n_active_vehicles"], "special_events": din["cell_state"]["special_events"]}
    cap = cons.get("tx_power_cap_dbm", 99); levels = [l for l in O["levels_dbm"] if l <= cap] or [min(O["levels_dbm"])]
    sol = {"s": pol["interference_coordination"]["strategy"], "p": {"urllc": pol["prb_priority"]["urllc_pct"], "embb": pol["prb_priority"]["embb_pct"], "mmtc": pol["prb_priority"]["mmtc_pct"]},
           "g": pol["mec_offloading"]["compute_share_pct"], "lvl": {c: int(pol["power_budget_dbm"][c]) for c in CL}, "q": None}
    # offload level per class is not stored in the policy; recover the best q per class by enumeration (cheap) for the CPLEX solution
    best_q = None
    for q0 in O["offload_levels"]:
        for q1 in O["offload_levels"]:
            for q2 in O["offload_levels"]:
                sol["q"] = {"urllc": q0, "embb": q1, "mmtc": q2}
                v = evaluate(sol, w, cons, o, levels)
                if v is not None and (best_q is None or v > best_q[0]): best_q = (v, dict(sol["q"]))
    if best_q is None: print("CPLEX solution infeasible in plain-Python model!", r["id"]); worse += 1; continue
    cplex_obj = best_q[0]; checked += 1
    # random search + local moves
    top = -1e9
    for _ in range(1500):
        c = {"s": rng.choice(B.STRATS), "p": rand_split(rng), "g": rand_split(rng), "lvl": {k: rng.choice(levels) for k in CL},
             "q": {k: rng.choice(O["offload_levels"]) for k in CL}}
        v = evaluate(c, w, cons, o, levels)
        if v is not None and v > top: top = v; cur = c
    for _ in range(600):                                  # local search from the best random point
        c = json.loads(json.dumps(cur)); k = rng.choice(["p", "g", "lvl", "q", "s"])
        if k in ("p", "g"):
            a, b = rng.sample(CL, 2); c[k][a] += 1; c[k][b] -= 1
        elif k == "lvl": c["lvl"][rng.choice(CL)] = rng.choice(levels)
        elif k == "q": c["q"][rng.choice(CL)] = rng.choice(O["offload_levels"])
        else: c["s"] = rng.choice(B.STRATS)
        v = evaluate(c, w, cons, o, levels)
        if v is not None and v > top: top = v; cur = c
    gap = top - cplex_obj; maxgap = max(maxgap, gap); worse += gap > 1e-6
print(f"checked {checked} samples: search beat CPLEX in {worse}; largest (search - CPLEX) objective gap {maxgap:.2e} (<= 0 means CPLEX never beaten)")
