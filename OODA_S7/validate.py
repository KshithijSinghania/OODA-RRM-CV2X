"""Validate a role dataset: assistant output vs JSON Schema, plus size stats.
Usage: python validate.py data/observe_S2.jsonl observe_summary"""
import sys, json, collections
import jsonschema

path, schema_name = sys.argv[1], sys.argv[2]
schema = json.load(open(f"schemas/{schema_name}.schema.json"))
v = jsonschema.Draft202012Validator(schema)
bad, n, toks = 0, 0, []
cnt = collections.defaultdict(collections.Counter)
for line in open(path):
    rec = json.loads(line); n += 1
    out = json.loads(rec["messages"][-1]["content"])
    errs = list(v.iter_errors(out))
    if errs:
        bad += 1
        if bad <= 3: print("INVALID", rec["id"], errs[0].message[:200])
    toks.append(len(rec["messages"][-1]["content"]) / 4)   # rough chars/4 token estimate
    cnt["split"][rec["split"]] += 1
    if "prb_priority" in out:
        import build_decide as B
        din = json.loads(rec["messages"][1]["content"]); ctx = din["context"]; cons = ctx["constraint_overrides"]; p = out["prb_priority"]
        N = din["cell_state"]["n_active_vehicles"]; errs2 = []
        if p["urllc_pct"] + p["embb_pct"] + p["mmtc_pct"] != 100: errs2.append("PRB sum")
        if sum(out["mec_offloading"]["compute_share_pct"].values()) != 100: errs2.append("compute sum")
        if p["urllc_pct"] < cons["urllc_min_prb_share_pct"]: errs2.append("urllc below min")
        if p["embb_pct"] > cons["embb_max_prb_share_pct"]: errs2.append("embb above max")
        lim = min(23, cons.get("tx_power_cap_dbm", 99))
        if max(out["power_budget_dbm"].values()) > lim and max(out["power_budget_dbm"].values()) > min(B.O["levels_dbm"]): errs2.append("power cap")
        if out["preemption"]["vehicle_ids"] != cons.get("preempt_vehicle_ids", []): errs2.append("preempt ids")
        sh = {"urllc": p["urllc_pct"], "embb": p["embb_pct"], "mmtc": p["mmtc_pct"]}
        lhs = sum(sh[c] / 100 * 10 ** (out["power_budget_dbm"][c] / 10) for c in sh)
        if lhs > 10 ** (B.cap_dbm(N, out["interference_coordination"]["strategy"]) / 10) * (1 + 1e-9): errs2.append("interference limit")
        if errs2:
            bad += 1
            if bad <= 3: print("CONSTRAINT VIOLATION", rec["id"], errs2)
        cnt["tag"][ctx["scenario_tag"]] += 1; cnt["interference"][out["interference_coordination"]["strategy"]] += 1
        cnt["mec mode"][out["mec_offloading"]["mode"]] += 1; cnt["max power dBm"][max(out["power_budget_dbm"].values())] += 1
    if "qos_weights" in out:
        w = out["qos_weights"]
        if abs(w["urllc"] + w["embb"] + w["mmtc"] - 1) > 0.011: bad += 1; print("WEIGHTS != 1", rec["id"])
        cnt["tag"][out["scenario_tag"]] += 1; cnt["conflict"][out["intent_resolution"]["conflict_type"]] += 1
        cnt["situation x tag"][(rec["situation"], out["scenario_tag"])] += 1
    if "qos_status" in out:
        cnt["severity"][out["qos_status"]["severity"]] += 1; cnt["load"][out["load_level"]] += 1
        for a in out["anomalies"]: cnt["anomaly"][a] += 1
        cnt["emergency_event"][bool(out["special_events"])] += 1
print(f"{n} records, {bad} invalid; output ~tokens mean {sum(toks)/n:.0f}, max {max(toks):.0f} (budget 512)")
for k, c in cnt.items(): print(k, dict(c))
