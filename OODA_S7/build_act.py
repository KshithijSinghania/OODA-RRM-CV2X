"""Build Act-DS from Decide samples.

Base: one sample per Decide sample. Execution feedback = the same cell's NEXT-window Observe summary (real trace data)
packaged as an enforcer report, plus synthetic enforcer faults (the traces are uncontrolled, so a policy's real effect
is never simulated -- ASSUMPTIONS.md 6d).  Adversarial: +1000 malformed policies, +1000 spoofed feedback reports.
Labels come from one deterministic teacher (act_teacher); injections are verified to be detected as intended.

Usage: python build_act.py
"""
import json, glob, os, copy, collections
import jsonschema
import config as C
from build_orient import rng_for

DSL = jsonschema.Draft202012Validator(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas", "decide_policy.schema.json"))))
SEV_BAD = ("degraded", "critical")
MAXP = C.SYSTEM_MODEL["max_vehicle_power_dbm"]

# ------------------------------------------------------------------ teacher
def check_policy(pol, cons):
    """Returns (reason, detail) -- reason in {None, malformed_policy, constraint_violation}."""
    if list(DSL.iter_errors(pol)): return "malformed_policy", "dsl"
    p = pol["prb_priority"]
    if p["urllc_pct"] + p["embb_pct"] + p["mmtc_pct"] != 100: return "malformed_policy", "prb_sum"
    if sum(pol["mec_offloading"]["compute_share_pct"].values()) != 100: return "malformed_policy", "compute_sum"
    cap = min(MAXP, cons.get("tx_power_cap_dbm", MAXP))
    ids = cons.get("preempt_vehicle_ids", [])
    if p["urllc_pct"] < cons["urllc_min_prb_share_pct"]: return "constraint_violation", "urllc_min"
    if p["embb_pct"] > cons["embb_max_prb_share_pct"]: return "constraint_violation", "embb_max"
    if max(pol["power_budget_dbm"].values()) > cap: return "constraint_violation", "power_cap"
    if pol["preemption"]["enabled"] != bool(ids) or pol["preemption"]["vehicle_ids"] != ids: return "constraint_violation", "preemption"
    return None, None

def check_feedback(fb, pol, expected_start):
    """Returns a spoof detail string or None."""
    if fb["policy_id"] != pol["policy_id"]: return "policy_id_echo"
    if not fb["auth_ok"]: return "auth"
    if fb["window_s"][0] != expected_start: return "stale_window"
    ob = fb["observed"]
    if not (0 <= ob["mean_loss_pct"] <= 100) or not (0 <= ob["frac_loss_gt_10pct"] <= 1) or not (0 <= fb["enforcer"]["prb_util_pct"] <= 100) or ob["mean_delay_ms"] < 0:
        return "impossible_value"
    if ob["severity"] == "nominal" and (ob["mean_loss_pct"] > C.ACT_NOMINAL_LOSS_MAX or ob["mean_delay_ms"] > C.ACT_NOMINAL_DELAY_MAX):
        return "inconsistent"
    return None

def translate(pol, cell_id):
    a1 = {"policy_type_id": "CV2X-RRM-POLICY-1", "policy_instance_id": pol["policy_id"], "scope": {"cell_id": cell_id},
          "statement": {"valid_for_ms": pol["valid_for_ms"], "class_priority": pol["prb_priority"]["priority_order"],
                        "preemption_enabled": pol["preemption"]["enabled"]}}
    p, pw, m, ic, pr = pol["prb_priority"], pol["power_budget_dbm"], pol["mec_offloading"], pol["interference_coordination"], pol["preemption"]
    e2 = [{"ran_function": "RC-PRB-QUOTA", "action": "SET_CLASS_PRB_QUOTA", "params": {"urllc_pct": p["urllc_pct"], "embb_pct": p["embb_pct"], "mmtc_pct": p["mmtc_pct"]}},
          {"ran_function": "RC-TX-POWER", "action": "SET_CLASS_POWER_CAP", "params": {"urllc_dbm": pw["urllc"], "embb_dbm": pw["embb"], "mmtc_dbm": pw["mmtc"]}}]
    if m["mode"] != "none":
        e2.append({"ran_function": "RC-MEC-OFFLOAD", "action": "SET_OFFLOAD_POLICY", "params": {"mode": m["mode"], "offload_ratio": m["offload_ratio"], "priority_class": m["priority_class"], "compute_share_pct": m["compute_share_pct"]}})
    if ic["strategy"] != "none":
        e2.append({"ran_function": "RC-INTERFERENCE", "action": "SET_COORDINATION", "params": {"strategy": ic["strategy"], "reuse_factor": ic["reuse_factor"]}})
    if pr["enabled"]:
        e2.append({"ran_function": "RC-PREEMPT", "action": "ENABLE_PREEMPTION", "params": {"vehicle_ids": pr["vehicle_ids"], "duration_ms": pol["valid_for_ms"]}})
    return a1, e2

def watch(pol):
    first = pol["prb_priority"]["priority_order"][0]
    w = {"urllc": ["delay", "packet_loss"], "embb": ["throughput", "packet_loss"], "mmtc": ["packet_loss", "throughput"]}[first]
    return w + (["jitter"] if pol["interference_coordination"]["strategy"] != "none" else [])

def act_teacher(inp):
    """Deterministic label function: input -> (output, reason_detail). Priority: reject policy > spoof > failure > violations > scenario > QoS."""
    pol, cons, fb = inp["policy"], inp["constraints"], inp["feedback"]
    reason, detail = check_policy(pol, cons)
    esc = lambda r, t: {"escalate": r != "none", "reason": r, "target": t}
    if reason:
        return {"a1_policy": None, "e2_control": [], "escalation": esc(reason, "decide"),
                "feedback_to_observe": {"policy_applied": False, "feedback_trusted": True, "watch_metrics": ["packet_loss", "delay"]}}, detail
    a1, e2 = translate(pol, inp["cell_id"]); trusted = True; applied = True; e = esc("none", "none"); detail = "clean"
    if fb is not None:
        sp = check_feedback(fb, pol, inp["expected_window_start_s"])
        if sp: trusted = False; e = esc("spoofed_feedback", "decide"); detail = "spoof:" + sp
        elif not fb["enforcer"]["applied"]: applied = False; e = esc("policy_execution_failure", "decide"); detail = "not_applied"
        elif fb["enforcer"]["violations"]: e = esc("constraint_violation", "decide"); detail = "enforcer_violation"
        elif fb["emergency_event_observed"] and not pol["preemption"]["enabled"]: e = esc("scenario_change", "orient"); detail = "emergency_appeared"
        elif fb["observed"]["severity"] in SEV_BAD:
            if inp["baseline_qos"]["severity"] in SEV_BAD: e = esc("qos_violation_persists", "decide"); detail = "qos_persists"
            else: e = esc("qos_violation_onset", "decide"); detail = "qos_onset"
    return {"a1_policy": a1, "e2_control": e2, "escalation": e,
            "feedback_to_observe": {"policy_applied": applied, "feedback_trusted": trusted, "watch_metrics": watch(pol)}}, detail

# ------------------------------------------------------------------ adversarial injections (return expected reason)
def corrupt_policy(pol, cons, rng):
    p = copy.deepcopy(pol); kinds = ["prb_sum", "unknown_enum", "missing_field", "bad_id", "compute_sum", "urllc_min", "embb_max", "power", "preemption"]
    k = kinds[rng.integers(len(kinds))]; cap = min(MAXP, cons.get("tx_power_cap_dbm", MAXP))
    if k == "prb_sum": p["prb_priority"]["embb_pct"] = min(100, p["prb_priority"]["embb_pct"] + int(rng.integers(3, 20))) if p["prb_priority"]["embb_pct"] < 80 else p["prb_priority"]["embb_pct"] - 7; exp = "malformed_policy"
    elif k == "unknown_enum": p["interference_coordination"]["strategy"] = str(rng.choice(["quantum_beam", "full_duplex_magic", "NONE"])); exp = "malformed_policy"
    elif k == "missing_field": del p[str(rng.choice(["power_budget_dbm", "mec_offloading", "interference_coordination", "valid_for_ms"]))]; exp = "malformed_policy"
    elif k == "bad_id": p["policy_id"] = str(rng.choice(["policy 1", "", "P_", "x-1234"])); exp = "malformed_policy"
    elif k == "compute_sum": cs = p["mec_offloading"]["compute_share_pct"]; cs["urllc"] = min(100, cs["urllc"] + int(rng.integers(2, 15))); exp = "malformed_policy"
    elif k == "urllc_min":
        u = int(rng.integers(5, max(6, cons["urllc_min_prb_share_pct"]))); d = p["prb_priority"]["urllc_pct"] - u
        if d <= 0: return corrupt_policy(pol, cons, rng)
        p["prb_priority"]["urllc_pct"] = u; p["prb_priority"]["mmtc_pct"] += d; exp = "constraint_violation"
    elif k == "embb_max":
        e = cons["embb_max_prb_share_pct"] + int(rng.integers(1, 15)); d = e - p["prb_priority"]["embb_pct"]
        if d <= 0 or p["prb_priority"]["mmtc_pct"] - d < 0 or e > 100: return corrupt_policy(pol, cons, rng)
        p["prb_priority"]["embb_pct"] = e; p["prb_priority"]["mmtc_pct"] -= d; exp = "constraint_violation"
    elif k == "power":
        cls = str(rng.choice(["urllc", "embb", "mmtc"]))
        if cap < MAXP: p["power_budget_dbm"][cls] = float(cap + int(rng.integers(1, int(MAXP - cap) + 1))); exp = "constraint_violation"
        else: p["power_budget_dbm"][cls] = float(MAXP + int(rng.integers(1, 8))); exp = "malformed_policy"
    else:   # preemption
        if p["preemption"]["enabled"]: p["preemption"] = {"enabled": False, "vehicle_ids": []}
        else: p["preemption"] = {"enabled": True, "vehicle_ids": ["AMBULANCE_99"]}
        exp = "constraint_violation"
    return p, k, exp

def corrupt_feedback(fb, pol, expected_start, rng):
    f = copy.deepcopy(fb); k = ["policy_id", "auth", "stale", "impossible", "inconsistent"][rng.integers(5)]
    if k == "policy_id": f["policy_id"] = "P-" + "".join(rng.choice(list("0123456789abcdef"), 8))
    elif k == "auth": f["auth_ok"] = False
    elif k == "stale": f["window_s"] = [expected_start - 5.0 * int(rng.integers(1, 6)), expected_start - 5.0 * int(rng.integers(0, 1))]
    elif k == "impossible":
        which = rng.integers(3)
        if which == 0: f["observed"]["mean_loss_pct"] = float(rng.choice([-5.0, 120.0, 340.0]))
        elif which == 1: f["enforcer"]["prb_util_pct"] = float(rng.choice([-10.0, 180.0, 400.0]))
        else: f["observed"]["frac_loss_gt_10pct"] = float(rng.choice([1.7, -0.2]))
    else:
        f["observed"]["severity"] = "nominal"
        if rng.random() < 0.5: f["observed"]["mean_loss_pct"] = round(float(rng.uniform(15, 70)), 2)
        else: f["observed"]["mean_delay_ms"] = round(float(rng.uniform(60, 150)), 2)
    return f, k

# ------------------------------------------------------------------ build
def main():
    obs = {}
    for f in sorted(glob.glob("data/observe_S*.jsonl")):
        for l in open(f): r = json.loads(l); obs[r["id"]] = json.loads(r["messages"][2]["content"])
    stats = collections.Counter(); all_recs = []
    mism = collections.Counter()
    for path in sorted(glob.glob("data/decide_S*.jsonl")):
        s = os.path.basename(path).split("_")[1].split(".")[0]; recs = []
        for line in open(path):
            r = json.loads(line); din = json.loads(r["messages"][1]["content"]); pol = json.loads(r["messages"][2]["content"])
            cs = din["cell_state"]; cons = din["context"]["constraint_overrides"]; rng = rng_for("act-" + r["id"])
            ws, we = cs["window_s"]; p = r["obs_id"].split("-"); nxt = obs.get(f"{p[0]}-{p[1]}-{p[2]}-{int(p[3]) + 5}")
            fb = None
            if nxt is not None:
                q = nxt["qos_status"]; ft = C.ACT_FAULT_P; x = rng.random(); enf = {"applied": True, "violations": [], "prb_util_pct": round(float(100 * min(1.0, nxt["resource_state"]["offered_load_ratio"])), 1)}
                if x < ft["applied_false"]: enf["applied"] = False; enf["violations"] = ["rb_allocation_rejected"]
                elif x < ft["applied_false"] + ft["violation_flags"]:
                    enf["violations"] = sorted(str(v) for v in rng.choice(["urllc_prr_below_target", "interference_limit_exceeded", "power_cap_exceeded", "mec_overload"], size=int(rng.integers(1, 3)), replace=False))
                fb = {"policy_id": pol["policy_id"], "window_s": nxt["window_s"], "auth_ok": True, "enforcer": enf,
                      "observed": {"severity": q["severity"], "mean_loss_pct": q["mean_loss_pct"], "mean_delay_ms": q["mean_delay_ms"], "frac_loss_gt_10pct": q["frac_loss_gt_10pct"]},
                      "emergency_event_observed": bool(nxt["special_events"])}
            inp = {"cell_id": cs["cell_id"], "policy": pol,
                   "constraints": {**{k: cons[k] for k in ("urllc_min_prb_share_pct", "embb_max_prb_share_pct") }, **({"tx_power_cap_dbm": cons["tx_power_cap_dbm"]} if "tx_power_cap_dbm" in cons else {}),
                                   "preempt_vehicle_ids": cons.get("preempt_vehicle_ids", []), "max_vehicle_power_dbm": MAXP},
                   "baseline_qos": {"severity": cs["qos"]["severity"], "mean_loss_pct": cs["qos"]["mean_loss_pct"]},
                   "expected_window_start_s": float(we), "feedback": fb}
            out, detail = act_teacher(inp)
            if out["escalation"]["reason"] in ("malformed_policy", "constraint_violation") and detail in ("dsl", "prb_sum", "compute_sum", "urllc_min", "embb_max", "power_cap", "preemption"):
                stats["FALSE_POSITIVE_CLEAN_POLICY_REJECTED"] += 1
            if out["escalation"]["reason"] == "spoofed_feedback": stats["FALSE_POSITIVE_CLEAN_FEEDBACK_SPOOFED"] += 1
            recs.append((r, inp, out, "clean", detail)); stats["clean"] += 1; stats["clean:" + out["escalation"]["reason"]] += 1
        # adversarial parents: drawn deterministically from this situation's base samples
        idx = list(range(len(recs))); rng = rng_for("act-adv-" + s); rng.shuffle(idx)
        recs_adv = []
        want_m = round(C.ACT_ADVERSARIAL["malformed"] * len(recs) / 7031); want_s = round(C.ACT_ADVERSARIAL["spoofed"] * len(recs) / 7031)
        nm = ns = 0
        for i in idx:
            r, inp, out, kind, detail = recs[i]; rr = rng_for("act-adv-i-" + r["id"])
            if nm < want_m:
                pol2, k, exp = corrupt_policy(inp["policy"], inp["constraints"], rr)
                inp2 = {**inp, "policy": pol2}; out2, det2 = act_teacher(inp2); nm += 1
                if out2["escalation"]["reason"] != exp: mism["malformed:" + k] += 1; nm -= 1; continue
                recs_adv.append((r, inp2, out2, "adv_malformed:" + k, det2)); stats["adv_malformed:" + k] += 1
            elif ns < want_s and inp["feedback"] is not None and inp["policy"]["policy_id"]:
                fb2, k = corrupt_feedback(inp["feedback"], inp["policy"], inp["expected_window_start_s"], rr)
                inp2 = {**inp, "feedback": fb2}; out2, det2 = act_teacher(inp2); ns += 1
                if out2["escalation"]["reason"] != "spoofed_feedback": mism["spoofed:" + k] += 1; ns -= 1; continue
                recs_adv.append((r, inp2, out2, "adv_spoofed:" + k, det2)); stats["adv_spoofed:" + k] += 1
            if nm >= want_m and ns >= want_s: break
        with open(f"data/act_{s}.jsonl", "w") as f:
            for j, (r, inp, out, kind, detail) in enumerate(recs + recs_adv):
                rec = {"id": r["id"][:-2] + ("-a" if kind == "clean" else f"-a-{kind.split(':')[0]}-{j}"), "agent": "act", "teacher": "rule", "kind": kind, "detail": detail,
                       "situation": r["situation"], "split": r["split"], "obs_id": r["obs_id"], "decide_id": r["id"],
                       "messages": [{"role": "system", "content": C.ACT_SYSTEM_PROMPT},
                                    {"role": "user", "content": json.dumps(inp, separators=(",", ":"))},
                                    {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))}]}
                f.write(json.dumps(rec) + "\n")
        print(f"{s}: {len(recs)} clean + {len(recs_adv)} adversarial")
    print("detection mismatches (injection NOT detected as intended; resampled):", dict(mism) or "none")
    for k in sorted(stats): print(f"  {k}: {stats[k]}")

if __name__ == "__main__":
    main()
