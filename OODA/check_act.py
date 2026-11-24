"""Independent audit of Act-DS. Re-implements the Act rules from the SYSTEM PROMPT (not from build_act.py) and checks every
record against its Decide / Orient / Observe parents, plus shortcut / leakage diagnostics. Exit 1 on any hard failure."""
import json, glob, re, sys, collections
import jsonschema
import config as C

DSL = jsonschema.Draft202012Validator(json.load(open("schemas/decide_policy.schema.json")))
ACT = jsonschema.Draft202012Validator(json.load(open("schemas/act_command.schema.json")))
MAXP = C.SYSTEM_MODEL["max_vehicle_power_dbm"]
load = lambda pat: [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
obs = {r["id"]: json.loads(r["messages"][2]["content"]) for r in load("data/observe_S*.jsonl")}
dec = {r["id"]: r for r in load("data/decide_S[0-9].jsonl")}
act = load("data/act_S*.jsonl")
fails = collections.Counter(); hard = collections.Counter()
def chk(name, ok, hardfail=True):
    if not ok: (hard if hardfail else fails)[name] += 1

def expected(inp):
    """Independent rules, written from the prompt: reject > spoof > failure/violation > new emergency > QoS."""
    pol, cons, fb = inp["policy"], inp["constraints"], inp["feedback"]
    # 1) type-check against DSL + constraints
    if list(DSL.iter_errors(pol)): return "malformed_policy", "decide"
    p = pol["prb_priority"]
    if p["urllc_pct"] + p["embb_pct"] + p["mmtc_pct"] != 100 or sum(pol["mec_offloading"]["compute_share_pct"].values()) != 100: return "malformed_policy", "decide"
    cap = min(cons["max_vehicle_power_dbm"], cons.get("tx_power_cap_dbm", 1e9))
    if (p["urllc_pct"] < cons["urllc_min_prb_share_pct"] or p["embb_pct"] > cons["embb_max_prb_share_pct"]
        or max(pol["power_budget_dbm"].values()) > cap
        or pol["preemption"]["vehicle_ids"] != cons["preempt_vehicle_ids"] or pol["preemption"]["enabled"] != bool(cons["preempt_vehicle_ids"])):
        return "constraint_violation", "decide"
    if fb is None: return "none", "none"
    # 3) spoofed feedback
    o, e = fb["observed"], fb["enforcer"]
    if (fb["policy_id"] != pol["policy_id"] or not fb["auth_ok"] or fb["window_s"][0] != inp["expected_window_start_s"]
        or not 0 <= o["mean_loss_pct"] <= 100 or not 0 <= o["frac_loss_gt_10pct"] <= 1 or not 0 <= e["prb_util_pct"] <= 100 or o["mean_delay_ms"] < 0
        or (o["severity"] == "nominal" and (o["mean_loss_pct"] > C.ACT_NOMINAL_LOSS_MAX or o["mean_delay_ms"] > C.ACT_NOMINAL_DELAY_MAX))):
        return "spoofed_feedback", "decide"
    # 4) genuine feedback
    if not e["applied"]: return "policy_execution_failure", "decide"
    if e["violations"]: return "constraint_violation", "decide"
    if fb["emergency_event_observed"] and not pol["preemption"]["enabled"]: return "scenario_change", "orient"
    if o["severity"] in ("degraded", "critical"):
        return ("qos_violation_persists" if inp["baseline_qos"]["severity"] in ("degraded", "critical") else "qos_violation_onset"), "decide"
    return "none", "none"

def e2_faithful(pol, out):
    p, pw, m, ic, pr = pol["prb_priority"], pol["power_budget_dbm"], pol["mec_offloading"], pol["interference_coordination"], pol["preemption"]
    by = {x["ran_function"]: x["params"] for x in out["e2_control"]}
    ok = by.get("RC-PRB-QUOTA") == {"urllc_pct": p["urllc_pct"], "embb_pct": p["embb_pct"], "mmtc_pct": p["mmtc_pct"]}
    ok &= by.get("RC-TX-POWER") == {"urllc_dbm": pw["urllc"], "embb_dbm": pw["embb"], "mmtc_dbm": pw["mmtc"]}
    ok &= ("RC-MEC-OFFLOAD" in by) == (m["mode"] != "none") and ("RC-INTERFERENCE" in by) == (ic["strategy"] != "none") and ("RC-PREEMPT" in by) == pr["enabled"]
    if pr["enabled"]: ok &= by["RC-PREEMPT"]["vehicle_ids"] == pr["vehicle_ids"]
    a = out["a1_policy"]; ok &= a is not None and a["policy_instance_id"] == pol["policy_id"] and a["statement"]["valid_for_ms"] == pol["valid_for_ms"]
    return ok

cnt = collections.Counter(); kinds = collections.Counter(); null_fb = collections.Counter(); reasons_split = collections.defaultdict(collections.Counter)
for r in act:
    inp = json.loads(r["messages"][1]["content"]); out = json.loads(r["messages"][2]["content"]); d = dec[r["decide_id"]]
    din = json.loads(d["messages"][1]["content"]); dpol = json.loads(d["messages"][2]["content"]); cs = din["cell_state"]
    kind = r["kind"].split(":")[0]; kinds[kind] += 1
    chk("schema", not list(ACT.iter_errors(out)))
    reason, target = expected(inp)
    chk("label_matches_independent_rules", (out["escalation"]["reason"], out["escalation"]["target"]) == (reason, target))
    chk("escalate_flag_consistent", out["escalation"]["escalate"] == (reason != "none"))
    rejected = reason in ("malformed_policy", "constraint_violation") and out["a1_policy"] is None
    if reason in ("malformed_policy",) or (reason == "constraint_violation" and out["a1_policy"] is None):
        chk("rejected_means_no_commands", out["a1_policy"] is None and out["e2_control"] == [] and out["feedback_to_observe"]["policy_applied"] is False)
    else:
        chk("e2_translation_faithful", e2_faithful(inp["policy"], out))
    chk("spoofed_feedback_is_untrusted", (reason == "spoofed_feedback") == (not out["feedback_to_observe"]["feedback_trusted"]) or reason in ("malformed_policy", "constraint_violation"))
    chk("same_split_situation_as_decide", r["split"] == d["split"] and r["situation"] == d["situation"])
    chk("cell_id_matches_decide", inp["cell_id"] == cs["cell_id"])
    chk("baseline_qos_matches_decide", inp["baseline_qos"] == {"severity": cs["qos"]["severity"], "mean_loss_pct": cs["qos"]["mean_loss_pct"]})
    chk("window_matches_decide", inp["expected_window_start_s"] == cs["window_s"][1])
    for k in ("urllc_min_prb_share_pct", "embb_max_prb_share_pct"): chk("constraints_match_orient", inp["constraints"][k] == din["context"]["constraint_overrides"][k])
    if kind == "clean":
        chk("clean_policy_equals_decide_output", inp["policy"] == dpol)
        chk("clean_never_malformed_or_spoofed", reason not in ("malformed_policy", "spoofed_feedback"))
        if inp["feedback"] is not None:          # feedback must be the real next-window Observe summary of the same cell
            p = r["obs_id"].split("-"); nxt = obs.get(f"{p[0]}-{p[1]}-{p[2]}-{int(p[3]) + 5}"); fo = inp["feedback"]["observed"]
            chk("feedback_equals_next_window_observe", nxt is not None and fo["severity"] == nxt["qos_status"]["severity"] and fo["mean_loss_pct"] == nxt["qos_status"]["mean_loss_pct"]
                and inp["feedback"]["window_s"] == nxt["window_s"])
    else:
        chk("adversarial_detected_as_intended", reason == ("spoofed_feedback" if "spoof" in kind else reason) and reason in ("malformed_policy", "constraint_violation", "spoofed_feedback"))
    null_fb[(kind, inp["feedback"] is None)] += 1
    reasons_split[r["split"]][out["escalation"]["reason"]] += 1
    txt = r["messages"][1]["content"]          # user turn only (the system prompt legitimately names the reasons)
    chk("no_label_metadata_in_input", not re.search(r"adv_|spoof|malformed|\"kind\"|\"detail\"", txt))
    chk("no_real_cell_or_situation_id", not re.search(r"\bC\d\d\b|\bS[1-4]\b", txt + r["messages"][2]["content"]))

# ---- threshold sanity: nominal-severity real-data maxima behind ACT_NOMINAL_*
nom = [o for o in obs.values() if o["scope"]["type"] == "cell" and o["qos_status"]["severity"] == "nominal"]
mx_l = max(o["qos_status"]["mean_loss_pct"] for o in nom); mx_d = max(o["qos_status"]["mean_delay_ms"] for o in nom)
print(f"nominal-severity cell windows: max mean loss {mx_l:.2f}% (threshold {C.ACT_NOMINAL_LOSS_MAX}), max mean delay {mx_d:.2f} ms (threshold {C.ACT_NOMINAL_DELAY_MAX})")
chk("nominal_thresholds_above_real_maxima", mx_l < C.ACT_NOMINAL_LOSS_MAX and mx_d < C.ACT_NOMINAL_DELAY_MAX)
print("records:", len(act), dict(kinds))
print("feedback==null by kind:", {f"{k}": f"{v}/{sum(c for (kk, _), c in null_fb.items() if kk == k)}" for (k, n), v in null_fb.items() if n})
print("reason mix per split:", {s: dict(c) for s, c in reasons_split.items()})
if fails: print("soft warnings:", dict(fails))
if hard: print("HARD FAILURES:", dict(hard)); sys.exit(1)
print("ALL ACT CHECKS PASSED")
