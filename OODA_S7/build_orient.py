"""Build Orient-DS: (Observe summary + intent registry + retrieved snippets) -> ContextObject.

The teacher is a deterministic rule set (documented below). Each Observe sample yields 1-3 samples with different
sampled intent contexts (operator mode, OEM contract, regulatory context), more for rare scenarios.
Splits are inherited from the Observe sample so no time block leaks across agents.

Usage: python build_orient.py            (reads data/observe_S*.jsonl, writes data/orient_S*.jsonl)
"""
import json, glob, hashlib, os
import numpy as np
import config as C

SEV = {"nominal": 0, "elevated": 1, "degraded": 2, "critical": 3}

def rng_for(key):
    return np.random.default_rng(int(hashlib.md5(f"{C.SEED}-{key}".encode()).hexdigest()[:12], 16))

# ---------------------------------------------------------------- scenario rules
def classify(o):
    if o["special_events"]:                                  # ambulance in scope or cars inside its 300 m alert zone
        return "emergency_preempt"
    enough = o["n_active_vehicles"] >= C.ORIENT_MIN_N        # small-sample guard
    if enough and (SEV[o["qos_status"]["severity"]] >= 2 or "standstill_cluster" in o["anomalies"]):
        return "intersection_dense"
    if o["load_level"] in ("high", "very_high"):
        return "intersection_dense"
    return "baseline"

# ---------------------------------------------------------------- intent context sampling
def sample_context(rng, tag):
    if tag == "emergency_preempt":
        reg = rng.choice(["none", "tmc_emergency_corridor", "regional_power_cap", "safety_zone_mandate"], p=[.45, .30, .10, .15])
    else:
        reg = rng.choice(["none", "regional_power_cap", "safety_zone_mandate"], p=[.70, .15, .15])
    mode = "energy_saving" if rng.random() < 0.25 else "normal"
    oem = {"name": "OEM-A", "min_embb_mbps": int(rng.choice([2, 5, 10]))} if rng.random() < 0.30 else None
    return str(reg), mode, oem

# ---------------------------------------------------------------- intent resolution
BASE_INTENTS = {"baseline": ["throughput_fairness"], "intersection_dense": ["safety_first"],
                "emergency_preempt": ["safety_first", "emergency_priority"]}

def resolve(tag, sev, reg, mode, oem):
    intents = list(BASE_INTENTS[tag])
    if reg != "none": intents.append("regulatory_compliance")
    if oem: intents.append("oem_qos_contract")
    if mode == "energy_saving": intents.append("energy_saving")
    suppressed, conflicts = set(), []
    if tag == "emergency_preempt" and oem:
        suppressed.add("oem_qos_contract"); conflicts.append("emergency_vs_oem_qos")
    if tag == "intersection_dense" and oem and SEV[sev] >= 2:
        suppressed.add("oem_qos_contract"); conflicts.append("safety_vs_throughput")
    if tag != "baseline" and mode == "energy_saving":
        suppressed.add("energy_saving"); conflicts.append("energy_vs_reliability")
    if tag != "baseline" and reg == "regional_power_cap":
        conflicts.append("regulation_vs_operator_slo")        # regulation wins: power cap applied, nothing dropped
    active = [i for i in C.INTENT_PRIORITY if i in intents and i not in suppressed]
    return active, sorted(suppressed), (conflicts[0] if conflicts else "none")

# ---------------------------------------------------------------- QoS weights & constraints
BASE_W = {"baseline": (0.40, 0.40, 0.20), "intersection_dense": (0.55, 0.30, 0.15), "emergency_preempt": (0.75, 0.15, 0.10)}
SEV_URLLC = {"nominal": 0.0, "elevated": 0.03, "degraded": 0.06, "critical": 0.10}

def weights(tag, sev, active, reg):
    u, e, m = BASE_W[tag]
    u += SEV_URLLC[sev]; e -= SEV_URLLC[sev]
    if "energy_saving" in active: u += .05; e -= .05
    if "oem_qos_contract" in active: e += .05; m -= .05
    if reg == "safety_zone_mandate": u += .05; e -= .05
    w = {k: max(0.05, v) for k, v in zip(("urllc", "embb", "mmtc"), (u, e, m))}
    s = sum(w.values()); w = {k: round(v / s, 2) for k, v in w.items()}
    w["urllc"] = round(1.0 - w["embb"] - w["mmtc"], 2)       # exact sum
    return w

def overrides(tag, sev, reg, ev):
    if tag == "baseline":   c = {"urllc_min_prb_share_pct": 20, "embb_max_prb_share_pct": 70, "max_policy_latency_ms": 500}
    elif tag == "intersection_dense":
        crit = sev == "critical"
        c = {"urllc_min_prb_share_pct": 35 if crit else 30, "embb_max_prb_share_pct": 40 if crit else 50, "max_policy_latency_ms": 200}
    else:
        c = {"urllc_min_prb_share_pct": 40, "embb_max_prb_share_pct": 30, "max_policy_latency_ms": 50,
             "preempt_vehicle_ids": [ev[0]["vehicle_id"]]}
        if reg == "tmc_emergency_corridor": c.update(urllc_min_prb_share_pct=50, embb_max_prb_share_pct=25)
    if reg == "safety_zone_mandate": c["urllc_min_prb_share_pct"] = min(100, c["urllc_min_prb_share_pct"] + 5)
    if reg == "regional_power_cap": c["tx_power_cap_dbm"] = 20
    return c

def confidence(tag, o, conflict):
    sev = o["qos_status"]["severity"]
    if tag == "emergency_preempt":
        c = 0.97 if o["special_events"][0]["in_scope"] else 0.88
    elif tag == "intersection_dense":
        c = 0.92 if SEV[sev] >= 2 else 0.80                  # dense by load only = borderline
    else:
        c = 0.95 if sev == "nominal" else 0.72               # baseline despite elevated QoS = borderline
    if conflict != "none": c -= 0.05
    return round(c, 2)

# ---------------------------------------------------------------- RAG
def rag_refs(tag, reg, conflict):
    refs = list(C.RAG_RELEVANT[tag])
    if reg in ("tmc_emergency_corridor", "safety_zone_mandate", "regional_power_cap"): refs.append("TMC-REG")
    if conflict != "none" and "TS28.312" not in refs: refs.append("TS28.312")
    return refs

def rationale(tag, o, reg, conflict, suppressed):
    q = o["qos_status"]
    if tag == "emergency_preempt":
        e = o["special_events"][0]
        t = (f"Emergency vehicle {e['vehicle_id']} {'in this scope' if e['in_scope'] else 'nearby'} with "
             f"{e['n_vehicles_within_300m']} vehicles in its 300 m alert zone (QoS {q['severity']}): preempt for right-of-way.")
    elif tag == "intersection_dense":
        t = f"Load {o['load_level']}, QoS {q['severity']} (mean loss {q['mean_loss_pct']}%, {o['n_active_vehicles']} vehicles): dense intersection, protect URLLC."
    else:
        t = f"Load {o['load_level']}, QoS {q['severity']}, no emergency event: baseline operation."
    if conflict != "none":
        t += f" Conflict {conflict}" + (f"; suppressed {', '.join(suppressed)} by priority." if suppressed else "; regulation takes precedence.")
    if reg != "none": t += f" Regulatory context {reg} applied."
    return t[:400]

# ---------------------------------------------------------------- main
def build_one(obs_rec, k):
    o = json.loads(obs_rec["messages"][2]["content"])
    tag = classify(o)
    rng = rng_for(f"{obs_rec['id']}-{k}")
    reg, mode, oem = sample_context(rng, tag)
    sev = o["qos_status"]["severity"]
    active, suppressed, conflict = resolve(tag, sev, reg, mode, oem)
    refs = rag_refs(tag, reg, conflict)
    # noisy retriever: each relevant snippet is retrieved with p=0.8 (>=1 kept) plus 2 random distractors, so the
    # snippets alone do not reveal the scenario tag; rag_refs = relevant snippets that were actually retrieved
    kept = [r for r in refs if rng.random() < 0.8] or [refs[0]]
    pool = [r for r in C.RAG_KB if r not in refs]
    shown = kept + [str(x) for x in rng.choice(pool, size=2, replace=False)]
    rng.shuffle(shown)
    refs = [r for r in refs if r in kept]
    inp = {"observe_summary": o,
           "intent_registry": {"operator_slos": {"max_loss_pct": C.LOSS_VIOLATION_PCT, "max_delay_ms": C.DELAY_VIOLATION_MS},
                               "operator_mode": mode, "oem_contract": oem, "regulatory_context": reg},
           "retrieved_snippets": [{"ref": r, "text": C.RAG_KB[r]} for r in shown]}
    out = {"scenario_tag": tag, "confidence": confidence(tag, o, conflict), "qos_weights": weights(tag, sev, active, reg),
           "constraint_overrides": overrides(tag, sev, reg, o["special_events"]), "active_intents": active,
           "intent_resolution": {"conflict_type": conflict, "suppressed_intents": suppressed},
           "rag_refs": refs, "rationale": rationale(tag, o, reg, conflict, suppressed)}
    return {"id": f"{obs_rec['id']}-v{k}", "agent": "orient", "situation": obs_rec["situation"], "split": obs_rec["split"],
            "obs_id": obs_rec["id"], "scenario_tag": tag,
            "messages": [{"role": "system", "content": C.ORIENT_SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(inp, separators=(",", ":"))},
                         {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))}]}

def main():
    for path in sorted(glob.glob("data/observe_S*.jsonl")):
        s = os.path.basename(path).split("_")[1].split(".")[0]
        n = 0
        with open(f"data/orient_{s}.jsonl", "w") as f:
            for line in open(path):
                rec = json.loads(line)
                summ = json.loads(rec["messages"][2]["content"])
                if summ["scope"]["type"] != "cell": continue          # OODA runs per RSU cell; network samples have no Orient/Decide counterpart
                tag = classify(summ)
                for k in range(C.ORIENT_VARIANTS[tag]):
                    f.write(json.dumps(build_one(rec, k)) + "\n"); n += 1
        print(f"{s}: {n} orient samples")

if __name__ == "__main__":
    main()
