"""Cross-stage consistency audit: Observe -> Orient -> Decide must tell one story. Exit code 1 on any failure."""
import json, glob, re, sys, collections
import system_model as SM

obs = {}
for f in sorted(glob.glob("data/observe_S*.jsonl")):
    for l in open(f): r = json.loads(l); obs[r["id"]] = r
ori = {r["obs_id"] + "|" + r["id"]: r for f in sorted(glob.glob("data/orient_S*.jsonl")) for r in map(json.loads, open(f))}
dec = [json.loads(l) for f in sorted(glob.glob("data/decide_S*.jsonl")) for l in open(f)]
ori_list = list(ori.values())
fails = collections.Counter(); notes = []
def chk(name, ok):
    if not ok: fails[name] += 1

for r in ori_list:                                   # Orient vs its Observe parent
    o = obs[r["obs_id"]]; so = json.loads(o["messages"][2]["content"]); io = json.loads(r["messages"][1]["content"])["observe_summary"]
    chk("orient_embeds_exact_observe_output", so == io)
    chk("orient_only_cell_scope", so["scope"]["type"] == "cell")
    chk("orient_same_split_as_observe", r["split"] == o["split"])
    chk("orient_same_situation", r["situation"] == o["situation"])
    ctx = json.loads(r["messages"][2]["content"])
    chk("emergency_tag_iff_event", (ctx["scenario_tag"] == "emergency_preempt") == bool(so["special_events"]))
    chk("preempt_ids_match_event", ctx["constraint_overrides"].get("preempt_vehicle_ids", []) == [e["vehicle_id"] for e in so["special_events"]])

by_ori = {r["id"]: r for r in ori_list}
for r in dec:                                        # Decide vs its Orient parent and Observe grandparent
    p = by_ori[r["id"][:-2]]; din = json.loads(r["messages"][1]["content"]); pol = json.loads(r["messages"][2]["content"])
    io = json.loads(p["messages"][1]["content"])["observe_summary"]; ctx = json.loads(p["messages"][2]["content"])
    chk("decide_context_equals_orient_output", din["context"] == ctx)
    cs = din["cell_state"]
    chk("decide_cell_state_matches_observe", all(cs[k] == io[k] for k in ("window_s", "n_active_vehicles", "load_level", "trend", "special_events", "anomalies"))
        and cs["cell_id"] == io["scope"]["cell_id"] and cs["v2x_messages"] == io["v2x_messages"])
    chk("decide_resource_state_from_observe", din["current_allocation"] == io["resource_state"]["current_allocation"]
        and all(cs["resource_state"][k] == io["resource_state"][k] for k in ("estimated", "offered_load_ratio", "interference_iot_db", "mec_load_ratio")))
    chk("decide_same_split_situation", r["split"] == p["split"] and r["situation"] == p["situation"])
    chk("decide_preempt_follows_event", pol["preemption"]["vehicle_ids"] == [e["vehicle_id"] for e in io["special_events"]])
for o in obs.values():                               # Observe: resource_state must be reproducible from the shared model; schema scope rules
    so = json.loads(o["messages"][2]["content"]); si = json.loads(o["messages"][1]["content"])
    if so["scope"]["type"] == "cell":
        chk("observe_resource_state_reproducible", so["resource_state"] == SM.resource_state({"n_active_vehicles": so["n_active_vehicles"], "special_events": so["special_events"]}))
        chk("observe_input_output_resource_state_equal", si["resource_state"] == so["resource_state"])
    else:
        chk("network_scope_has_no_resource_state", so["resource_state"] is None)
    txt = " ".join(m["content"] for m in o["messages"])
    chk("no_real_cell_id_in_text", not re.search(r"\bC\d\d\b", txt))
    chk("no_situation_id_in_text", not re.search(r'"S[1-4]"|\bS[1-4]\b', txt))
    chk("no_legacy_words", not re.search(r"\b(zone_id|global|ALL_ZONES)\b", txt.replace('"zone_id":"ALL"', "")) and "whole area" not in txt)

# terminology + coverage report
txt_o = " ".join(m["content"] for r in list(obs.values())[:500] for m in r["messages"])
txt_r = " ".join(m["content"] for r in ori_list[:500] for m in r["messages"])
txt_d = " ".join(m["content"] for r in dec[:500] for m in r["messages"])
for name, t in (("observe", txt_o), ("orient", txt_r), ("decide", txt_d)):
    notes.append(f"{name:8s} uses: 'zone'={len(re.findall(r'[Zz]one', t))} 'cell'={len(re.findall(r'[Cc]ell', t))} 'cell_id'={t.count('cell_id')}")
print("\n".join(notes))
print(f"samples: observe {len(obs)} (network {sum(1 for k in obs if '-network-' in k)}, cell {sum(1 for k in obs if '-cell-' in k)}), orient {len(ori_list)}, decide {len(dec)}")
print(f"orient parents: {len({r['obs_id'] for r in ori_list})} distinct cell summaries; decide parents: {len({r['obs_id'] for r in dec})}")
if fails:
    print("FAILED:", dict(fails)); sys.exit(1)
print("ALL CONSISTENCY CHECKS PASSED")
