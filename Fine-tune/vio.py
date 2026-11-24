import json, collections
c = collections.Counter(); sev = collections.Counter()
for l in open(r"..\OODA_S7\data\act_S7.jsonl"):
    fb = json.loads(json.loads(l)["messages"][1]["content"])["feedback"]
    if fb:
        c.update(fb["enforcer"]["violations"]); sev[fb["observed"]["severity"]] += 1
print(c); print(sev)
