"""How well does the deterministic guard alone separate hard cases? Checks every Act record (all splits; test split reported separately).
Guard is rule-based and the labels come from the rule teacher, so agreement is by construction; the point is that it is model-independent."""
import json, glob, collections
import act_guard as G
rows = [json.loads(l) for f in sorted(glob.glob('data/act_S?.jsonl')) for l in open(f, encoding='utf-8')]
res = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    inp = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content'])
    reason, target = G.verdict(inp); kind = r['kind'].split(':')[0]; g = gold['escalation']
    hard_gold = r['kind'] != 'clean'      # clean windows may carry enforcer-reported violations: those are deferred to the Act model
    for tag in ('all', r['split']):
        c = res[(tag, kind)]; c['n'] += 1
        c['guard_fires'] += reason is not None
        c['reason_matches_gold'] += (reason == g['reason']) if hard_gold else (reason is None)
        c['final_exact'] += G.apply(gold, inp) == gold            # applying the guard to a correct model output must not change it
for (tag, kind), c in sorted(res.items()):
    print(f"{tag:6s} {kind:14s} n={c['n']:5d}  guard fires {c['guard_fires']/c['n']:6.1%}  guard decision == gold {c['reason_matches_gold']/c['n']:6.1%}  guard leaves a correct output unchanged {c['final_exact']/c['n']:6.1%}")
