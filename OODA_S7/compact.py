"""Compact agent output formats + lossless expanders (the latency fix for Orient and Decide).

The LLM writes only the DECISION values (about 25-35 tokens instead of 140-160); everything that is a deterministic consequence of those values and of the
agent's own input (valid_for_ms from the scenario, priority order from the shares, preemption from the constraints, confidence / RAG refs / rationale text from
the observe summary and retrieved snippets, MEC priority class from the compute shares, policy_id from the glue ...) is filled in by code.
decode(encode(x)) == x is verified on every record by  python compact.py
Formats   ORIENT  base|0.4,0.45,0.15|20,70,500|oq+tf|-|-[|pre:veh12][|cap:20]      tag | qos weights u,e,m | urllc_min,embb_max,max_latency | intents | conflict | suppressed
          DECIDE  45,40,15|10,10,10|n|90,5,5                                      prb u,e,m | power dBm u,e,m | strategy | mec compute share u,e,m"""
import json, re, glob, sys, collections
import config as C
import build_orient as BO

TAG = {'baseline': 'base', 'intersection_dense': 'dense', 'emergency_preempt': 'emerg'}
INT = {'throughput_fairness': 'tf', 'safety_first': 'sf', 'regulatory_compliance': 'rc', 'oem_qos_contract': 'oq', 'energy_saving': 'es', 'emergency_priority': 'ep'}
CONF = {'none': '-', 'energy_vs_reliability': 'er', 'safety_vs_throughput': 'st', 'emergency_vs_oem_qos': 'eo', 'regulation_vs_operator_slo': 'ro'}
STRAT = {'none': 'n', 'power_backoff_edge': 'p', 'sidelink_mode1_scheduling': 's', 'frequency_reuse': 'f'}
VALID_MS = {'baseline': 2500, 'intersection_dense': 1000, 'emergency_preempt': 250}
CLS = ('urllc', 'embb', 'mmtc')
inv = lambda d: {v: k for k, v in d.items()}
def _num(x): return str(int(x)) if float(x) == int(x) else repr(float(x))

# ----------------------------------------------------------------------------- ORIENT
def encode_orient(o):
    c = o['constraint_overrides']
    s = '|'.join([TAG[o['scenario_tag']], ','.join(_num(o['qos_weights'][k]) for k in CLS),
                  f"{c['urllc_min_prb_share_pct']},{c['embb_max_prb_share_pct']},{c['max_policy_latency_ms']}",
                  '+'.join(INT[i] for i in o['active_intents']) or '-', CONF[o['intent_resolution']['conflict_type']],
                  '+'.join(INT[i] for i in o['intent_resolution']['suppressed_intents']) or '-'])
    if 'preempt_vehicle_ids' in c: s += '|pre:' + '+'.join(c['preempt_vehicle_ids'])
    if 'tx_power_cap_dbm' in c: s += '|cap:' + _num(c['tx_power_cap_dbm'])
    return s

def decode_orient(s, inp):
    """inp = the Orient agent's input dict (observe_summary, intent_registry, retrieved_snippets)."""
    f = s.strip().split('|')
    tag = inv(TAG)[f[0]]; w = [float(x) for x in f[1].split(',')]; um, em, lat = [int(x) for x in f[2].split(',')]
    active = [inv(INT)[x] for x in f[3].split('+')] if f[3] != '-' else []
    conflict = inv(CONF)[f[4]]; supp = [inv(INT)[x] for x in f[5].split('+')] if f[5] != '-' else []
    ov = {'urllc_min_prb_share_pct': um, 'embb_max_prb_share_pct': em, 'max_policy_latency_ms': lat}
    for x in f[6:]:
        if x.startswith('pre:'): ov['preempt_vehicle_ids'] = x[4:].split('+')
        elif x.startswith('cap:'): ov['tx_power_cap_dbm'] = float(x[4:]) if '.' in x else int(x[4:])
        else: raise ValueError('bad orient field ' + x)
    o = inp['observe_summary']; reg = inp['intent_registry']['regulatory_context']
    shown = {sn['ref'] for sn in inp['retrieved_snippets']}
    refs = [r for r in BO.rag_refs(tag, reg, conflict) if r in shown]
    return {'scenario_tag': tag, 'confidence': BO.confidence(tag, o, conflict), 'qos_weights': dict(zip(CLS, w)), 'constraint_overrides': ov,
            'active_intents': active, 'intent_resolution': {'conflict_type': conflict, 'suppressed_intents': supp}, 'rag_refs': refs,
            'rationale': BO.rationale(tag, o, reg, conflict, supp)}

# ----------------------------------------------------------------------------- DECIDE
def encode_decide(p):
    pr, pw, m = p['prb_priority'], p['power_budget_dbm'], p['mec_offloading']['compute_share_pct']
    return '|'.join([f"{pr['urllc_pct']},{pr['embb_pct']},{pr['mmtc_pct']}", ','.join(_num(pw[k]) for k in CLS), STRAT[p['interference_coordination']['strategy']],
                     ','.join(str(m[k]) for k in CLS)])

def decode_decide(s, ctx, policy_id='P-00000000'):
    """ctx = the Orient ContextObject (scenario_tag, constraint_overrides)."""
    a, b, st, c = s.strip().split('|')
    u, e, m = [int(x) for x in a.split(',')]; pw = [float(x) for x in b.split(',')]; cs = [int(x) for x in c.split(',')]
    share = dict(zip(CLS, (u, e, m))); ids = ctx['constraint_overrides'].get('preempt_vehicle_ids', [])
    cshare = dict(zip(CLS, cs))
    return {'policy_id': policy_id, 'valid_for_ms': VALID_MS[ctx['scenario_tag']],
            'prb_priority': {'urllc_pct': u, 'embb_pct': e, 'mmtc_pct': m, 'priority_order': sorted(CLS, key=lambda k: -share[k])},
            'power_budget_dbm': dict(zip(CLS, pw)),
            'mec_offloading': {'mode': 'full', 'offload_ratio': 1.0, 'priority_class': max(cshare, key=cshare.get), 'compute_share_pct': cshare},
            'interference_coordination': {'strategy': inv(STRAT)[st], 'reuse_factor': 1},
            'preemption': {'enabled': bool(ids), 'vehicle_ids': ids}}

# ----------------------------------------------------------------------------- system prompts for the compact agents
ORIENT_COMPACT_SYSTEM = ("You are the ORIENT agent of a C-V2X radio resource management system. Given the Observe summary, the intent registry and retrieved standard snippets, "
    "classify the scenario and resolve the intents. Return ONLY one line:  tag|wu,we,wm|urllc_min,embb_max,max_latency_ms|intents|conflict|suppressed[|pre:ids][|cap:dBm]\n"
    "tag: base, dense or emerg. wu,we,wm: QoS weights of URLLC, eMBB, mMTC (sum 1). intents/suppressed: codes joined by +, '-' if none "
    "(tf throughput_fairness, sf safety_first, rc regulatory_compliance, oq oem_qos_contract, es energy_saving, ep emergency_priority). "
    "conflict: '-' none, er energy_vs_reliability, st safety_vs_throughput, eo emergency_vs_oem_qos, ro regulation_vs_operator_slo. "
    "pre: vehicle ids to preempt joined by +. cap: transmit-power cap in dBm.")
DECIDE_COMPACT_SYSTEM = ("You are the DECIDE agent of a C-V2X radio resource management system. Input: the Orient ContextObject, the state of ONE RSU cell (with its co-located MEC server), "
    "the cell's current allocation snapshot and its system model. Return ONLY one line:  u,e,m|pu,pe,pm|strategy|cu,ce,cm\n"
    "u,e,m: integer PRB shares of URLLC, eMBB, mMTC in % (sum 100, honouring constraint_overrides). pu,pe,pm: maximum vehicle power per class in dBm (never above max_vehicle_power_dbm or tx_power_cap_dbm). "
    "strategy: n none, p power_backoff_edge, s sidelink_mode1_scheduling, f frequency_reuse. cu,ce,cm: integer MEC compute shares in % (sum 100). "
    "The mean transmit power across PRBs must respect the interference limit implied by cell load and strategy.")

if __name__ == '__main__':
    bad = collections.Counter(); n = collections.Counter(); toks = collections.defaultdict(list)
    for f in sorted(glob.glob('data/orient_S?.jsonl')):
        for l in open(f):
            r = json.loads(l); inp = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content']); n['orient'] += 1
            s = encode_orient(gold); toks['orient'].append((len(r['messages'][2]['content']), len(s)))
            try: bad['orient'] += decode_orient(s, inp) != gold
            except Exception as e: bad['orient'] += 1; print('orient err', e)
    for f in sorted(glob.glob('data/decide_S?.jsonl')):
        for l in open(f):
            r = json.loads(l); u = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content']); n['decide'] += 1
            s = encode_decide(gold); toks['decide'].append((len(r['messages'][2]['content']), len(s)))
            try: bad['decide'] += decode_decide(s, u['context'], gold['policy_id']) != gold
            except Exception as e: bad['decide'] += 1; print('decide err', e)
    for k in n:
        a = sum(x for x, _ in toks[k]) / n[k]; b = sum(y for _, y in toks[k]) / n[k]
        print(f'{k}: {n[k]} records, round-trip mismatches {bad[k]}; output chars {a:.0f} -> {b:.0f} ({a/b:.1f}x shorter); example: {s}')
