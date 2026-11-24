#!/usr/bin/env python3
r"""Coverage-complete Decide training data, labelled by the REAL teacher (build_decide.optimize, CPLEX).
  python make_decide_coverage.py --ooda ..\OODA --codebook ..\OODA_cov\codebook2c.json --out ..\OODA_dcov [--max_problems 5000] [--procs 3]
Writes <out>\data_compact2_dcov\ (copy of data_compact2 + decide_S9.jsonl) and <out>\codebook2d.json (new MEC codes APPENDED). Nothing in --ooda is modified."""
import argparse, collections, glob, itertools, json, multiprocessing as mp, os, random, shutil, sys, time
B = None
def _init(ooda, threads=1):
    global B
    sys.path.insert(0, ooda); os.chdir(ooda)
    import build_decide as B_; B = B_
    try:
      if threads > 0:                       # one CPLEX thread per worker: N workers x all cores oversubscribes the CPU
        from docplex.mp import model as _M; _o = _M.Model.__init__
        def _p(self, *x, **k): _o(self, *x, **k); self.parameters.threads = threads
        _M.Model.__init__ = _p
    except Exception as e: print('could not limit CPLEX threads:', e)
def _solve(job):
    prompt, w, ov, obs = job
    try:
        r = B.optimize(w, ov, obs)
        return prompt, (None if r is None else B.to_policy(r, ov, prompt)), None
    except Exception as e: return prompt, None, repr(e)
def parse(s):
    d = dict(x.split('=', 1) for x in s.split())
    w = dict(zip(('urllc', 'embb', 'mmtc'), (float(x) for x in d['w'].split(','))))
    um, em, lat = (int(x) for x in d['c'].split(','))
    ov = {'urllc_min_prb_share_pct': um, 'embb_max_prb_share_pct': em, 'max_policy_latency_ms': lat}
    if d['cap'] != '-': ov['tx_power_cap_dbm'] = float(d['cap']) if '.' in d['cap'] else int(d['cap'])
    if d['pre'] == '1': ov['preempt_vehicle_ids'] = ['AMBULANCE_01']
    ev = [] if d['ev'] == '-' else [{'vehicle_id': 'AMBULANCE_01', 'in_scope': bool(int(d['ev'].split(',')[0])), 'n_vehicles_within_300m': int(d['ev'].split(',')[1])}]
    return d['tag'], w, ov, {'n_active_vehicles': int(d['n']), 'special_events': ev}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ooda', required=True); ap.add_argument('--codebook', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max_problems', type=int, default=5000); ap.add_argument('--procs', type=int, default=3); ap.add_argument('--check', type=int, default=150)
    ap.add_argument('--extrap_n', type=int, default=80, help='all n >= this are held out entirely (extrapolation test)'); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--threads', type=int, default=1, help='CPLEX threads per worker, 0 = do not limit'); ap.add_argument('--min_faith', type=float, default=0.97)
    ap.add_argument('--dense_n', type=int, default=0, help='sample EVERY n in 1..dense_n and EVERY vehicles-within-300m count 0..n (default 0 = the sparse lists)'); a = ap.parse_args()
    ooda, out = os.path.abspath(a.ooda), os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, 'data_compact2_dcov')): sys.exit(f'{out}\\data_compact2_dcov already exists - choose a new --out folder')
    cbp = os.path.abspath(a.codebook); sys.path.insert(0, ooda); os.chdir(ooda)
    import compact as K, compact2 as C2, build_orient as BO, build_decide as BD
    cb = json.load(open(cbp)); rng = random.Random(a.seed)
    pool = mp.Pool(a.procs, _init, (ooda, a.threads))
    # ---- 1. faithfulness: the teacher called from the PROMPT must reproduce the label of existing records
    rec = [json.loads(l) for f in sorted(glob.glob('data_compact2/decide_S?.jsonl')) for l in open(f, encoding='utf-8')]
    rng.shuffle(rec); rec = rec[:a.check]; jobs = []
    for r in rec:
        tag, w, ov, obs = parse(r['messages'][1]['content']); jobs.append((r['messages'][1]['content'], w, ov, obs))
    print(f'checking {len(jobs)} existing records with the teacher ...', flush=True)
    t0 = time.time(); res = pool.map(_solve, jobs); dt = (time.time() - t0) / max(1, len(jobs)) * a.procs
    cbx = dict(cb); cbx['mec'] = list(cb['mec']); ok = bad = fail = 0; diffs = []
    for r, (p, pol, err) in zip(rec, res):
        if pol is None: fail += 1; continue
        try: code = C2.encode_decide2(pol, cbx)
        except ValueError: bad += 1; continue
        same = code == r['messages'][2]['content']; ok += same; bad += not same
        if not same and len(diffs) < 14:
            g = r['messages'][2]['content']; diffs.append((r['messages'][1]['content'], g, code, [n for n, sl in (('PRB', slice(0, 6)), ('power', slice(6, 9)), ('strat', slice(9, 10)), ('MEC', slice(10, 12))) if g[sl] != code[sl]]))
    print(f'faithfulness: teacher reproduces the label of {ok}/{len(rec)} existing records ({bad} differ, {fail} unsolved); ~{dt:.2f} s per solve per process')
    for pr, g, c, f in diffs: print(f'  DIFF fields={f}\n    prompt  : {pr}\n    stored  : {g}\n    teacher : {c}')
    if ok < a.min_faith * len(rec): sys.exit('teacher-from-prompt does NOT reproduce the existing labels - DO NOT TRUST (paste this output)')
    # ---- 2. realistic contexts: enumerate the ORIENT teacher over its whole input grid (same as make_orient_coverage)
    NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 80]; N300 = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 80]
    if a.dense_n: NS = sorted(set(range(1, a.dense_n + 1)) | {x for x in NS if x > a.dense_n})
    LOADS, QOS = ['low', 'moderate', 'high', 'very_high'], ['nominal', 'elevated', 'degraded', 'critical']
    REG = ['none', 'regional_power_cap', 'safety_zone_mandate', 'tmc_emergency_corridor']
    nb = lambda n: 0 if n <= 4 else 1 if n <= 8 else 2 if n <= 12 else 3 if n <= 20 else 4 if n <= 40 else 5
    ctxs = {}                                                       # (tag, w, ov) -> set of n
    for n, load, qos, ss, ev, mode, oem, reg in itertools.product(NS, LOADS, QOS, [0, 1], [0, 1], ['normal', 'energy_saving'], [0, 1], REG):
        evs = [{'vehicle_id': 'AMBULANCE_01', 'in_scope': True, 'n_vehicles_within_300m': 5}] if ev else []
        o = {'special_events': evs, 'n_active_vehicles': n, 'load_level': load, 'qos_status': {'severity': qos, 'mean_loss_pct': 0.0}, 'anomalies': ['standstill_cluster'] if ss else []}
        tag = BO.classify(o); active, supp, conf = BO.resolve(tag, qos, reg, mode, {'name': 'OEM-A', 'min_embb_mbps': 5} if oem else None)
        w = BO.weights(tag, qos, active, reg); ov = BO.overrides(tag, qos, reg, evs)
        ctxs.setdefault((tag, json.dumps(w, sort_keys=True), json.dumps(ov, sort_keys=True)), set()).add(n)
    print(f'orient teacher gives {len(ctxs)} distinct (tag, weights, constraints incl. cap/preempt) contexts; by tag:', dict(collections.Counter(k[0] for k in ctxs)))
    probs = {}                                                      # prompt -> (ctx id, n, job)
    for ci, ((tag, ws, ovs), ns) in enumerate(sorted(ctxs.items())):
        w, ov = json.loads(ws), json.loads(ovs)
        for n in sorted(ns):
            for evs in ([[]] if tag != 'emergency_preempt' else [[{'vehicle_id': 'AMBULANCE_01', 'in_scope': s, 'n_vehicles_within_300m': m}] for s in (False, True) for m in (range(n + 1) if n <= a.dense_n else [x for x in N300 if x <= n])]):
                obs = {'n_active_vehicles': n, 'special_events': evs}
                p = C2.decide_input2({'scenario_tag': tag, 'qos_weights': w, 'constraint_overrides': ov}, obs)
                probs[p] = (ci, tag, n, (p, w, ov, obs))
    strata = collections.defaultdict(list)
    for p, v in probs.items(): strata[(v[1], nb(v[2]))].append(p)
    per = max(1, a.max_problems // len(strata)); pick = []
    for k in sorted(strata): rng.shuffle(strata[k]); pick += strata[k][:per]
    print(f'{len(probs)} distinct prompts in the grid -> {len(pick)} sampled ({per} per tag x n-bucket stratum, {len(strata)} strata); estimated solve time ~{len(pick) * dt / a.procs / 60:.0f} min on {a.procs} process(es)')
    # ---- 3. solve with the real teacher
    t0 = time.time(); solved = {}; errs = collections.Counter(); unsolved = 0
    for i, (p, pol, err) in enumerate(pool.imap_unordered(_solve, [probs[p][3] for p in pick], chunksize=8)):
        if pol is None: unsolved += 1; errs[err or 'infeasible/non-optimal'] += 1
        else: solved[p] = pol
        if (i + 1) % 250 == 0: print(f'  {i + 1}/{len(pick)} solved ({time.time() - t0:.0f} s)', flush=True)
    pool.close()
    print(f'solved {len(solved)}, unsolved {unsolved}', dict(errs.most_common(3)) if errs else '')
    # ---- 4. codes (append new MEC combos), grouped hold-out
    rows = []
    for p, pol in solved.items():
        m = [pol['mec_offloading']['compute_share_pct'][k] for k in K.CLS]
        if m not in cbx['mec']: cbx['mec'].append(m)
        rows.append((p, probs[p][0], probs[p][1], probs[p][2], C2.encode_decide2(pol, cbx)))
    assert len(cbx['mec']) < 100, 'MEC codebook would need 3 digits'
    gk = lambda r: (r[1], nb(r[3])); keys = sorted({gk(r) for r in rows}); random.Random(a.seed + 1).shuffle(keys)
    test_k, val_k = set(keys[:int(.14 * len(keys))]), set(keys[int(.14 * len(keys)):int(.20 * len(keys))])
    def split(r): return 'test' if (gk(r) in test_k or r[3] >= a.extrap_n) else 'val' if gk(r) in val_k else 'train'
    for _ in range(50):                                             # every context and n-bucket must still appear in training
        tr = [r for r in rows if split(r) == 'train']; miss = ({r[1] for r in rows if r[3] < a.extrap_n} - {r[1] for r in tr}) | {('nb', nb(r[3])) for r in rows if r[3] < a.extrap_n} - {('nb', nb(r[3])) for r in tr}
        if not miss: break
        for g in list(test_k | val_k):
            if any((r[1] in miss or ('nb', nb(r[3])) in miss) for r in rows if gk(r) == g): test_k.discard(g); val_k.discard(g)
    sp = collections.Counter(split(r) for r in rows)
    print('records by split:', dict(sp), f'  (n >= {a.extrap_n} fully held out as extrapolation test)')
    trp = {r['messages'][1]['content'] for f in glob.glob('data_compact2/decide_S?.jsonl') for r in map(json.loads, open(f, encoding='utf-8')) if r['split'] != 'test'}
    os.makedirs(out, exist_ok=True); shutil.copytree(os.path.join(ooda, 'data_compact2'), os.path.join(out, 'data_compact2_dcov'))
    json.dump(cbx, open(os.path.join(out, 'codebook2d.json'), 'w'), indent=1)
    with open(os.path.join(out, 'data_compact2_dcov', 'decide_S9.jsonl'), 'w', encoding='utf-8') as fo:
        for i, (p, ci, tag, n, code) in enumerate(rows):
            fo.write(json.dumps({'id': f'S9d-syn-{i}', 'agent': 'decide', 'situation': 'S9', 'split': split((p, ci, tag, n, code)), 'obs_id': f'S9d-syn-{i}', 'scenario_tag': tag,
                                 'input_seen_in_train': p in trp, 'extrap': n >= a.extrap_n,
                                 'messages': [{'role': 'system', 'content': C2.DECIDE_SYSTEM2}, {'role': 'user', 'content': p}, {'role': 'assistant', 'content': code}]}) + '\n')
    print('emergency training prompts by n bucket:', dict(sorted(collections.Counter(nb(r[3]) for r in rows if r[2] == 'emergency_preempt' and split(r) == 'train').items())))
    print(f'MEC codebook: {len(cb["mec"])} -> {len(cbx["mec"])}\n\nwritten: {out}\\data_compact2_dcov  and  {out}\\codebook2d.json')
if __name__ == '__main__': main()
