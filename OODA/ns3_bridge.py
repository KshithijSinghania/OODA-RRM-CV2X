r"""Live OODA <-> ns-3 bridge (used by ooda_loop.py --cl_mode ns3; put this file next to ooda_loop.py).
Each 5 s window ns-3 (patched with patch_ctrl_loop.py, run with --ctrlDir=DIR) writes DIR/req_KKKK_{cell,ue}.csv + .ready and waits.
For every (window, cell) chain ooda_loop runs:
  begin(go)        wait for ns-3's measurement of that window; overwrite the QoS fields of the Observe INPUT telemetry with the
                   MEASURED ns-3 values of the vehicles served by that cell (mobility/SUMO fields stay as in the dataset: same trajectories)
  Observe(rules) -> Orient(LLM) -> Decide(LLM) -> Act(LLM, no feedback: its training feedback is next-window, i.e. future)
  set_policy(pol)  policy -> that cell's TxPower (same mapping as policy_to_schedule.py), sent to ns-3 when the window is complete;
                   it governs the NEXT window (causal).
Optional --ns3_correct: a global correction (block hill-climbing on measured network loss) added on top of the OODA powers (OODA + feedback).
Optional --ns3_rule rsrp_teacher.json (+ --ns3_rule_mode protect|rule): the RSRP rule (fit_rsrp_teacher.py; each cell's depth from the 10th-percentile
RSRP of its vehicles) sets every cell's power offset. 'rule' = the rule alone (OODA decisions ignored; reproduces the standalone rule runs);
'protect' = the same, except that where the Orient agent tags the cell's situation as anything but 'baseline' (emergency, dense intersection, ...) the cell
never cuts deeper than the OODA policy's own power (offset = max(rule, OODA)): the LLM chain decides WHERE to protect, the rule decides HOW DEEP.
Optional --ns3_enforcer (ns3_enforcer.py): the trained DRL enforcer runs per decided cell on its analytic cell model; its power residual vs the OODA policy
(policy-follow) is added to the offset above. Only the enforcer's POWER is transferred; its PRB shares have nothing to act on in this ns-3 setup."""
import os, re, csv, json, math, time
import config as C

def _pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q; lo = int(math.floor(i)); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)

def _sev(f):
    for hi, name in C.SEVERITY_BINS:
        if f < hi: return name
    return C.SEVERITY_BINS[-1][1]

class _Hill:
    def __init__(s, start, step, min_step, block, lo, hi):
        s.off, s.dir, s.step, s.min_step, s.block, s.lo, s.hi = start, -1.0, step, min_step, block, lo, hi
        s.prev, s.acc, s.rev = None, [0.0, 0.0, 0], 0
    def update(s, tx, lost):
        s.acc[0] += tx; s.acc[1] += lost; s.acc[2] += 1
        if s.acc[2] < s.block: return
        L = s.acc[1] / s.acc[0] if s.acc[0] else 0.0; s.acc = [0.0, 0.0, 0]
        if s.prev is not None and L > s.prev:
            s.dir = -s.dir; s.rev += 1
            if s.rev % 2 == 0: s.step = max(s.min_step, s.step / 2)
        s.prev = L; s.off = min(s.hi, max(s.lo, s.off + s.dir * s.step))

class _Rule:
    """RSRP rule (same logic as ctrl_server.py --policy rsrp): per cell, 10th-percentile RSRP of its served vehicles normalised to default power
    -> offset from the table; a new target must persist `persist` windows before it is applied."""
    def __init__(s, path, persist):
        s.T = json.load(open(path)); s.persist = persist; s.off = {}; s.pend = {}
    def update(s, ues, sites, applied):
        for site in sites:
            o_now = s.off.get(site, 0.0); a_now = applied.get(site, 0.0); rs = []
            for u in ues:
                if int(u['cell']) != site or float(u['tx_packets']) <= 0: continue
                v = u['rsrp_dbm']
                if v in ('', 'nan', '-nan'): continue
                v = float(v)
                if math.isfinite(v): rs.append(v - a_now)
            if not rs: continue
            x = _pct(rs, 0.10) if len(rs) >= s.T['min_ues'] else sum(rs) / len(rs)
            tgt = s.T['offsets'][sum(x > e for e in s.T['edges'])]
            if tgt == o_now: s.pend.pop(site, None); continue
            pt, n = s.pend.get(site, (None, 0)); n = n + 1 if pt == tgt else 1; s.pend[site] = (tgt, n)
            if n >= s.persist: s.off[site] = tgt; s.pend.pop(site, None)

class NS3Loop:
    def __init__(s, a):
        s.D = a.ns3_dir; s.tstart = a.ns3_tstart; s.t1 = a.ns3_t1; s.base = (a.ns3_txMacro, a.ns3_txRsu)
        s.full = a.ns3_full_dbm; s.timeout = a.ns3_timeout; s.delay_thr = a.ns3_delay_thr if a.ns3_delay_thr is not None else getattr(C, 'DELAY_VIOLATION_MS', 30.0)
        s.hill = _Hill(0.0, 3.0, 1.0, 2, -6.0, 13.0) if a.ns3_correct else None
        s.next_k = 0; s.cur_k = None; s.site = None; s.pending = {}; s.last_delta = {}; s.prev = {}; s.cache = {}; s.cur_pol_id = None
        s.rule = _Rule(a.ns3_rule, getattr(a, 'ns3_persist', 2)) if getattr(a, 'ns3_rule', None) else None
        s.rule_mode = getattr(a, 'ns3_rule_mode', 'protect'); s.applied = {}; s.last_tag = {}; s.ptag = {}
        s.enf = None; s.gobs = {}; s.enf_pend = {}; s.enf_res = {}; s.enf_err = 0
        if getattr(a, 'ns3_enforcer', False):
            import ns3_enforcer
            s.enf = ns3_enforcer.NS3Enforcer(a.cl_enforcer_dir, a.cl_runs or os.path.join(a.cl_enforcer_dir, 'runs'), kind=a.cl_enforcer, agent_seed=a.cl_agent_seed,
                                             log_path=os.path.join(s.D, 'enforcer_log.csv'))
        os.makedirs(s.D, exist_ok=True)
        s.log = open(os.path.join(s.D, 'ooda_ctrl_log.csv'), 'a')
        s.log.write('k,t_end,warmup,net_tx,net_lost,net_loss_pct,n_cells_decided,correction_db,rule_mean_offset_db,n_cells_protected,enf_mean_residual_db\n'); s.log.flush()
    # ---------- ordering
    @staticmethod
    def _w0(go): return float(json.loads(go['messages'][1]['content'])['window_s'][0])
    @staticmethod
    def _site(go):
        m = re.search(r'(\d+)\s*$', str(go.get('cell') or json.loads(go['messages'][1]['content'])['scope']['cell_id']))
        return int(m.group(1))
    def group(s, chains):
        sel = [c for c in chains if s.tstart <= s._w0(c[0]) < s.t1]
        if s.enf:                                             # dataset telemetry of every window: the enforcer's world demand for the window a policy will govern
            for c in sel:
                try: s.gobs[(s._site(c[0]), int(round((s._w0(c[0]) - s.tstart) / 5.0)))] = json.loads(c[0]['messages'][2]['content'])
                except Exception: pass
        sel.sort(key=lambda c: (s._w0(c[0]), s._site(c[0])))
        s.eps = sorted({s._w0(c[0]) for c in sel}); s.W = 1
        return sel
    # ---------- ns-3 files
    def _wait(s, k):
        if k in s.cache: return s.cache[k]
        rdy = os.path.join(s.D, f'req_{k:04d}.ready'); t0 = time.time()
        while not os.path.exists(rdy):
            if os.path.exists(os.path.join(s.D, 'end.flag')): raise SystemExit(f'ns-3 ended before window {k}')
            if time.time() - t0 > s.timeout: raise SystemExit(f'timeout waiting for {rdy} (did ns-3 crash?)')
            time.sleep(0.2)
        cells = {int(r['cell']): r for r in csv.DictReader(open(os.path.join(s.D, f'req_{k:04d}_cell.csv')))}
        ues = list(csv.DictReader(open(os.path.join(s.D, f'req_{k:04d}_ue.csv'))))
        s.cache[k] = (cells, ues); return s.cache[k]
    def _respond(s, k):
        cells, _ = s._wait(k)
        for site, d in s.pending.pop(k, {}).items(): s.last_delta[site] = d
        for site, t in s.ptag.pop(k, {}).items(): s.last_tag[site] = t
        for site, r_ in s.enf_pend.pop(k, {}).items(): s.enf_res[site] = r_
        if s.rule: s.rule.update(s._wait(k)[1], sorted(cells), s.applied)       # measured window k, normalised by the offset that was in force
        r0 = next(iter(cells.values())); warm = r0['warmup'] == '1'
        tx = sum(float(r['tx_packets']) for r in cells.values()); lost = sum(max(0.0, float(r['tx_packets']) - float(r['rx_packets'])) for r in cells.values())
        if s.hill and not warm: s.hill.update(tx, lost)
        corr = s.hill.off if s.hill else 0.0
        tmp = os.path.join(s.D, f'resp_{k:04d}.tmp')
        with open(tmp, 'w') as f:
            f.write('site,tx_dbm\n')
            nprot = 0
            for site in sorted(cells):
                b = s.base[0] if site == 0 else s.base[1]; d = s.last_delta.get(site, 0.0)
                if s.rule:
                    r = s.rule.off.get(site, 0.0)
                    prot = s.rule_mode == 'protect' and site in s.last_delta and s.last_tag.get(site) not in (None, 'baseline')
                    off = max(r, d) if prot else r; nprot += bool(prot)
                else: off = d
                off += s.enf_res.get(site, 0.0) if site in s.last_delta else 0.0       # DRL enforcer residual (0 without --ns3_enforcer)
                tx_site = min(b, max(b - 18.0, b + off + corr)); s.applied[site] = tx_site - b
                f.write(f'{site},{tx_site:.4f}\n')
        os.replace(tmp, os.path.join(s.D, f'resp_{k:04d}.csv'))
        s.log.write(f"{k},{r0['t_end']},{int(warm)},{tx:.0f},{lost:.0f},{100*lost/tx if tx else 0:.4f},{len(s.last_delta)},{corr:.3f},{(sum(s.rule.off.values()) / max(1, len(cells))) if s.rule else float('nan'):.3f},{nprot if s.rule else 0},{(sum(s.enf_res.get(x_, 0.0) for x_ in s.last_delta) / max(1, len(s.last_delta))) if s.enf else float('nan'):.3f}\n"); s.log.flush()
        s.next_k = k + 1
    # ---------- hooks called by ooda_loop
    def begin(s, go):
        k = int(round((s._w0(go) - s.tstart) / 5.0)); site = s._site(go)
        while s.next_k < k: s._respond(s.next_k)              # close earlier windows (also ones with no chain)
        cells, ues = s._wait(k); s.cur_k, s.site = k, site
        inp = json.loads(go['messages'][1]['content'])
        mine = [u for u in ues if int(u['cell']) == site and float(u['tx_packets']) > 0]
        if mine:
            loss = [float(u['loss_pct']) for u in mine]; dly = [float(u['mean_delay_ms']) for u in mine]
            txs = sum(float(u['tx_packets']) for u in mine); rxs = sum(float(u['rx_packets']) for u in mine)
            n = len(mine); lm = sum(loss) / n
            inp['loss_pct'] = dict(inp.get('loss_pct') or {}, mean=round(lm, 3), p95=round(_pct(loss, 0.95), 3))
            inp['delay_ms'] = dict(inp.get('delay_ms') or {}, mean=round(sum(dly) / n, 3))
            inp['frac_loss_gt_10pct'] = round(sum(x > 10 for x in loss) / n, 4)
            inp['frac_delay_gt_thr'] = round(sum(x > s.delay_thr for x in dly) / n, 4)
            inp['frac_delay_gt_100ms'] = round(sum(x > 100 for x in dly) / n, 4)
            inp['n_loss_gt_50pct'] = int(sum(x > 50 for x in loss))
            inp['delivery_ratio'] = round(rxs / txs, 5) if txs else 1.0
            c = cells.get(site)
            if c and float(c['ues']) > 0: inp['throughput_mbps_mean'] = round(float(c['thr_mbps']) / float(c['ues']), 4)
            inp['worst_vehicles'] = [{'id': u['vehicle_id'], 'loss_pct': float(u['loss_pct'])} for u in sorted(mine, key=lambda u: -float(u['loss_pct']))[:3]]
            if site in s.prev: inp['prev_windows_mean'] = s.prev[site]
            s.prev[site] = {'loss_pct': lm, 'n_vehicles': inp.get('n_active_vehicles', n)}
        go['messages'][1]['content'] = json.dumps(inp)
        return {'k': k, 'site': site, 'measured_vehicles': len(mine)}
    def override_obs(s, obs): return None                      # not used: the INPUT telemetry is replaced in begin()
    def set_policy(s, pol, ctx):
        s.cur_pol_id = pol.get('policy_id')
        x = pol['prb_priority']; l = pol['power_budget_dbm']
        sh = {'urllc': x['urllc_pct'], 'embb': x['embb_pct'], 'mmtc': x['mmtc_pct']}; tot = sum(sh.values()) or 1.0
        pavg = sum(sh[c] / tot * 10 ** (l[c] / 10) for c in sh)
        d = 10 * math.log10(max(pavg, 1e-9) / 10 ** (s.full / 10))
        s.pending.setdefault(s.cur_k, {})[s.site] = d; s.ptag.setdefault(s.cur_k, {})[s.site] = (ctx or {}).get('scenario_tag')
        out = {'k': s.cur_k, 'site': s.site, 'delta_db': round(d, 3), 'tag': (ctx or {}).get('scenario_tag')}
        if s.enf:
            g = s.gobs.get((s.site, s.cur_k + 1)) or s.gobs.get((s.site, s.cur_k))        # next window's telemetry if the dataset has it, else this window's
            try:
                r_, inf = s.enf.run(s.site, s.cur_k, g, pol, ctx) if g is not None else (0.0, None)
                if inf is None and g is None: raise RuntimeError('no dataset telemetry for this cell/window')
            except Exception as e:
                r_, inf = 0.0, None; s.enf.n_fail += 1
                if s.enf.n_fail <= 3: print('ENFORCER failed:', repr(e), flush=True)
            s.enf_pend.setdefault(s.cur_k, {})[s.site] = r_; out['enforcer'] = inf
        return out
    def feedback(s, obs):
        # Act was trained with feedback = the outcome of THIS policy in the NEXT window (future at decision time), so live Act runs
        # without feedback (an in-distribution case: last window of each trace). Escalation on measured outcomes = post-hoc analysis.
        return None
    def measured_feedback(s, obs):   # kept for post-hoc use; NOT passed to Act live (the guard would flag it as stale)
        q = obs['qos_status']
        return {'policy_id': s.cur_pol_id, 'window_s': obs['window_s'], 'auth_ok': True,
                'enforcer': {'applied': True, 'violations': [], 'prb_util_pct': round(100 * float((obs.get('resource_state') or {}).get('offered_load_ratio', 0) or 0), 1)},
                'observed': {'severity': q['severity'], 'mean_loss_pct': q['mean_loss_pct'], 'mean_delay_ms': q['mean_delay_ms'], 'frac_loss_gt_10pct': q['frac_loss_gt_10pct']},
                'emergency_event_observed': bool(obs.get('special_events'))}
    def finish_all(s):
        if s.enf: print(s.enf.summary(), flush=True)
        if s.cur_k is not None:
            while s.next_k <= s.cur_k: s._respond(s.next_k)
        while True:                                            # answer the remaining windows (no new decisions) until ns-3 ends
            rdy = os.path.join(s.D, f'req_{s.next_k:04d}.ready'); t0 = time.time()
            while not os.path.exists(rdy):
                if os.path.exists(os.path.join(s.D, 'end.flag')): print('ns-3 finished; live loop done'); return
                if time.time() - t0 > s.timeout: print('timeout waiting for ns-3; stopping'); return
                time.sleep(0.2)
            s._respond(s.next_k)
