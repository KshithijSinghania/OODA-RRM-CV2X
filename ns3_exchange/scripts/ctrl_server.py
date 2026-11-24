#!/usr/bin/env python3
r"""Controller side of the ns-3 closed loop (pairs with patch_ctrl_loop.py). Pure Python, no packages needed.
ns-3 writes DIR/req_KKKK_cell.csv + req_KKKK_ue.csv + req_KKKK.ready after each 5 s window and waits for DIR/resp_KKKK.csv.
Policies
  flat   --offset X          constant offset X dB vs default on every site (validation: must reproduce a static run)
  replay --sched FILE        apply an open-loop schedule (t_start,site,tx_dbm) at the matching window end (validation: must reproduce the schedule run)
  hill   [--start X]         ONE global offset tuned by block hill-climbing on measured network packet loss (simple feedback baseline;
                             knows nothing about the scenario; start X = where it begins, e.g. 0 = default, -13 = S7-tuned value)
  hybrid --sched FILE        OODA + feedback: the OODA schedule sets each cell's power (per-cell pattern, changes per window) and a
                             global CORRECTION (hill-climbed on measured loss, starts at 0) is added on top. tx is clipped to <= default.
  rsrp   --table FILE        RSRP teacher (fit_rsrp_teacher.py): each cell's offset from the 10th-percentile RSRP of its served
                             vehicles (normalised to default power); a new target must persist --persist windows before it is applied.
  model  --model FILE        TEACHER v2 (teacher_model.py, calibrated multi-cell SINR model, fitted on S7): every window, from each vehicle's
                             measured position / serving cell / RSRP, predict loss for candidate per-cell offsets (incl. A3 re-attachment)
                             and pick them by coordinate descent; --lam = cost (pp of network loss) per cell whose power changes.
Writes DIR/ctrl_log.csv (one line per window). Exits when DIR/end.flag appears.
usage: python3 ctrl_server.py --dir CTRL_DIR --policy hill --start 0"""
import os, csv, time, argparse, math, json
ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--policy', choices=['flat', 'replay', 'hill', 'hybrid', 'rsrp', 'model'], required=True)
ap.add_argument('--offset', type=float, default=0.0); ap.add_argument('--sched')
ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--step', type=float, default=3.0); ap.add_argument('--min_step', type=float, default=1.0)
ap.add_argument('--block', type=int, default=2, help='windows per hill-climbing step'); ap.add_argument('--lo', type=float, default=-18.0); ap.add_argument('--hi', type=float, default=0.0)
ap.add_argument('--txMacro', type=float, default=37.0); ap.add_argument('--txRsu', type=float, default=30.0)
ap.add_argument('--timeout', type=float, default=7200.0, help='s to wait for the next request before giving up')
ap.add_argument('--table', help='rsrp_teacher.json'); ap.add_argument('--persist', type=int, default=2)
ap.add_argument('--model', help='teacher_model.json'); ap.add_argument('--lam', type=float, default=0.05); ap.add_argument('--hyst', type=float, default=3.0)
ap.add_argument('--cands', default='0,-3,-6,-9,-10.75,-13,-15')
a = ap.parse_args()
D = a.dir; os.makedirs(D, exist_ok=True)
base = lambda s: a.txMacro if s == 0 else a.txRsu
sched = {}
if a.policy in ('replay', 'hybrid'):
    for r in csv.DictReader(open(a.sched)): sched.setdefault(round(float(r['t_start']), 3), []).append((int(r['site']), float(r['tx_dbm'])))

class Hill:
    """block hill-climbing on network loss: after each block of windows compare its loss with the previous block's;
    better -> keep direction, worse -> reverse (step halved on every 2nd reversal, not below min_step). Starts by moving DOWN."""
    def __init__(s): s.off, s.dir, s.step, s.prev, s.acc, s.rev = a.start, -1.0, a.step, None, [0.0, 0.0, 0], 0
    def update(s, tx, lost):
        s.acc[0] += tx; s.acc[1] += lost; s.acc[2] += 1
        if s.acc[2] < a.block: return
        L = s.acc[1] / s.acc[0] if s.acc[0] else 0.0; s.acc = [0.0, 0.0, 0]
        if s.prev is not None and L > s.prev:
            s.dir = -s.dir; s.rev += 1
            if s.rev % 2 == 0: s.step = max(a.min_step, s.step / 2)   # halve only on every 2nd reversal
        s.prev = L; s.off = min(a.hi, max(a.lo, s.off + s.dir * s.step))
hill = Hill() if a.policy in ('hill', 'hybrid') else None
cur = {}   # hybrid: last scheduled tx per site
RT = json.load(open(a.table)) if a.policy == 'rsrp' else None
r_off, r_pend = {}, {}   # rsrp: current offset per site, (pending target, count)
TM = json.load(open(a.model)) if a.policy == 'model' else None
m_off = None                                               # model: current offsets (np array)
def model_step(ues):
    import numpy as np
    global m_off
    SI = np.array(TM['sites']); B = np.array(TM['base']); nS = len(SI)
    if m_off is None: m_off = np.zeros(nS)
    u = [r for r in ues if float(r['tx_packets']) > 0 and 0 <= int(r['cell']) < nS]
    if not u: return m_off.copy()
    x = np.array([float(r['x']) for r in u]); y = np.array([float(r['y']) for r in u]); sv = np.array([int(r['cell']) for r in u])
    tx = np.array([float(r['tx_packets']) for r in u])
    rs = np.array([float(r['rsrp_dbm']) if r['rsrp_dbm'] not in ('', 'nan', '-nan') else np.nan for r in u])
    i = np.arange(len(u)); d = np.sqrt((x[:, None] - SI[None, :, 0]) ** 2 + (y[:, None] - SI[None, :, 1]) ** 2 + 23.5 ** 2)
    PL = TM['pl_a'] + TM['pl_b'] * np.log10(d)
    corr = np.where(np.isfinite(rs), rs - (B[sv] + m_off[sv] - PL[i, sv]), 0.0)      # measured - predicted on the serving link
    n0 = 10 ** (TM['n0_dbm'] / 10)
    def J(o):
        R = B[None, :] + o[None, :] - PL; R[i, sv] += corr
        best = R.argmax(1); serv = np.where(R[i, best] > R[i, sv] + a.hyst, best, sv)
        act = np.bincount(serv, minlength=nS).astype(float); lin = 10 ** (R / 10)
        W = np.minimum(act * TM['eta'], TM['cap']); I = (lin * W[None, :]).sum(1) - lin[i, serv] * W[serv]   # all cells but the serving one
        sinr = 10 * np.log10(lin[i, serv] / (np.maximum(I, 0) + n0))
        loss = TM['c0'] + TM['c1'] / (1 + np.exp((sinr - TM['s0']) / TM['w']))
        return 100 * (tx * loss).sum() / tx.sum() + a.lam * np.sum(np.abs(o - m_off) > 1e-9)
    cands = [float(c) for c in a.cands.split(',')]; o = m_off.copy(); cur = J(o)
    active = sorted(set(sv.tolist()))
    for _ in range(3):
        moved = False
        for j in active:
            for c in cands:
                if c == o[j]: continue
                t = o.copy(); t[j] = c; v = J(t)
                if v < cur - 1e-9: o, cur, moved = t, v, True
        if not moved: break
    m_off = o; return o
def _pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q; lo = int(i); hi = min(lo + 1, len(v) - 1); return v[lo] + (v[hi] - v[lo]) * (i - lo)
log = open(os.path.join(D, 'ctrl_log.csv'), 'a'); log.write('k,t_end,warmup,net_tx,net_lost,net_loss_pct,offset_db,n_sites_set\n'); log.flush()
k = 0; t_wait = time.time()
while True:
    rdy = os.path.join(D, f'req_{k:04d}.ready')
    if not os.path.exists(rdy):
        if os.path.exists(os.path.join(D, 'end.flag')): print('end.flag seen, exiting'); break
        if time.time() - t_wait > a.timeout: print('timeout waiting for', rdy); break
        time.sleep(0.2); continue
    rows = list(csv.DictReader(open(os.path.join(D, f'req_{k:04d}_cell.csv'))))
    t_end = float(rows[0]['t_end']); warm = rows[0]['warmup'] == '1'
    tx = sum(float(r['tx_packets']) for r in rows); lost = sum(max(0.0, float(r['tx_packets']) - float(r['rx_packets'])) for r in rows)
    out = []
    if a.policy == 'flat':
        out = [(s, base(s) + a.offset) for s in range(len(rows))]; off = a.offset
    elif a.policy == 'replay':
        out = sched.get(round(t_end, 3), []); off = float('nan')
    elif a.policy == 'hill':
        if not warm: hill.update(tx, lost)          # do not learn from the attach transient
        off = hill.off; out = [(s, base(s) + off) for s in range(len(rows))]
    elif a.policy == 'rsrp':
        ues = list(csv.DictReader(open(os.path.join(D, f'req_{k:04d}_ue.csv'))))
        for s_ in range(len(rows)):
            o_now = r_off.get(s_, 0.0)
            rs = [float(u['rsrp_dbm']) - o_now for u in ues if int(u['cell']) == s_ and float(u['tx_packets']) > 0
                  and u['rsrp_dbm'] not in ('', 'nan', '-nan') and math.isfinite(float(u['rsrp_dbm']))]
            if not rs: continue
            x = _pct(rs, 0.10) if len(rs) >= RT['min_ues'] else sum(rs) / len(rs)
            tgt = RT['offsets'][sum(x > e for e in RT['edges'])]
            if tgt == o_now: r_pend.pop(s_, None); continue
            pt, n = r_pend.get(s_, (None, 0)); n = n + 1 if pt == tgt else 1; r_pend[s_] = (tgt, n)
            if n >= a.persist: r_off[s_] = tgt; r_pend.pop(s_, None)
        off = sum(r_off.values()) / len(rows) if rows else 0.0
        out = [(s_, base(s_) + r_off.get(s_, 0.0)) for s_ in range(len(rows))]
    elif a.policy == 'model':
        ues = list(csv.DictReader(open(os.path.join(D, f'req_{k:04d}_ue.csv'))))
        o = model_step(ues) if not warm else [0.0] * len(rows); off = float(sum(o) / len(o))   # hold default during the attach transient
        out = [(s_, base(s_) + float(o[s_])) for s_ in range(len(rows))]
    else:  # hybrid
        for s_, v in sched.get(round(t_end, 3), []): cur[s_] = v
        if not warm: hill.update(tx, lost)
        off = hill.off
        out = [(s, min(base(s), max(base(s) - 18.0, cur.get(s, base(s)) + off))) for s in range(len(rows))]
    tmp = os.path.join(D, f'resp_{k:04d}.tmp')
    with open(tmp, 'w') as f:
        f.write('site,tx_dbm\n'); [f.write(f'{s},{v:.4f}\n') for s, v in out]
    os.replace(tmp, os.path.join(D, f'resp_{k:04d}.csv'))
    log.write(f'{k},{t_end},{int(warm)},{tx:.0f},{lost:.0f},{100*lost/tx if tx else 0:.4f},{off:.3f},{len(out)}\n'); log.flush()
    k += 1; t_wait = time.time()
