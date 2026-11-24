r"""Teacher v2, step 1: calibrate a MULTI-CELL radio model on S7 ns-3 runs only, then test it on S5 (never seen).
  cd ~/exchange/results && python3 ~/exchange/scripts/teacher_model.py          (needs numpy)
Model (research: calibrated twin + interference-coupled power control, Dreifuerst 2021 / Shen&Yu 2018 / AutoPlan 2025):
  RSRP_kj = P_j - PL(d_kj),  PL = a + b*log10(d)                         (fit on measured serving RSRP vs distance)
  SINR_k  = S_k / (eta * sum_{j!=s} act_j * R_kj + N0)                   (fit eta, N0 on measured SINR; act_j = load of cell j)
  loss_k  = c0 + c1*sigmoid(-(SINR_k - s0)/w)                            (fit on measured per-vehicle loss)
Uniform power cuts scale S and I together but not N0 -> they identify the noise-vs-interference balance, i.e. the point where cutting
power stops helping (S7 ~ -13 dB, S5 ~ -6 dB measured). GATE: predicted S5 curve (from the S7 fit) must have a shallower optimum than S7.
Writes teacher_model.json (used by the live 'model' teacher)."""
import csv, os, json, math
import numpy as np
SITES = np.array([[1824.7, 395.1], [1492.5, 394.0], [1725.5, 722.1], [2196.9, 702.4], [1882.2, 608.9], [1404.6, 666.8],
                  [353.4, 767.4], [1026.1, 687.0], [1092.5, 323.6], [687.7, 283.3], [234.0, 259.0], [2006.1, 1299.7]])
BASE = np.array([37.0] + [30.0] * 11)
S7 = {6: ['sw_high'], 0: ['sw_base', 'sw_base_seed2', 'sw_base_seed3', 'hp_base'], -3: ['hp_m3'], -6: ['sw_low', 'hp_m6_s2', 'hp_m6_s3'],
      -9: ['hp_m9_s2', 'hp_m9_s3'], -10.75: ['hp_static11_s2', 'hp_static11_s3'], -13: ['b2_static13', 'hp_static13_s2', 'hp_static13_s3'],
      -15: ['hp_m15'], -18: ['hp_m18']}
S5 = {0: ['s5_base', 's5_base_s2', 's5_base_s3'], -3: ['s5_m3'], -6: ['s5_m6', 's5_m6_s2', 's5_m6_s3'], -10.75: ['s5_m11'],
      -13: ['s5_m13', 's5_m13_s2', 's5_m13_s3'], -15: ['s5_m15']}
def fnum(x):
    try: v = float(x); return v if math.isfinite(v) else np.nan
    except ValueError: return np.nan
def load(runs):
    R = []
    for off, ps in runs.items():
        for p in ps:
            f = p + '_ns3_5s.csv'
            if not os.path.exists(f): print('  missing', f); continue
            for r in csv.DictReader(open(f)):
                if r['warmup'] != '0' or float(r['tx_packets']) <= 0: continue
                R.append((off, p, float(r['window_start']), r['vehicle_id'], int(float(r['serving_cell'])), float(r['x']), float(r['y']),
                          fnum(r['rsrp_dbm']), fnum(r['sinr_db']), float(r['tx_packets']), float(r['lost_packets']), float(r['handovers'])))
    a = np.array([(o, w, s, x, y, rs, si, tx, lo, ho) for (o, p, w, v, s, x, y, rs, si, tx, lo, ho) in R], float)
    runs_id = np.array([p for (o, p, *_) in R]); return a, runs_id
def dist(x, y):   # (n,12) 3-D distance to every site (gNB 25 m, UE 1.5 m)
    dx = x[:, None] - SITES[None, :, 0]; dy = y[:, None] - SITES[None, :, 1]; return np.sqrt(dx ** 2 + dy ** 2 + 23.5 ** 2)
def interference_terms(A, ids, a_, b_):
    """per row: serving RSRP S (measured), and sum over other sites of act_j * predicted RSRP (linear mW)."""
    off, w, s, x, y = A[:, 0], A[:, 1], A[:, 2].astype(int), A[:, 3], A[:, 4]
    P = BASE[None, :] + off[:, None]; R = P - (a_ + b_ * np.log10(dist(x, y)))       # predicted RSRP to all sites (dBm)
    act = np.zeros((len(A), 12))                                                      # load of each cell = served vehicles in that run+window
    key = {}
    for i, (rid, ww, ss) in enumerate(zip(ids, w, s)):
        if ss < 12: key[(rid, ww, ss)] = key.get((rid, ww, ss), 0) + 1
    for i, (rid, ww) in enumerate(zip(ids, w)):
        for j in range(12): act[i, j] = key.get((rid, ww, j), 0)
    lin = 10 ** (R / 10); mask = np.ones_like(lin); mask[np.arange(len(A)), np.clip(s, 0, 11)] = 0
    return lin, act, mask
def sig(z): return 1 / (1 + np.exp(-z))
print('loading S7 (fit) ...'); A7, I7 = load(S7)
print('loading S5 (test only) ...'); A5, I5 = load(S5)
if A7.ndim != 2 or A5.ndim != 2: raise SystemExit('no S7 or no S5 runs found: run this inside ~/exchange/results')
ok = lambda A: (A[:, 2] < 12) & np.isfinite(A[:, 5]) & np.isfinite(A[:, 6]) & np.isfinite(A[:, 3]) & np.isfinite(A[:, 4])
A7f, I7f = A7[ok(A7)], I7[ok(A7)]
# 1. path loss on serving link: PL = P_serv - RSRP
d7 = dist(A7f[:, 3], A7f[:, 4])[np.arange(len(A7f)), A7f[:, 2].astype(int)]
PL = (BASE[A7f[:, 2].astype(int)] + A7f[:, 0]) - A7f[:, 5]
X = np.c_[np.ones(len(d7)), np.log10(d7)]; (pa, pb), *_ = np.linalg.lstsq(X, PL, rcond=None)
res = PL - X @ [pa, pb]; print(f'\n1. path loss  PL = {pa:.1f} + {pb:.1f}*log10(d)   residual std {res.std():.1f} dB   R2 {1 - res.var() / PL.var():.2f}   (n={len(PL)})')
# 2. SINR: S measured (linear), I = eta * sum act_j * R_j over other sites, N0 free -> grid fit on measured SINR (dB)
def sinr_pred(A, ids, eta, n0, cap):
    lin, act, mask = interference_terms(A, ids, pa, pb)
    S = 10 ** (A[:, 5] / 10); I = (lin * np.minimum(act * eta, cap) * mask).sum(1)
    return 10 * np.log10(S / (I + 10 ** (n0 / 10)))
best = None
lin7, act7, mask7 = interference_terms(A7f, I7f, pa, pb); S7l = 10 ** (A7f[:, 5] / 10)
for n0 in np.arange(-150, -79, 1.0):
    for eta in 10 ** np.arange(-3, 0.01, 0.25):
        for cap in (0.25, 0.5, 1.0):
            I = (lin7 * np.minimum(act7 * eta, cap) * mask7).sum(1); p = 10 * np.log10(S7l / (I + 10 ** (n0 / 10)))
            e = np.mean((p - A7f[:, 6]) ** 2)
            if best is None or e < best[0]: best = (e, n0, eta, cap)
_, N0, ETA, CAP = best
p7 = sinr_pred(A7f, I7f, ETA, N0, CAP); r7 = p7 - A7f[:, 6]
A5f, I5f = A5[ok(A5)], I5[ok(A5)]; p5 = sinr_pred(A5f, I5f, ETA, N0, CAP); r5 = p5 - A5f[:, 6]
print(f'2. SINR model  N0 {N0:.0f} dBm  eta {ETA:.3g}/vehicle (cap {CAP})   S7 fit: RMSE {np.sqrt(np.mean(r7**2)):.1f} dB R2 {1 - r7.var()/A7f[:,6].var():.2f}'
      f'   S5 (unseen): RMSE {np.sqrt(np.mean(r5**2)):.1f} dB R2 {1 - r5.var()/A5f[:,6].var():.2f}   bias S5 {r5.mean():+.1f} dB')
# 3. loss curve vs predicted SINR (tx-weighted least squares, grid over s0,w; linear for c0,c1)
y7 = A7f[:, 8] / A7f[:, 7]; wt = A7f[:, 7]; bestL = None
for s0 in np.arange(-10, 20, 0.5):
    for wd in (0.5, 1, 1.5, 2, 3, 4, 6):
        z = sig(-(p7 - s0) / wd); M = np.c_[np.ones_like(z), z]; W = np.sqrt(wt)
        c, *_ = np.linalg.lstsq(M * W[:, None], y7 * W, rcond=None); c = np.maximum(c, 0)
        e = np.sum(wt * (M @ c - y7) ** 2)
        if bestL is None or e < bestL[0]: bestL = (e, s0, wd, c[0], c[1])
_, S0, WD, C0, C1 = bestL
print(f'3. loss model  loss = {C0:.4f} + {C1:.3f}*sigmoid(-(SINR-{S0:.1f})/{WD})')
# 4. GATE: predict each scenario's flat-offset curve from its DEFAULT runs (positions, serving, loads), shifting all powers by D
def curve(A, ids, offs):
    base = A[(A[:, 0] == 0) & ok(A)]; bid = ids[(A[:, 0] == 0) & ok(A)]
    lin, act, mask = interference_terms(base, bid, pa, pb); S = 10 ** (base[:, 5] / 10); I = (lin * np.minimum(act * ETA, CAP) * mask).sum(1)
    out = {}
    for D in offs:
        g = 10 ** (D / 10); sp = 10 * np.log10(S * g / (I * g + 10 ** (N0 / 10)))
        out[D] = 100 * np.sum(base[:, 7] * (C0 + C1 * sig(-(sp - S0) / WD))) / base[:, 7].sum()
    return out
def measured(A):
    m = {}
    for D in sorted(set(A[:, 0])):
        s = A[A[:, 0] == D]; m[D] = 100 * s[:, 8].sum() / s[:, 7].sum()
    return m
grid = [6, 3, 0, -3, -6, -9, -10.75, -13, -15, -18]
for name, A, ids in (('S7 (fit)', A7, I7), ('S5 (UNSEEN)', A5, I5)):
    pc, mc = curve(A, ids, grid), measured(A)
    print(f'\n4. {name}: offset   ' + ''.join(f'{d:>7}' for d in grid))
    print('   predicted loss %  ' + ''.join(f'{pc[d]:7.2f}' for d in grid) + f'    predicted best (<=0 dB) {min((d for d in pc if d <= 0), key=pc.get)} dB')
    print('   measured loss %   ' + ''.join(f'{mc[d]:7.2f}' if d in mc else '      -' for d in grid) + f'    measured best (<=0 dB) {min((d for d in mc if d <= 0), key=mc.get)} dB')
    common = [d for d in grid if d in mc]
    if len(common) > 2:
        rp = np.argsort(np.argsort([pc[d] for d in common])); rm = np.argsort(np.argsort([mc[d] for d in common]))
        print(f'   rank correlation predicted vs measured: {np.corrcoef(rp, rm)[0,1]:+.2f}')
json.dump(dict(pl_a=pa, pl_b=pb, n0_dbm=N0, eta=ETA, cap=CAP, s0=S0, w=WD, c0=C0, c1=C1, sites=SITES.tolist(), base=BASE.tolist(), fitted_on='S7'),
          open('teacher_model.json', 'w'), indent=1)
print('\nwrote teacher_model.json.  GATE passes if the S5 predicted best offset is clearly shallower than S7\'s and the S5 rank correlation is positive.')
