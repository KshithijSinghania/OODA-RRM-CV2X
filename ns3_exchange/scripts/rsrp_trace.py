r"""What did a controller do? Reads the per-window replies (resp_KKKK.csv) of closed-loop runs and prints, per site, the offset vs default power.
  python3 rsrp_trace.py ~/ns3/cl/ns-3.48/ctrl/cl_s7_rsrp_s1 ~/ns3/cl/ns-3.48/ctrl/cl_s5_rsrp_s1 ...     (one folder per run; pure python)
Shows: mean offset per site, share of windows at each offset, how many times each site's power changed, how many sites change per window."""
import sys, os, csv, glob, collections
BASE = lambda s: 37.0 if s == 0 else 30.0
for d in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(d, 'resp_*.csv')))
    if len(fs) < 3: print(f'{d}: only {len(fs)} reply files, skipped'); continue
    W = []
    for f in fs: W.append({int(r['site']): float(r['tx_dbm']) - BASE(int(r['site'])) for r in csv.DictReader(open(f))})
    sites = sorted(W[0]); n = len(W)
    chg = {s: sum(abs(W[i][s] - W[i - 1][s]) > 1e-6 for i in range(1, n)) for s in sites}
    perwin = [sum(abs(W[i][s] - W[i - 1][s]) > 1e-6 for s in sites) for i in range(1, n)]
    print(f'\n== {os.path.basename(d.rstrip("/"))}: {n} windows ==')
    print(f"{'site':>5s}{'mean off':>10s}{'min':>7s}{'max':>7s}{'changes':>9s}   share of windows at each offset")
    for s in sites:
        v = [w[s] for w in W]; c = collections.Counter(round(x, 2) for x in v)
        print(f'{s:5d}{sum(v)/n:10.1f}{min(v):7.1f}{max(v):7.1f}{chg[s]:9d}   ' + '  '.join(f'{k:g}:{100*c[k]//n}%' for k in sorted(c)))
    print(f'network mean offset {sum(sum(w.values())/len(w) for w in W)/n:.1f} dB;  sites changed per window: mean {sum(perwin)/len(perwin):.1f}, max {max(perwin)}')
