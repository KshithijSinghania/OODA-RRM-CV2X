import csv, sys, argparse, math
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('prefixes', nargs='+'); ap.add_argument('--amb', default='AMBULANCE_01'); ap.add_argument('--radius', type=float, default=300.0)
ap.add_argument('--boot', type=int, default=2000); a = ap.parse_args()
rng = np.random.default_rng(0)
def load(p):
    rows = [r for r in csv.DictReader(open(p + '_ns3_5s.csv')) if r['warmup'] == '0' and float(r['tx_packets']) > 0]
    pos = {r['window_start']: (float(r['x']), float(r['y'])) for r in rows if r['vehicle_id'] == a.amb}
    out = {}
    for r in rows:
        k = (r['vehicle_id'], r['window_start']); me = r['vehicle_id'] == a.amb
        zone = (not me) and r['window_start'] in pos and math.hypot(float(r['x']) - pos[r['window_start']][0], float(r['y']) - pos[r['window_start']][1]) <= a.radius
        out[k] = dict(tx=float(r['tx_packets']), lost=float(r['lost_packets']), loss=float(r['packet_loss_percent']), delay=float(r['mean_delay_ms']),
                      ho=float(r['handovers']), amb=me, zone=zone)
    return out
R = {p: load(p) for p in a.prefixes}; base = R[a.prefixes[0]]
SUB = {'ALL': lambda d: True, 'ALERT zone': lambda d: d['zone'], 'AMBULANCE': lambda d: d['amb']}
def wl(ds): t = sum(d['tx'] for d in ds); return 100.0 * sum(d['lost'] for d in ds) / t if t else float('nan')
for sn, f in SUB.items():
    keys = [k for k, d in base.items() if f(d)]
    print(f'\n##### {sn}: {len(keys)} vehicle-windows in the base run #####')
    print(f"{'run':36s}{'loss % (pkt-wtd)':>17s}{'windows >5%':>13s}{'mean delay ms':>15s}{'HO windows':>12s}{'paired diff vs base (pp) [95% CI]':>40s}")
    for p in a.prefixes:
        ck = [k for k in keys if k in R[p]]; ds = [R[p][k] for k in ck]
        if not ds: print(f'{p:36s}  no rows'); continue
        diff = np.array([R[p][k]['loss'] - base[k]['loss'] for k in ck])
        if p == a.prefixes[0] or len(diff) < 5: ci = '-'
        else:
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.boot)]
            ci = f'{diff.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]'
        print(f"{p:36s}{wl(ds):17.3f}{100*np.mean([d['loss']>5 for d in ds]):12.1f}%{np.mean([d['delay'] for d in ds]):15.2f}{int(sum(d['ho']>0 for d in ds)):12d}{ci:>40s}")
