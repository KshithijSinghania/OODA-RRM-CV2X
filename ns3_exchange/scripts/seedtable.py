r"""Seed-matched comparison table (no extra packages).
  python3 seedtable.py --base "Default=s5_base,s5_base_s2,s5_base_s3" "Flat -6=s5_m6,s5_m6_s2,s5_m6_s3" "Live OODA=live_s5_ooda_s1,live_s5_ooda_s2,live_s5_ooda_s3"
Each setting lists its runs in SEED ORDER (seed 1, 2, 3, ...). Differences are taken seed by seed against the --base setting (same seed),
only for seeds present in both. Whole run = all non-warmup windows; 2nd half = windows starting at or after --half (steady state).
Metrics: packet-weighted loss %, share of vehicle-windows with loss > 5 %, mean delay ms, vehicle-windows with a handover."""
import csv, sys, argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('settings', nargs='+')
ap.add_argument('--half', type=float, default=150.0, help='SUMO time where the 2nd half starts'); ap.add_argument('--dir', default='.')
a = ap.parse_args()
def parse(s): lab, runs = s.split('=', 1); return lab, [r.strip() for r in runs.split(',')]
def stats(prefix, t0):
    f = os.path.join(a.dir, prefix + '_ns3_5s.csv')
    if not os.path.exists(f): return None
    tx = lost = n = bad = ho = 0; dl = 0.0
    for r in csv.DictReader(open(f)):
        if r['warmup'] != '0' or float(r['tx_packets']) <= 0 or float(r['window_start']) < t0: continue
        t = float(r['tx_packets']); tx += t; lost += float(r['lost_packets']); n += 1
        bad += float(r['packet_loss_percent']) > 5; dl += float(r['mean_delay_ms']); ho += float(r['handovers']) > 0
    return dict(loss=100 * lost / tx, bad=100 * bad / n, delay=dl / n, ho=ho) if n else None
sets = [parse(a.base)] + [parse(s) for s in a.settings]
for part, t0 in (('WHOLE RUN', 0.0), (f'2nd HALF (windows from t={a.half:g} s)', a.half)):
    S = {lab: [stats(p, t0) for p in runs] for lab, runs in sets}
    B = S[sets[0][0]]
    print(f'\n######## {part} ########')
    print(f"{'setting':22s}{'seeds':>6s}{'loss %':>8s}{'>5% win':>9s}{'delay ms':>9s}{'HO win':>7s}   per-seed loss diff vs base (pp)        mean diff  n_better")
    for lab, runs in sets:
        v = S[lab]; ok = [x for x in v if x]
        if not ok: print(f'{lab:22s}  no data'); continue
        m = lambda k: sum(x[k] for x in ok) / len(ok)
        d = [(v[i]['loss'] - B[i]['loss']) if (i < len(B) and v[i] and B[i]) else None for i in range(len(v))]
        dd = [x for x in d if x is not None]
        ds = ' '.join('  n/a ' if x is None else f'{x:+6.2f}' for x in d)
        tail = '' if lab == sets[0][0] else f'{sum(dd)/len(dd):+8.2f}   {sum(x < 0 for x in dd)}/{len(dd)}' if dd else ''
        print(f"{lab:22s}{len(ok):6d}{m('loss'):8.2f}{m('bad'):8.1f}%{m('delay'):9.2f}{m('ho'):7.0f}   {ds:<36s}{tail}")
print('\nnoise reference: seed-to-seed spread of the default runs is about 0.2-0.4 pp; n_better = seeds where the setting beats default.')
