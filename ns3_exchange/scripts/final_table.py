r"""Presentation table in the format of the midterm report (Table 2): packet loss, throughput, delay, jitter - per controller, averaged over seeds.
  cd ~/exchange/results && python3 ~/exchange/scripts/final_table.py --title S7 "Default=sw_base,sw_base_seed2,sw_base_seed3" "Flat -6 dB=sw_low,hp_m6_s2,hp_m6_s3" ...
The first setting is the reference (Default). Runs of one setting are listed in seed order. Whole run = every non-warm-up vehicle-window with traffic.
Per run: loss % = lost/tx packets (packet-weighted); PRR % = 100 - loss; throughput = mean over vehicle-windows of the per-vehicle downlink rate in Mbps;
delay / jitter = mean over vehicle-windows (same definition as seedtable.py). Network throughput = sum over vehicles of the rate, averaged over windows (Mbps).
Every value is the mean over the seeds of the setting; 'sd' = spread of the run losses over seeds. Also writes DIR/final_table_<title>.csv.
Note: every vehicle sends a constant 540 B/packet x 100 packet/s = 0.432 Mbps (512 B payload + 28 B UDP/IP), so throughput = 0.432 x (1 - loss): it follows loss and
is not a capacity measurement."""
import csv, sys, argparse, os, math
ap = argparse.ArgumentParser(); ap.add_argument('settings', nargs='+'); ap.add_argument('--title', default='scenario'); ap.add_argument('--dir', default='.')
ap.add_argument('--half', type=float, default=0.0, help='only windows starting at or after this SUMO time (0 = whole run)')
a = ap.parse_args()
def parse(s): lab, runs = s.split('=', 1); return lab, [r.strip() for r in runs.split(',') if r.strip()]
def run_metrics(prefix):
    f = os.path.join(a.dir, prefix + '_ns3_5s.csv')
    if not os.path.exists(f): return None
    tx = lost = n = 0; thr = dl = jit = 0.0; per_win = {}
    for r in csv.DictReader(open(f)):
        if r['warmup'] != '0' or float(r['tx_packets']) <= 0 or float(r['window_start']) < a.half: continue
        t = float(r['tx_packets']); tx += t; lost += float(r['lost_packets']); n += 1
        th = float(r['throughput_mbps']); thr += th; dl += float(r['mean_delay_ms']); jit += float(r['mean_jitter_ms'])
        per_win[r['window_start']] = per_win.get(r['window_start'], 0.0) + th
    if not n: return None
    return dict(loss=100 * lost / tx, thr=thr / n, delay=dl / n, jit=jit / n, net=sum(per_win.values()) / len(per_win), n=n)
sets = [parse(s) for s in a.settings]
rows = []; ref = None
print(f'\n### {a.title}  (whole run, mean over seeds{"" if not a.half else f", windows from t={a.half:g} s"})')
print(f"{'Controller':24s}{'seeds':>6s}{'Loss %':>8s}{'(sd)':>7s}{'PRR %':>8s}{'Thr Mbps/veh':>13s}{'Net Mbps':>9s}{'Delay ms':>9s}{'Jitter ms':>10s}{'dLoss pp':>9s}{'dLoss %':>8s}")
for lab, runs in sets:
    ms = [m for m in (run_metrics(p) for p in runs) if m]
    if not ms: print(f'{lab:24s}  no data ({runs[0]} ...)'); continue
    mean = lambda k: sum(m[k] for m in ms) / len(ms)
    sd = math.sqrt(sum((m['loss'] - mean('loss')) ** 2 for m in ms) / (len(ms) - 1)) if len(ms) > 1 else float('nan')
    v = dict(label=lab, seeds=len(ms), loss=mean('loss'), sd=sd, prr=100 - mean('loss'), thr=mean('thr'), net=mean('net'), delay=mean('delay'), jit=mean('jit'))
    if ref is None: ref = v
    v['dl'] = v['loss'] - ref['loss']; v['dlp'] = 100 * v['dl'] / ref['loss']; rows.append(v)
    print(f"{lab:24s}{v['seeds']:6d}{v['loss']:8.2f}{v['sd']:7.2f}{v['prr']:8.2f}{v['thr']:13.4f}{v['net']:9.2f}{v['delay']:9.2f}{v['jit']:10.3f}{v['dl']:+9.2f}{v['dlp']:+7.0f}%")
with open(os.path.join(a.dir, f'final_table_{a.title}.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['controller', 'seeds', 'loss_pct', 'loss_sd_pp', 'prr_pct', 'throughput_mbps_per_vehicle', 'network_throughput_mbps', 'delay_ms', 'jitter_ms', 'loss_diff_vs_default_pp', 'loss_diff_vs_default_pct'])
    for v in rows: w.writerow([v['label'], v['seeds'], f"{v['loss']:.3f}", f"{v['sd']:.3f}", f"{v['prr']:.3f}", f"{v['thr']:.4f}", f"{v['net']:.3f}", f"{v['delay']:.3f}", f"{v['jit']:.4f}", f"{v['dl']:.3f}", f"{v['dlp']:.1f}"])
