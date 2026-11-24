r"""Adds --cl_mode ns3 (live OODA chain controlling ns-3 through ns3_bridge.py) to an ooda_loop.py that already has the closed-loop patch.
Usage (PowerShell, from Fine-tune):  .\ft\Scripts\python.exe patch_ns3_loop.py ..\OODA\ooda_loop.py      (backup: ooda_loop.py.bak_ns3)"""
import sys, shutil
p = sys.argv[1]; src = open(p, encoding='utf-8').read()
if "'ns3'" in src: sys.exit('already patched for ns3')
if "--cl_mode" not in src: sys.exit('apply patch_closed_loop.py first')
def sub(old, new):
    global src
    assert src.count(old) == 1, f'anchor not unique / not found ({src.count(old)}x): {old[:80]!r}'
    src = src.replace(old, new)
sub("choices=['off', 'dataset', 'enforcer']", "choices=['off', 'dataset', 'enforcer', 'ns3']")
sub("ap.add_argument('--cl_enforcer_dir', default=",
    "ap.add_argument('--ns3_dir', default=None, help='ns-3 --ctrlDir folder as seen from this machine (e.g. C:\\\\ns3ctrl\\\\live_s7_s1)')\n"
    "ap.add_argument('--ns3_tstart', type=float, default=90.0); ap.add_argument('--ns3_t1', type=float, default=200.0)\n"
    "ap.add_argument('--ns3_txMacro', type=float, default=37.0); ap.add_argument('--ns3_txRsu', type=float, default=30.0); ap.add_argument('--ns3_full_dbm', type=float, default=23.0)\n"
    "ap.add_argument('--ns3_correct', action='store_true', help='add the feedback correction (hill-climbing on measured loss) on top of the OODA powers')\n"
    "ap.add_argument('--ns3_timeout', type=float, default=7200.0); ap.add_argument('--ns3_delay_thr', type=float, default=None, help='ms threshold behind frac_delay_gt_thr (default config value or 20)')\n"
    "ap.add_argument('--cl_enforcer_dir', default=")
sub("    import cl_env\n    CL = cl_env.ClosedLoop(", "    import ns3_bridge\n    if a.cl_mode != 'ns3': import cl_env\n    CL = ns3_bridge.NS3Loop(a) if a.cl_mode == 'ns3' else cl_env.ClosedLoop(")
sub("if CL and a.cl_mode == 'enforcer':\n            fb_cl = CL.feedback(obs)", "if CL and a.cl_mode in ('enforcer', 'ns3'):\n            fb_cl = CL.feedback(obs)")
shutil.copy(p, p + '.bak_ns3'); open(p, 'w', encoding='utf-8').write(src); print('patched', p, '(backup', p + '.bak_ns3)')
