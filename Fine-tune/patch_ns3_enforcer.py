r"""Adds --ns3_enforcer to an ooda_loop.py that already has patch_ns3_loop.py applied (the bridge reads it; the enforcer folder/weights/kind/seed come from the
existing --cl_enforcer_dir, --cl_runs, --cl_enforcer (b2s|b2), --cl_agent_seed). Usage (PowerShell, from Fine-tune):
  .\ft\Scripts\python.exe patch_ns3_enforcer.py ..\OODA\ooda_loop.py      (backup: ooda_loop.py.bak_enf)"""
import sys, shutil
p = sys.argv[1]; src = open(p, encoding='utf-8').read()
if "--ns3_enforcer" in src: sys.exit('already patched for the enforcer')
anchor = "ap.add_argument('--ns3_correct',"
if src.count(anchor) != 1: sys.exit(f'anchor not found / not unique ({src.count(anchor)}x): apply patch_ns3_loop.py first')
new = "ap.add_argument('--ns3_enforcer', action='store_true', help='DRL enforcer in the ns-3 loop: its power residual vs the OODA policy is added to each cell\\'s offset (see ns3_enforcer.py)')\n"
src = src.replace(anchor, new + anchor)
shutil.copy(p, p + '.bak_enf'); open(p, 'w', encoding='utf-8').write(src); print('patched', p, '(backup', p + '.bak_enf)')
