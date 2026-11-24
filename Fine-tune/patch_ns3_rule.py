r"""Adds --ns3_rule / --ns3_rule_mode / --ns3_persist to an ooda_loop.py that already has patch_ns3_loop.py applied (the bridge reads them).
Usage (PowerShell, from Fine-tune):  .\ft\Scripts\python.exe patch_ns3_rule.py ..\OODA\ooda_loop.py      (backup: ooda_loop.py.bak_rule)"""
import sys, shutil
p = sys.argv[1]; src = open(p, encoding='utf-8').read()
if "--ns3_rule" in src: sys.exit('already patched for the rule')
anchor = "ap.add_argument('--ns3_correct',"
if src.count(anchor) != 1: sys.exit(f'anchor not found / not unique ({src.count(anchor)}x): apply patch_ns3_loop.py first')
new = ("ap.add_argument('--ns3_rule', default=None, help='rsrp_teacher.json: the RSRP rule sets every cell\\'s power depth (see ns3_bridge.py)')\n"
       "ap.add_argument('--ns3_rule_mode', default='protect', choices=['protect', 'rule'], help=\"protect = rule, but non-baseline cells never cut deeper than the OODA policy; rule = rule only\")\n"
       "ap.add_argument('--ns3_persist', type=int, default=2, help='windows a new rule target must persist before it is applied')\n")
src = src.replace(anchor, new + anchor)
shutil.copy(p, p + '.bak_rule'); open(p, 'w', encoding='utf-8').write(src); print('patched', p, '(backup', p + '.bak_rule)')
