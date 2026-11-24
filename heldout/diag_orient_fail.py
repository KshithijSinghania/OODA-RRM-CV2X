r"""Why did 6 Orient outputs fail validation?  Run on HP from Fine-tune folder:
  .\ft\Scripts\python.exe ..\OODA\heldout\diag_orient_fail.py --codebook "..\OODA_cov\codebook2c.json" --trace loop_S7_cov.jsonl loop_S5_cov.jsonl
"""
import json, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--codebook'); ap.add_argument('--trace',nargs='+'); a=ap.parse_args()
cb=json.load(open(a.codebook))
print('codebook keys:',list(cb.keys()))
for k,v in cb.items():
    try: print(' ',k,'len',len(v))
    except: pass
def get(name,i):
    for k in cb:
        if name in k.lower():
            v=cb[k]
            try: return v[i] if isinstance(v,list) else v[str(i)]
            except Exception as e: return f'<index {i} missing: {e}>'
    return '<no table matching '+name+'>'
for f in a.trace:
    for l in open(f):
        t=json.loads(l)
        if t.get('orient_ok') is False:
            raw=t['raw']; ia,ib=int(raw[:3]),int(raw[3:])
            print('\n',t['obs_id'],'raw',raw)
            print('  orient_a[%d] ='%ia, json.dumps(get('orient_a',ia))[:400])
            print('  orient_b[%d] ='%ib, json.dumps(get('orient_b',ib))[:400])
            print('  extra fields:',{k:str(v)[:200] for k,v in t.items() if k not in('sec','tok','raw')})
